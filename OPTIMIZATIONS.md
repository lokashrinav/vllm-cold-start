# Foundry CUDA Graph Serialization — Complete Technical Guide

## Context

This document describes **everything** the Foundry integration does to
eliminate CUDA graph capture overhead from vLLM cold starts. The target
workload is **Qwen/Qwen2.5-7B-Instruct** on **NVIDIA A100 GPUs** running on
**Modal**, benchmarked with `python -m modal run profiling/modal_foundry_benchmark.py`.

All code lives in `src/vllm_profile_cache/foundry_graphs.py` unless stated
otherwise.

---

## Table of Contents

1. [What problem does this solve?](#1-what-problem-does-this-solve)
2. [How Foundry's deterministic VMM works](#2-how-foundrys-deterministic-vmm-works)
3. [On-disk cache structure](#3-on-disk-cache-structure)
4. [Cache key computation and invalidation](#4-cache-key-computation-and-invalidation)
5. [The entry point: `cached_vllm_init_with_foundry`](#5-the-entry-point-cached_vllm_init_with_foundry)
6. [Region setup: `setup_foundry_regions`](#6-region-setup-setup_foundry_regions)
7. [The `_FoundryPatchState` dataclass](#7-the-_foundrypatchstate-dataclass)
8. [The SHM pre-copy mechanism](#8-the-shm-pre-copy-mechanism)
9. [How Foundry graph loading works internally (C++)](#9-how-foundry-graph-loading-works-internally-c)
10. [Every monkey-patch explained](#10-every-monkey-patch-explained)
11. [The save path: `_finalize_save`](#11-the-save-path-_finalize_save)
12. [The SIGABRT handler](#12-the-sigabrt-handler)
13. [Guard flag deduplication](#13-guard-flag-deduplication)
14. [Architecture: full save path call tree](#14-architecture-full-save-path-call-tree)
15. [Architecture: full load path call tree](#15-architecture-full-load-path-call-tree)
16. [Optimization deep-dives](#16-optimization-deep-dives)
17. [Passthrough recording & address patching](#17-passthrough-recording--address-patching)
18. [Multi-GPU support](#18-multi-gpu-support)
19. [Abandoned optimizations](#19-abandoned-optimizations)
20. [Timeline visualizations](#20-timeline-visualizations)
21. [Final results](#21-final-results)
22. [Key files modified](#22-key-files-modified)

---

## 1. What problem does this solve?

Every vLLM cold start does three expensive things:

| Step | What it does | Time (A100, 7B model) |
|------|--------------|-----------------------|
| `determine_available_memory()` | Full forward pass (`profile_run`) + `profile_cudagraph_memory()` to measure VRAM usage | ~16 s |
| `compile_or_warm_up_model()` | Triton JIT warmup, kernel warmup, `_dummy_run`, `_dummy_sampler_run` | ~10 s |
| `capture_model()` | Captures 35 CUDA graphs (one per batch size) via `_warmup_and_capture` x 35 | ~3 s |

**Total: ~30 s** of `init_engine` time, every single cold start.

Foundry eliminates this by **serializing** the CUDA graphs to disk on the first
cold start, then **deserializing** them on subsequent starts. The integration
also skips the profiling forward pass and warmup passes since their results are
cached in metadata.

---

## 2. How Foundry's deterministic VMM works

CUDA graphs encode GPU memory addresses into their node operations (memcpy
source/destination, kernel arguments). For serialized graphs to replay
correctly, every GPU tensor must be at the **exact same address** as when the
graph was captured.

Foundry achieves this with a **deterministic bump allocator**:

```
set_allocation_region(base=0x10000000000, size=36GB)
```

This does:
1. Reserves a 36 GB contiguous block of GPU virtual address space starting at
   `0x10000000000` (64 GB offset — safe for A100's ~128 TB VA range).
2. Hooks `cuMemAlloc` via `LD_PRELOAD=libcuda_hook.so` to intercept all GPU
   memory allocations.
3. Maintains a **bump cursor** starting at `base`. Every `cuMemAlloc(size)`
   call returns `cursor` and advances `cursor += align(size)`.

Because:
- PyTorch model initialization is deterministic (same architecture = same
  allocation sequence)
- The bump cursor starts at the same address every time
- Each allocation gets the same offset

...every tensor ends up at the same GPU address on every run.

### The bump cursor is monotonic

The cursor **only moves forward**. Calling `fdry.set_current_alloc_offset()`
with a value less than the current cursor is rejected:
```
[HOOK] WARNING: New offset 0x200000 is less than current offset 0x400000, skipping
```
This is critical — it means we can advance the cursor to skip profiling phases,
but we can never undo an allocation. Any "temporary" GPU allocation permanently
shifts the cursor. (This constraint drove the abandonment of the CUDA graph
driver warmup optimization — see section 17.)

### Key Foundry APIs used

| API | Purpose |
|-----|---------|
| `fdry.set_allocation_region(base, size)` | Start the bump allocator at `base`. Must be called before any `cuMemAlloc`. |
| `fdry.get_current_alloc_offset()` | Read current cursor position (bytes from start of region). |
| `fdry.set_current_alloc_offset(offset)` | Advance cursor forward to `offset`. Rejects backward moves. |
| `fdry.preallocate_region(size)` | Map `size` bytes of virtual addresses with a single `cuMemCreate` + `cuMemMap`. Makes subsequent `cuMemAlloc` a fast pointer bump (no driver call needed). Also required for `cuGraphAddMemcpyNode` address validation. |
| `fdry.parse_size("36GB")` | Parse human-readable size string to bytes. |
| `fdry.CUDAGraph()` | Drop-in replacement for `torch.cuda.CUDAGraph`. Captures graph topology + kernel params via Foundry's hook. |
| `fdry.graph(g)` | Context manager: calls `g.capture_begin()` on enter, `g.capture_end()` on exit. |
| `fdry.CUDAGraph.start_graph_builds(paths, num_threads)` | Async: spawns background C++ thread for Phase 1 (file I/O + parse) + Phase 2a (template building). Returns a `PendingGraphLoads` handle. Returns to Python in ~13 ms. |
| `fdry.CUDAGraph.finish_graph_loads(pending)` | Blocks on `PendingGraphLoads.build_complete_` future, then runs Phase 2b (allocator replay) + Phase 2c (output tensor reconstruction). Returns list of `(graph, output)` pairs. |
| `fdry.pack_fatbins_to_folder(dir)` | Save CUDA kernel binaries (fatbins) that the hook intercepted during capture. These are needed for `query_function_handle()` during graph rebuilding. |
| `fdry.set_pack_fatbins_on_exit(False)` | Disable auto-packing on process exit (we pack eagerly after each save). |
| `fdry.stop_allocation_region()` | Temporarily disable the bump allocator hook (`tls_storage.enabled = false` in `hook.cpp:2924`). Subsequent allocations go through the default CUDA allocator. |
| `fdry.resume_allocation_region()` | Re-enable the bump allocator hook (`tls_storage.enabled = true` in `hook.cpp:2939`). |

---

## 3. On-disk cache structure

```
/root/.cache/foundry-graphs/<cache_key>/
  metadata.json               # Model ID, cache key, region base, cursor deltas,
                               # kv_cache_memory, graph descriptors, timestamps
  .save_complete               # Marker file — only written after all graphs saved
  wrapper_graphs/
    graph_0.json               # Per-graph: topology (nodes, edges), allocator events,
    graph_0.cugraph            #   output tensor metadata (JSON) + binary kernel data
    graph_1.json               #   (.cugraph: raw struct data — kernel params, arg
    graph_1.cugraph            #    buffers, function hashes, all as binary memcpy)
    ...
    graph_34.json
    graph_34.cugraph
    graph_manifest.json        # Pre-computed topology groups: which graphs share
                               #   the same node structure (template assignment)
  hook_archive/
    <fatbin_hash>.fatbin        # Packed CUDA kernel binaries intercepted by the hook.
    ...                         # Used by query_function_handle() during rebuild to
                               #   resolve CUfunction/CUkernel handles from saved hashes.
```

### `metadata.json` contents

```json
{
  "model_id": "Qwen/Qwen2.5-7B-Instruct",
  "num_graphs": 35,
  "cache_key": "a1b2c3d4e5f67890",
  "region_size": "36GB",
  "region_base": 68719476736,
  "pre_capture_offset": 15032385536,
  "cursor_after_graph0": 15098920960,
  "cursor_positions": [15032385536, 15032385536, ...],
  "profile_cudagraph_estimate": 12345678.0,
  "profile_cudagraph_cursor_delta": 67108864,
  "available_kv_cache_memory": 9876543210,
  "determine_cursor_delta": 268435456,
  "passthrough_events": [
    {"source": "cuMemAlloc_v2", "ptr": 140234567890, "size": 2097152},
    {"source": "cuMemAlloc_v2", "ptr": 140236665042, "size": 4194304},
    ...
  ],
  "descs": [
    {
      "repr": "BatchDescriptor(num_tokens=1, num_reqs=1, uniform=True, ...)",
      "cg_mode": "FULL",
      "num_tokens": 1,
      "num_reqs": 1,
      "uniform": true
    },
    ...
  ],
  "created_at": 1715000000.0
}
```

Fields used during load:
- `region_base` — passed to `setup_foundry_regions(force_base=...)` so the bump
  allocator starts at the same address
- `available_kv_cache_memory` — returned by the skipped `determine_available_memory()`
- `determine_cursor_delta` — how far to advance the cursor to simulate the
  profiling phase
- `profile_cudagraph_estimate` — returned by the skipped `profile_cudagraph_memory()`
- `profile_cudagraph_cursor_delta` — how far to advance the cursor to simulate
  graph memory estimation
- `pre_capture_offset` — cursor position just before graph capture started
- `cursor_after_graph0` — cursor position after first graph was captured

### `.save_complete` marker

This file is only written **after all graphs are saved and metadata is flushed**.
If the process crashes during save (e.g., due to the CGE BUILD thread aborting),
the marker won't exist. On next run, `is_load = has_any and pre_cache.is_save_complete()`
will be `False`, forcing a fresh save.

### Binary format (`.cugraph` files)

The `.cugraph` files use a compact binary format (`binary_format` namespace in
`CUDAGraphParallel.cpp`). Structure:

```
[Header: 64 bytes]
  magic, version, num_nodes, flags, section offsets

[Node Table: num_nodes * sizeof(BinNodeEntry)]
  Per node: type (KERNEL/MEMCPY/MEMSET/EVENT_RECORD/EVENT_WAIT),
            node_id, dependency count
  For KERNEL nodes: blockDim, gridDim, sharedMemBytes,
                    function_name offset/length, binary_hash,
                    param_index_offset, arg_buffer_offset/size,
                    kernel_node_attrs flags

[Param Index + Param Data]
  Per-kernel parameter table: offset + size for each param
  Raw kernel parameter bytes (direct memcpy, no hex encoding)

[Arg Buffer Data]
  CUDA launch param buffers (CU_LAUNCH_PARAM_BUFFER_POINTER)

[String Table]
  Null-terminated function names

[Common Kernel Attrs (optional)]
  Shared attributes applied to all kernel nodes in the graph

[Topology Key (optional)]
  Pre-computed topology signature for template matching
```

The binary format is ~10x faster than the JSON path because:
- No JSON parsing for node data
- No hex decode for kernel parameters (raw `memcpy` instead)
- Direct struct reads with known offsets

---

## 4. Cache key computation and invalidation

`build_graph_cache_key()` (line 151) produces a SHA-256 hash (truncated to 16
hex chars) from:

```python
payload = {
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "region_size": "36GB",
    "vllm_version": "0.20.1",
    "torch_version": "2.6.0",
    "cuda_version": "12.4",
    "kwargs": {
        "dtype": "auto",
        "gpu_memory_utilization": 0.8,
        "max_model_len": 4096,
        "max_num_batched_tokens": 4096,
        "max_num_seqs": 256,
        "tensor_parallel_size": 1,
        "enforce_eager": false,
        ...
    }
}
```

The cache auto-invalidates when **any** of these change:
- Model name
- vLLM version (graph capture logic changes between versions)
- PyTorch version (kernel binary format changes)
- CUDA version (driver API changes)
- Any vLLM config kwarg that affects graph shape (batch sizes, memory
  utilization, max sequence length, etc.)

The kwargs checked are defined in `GRAPH_CACHE_KEY_KWARGS` (line 62):
```python
GRAPH_CACHE_KEY_KWARGS = (
    "dtype", "gpu_memory_utilization", "max_model_len",
    "max_num_batched_tokens", "max_num_seqs", "tensor_parallel_size",
    "pipeline_parallel_size", "enforce_eager", "kv_cache_dtype",
    "quantization",
)
```

### Region base mismatch handling

If the saved `region_base` (e.g., `0x10000000000`) is unavailable on the new
machine (another process reserved that VA range), the code:
1. Clears the graph cache (addresses won't match)
2. Tries alternative addresses from `CANDIDATE_REGION_BASES`
3. Falls back to standard vLLM if all fail

```python
CANDIDATE_REGION_BASES = [
    0x10000000000,   # 64 GB (primary)
    0x8000000000,    # 32 GB
    0x18000000000,   # 96 GB
    0x4000000000,    # 16 GB
    0x20000000000,   # 128 GB
]
```

---

## 5. The entry point: `cached_vllm_init_with_foundry`

This is the drop-in replacement for `vllm.LLM()`. Here's the complete flow
(line 1037):

```
cached_vllm_init_with_foundry(model, graph_cache_dir, region_size, **vllm_kwargs)
  │
  ├── 1. is_foundry_available()?
  │     Check: `import foundry` works AND `LD_PRELOAD` contains `libcuda_hook.so`
  │     If no: fall back to standard `vllm.LLM()`, no graph caching
  │
  ├── 2. resolve_graph_cache_dir()
  │     Compute cache_key = sha256(model + versions + kwargs)[:16]
  │     effective_cache_dir = f"{graph_cache_dir}/{cache_key}"
  │
  ├── 3. Load metadata and determine save vs load
  │     pre_cache = FoundryGraphCache(effective_cache_dir)
  │     saved_meta = pre_cache.load_metadata()
  │     saved_base = saved_meta["region_base"]  # GPU VA base from save run
  │     has_any = has_cached_graphs() or has_cached_wrapper_graphs()
  │
  │     Safety check: if graphs exist but metadata has no region_base
  │     AND .save_complete exists, the cache was saved with non-deterministic
  │     addresses (old version). Clear it.
  │
  ├── 4. setup_foundry_regions()
  │     Load path: force_base=saved_base (must match save run)
  │     Save path: try CANDIDATE_REGION_BASES in order
  │     If saved_base fails: clear cache, try alternatives
  │     If all fail: fall back to standard vLLM
  │
  ├── 5. Determine save vs load mode
  │     is_load = has_any AND is_save_complete() AND NOT force_save
  │     If force_save: clear existing cache
  │
  ├── 6. Extract saved profiling data from metadata
  │     pcg_est = saved_meta["profile_cudagraph_estimate"]
  │     pcg_delta = saved_meta["profile_cudagraph_cursor_delta"]
  │     kv_mem = saved_meta["available_kv_cache_memory"]
  │     det_delta = saved_meta["determine_cursor_delta"]
  │
  ├── 7. patch_vllm_for_foundry()
  │     Installs all monkey-patches (Patches 0, 0b, 1, 1b, 1b2, 1b3, 2, 3)
  │     Starts SHM pre-copy thread (load mode only)
  │     Returns _FoundryPatchState
  │
  ├── 8. Restore cursor metadata from save run
  │     state.saved_offset_for_load = saved_meta["pre_capture_offset"]
  │     state.cursor_after_graph0 = saved_meta["cursor_after_graph0"]
  │     state.cursor_positions = saved_meta["cursor_positions"]
  │
  ├── 9. Register atexit(state.finalize)
  │     Ensures metadata is written even if LLM() construction fails
  │
  └── 10. LLM(model, **vllm_kwargs)
        All patches are active. See sections 14/15 for full call trees.
        Returns the constructed LLM instance.
```

---

## 6. Region setup: `setup_foundry_regions`

Line 375. Called once before any GPU allocation.

```python
def setup_foundry_regions(region_size="36GB", force_base=None):
```

Step by step:

1. **Disable PyTorch expandable segments:**
   ```python
   os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")
   ```
   PyTorch's expandable segments feature uses `cuMemAddressReserve` internally,
   which can conflict with Foundry's `set_allocation_region` (both try to
   reserve contiguous VA ranges). Disabling it ensures PyTorch's caching
   allocator falls through to `cuMemAlloc`, which Foundry intercepts.

2. **Initialize CUDA context:**
   ```python
   if not torch.cuda.is_initialized():
       torch.cuda.init()
   ```
   Foundry's `set_allocation_region` needs an active CUDA context to reserve
   GPU virtual address space.

3. **Try candidate base addresses:**
   ```python
   for base in bases_to_try:
       fdry.set_allocation_region(base, size_bytes)
   ```
   On the load path, `force_base=saved_base` means only one address is tried
   (it must match the save run). On the save path, all candidates are tried.

4. **Verify the region works:**
   ```python
   test = torch.empty(1024, device="cuda")
   ptr = test.data_ptr()
   in_region = used_base <= ptr < (used_base + size_bytes)
   ```
   Allocates a small test tensor and checks its address falls within the
   reserved region. If it doesn't, the hook isn't working (e.g., wrong
   `LD_PRELOAD`, or another allocator took precedence).

5. **Disable auto fatbin packing:**
   ```python
   fdry.set_pack_fatbins_on_exit(False)
   ```
   By default, Foundry packs fatbins in an `atexit` handler. We pack them
   eagerly after each graph save instead (see section 11), so the atexit
   handler is unnecessary and can race with process shutdown.

6. **Install SIGABRT handler:**
   ```python
   _install_sigabrt_handler()
   ```
   See section 12.

---

## 7. The `_FoundryPatchState` dataclass

Line 220. This is the shared mutable state that all patches read and write.
One instance is created per `patch_vllm_for_foundry()` call.

```python
@dataclass
class _FoundryPatchState:
    # ---- Identity ----
    cache: FoundryGraphCache       # Disk cache manager
    model_id: str                  # e.g., "Qwen/Qwen2.5-7B-Instruct"
    cache_key: str                 # SHA-256 hash prefix
    region_size: str               # e.g., "36GB"
    region_base: Optional[int]     # GPU VA base address (e.g., 0x10000000000)

    # ---- Save state ----
    saved_descs: list[Any]         # BatchDescriptors of saved graphs (save mode)
    save_seconds: float            # Wall time for all graph.save() calls
    finalized: bool                # True after _finalize_save() runs

    # ---- Load state ----
    loaded_graphs: list[tuple]     # (graph, output) pairs consumed during load
    _graph_files: Optional[list]   # Sorted list of graph_N.json paths on disk
    load_index: int                # Next graph index to serve from cache
    load_seconds: float            # Wall time for graph loading
    _preloaded_graphs: Optional[list]  # All (graph, output) from CUDAGraph.load()

    # ---- Cursor tracking (save mode, stored in metadata) ----
    pre_capture_offset: Optional[int]        # Cursor just before capture_model
    cursor_after_graph0: Optional[int]       # Cursor after first graph captured
    cursor_positions: list[int]              # Cursor before each graph capture
    saved_offset_for_load: Optional[int]     # pre_capture_offset from metadata

    # ---- Profiling data (recorded in save, replayed in load) ----
    profile_cudagraph_estimate: Optional[float]   # Return value of profile_cudagraph_memory()
    profile_cudagraph_cursor_delta: Optional[int] # Cursor delta from profile_cudagraph_memory()
    available_kv_cache_memory: Optional[int]      # Return value of determine_available_memory()
    determine_cursor_delta: Optional[int]         # Cursor delta from determine_available_memory()

    # ---- Passthrough recording (non-Foundry allocations) ----
    passthrough_events: Optional[list]     # Save/load: recorded non-bump allocations
    _extended_recording: bool              # True while passthrough recording spans init→capture

    # ---- Guard flags (prevent duplicate work across patches) ----
    _preallocated_early: bool              # True after preallocate_region() called
    _preparse_pending: Optional[Any]       # PendingGraphLoads handle (or None)
    _preparse_iter: Optional[Any]          # (unused legacy field)

    # ---- Misc ----
    manager_handled: bool          # Legacy: CudaGraphManager path handled
    _dummy_blocks: list            # Legacy: dummy allocation blocks
```

### Key methods

**`load_all()`** — Populates `_graph_files` by globbing `wrapper_graphs/graph_*.json`
and sorting by index number. Idempotent (returns immediately if already called).

**`next_loaded_graph()`** — Returns the next `(graph, output)` pair from
`_preloaded_graphs`. Increments `load_index`. Used by Patch 2's load path
(CUDAGraphWrapper.__call__) as a sequential iterator.

**`save_graph(desc, graph, output)`** — Saves one graph to disk, packs fatbins
immediately, and writes metadata (including passthrough events). Called during
the save path for each captured graph. Eagerly flushes everything to survive
potential SIGABRT.

**`finalize()`** — Final metadata write (including passthrough events) + logging.
Registered as an `atexit` handler via `atexit.register(state.finalize)` in the
entry point.

---

## 8. The SHM pre-copy mechanism

Line 493. Graph cache files may live on a network-mounted volume (e.g., Modal
Volume backed by S3). Reading them directly during `start_graph_builds` would
add network latency to the C++ I/O thread.

```python
if _is_load_mode and cache.has_cached_wrapper_graphs():
    state.load_all()
    if state._graph_files:
        _shm_dir = Path("/dev/shm/foundry_graphs")

        def _precopy():
            _shm_dir.mkdir(parents=True, exist_ok=True)
            # Copy manifest
            manifest = src_dir / "graph_manifest.json"
            if manifest.exists():
                shutil.copy2(str(manifest), str(_shm_dir / manifest.name))
            # Copy each graph's JSON + binary
            for src in state._graph_files:
                cg = src.with_suffix(".cugraph")
                if cg.exists():
                    shutil.copy2(str(cg), str(_shm_dir / cg.name))
                shutil.copy2(str(src), str(_shm_dir / src.name))
            state._shm_graph_dir = _shm_dir

        state._precopy_thread = threading.Thread(target=_precopy, daemon=True)
        state._precopy_thread.start()
```

**What gets copied:**
- `graph_manifest.json` — topology group assignments (which graphs share templates)
- `graph_N.json` — per-graph metadata, allocator events, output tensor info
- `graph_N.cugraph` — per-graph binary kernel data

**Timing:** The thread starts during `patch_vllm_for_foundry()`, which runs
BEFORE `LLM()` construction. By the time Patch 3's `_hooked_load_weights`
calls `precopy_thread.join()`, the model architecture has already been
initialized (several seconds). The SHM copy is typically done by then.

**Why `/dev/shm`:** This is Linux's tmpfs — backed by kernel memory, no disk
I/O. File reads from tmpfs are essentially `memcpy` from kernel page cache.
For the C++ thread in `start_graph_builds` that reads graph files in parallel,
this eliminates any storage latency.

**Fallback:** If the copy fails (e.g., `/dev/shm` doesn't exist or is full),
`state._shm_graph_dir` is set to `None`, and `start_graph_builds` reads from
the original cache directory paths instead.

---

## 9. How Foundry graph loading works internally (C++)

File: `foundry/csrc/CUDAGraphParallel.cpp`

`start_graph_builds()` (line ~2000) and `finish_graph_loads_impl()` (line ~2402)
form a two-stage pipeline:

### Phase 1: Parse + prepare (in `start_graph_builds`)

**Phase 1a — Parallel file I/O:**
The C++ code reads all graph files in parallel using a `SimpleThreadPool` with
`num_threads` workers. For each graph:
1. Read the `.json` file (or `.cugraph` binary if available)
2. Parse graph metadata: node count, node types, topology key
3. Extract `allocator_events` and `output_tensors` from the JSON root
4. Create a `CUDAGraph` object with the parsed data

**Phase 1b — Topology grouping:**
Graphs with identical node structure (same count and types of kernel/memcpy/
memset nodes) are grouped into **topology groups**. The first graph in each
group becomes the **template** — it gets fully instantiated. All other graphs
in the group are **on-demand** — they reuse the template's instantiated graph
structure and only need parameter updates at replay time.

For Qwen2.5-7B-Instruct: 35 graphs collapse into **5 topology groups** (7
graphs each, same architecture but different batch sizes).

The grouping is read from `graph_manifest.json` if available (pre-computed
during save). Otherwise it's computed from node data:
```
topology_key = "K:423:S64:S64:..." (node type + count + shared mem for each kernel)
```

**Phase 1 returns to Python in ~13 ms** — all slow work is deferred to Phase 2.

### Phase 2a: Template building (background C++ thread)

A detached `std::thread` is spawned to do the actual CUDA driver work:

```cpp
std::thread bg_thread([...] {
    // Phase 2a: Build templates sequentially
    for (auto& [key, indices] : topology_groups) {
        size_t tmpl_idx = indices[0];

        if (bin_files[tmpl_idx].valid()) {
            // Binary-native path: ~10x faster than JSON
            result = CUDAGraph::build_graph_from_binary(
                bin_files[tmpl_idx], all_parsed[tmpl_idx].graph,
                main_ctx, &tmpl);
        } else {
            // JSON fallback
            result = CUDAGraph::build_graph_from_parsed(...);
        }

        shared_exec = std::make_shared<SharedGraphExec>();
        result.graph->transfer_to_shared_exec(shared_exec, tmpl);
    }

    // Phase 2c: Link on-demand graphs to their templates
    for (size_t i = 0; i < num; ++i) {
        CUDAGraph::link_on_demand_shared_exec(
            *all_parsed[i].graph, shared_execs.at(template_for[i]));
    }

    build_promise->set_value();  // signals finish_graph_loads
});
bg_thread.detach();
```

**What `build_graph_from_binary` does** (line 822):
For each node in the binary data:
1. `NODE_KERNEL`: Resolve `CUfunction`/`CUkernel` handle from `binary_hash` +
   `function_name` via `query_function_handle()`. Set grid/block dims, shared
   memory, kernel params. Call `cuGraphAddKernelNode()`.
2. `NODE_MEMCPY`: Set src/dst addresses, element size, copy kind. Call
   `cuGraphAddMemcpyNode()`. **This is where address validation happens** —
   src/dst must be mapped GPU addresses.
3. `NODE_MEMSET`: Set dst address, value, size. Call `cuGraphAddMemsetNode()`.
4. `NODE_EVENT_RECORD/WAIT`: Create events, add record/wait nodes.

After all nodes: `cuGraphInstantiate()` — creates an executable graph from the
topology. **This is the expensive step** — 480-760 ms for the first template
(423 nodes), 30-45 ms for subsequent templates.

**On-demand prep runs in parallel with template building:**
While the main thread builds templates sequentially (constrained by CUDA driver
per-device mutex on `cuGraphInstantiate`), a `SimpleThreadPool` prepares
on-demand graphs in parallel. On-demand prep is CPU-only (JSON parse, hex decode,
function handle lookup — no CUDA driver calls), so it doesn't contend with
template building.

### Phase 2b: Allocator replay (in `finish_graph_loads`)

```cpp
// Wait for background build to complete
pending->build_complete_.get();  // blocks until Phase 2a signals

// For each graph:
foundry::replay_hook_events_from_json(alloc_events);
```

`replay_hook_events_from_json` advances the bump allocator cursor to match the
positions recorded during save. This ensures that subsequent allocations (KV
cache, etc.) get the same addresses they had during the save run.

### Phase 2c: Output tensor reconstruction (in `finish_graph_loads`)

```cpp
results.push_back(make_load_result_from_extracted(
    entry.graph, entry.output_tensors_meta, reconstruct_fn));
```

Reconstructs PyTorch tensors from saved metadata (dtype, shape, stride,
data_ptr). These are the output tensors that the model's forward pass would
have returned — they're pre-allocated at the same GPU addresses.

### Generator state registration

```cpp
if (pending->registry && !entry.generators_meta.is_null()) {
    for (auto& gen : generators_meta) {
        auto state = registry->get_state_from_id(state_id, seed);
        entry.graph->register_generator_state(state, wholegraph_increment);
    }
}
```

CUDA graphs that use random number generation (dropout, etc.) need generator
state registered before allocator replay. This happens per-graph in
`finish_graph_loads`, before the allocator events are replayed.

---

## 10. Every monkey-patch explained

### Patch 0: `GPUModelRunner.profile_cudagraph_memory`

**Location:** ~line 990

**What the original does:** Allocates dummy tensors sized to the maximum CUDA
graph memory footprint, measures peak VRAM, then frees them. Returns the
estimated graph memory in bytes.

**Save mode:** Wraps the original to record:
- `profile_cudagraph_estimate` — the memory estimate returned (float, bytes)
- `profile_cudagraph_cursor_delta` — how far the bump cursor advanced (int, bytes)

These are saved to `metadata.json`.

**Load mode:** Replaces entirely:
```python
def _skip_profile_cudagraph(self):
    saved = state.profile_cudagraph_estimate
    delta = state.profile_cudagraph_cursor_delta
    pre = fdry.get_current_alloc_offset()
    if delta is not None and delta > 0:
        fdry.set_current_alloc_offset(pre + delta)
    return saved
```

**Why cursor advancement matters:** `profile_cudagraph_memory()` internally
allocates temporary GPU memory (dummy tensors for graph memory estimation).
Even though those tensors are freed, the bump cursor only moves forward (it's
a bump allocator — `cuMemFree` is a no-op for the cursor). If we skip the
function without advancing the cursor, all subsequent allocations shift to
lower addresses, breaking address matching for graph replay.

The cursor delta is typically 64-256 MB depending on the model and graph
configuration.

---

### Patch 0b: `GPUWorker.determine_available_memory`

**Location:** ~line 1030

**What the original does:** Runs a full forward pass (`profile_run()`, ~16 s),
then calls `profile_cudagraph_memory()`, measures free VRAM, and returns bytes
available for KV cache.

**Save mode:** Wraps the original to record:
- `available_kv_cache_memory` — bytes available for KV cache (int)
- `determine_cursor_delta` — total cursor movement from profiling (int, bytes)

The Foundry allocator is stopped during profiling (`fdry.stop_allocation_region()`)
so cuBLAS workspace init and other runtime allocations go through the default
CUDA allocator. Then resumed after.

**Load mode:** Runs the original with Foundry disabled, returns saved value:
```python
def _skip_determine(self):
    fdry.stop_allocation_region()
    try:
        result = _original_determine(self)  # side effects happen normally
    finally:
        fdry.resume_allocation_region()
    return _saved_kv_mem  # use saved value for cursor alignment
```

This lets `profile_run()` execute normally (initializing cuBLAS workspaces,
CUDA runtime buffers, etc.) but with the Foundry allocator disabled so those
allocations don't disturb the bump cursor. The saved KV cache memory value
is returned instead of the fresh result, ensuring cursor alignment matches
the save run.

**Why not skip entirely:** cuBLAS workspace initialization happens as a side
effect of `profile_run`. These workspaces get referenced by CUDA graph kernel
parameters (baked into saved graphs). If they don't exist at graph load time,
graph replay faults with XID 31 (GPU memory violation). Running the profiling
with Foundry disabled ensures all runtime state is initialized while keeping
the bump allocator deterministic.

---

### Patch 1: `GPUModelRunner.capture_model`

**Location:** ~line 1084

This is the core patch. In save mode it sets a phase flag so Patch 2 knows
when to intercept captures. In load mode it does the actual graph restoration.

**Save mode flow:**
1. Set `_capture_phase[0] = True` — signals Patch 2 to intercept
2. Record `pre_capture_offset` (cursor before any graph capture)
3. `torch.cuda.synchronize()` + `torch.cuda.empty_cache()` — flush pending work
4. Call original `capture_model()` (runs `_warmup_and_capture` x 35)
5. Set `_capture_phase[0] = False`
6. Call `_finalize_save()` — see section 11

**Load mode flow:**
1. **NCCL warmup** (if multi-GPU and extended recording active):
   ```python
   if state._extended_recording and _is_load_mode:
       if torch.distributed.is_initialized():
           warmup = torch.zeros(1024, device=f"cuda:{dev}")
           torch.distributed.all_reduce(warmup)
   ```
   This captures NCCL's internal allocations as passthrough events before
   recording ends.
2. **End extended passthrough recording** — capture all non-Foundry allocation
   events from init_device through this point.
3. **Build address remap table** from save/load passthrough events:
   ```python
   addr_ranges = _build_passthrough_addr_map(save_pt, load_pt)
   ```
4. **Patch graph JSON addresses** — binary search each kernel parameter
   against the remap table:
   ```python
   patched_path, n_patches = _patch_graph_json_addresses(path, addr_ranges)
   ```
5. **Pre-map non-Foundry addresses** — scan graphs for pointer-sized values
   outside the Foundry region, reserve + map physical memory via raw CUDA VMM:
   ```python
   _premap_non_foundry_addresses(load_paths, dev, region_base, region_size)
   ```
6. **Load each graph** via `fdry.CUDAGraph.load(path, pool)` sequentially.
7. **Direct-populate** — bypass the entire capture loop:
   ```python
   wrapper = self.model  # CUDAGraphWrapper instance
   graph_idx = 0
   for _, batch_descs in self.cudagraph_dispatcher.get_capture_descs():
       for desc in batch_descs:
           graph, output = state._preloaded_graphs[graph_idx]
           entry = CUDAGraphEntry(batch_descriptor=desc)
           entry.cudagraph = graph
           entry.output = output
           wrapper.concrete_cudagraph_entries[desc] = entry
           graph_idx += 1
   ```
8. Return 0 (no memory used for capture)

**Direct-populate explained:** Normally `capture_model()` calls
`_warmup_and_capture()` for each of 35 batch sizes. Each call does N warmup
forward passes then one capture pass (~23 ms each x 35 = ~0.8 s). Direct-populate
skips the entire capture loop by writing entries directly into the wrapper's
`concrete_cudagraph_entries` dictionary. The `BatchDescriptor` is a frozen
dataclass (hashable via `@dataclass(frozen=True)`) with fields: `num_tokens`,
`num_reqs`, `uniform`, `has_lora`, `num_active_loras`. It serves as the dict key.

The iteration order from `cudagraph_dispatcher.get_capture_descs()` matches
the save order, so `graph_idx` maps 1:1 to the saved graph files.

---

### Patch 1b: `GPUWorker.compile_or_warm_up_model`

**Location:** ~line 1279

**What the original does** (vLLM `gpu_worker.py:668-685`):
1. Compile-size warmup — Triton JIT for various batch sizes
2. `maybe_remove_all_loras()` — remove LoRA adapters during capture
3. `kernel_warmup()` — warm up attention kernels (no-op on A100 + FlashAttention)
4. `capture_model()` — capture CUDA graphs for all batch sizes
5. `_dummy_run(NONE mode)` — one forward pass without graph capture (~50 ms)
6. `_dummy_sampler_run()` — compile sampling Triton kernels

**Both modes:** Wraps the original with timing instrumentation only. The original
function body runs unchanged — all the real work (skipping warmup, loading
graphs, etc.) is handled by the patches on the functions *called by*
`compile_or_warm_up_model`:

```python
def _timed_compile_warmup(self):
    t0 = time.perf_counter()
    result = _original_compile_warmup(self)
    elapsed = time.perf_counter() - t0
    print(f"[FOUNDRY] compile_or_warm_up_model: {elapsed:.2f}s")
    return result
```

In load mode, the heavy lifting happens inside the callees:
- `capture_model()` → Patch 1 loads graphs + direct-populates
- `_warmup_and_capture()` → Patch 1b2 reduces to single pass (safety net)
- `_dummy_sampler_run()` → Patch 1b3 skips Triton JIT compilation

---

### Patch 1b2: Skip warmup passes in capture loop

**Location:** ~line 1305

```python
if _is_load_mode and hasattr(GPUModelRunner, '_warmup_and_capture'):
    _orig_warmup_capture = GPUModelRunner._warmup_and_capture

    def _skip_warmup_capture(self, desc, cudagraph_runtime_mode, **kwargs):
        self._dummy_run(
            desc.num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            uniform_decode=desc.uniform,
            is_graph_capturing=True,
            skip_eplb=True,
            remove_lora=False,
            num_active_loras=getattr(desc, 'num_active_loras', 0),
        )
```

**What the original does:** Runs N warmup forward passes before each graph
capture to stabilize PyTorch's CUDA caching allocator (ensure consistent
allocation patterns). Then runs one capture pass.

**Why skip it:** In load mode, Patch 1 bypasses the capture loop entirely via
direct-populate. This patch is a safety net — if any code path still calls
`_warmup_and_capture` in load mode, it reduces the work to a single pass.

---

### Patch 1b3: Skip `_dummy_sampler_run`

**Location:** ~line 1326

```python
if _is_load_mode:
    def _skip_dummy_sampler_run(self, *args, **kwargs):
        return None
    GPUModelRunner._dummy_sampler_run = _skip_dummy_sampler_run
```

**What the original does:** Compiles Triton sampling kernels (top-k sort on
`vocab_size x 256` logits) for **all possible batch sizes**. This is pure
Triton JIT compilation — ~15 seconds of CPU work generating PTX/SASS.

**Why skip it:** In load mode, we've already skipped `profile_run()` (which
would have compiled some of these kernels). Running `_dummy_sampler_run` would
trigger first-time Triton JIT for ALL batch sizes (15 s). Instead, kernels
compile lazily on first real inference — only the variants actually needed
compile. Cost: ~1.4 s added to first inference instead of 15 s to init.

**The tradeoff:**
- Save: 15 s from init_engine
- Cost: +1.4 s on first inference (only actually-needed kernel variants compile)
- Net: 13.6 s improvement

---

### Patch 2: `CUDAGraphWrapper.__call__`

**Location:** ~line 1341

This is the low-level intercept that replaces PyTorch's CUDA graph capture
with Foundry's.

**Filters (no intercept, pass through to original):**
- `_capture_phase[0]` is False — not in capture phase
- `runtime_mode` is `NONE` — no graph capture mode active
- `runtime_mode` is `PIECEWISE` — torch.compile graphs, not our target
- `forward_context` not available — can't determine batch descriptor
- `cudagraph_runtime_mode != self.runtime_mode` — mode mismatch

**Save mode flow:**
1. Record cursor position (`_pre_cursor`)
2. Record input tensor addresses for debugging
3. Call `validate_cudagraph_capturing_enabled()`
4. Create `fdry.CUDAGraph()` — Foundry's drop-in for `torch.cuda.CUDAGraph`
5. Set graph pool ID (for CUDA's internal memory management):
   ```python
   if self.graph_pool is not None:
       cuda_graph_mod.set_graph_pool_id(self.graph_pool)
   else:
       cuda_graph_mod.set_graph_pool_id(
           cuda_graph_mod.current_platform.graph_pool_handle()
       )
   ```
6. Capture the graph:
   ```python
   cuda_graph_mod.get_offloader().sync_prev_onload()
   with fdry.graph(graph):
       output = self.runnable(*args, **kwargs)
       cuda_graph_mod.get_offloader().join_after_forward()
       if self.cudagraph_options.weak_ref_output:
           output = cuda_graph_mod.weak_ref_tensors(output)
   ```
7. Store entry: `entry.output = output`, `entry.cudagraph = graph`
8. Append to `_pending_graphs` for batch save later
9. Record cursor position for metadata

**Load mode flow:**
1. Call `state.next_loaded_graph()` — sequential iterator over `_preloaded_graphs`
2. If graph available: `entry.cudagraph = graph`, `entry.output = output`, return
3. If cache exhausted: set `_load_abandoned = True`, fall back to original capture

**Note:** In the optimized load path (Patch 1 direct-populate), this patch's
load path is NOT used — direct-populate writes entries without going through
`CUDAGraphWrapper.__call__`. This load path exists as a fallback for any code
path that still triggers the capture loop.

---

### Patch 3: Hook `BaseModelLoader.load_model` (currently passthrough)

**Location:** ~line 1500

Hooks `BaseModelLoader.load_model` to wrap `load_weights` with debug logging.
Load-mode only.

```python
if _is_load_mode:
    from vllm.model_executor.model_loader.base_loader import BaseModelLoader
    _original_base_load_weights = BaseModelLoader.load_model

    def _load_model_with_early_builds(self_loader, **kwargs):
        _orig_load_weights = self_loader.load_weights

        def _hooked_load_weights(model, model_config):
            print("[FOUNDRY] Calling _orig_load_weights (no prealloc/early builds)...")
            result = _orig_load_weights(model, model_config)
            return result

        self_loader.load_weights = _hooked_load_weights
        return _original_base_load_weights(self_loader, **kwargs)

    BaseModelLoader.load_model = _load_model_with_early_builds
```

**Current state:** Passthrough — no preallocation or early graph builds. Graph
loading happens entirely in Patch 1 via per-graph `fdry.CUDAGraph.load()`.

**Why the hook remains:** The position (between `initialize_model()` and
`load_weights()`) is ideal for overlapping background work with the ~66s weight
download. If Foundry's `start_graph_builds` / `finish_graph_loads` pipeline is
re-enabled, this hook can fire `preallocate_region()` + `start_graph_builds()`
before weight loading begins, giving Phase 2a 66 seconds of lead time.

**Why `BaseModelLoader.load_model` is the right hook point:**

`BaseModelLoader.load_model()` (in `vllm/model_executor/model_loader/base_loader.py:43`)
does:
```python
def load_model(self, **kwargs):
    model = self.initialize_model(**kwargs)  # creates architecture, allocates tensors
    self.load_weights(model, model_config)    # downloads + copies weights
    return model
```

After `initialize_model()`, all parameter tensors exist at deterministic GPU
addresses and the bump cursor is at the "post-model-init" position — but weight
data hasn't been downloaded yet (66 s of network I/O ahead).

---

## 11. The save path: `_finalize_save`

Line 914. Called at the end of `capture_model()` in save mode, after all 35
graphs have been captured.

```python
def _finalize_save():
    if not _pending_graphs:
        return

    cache.wrapper_graphs_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    # Save all 35 graphs to disk
    for i, (desc, graph, output) in enumerate(_pending_graphs):
        path = str(cache.wrapper_graphs_dir / f"graph_{i}.json")
        try:
            graph.save(path, output_tensors=output)
        except (TypeError, Exception):
            graph.save(path)
        state.saved_descs.append(desc)

    state.save_seconds = time.perf_counter() - t0

    # Pack fatbins immediately (don't wait for atexit)
    try:
        cache.hook_archive.mkdir(parents=True, exist_ok=True)
        fdry.pack_fatbins_to_folder(str(cache.hook_archive))
    except Exception as e:
        logger.warning("Failed to pack fatbins: %s", e)

    # Write metadata with all cursor positions and profiling data
    cache.save_metadata(
        state.saved_descs,
        model_id=model_id,
        cache_key=cache_key,
        region_size=region_size,
        region_base=state.region_base,
        pre_capture_offset=state.pre_capture_offset,
        cursor_after_graph0=state.cursor_after_graph0,
        cursor_positions=state.cursor_positions,
        profile_cudagraph_estimate=state.profile_cudagraph_estimate,
        profile_cudagraph_cursor_delta=state.profile_cudagraph_cursor_delta,
        available_kv_cache_memory=state.available_kv_cache_memory,
        determine_cursor_delta=state.determine_cursor_delta,
    )
    cache.write_save_complete_marker()
    state.finalized = True
```

**Why `graph.save()` is called with `output_tensors=output`:**
The `output_tensors` parameter tells Foundry to extract tensor metadata
(dtype, shape, stride, data_ptr) from the model's output tensors and serialize
it into the graph file. On load, `finish_graph_loads` uses this metadata to
reconstruct the output tensors at the same GPU addresses.

Without `output_tensors`, the loaded graph has no output — it can replay
(execute the GPU kernels) but the Python side can't read the results. The
`try/except` handles older Foundry versions that don't support this parameter.

**Why fatbins are packed eagerly:**
`fdry.pack_fatbins_to_folder()` saves the CUDA kernel binaries (fatbins) that
the hook intercepted during graph capture. These are needed during load for
`query_function_handle()` to resolve kernel function pointers.

Fatbins are packed eagerly (immediately after graph save) rather than in an
`atexit` handler because the Foundry hook's `[CGE BUILD]` background thread
can `abort()` at any time when it encounters unrecognized kernel hashes. If
we waited for `atexit`, the fatbins might never be packed. Eager packing
ensures they survive process crashes.

**Why `save_graph()` on `_FoundryPatchState` ALSO packs fatbins:**
The `state.save_graph()` method (line 282) packs fatbins after **each individual
graph save** — even more aggressive than `_finalize_save` (which packs once
at the end). This is because the SIGABRT can happen at any moment during the
capture loop. If we save graphs 0-15 and the process aborts during graph 16,
the per-save packing ensures fatbins for graphs 0-15 are on disk.

In practice, `_finalize_save` is the primary save path, but `state.save_graph()`
exists as an extra-safe fallback.

---

## 12. The SIGABRT handler

Line 342. This is a **C-level** signal handler (not Python's `signal` module)
installed via `ctypes`.

```python
def _install_sigabrt_handler():
    libc = ctypes.CDLL(ctypes.util.find_library("c"))
    SIGABRT = 6
    _HANDLER = ctypes.CFUNCTYPE(None, ctypes.c_int)

    @_HANDLER
    def _on_sigabrt(_signum):
        libc.fflush(None)    # flush all stdio buffers
        libc._exit(0)         # immediate exit, skip atexit handlers

    libc.signal(SIGABRT, _on_sigabrt)
    _install_sigabrt_handler._prevent_gc = _on_sigabrt  # prevent GC of callback
```

**Why it exists:**
During graph capture (save mode), Foundry's hook spawns a `[CGE BUILD]`
background thread that processes captured kernel binaries. If a kernel binary
hash doesn't match any known fatbin (e.g., a JIT-compiled Triton kernel with
a new hash), the thread calls `abort()`.

Without the handler, `abort()` raises `SIGABRT`, which terminates the process
with a core dump. Any graph files already written to disk survive, but:
- The process exit code is non-zero (crash)
- Core dumps on Modal can be confusing
- The `.save_complete` marker may not be written

With the handler:
- `fflush(None)` ensures all log output is written
- `_exit(0)` exits cleanly with code 0
- Graph files on disk are intact
- The `.save_complete` marker was written eagerly (in `_finalize_save`)

**Why C-level and not Python-level:**
Python's `signal.signal(SIGABRT, handler)` only works for signals delivered to
the main thread. The `[CGE BUILD]` thread is a native C++ thread (created via
`pthread_create`). When it calls `abort()`, the SIGABRT is delivered to the
calling thread, which Python's signal handler can't intercept. A C-level
handler installed via `libc.signal()` catches it regardless of which thread
raises it.

**`_prevent_gc`:**
The `@_HANDLER` decorator creates a C callback via `ctypes`. If the Python
reference to this callback is garbage collected, the function pointer becomes
dangling and the next SIGABRT would segfault. Storing it as a function attribute
prevents GC.

---

## 13. Vestigial guard flags

`_FoundryPatchState` still carries two guard flags from an earlier architecture
where graph pre-loading could be triggered from multiple locations:

```python
state._preallocated_early: bool         # No longer read/written (always False)
state._preparse_pending: Optional[Any]  # Set to None after graph loading (cleanup only)
```

In the current architecture, graph loading happens in a single location:
**Patch 1** (`_flagged_capture_model`) handles the entire load flow —
passthrough matching, address patching, per-graph `fdry.CUDAGraph.load()`,
and direct-populate into `CUDAGraphWrapper.concrete_cudagraph_entries`.

**Patch 3** (`_load_model_with_early_builds`) is currently a **passthrough** —
it hooks `BaseModelLoader.load_model` and wraps `load_weights` with debug
logging but performs no preallocation or early graph builds:

```python
def _hooked_load_weights(model, model_config):
    print("[FOUNDRY] Calling _orig_load_weights (no prealloc/early builds)...")
    result = _orig_load_weights(model, model_config)
    return result
```

The guard flags remain in the dataclass for future use (the Patch 3 hook is
positioned to overlap graph template building with weight download if the
Foundry `start_graph_builds` / `finish_graph_loads` pipeline is re-enabled).

---

## 14. Architecture: full save path call tree

### Single GPU (save)

```
cached_vllm_init_with_foundry(model="Qwen/Qwen2.5-7B-Instruct")
  │
  ├── is_foundry_available() → True
  ├── resolve_graph_cache_dir() → cache_key="a1b2c3d4"
  ├── FoundryGraphCache(dir) → has_cached=False
  ├── setup_foundry_regions("36GB")
  │     ├── PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
  │     ├── torch.cuda.init()
  │     ├── fdry.set_allocation_region(0x10000000000, 36GB)
  │     ├── Verify: torch.empty(1024, device="cuda").data_ptr() in region
  │     ├── fdry.set_pack_fatbins_on_exit(False)
  │     └── _install_sigabrt_handler()
  │
  ├── patch_vllm_for_foundry(load_mode=False)
  │     ├── Patch 0: instrument profile_cudagraph_memory (record result + cursor delta)
  │     ├── Patch 0b: instrument determine_available_memory
  │     │     └── fdry.stop_allocation_region() → run original → fdry.resume_allocation_region()
  │     │         Records result and cursor delta for metadata
  │     ├── Patch 1: wrap capture_model with phase flag + _finalize_save
  │     ├── Patch 1b: wrap compile_or_warm_up_model with timing
  │     ├── Patch 2: intercept CUDAGraphWrapper.__call__
  │     │     └── Save: fdry.CUDAGraph capture instead of torch.cuda.CUDAGraph
  │     └── (Patches 1b2, 1b3, 3 are load-mode only — not installed)
  │
  ├── atexit.register(state.finalize)
  │
  └── LLM(model, **kwargs)
        ├── init_device()
        │     └── request_memory() — checks free GPU memory
        ├── load_model()
        │     ├── initialize_model() — creates model architecture (~5s)
        │     └── load_weights() — downloads + copies weights (~85s)
        ├── determine_available_memory() ← Patch 0b instruments
        │     ├── fdry.stop_allocation_region() (allow non-deterministic allocations)
        │     ├── profile_run() — full forward pass (~16s)
        │     ├── profile_cudagraph_memory() ← Patch 0 records result + cursor delta
        │     ├── fdry.resume_allocation_region()
        │     └── Records available_kv_cache_memory + determine_cursor_delta
        ├── compile_or_warm_up_model() ← Patch 1b adds timing
        │     ├── kernel_warmup()
        │     ├── capture_model() ← Patch 1 sets phase flag
        │     │     └── _warmup_and_capture x 35 batch sizes
        │     │           └── CUDAGraphWrapper.__call__ ← Patch 2 intercepts
        │     │                 ├── fdry.CUDAGraph() — create Foundry graph
        │     │                 ├── with fdry.graph(g): runnable(*args) — capture
        │     │                 └── _pending_graphs.append((desc, graph, output))
        │     ├── _finalize_save() ← called by Patch 1 after capture
        │     │     ├── graph.save(path, output_tensors=output) x 35
        │     │     ├── fdry.pack_fatbins_to_folder(hook_archive)
        │     │     ├── cache.save_metadata(cursor deltas, profiling data)
        │     │     └── cache.write_save_complete_marker()
        │     ├── _dummy_run(NONE) — forward pass without graphs
        │     └── _dummy_sampler_run() — compile sampling kernels
        └── Ready to serve
```

### Multi-GPU (save, tp=2)

```
cached_vllm_init_with_foundry(model="...", tensor_parallel_size=2)
  │
  ├── _cached_vllm_init_multi_gpu()
  │     ├── resolve_graph_cache_dir() → cache_key
  │     ├── Check rank_0/ and rank_1/ caches → global save mode
  │     ├── Patch GPUWorker.init_device → _foundry_init_device
  │     ├── Set disable_custom_all_reduce=True (force NCCL)
  │     │
  │     └── LLM(model, tensor_parallel_size=2)
  │           │
  │           ├── Worker 0 (subprocess):
  │           │     ├── _foundry_init_device(rank=0):
  │           │     │     ├── Per-rank cache: {base_dir}/rank_0/
  │           │     │     ├── setup_foundry_regions()
  │           │     │     ├── patch_vllm_for_foundry(rank_cache, load_mode=False)
  │           │     │     ├── fdry.stop_allocation_region()
  │           │     │     ├── fdry.start_passthrough_record()  ← start recording
  │           │     │     ├── original init_device() → NCCL init, cuBLAS workspace
  │           │     │     ├── fdry.resume_allocation_region()
  │           │     │     └── _extended_recording = True
  │           │     │
  │           │     ├── determine_available_memory() ← Patch 0b
  │           │     │     ├── Pause passthrough recording during profiling (Patch 0)
  │           │     │     ├── fdry.stop_allocation_region() → run original
  │           │     │     ├── fdry.resume_allocation_region()
  │           │     │     └── Resume passthrough recording
  │           │     │
  │           │     └── compile_or_warm_up_model() ← Patch 1b
  │           │           └── capture_model() ← Patch 1
  │           │                 ├── End extended passthrough recording
  │           │                 ├── Save passthrough_events to state
  │           │                 ├── _warmup_and_capture x 35 batch sizes
  │           │                 │     └── CUDAGraphWrapper.__call__ ← Patch 2
  │           │                 │           ├── fdry.CUDAGraph() capture
  │           │                 │           └── _pending_graphs.append(...)
  │           │                 └── _finalize_save()
  │           │                       ├── graph.save() x 35
  │           │                       ├── Save passthrough_events in metadata
  │           │                       └── write_save_complete_marker()
  │           │
  │           └── Worker 1 (subprocess):
  │                 └── (same as Worker 0, with rank_1/ cache)
```

---

## 15. Architecture: full load path call tree

### Single GPU

```
cached_vllm_init_with_foundry(model="Qwen/Qwen2.5-7B-Instruct")
  │
  ├── is_foundry_available() → True
  ├── resolve_graph_cache_dir() → cache_key="a1b2c3d4"
  ├── FoundryGraphCache(dir) → has_cached=True, is_save_complete=True
  ├── load_metadata() → {region_base, kv_mem, cursor deltas, passthrough_events, ...}
  ├── Load CUDA modules from hook_archive (fdry.load_cuda_modules_and_libraries)
  ├── setup_foundry_regions("36GB", force_base=0x10000000000)
  │
  ├── patch_vllm_for_foundry(load_mode=True)
  │     ├── SHM pre-copy thread starts → copies graph files to /dev/shm
  │     ├── Patch 0: skip profile_cudagraph_memory (return saved estimate)
  │     ├── Patch 0b: run determine_available_memory with Foundry disabled, return saved value
  │     ├── Patch 1: capture_model → passthrough matching + addr patching + graph loading + direct-populate
  │     ├── Patch 1b: compile_or_warm_up_model → timing instrumentation
  │     ├── Patch 1b2: skip warmup passes (safety net)
  │     ├── Patch 1b3: skip _dummy_sampler_run
  │     ├── Patch 2: CUDAGraphWrapper.__call__ → return preloaded graph
  │     └── Patch 3: hook BaseModelLoader.load_model (passthrough currently)
  │
  ├── Restore cursor metadata from saved_meta
  ├── atexit.register(state.finalize)
  │
  └── LLM(model, **kwargs)
        ├── init_device()
        │     └── request_memory() — checks free GPU memory
        │
        ├── load_model()
        │     ├── initialize_model() — creates model architecture (~5s)
        │     └── load_weights() — downloads + copies weights (~85s)
        │
        ├── determine_available_memory() ← Patch 0b
        │     ├── fdry.stop_allocation_region()
        │     ├── Run original determine_available_memory (cuBLAS init, etc.)
        │     ├── fdry.resume_allocation_region()
        │     └── Return saved_kv_cache_memory
        │
        ├── compile_or_warm_up_model() ← Patch 1b
        │     └── capture_model() ← Patch 1
        │           ├── End extended passthrough recording
        │           ├── _build_passthrough_addr_map(save_events, load_events)
        │           ├── _patch_graph_json_addresses() × N graphs
        │           ├── _premap_non_foundry_addresses()
        │           ├── fdry.CUDAGraph.load(path, pool) × N graphs
        │           └── Direct-populate:
        │                 for each (desc, graph, output):
        │                   entry = CUDAGraphEntry(batch_descriptor=desc)
        │                   entry.cudagraph = graph
        │                   entry.output = output
        │                   wrapper.concrete_cudagraph_entries[desc] = entry
        │
        └── Ready to serve
              First inference: +1.4s kernel JIT (sampling Triton kernels compile lazily)
```

### Multi-GPU (tp=2)

```
cached_vllm_init_with_foundry(model="...", tensor_parallel_size=2)
  │
  ├── _cached_vllm_init_multi_gpu()
  │     ├── resolve_graph_cache_dir() → cache_key
  │     ├── Check rank_0/ and rank_1/ caches → global_load_mode
  │     ├── Patch GPUWorker.init_device → _foundry_init_device
  │     ├── Set disable_custom_all_reduce=True
  │     │
  │     └── LLM(model, tensor_parallel_size=2)
  │           │
  │           ├── Worker 0 (subprocess):
  │           │     ├── _foundry_init_device(rank=0):
  │           │     │     ├── Per-rank cache: {base_dir}/rank_0/
  │           │     │     ├── Load CUDA modules from hook_archive
  │           │     │     ├── setup_foundry_regions()
  │           │     │     ├── patch_vllm_for_foundry(rank_cache)
  │           │     │     ├── fdry.stop_allocation_region()
  │           │     │     ├── fdry.start_passthrough_record()
  │           │     │     ├── original init_device() → NCCL init
  │           │     │     ├── fdry.resume_allocation_region()
  │           │     │     └── _extended_recording = True
  │           │     └── (rest uses patched methods, same as single-GPU)
  │           │
  │           └── Worker 1 (subprocess):
  │                 └── (same as Worker 0, with rank_1/ cache)
```

---

## 16. Optimization deep-dives

### Optimization 1 — Skip `torch.cuda.synchronize()` and `empty_cache()` in load mode

**Location:** ~line 1084 (inside Patch 1)

**Before:**
```python
torch.cuda.synchronize()
torch.cuda.empty_cache()
```
These ran unconditionally before graph capture/load.

**After:**
```python
if not _is_load_mode:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
```

**Why:** In load mode, no forward passes have run. The CUDA stream has no
pending work — `synchronize()` returns immediately but still makes a driver
round-trip. The CUDA caching allocator has no cached blocks — `empty_cache()`
is a no-op but still acquires the allocator mutex and calls `cudaFree`. Skipping
both removes two unnecessary CUDA driver round-trips.

**Impact:** ~2-5 ms saved (minor, but free).

---

### Optimization 2 — `compile_or_warm_up_model` timing wrapper

See [Patch 1b](#patch-1b-gpuworkercompile_or_warm_up_model).

Patch 1b wraps the original function with timing instrumentation. The actual
optimizations happen in the callees: Patch 1 (graph loading + direct-populate),
Patch 1b2 (skip warmup passes), and Patch 1b3 (skip Triton JIT).

---

### Optimization 3 — Early graph builds (Patch 3, currently disabled)

See [Patch 3](#patch-3-early-graph-builds-before-weight-loading).

Patch 3 hooks `BaseModelLoader.load_model` and wraps `load_weights`. The
original design overlapped `start_graph_builds` (Foundry Phase 2a template
building) with weight download. This optimization is **currently disabled** —
the hook is a passthrough with debug logging only. Graph loading now happens
entirely in Patch 1 via per-graph `fdry.CUDAGraph.load()`.

The hook remains in place for future re-enablement: the Foundry
`start_graph_builds` / `finish_graph_loads` pipeline could be restored to
overlap template building with the ~66s weight download if graph loading
latency becomes a bottleneck again.

---

## 17. Passthrough recording & address patching

### The problem

Foundry's bump allocator makes model weights and KV cache land at deterministic
addresses. But not everything goes through the bump allocator. NCCL workspace
buffers, cuBLAS handles, and CUDA runtime scratch memory are allocated by
libraries on threads where `tls_storage.enabled = false` (Foundry's bump
allocator is thread-local). These land at non-deterministic addresses.

During CUDA graph capture, kernel parameters encode these addresses. On reload,
the addresses are different, causing XID 31 GPU faults.

### Passthrough recording

Foundry provides APIs to record non-bump allocations:

- `fdry.stop_allocation_region()` — disable bump allocator on current thread
- `fdry.start_passthrough_record()` — start recording all allocations
- `fdry.pause_passthrough_record()` / `fdry.resume_passthrough_record()` — temporarily
  pause during profiling phases to avoid capturing profiling-only allocations
- `fdry.end_passthrough_record()` — stop recording
- `fdry.get_passthrough_events()` — return list of `{source, ptr, size}` dicts

Events are recorded during `init_device()` (NCCL init, cuBLAS workspace creation)
and through `capture_model()` via extended recording (`_extended_recording` flag).

### Address matching: `_build_passthrough_addr_map`

Size-based greedy matching: group load events by size, then for each save event,
consume the first unmatched load event with the same size. This handles event
count and ordering differences between save/load runs.

Returns `[(old_base, old_end, new_base, delta), ...]` sorted by `old_base`
for binary-search patching.

### Graph JSON patching: `_patch_graph_json_addresses`

For each graph JSON file:
1. Parse all nodes
2. For `KernelNode`: iterate `kernelParams[].value_hex` and `extra_argBuffer_hex`,
   extract 8-byte little-endian integers, binary search in range table
3. For `MemsetNode`: remap `params.dst`
4. For `MemcpyNode`: remap `params.srcDevice` and `params.dstDevice`
5. Write patched JSON to temp file (`.patched_graph_N.json`)

### Non-Foundry address pre-mapping: `_premap_non_foundry_addresses`

For addresses in graph kernel parameters that are outside the Foundry region
AND not covered by passthrough matching:

1. Scan all graph JSONs for pointer-sized values
2. Filter: page-aligned, in plausible GPU range (4GB–128TB), outside Foundry region
3. Merge into 2MB-aligned ranges
4. Stop Foundry allocator, use raw CUDA VMM via ctypes:
   - `cuMemAddressReserve(base, size, granularity, addr, 0)`
   - `cuMemCreate(&handle, size, &prop, 0)`
   - `cuMemMap(addr, size, 0, handle, 0)`
   - `cuMemSetAccess(addr, size, &desc, 1)`
5. Resume Foundry allocator

These are scratch buffers — zero-initialized memory is sufficient.

### Diagnostics: `_diagnose_unpatched_addresses`

After patching, scans all graph files for non-Foundry addresses that are NOT
covered by any passthrough range. Logs them with reference counts for debugging.

---

## 18. Multi-GPU support

### Architecture

`_cached_vllm_init_multi_gpu()` handles `tensor_parallel_size > 1`:

1. **Parent process** checks all rank caches to determine global mode
2. Patches `GPUWorker.init_device` → `_foundry_init_device`
3. Sets `disable_custom_all_reduce=True` (forces NCCL path)
4. Each worker subprocess runs `_foundry_init_device`:
   - Per-rank cache dir: `{base_dir}/rank_{rank}/`
   - `setup_foundry_regions()` per-GPU
   - `patch_vllm_for_foundry()` with rank-specific cache
   - Passthrough recording: stop allocator → start recording → `init_device()` →
     resume allocator → keep extended recording active
   - NCCL warmup allreduce during extended recording (load mode)

### Global mode decision

All ranks must agree on save vs load to avoid NCCL deadlocks. `profile_run()`
involves NCCL collectives — if one rank skips it and another doesn't, they hang.

### Cache structure

```
{graph_cache_dir}/{cache_key}/
├── rank_0/
│   ├── .save_complete
│   ├── metadata.json        # Per-rank passthrough events
│   ├── hook_archive/
│   └── wrapper_graphs/
├── rank_1/
│   └── (same structure)
└── ...
```

---

## 19. Abandoned optimizations

### Abandoned: CUDA graph driver warmup

Observed that the first `cuGraphInstantiate` in a process pays a 480-760 ms
penalty. Subsequent templates with identical node counts take 30-45 ms.
Hypothesis: this is a one-time CUDA driver JIT cost that could be amortized
with a dummy graph.

**Attempt 1 — Dummy graph with cursor reset:**

Created a minimal graph:
```python
fdry.stop_allocation_region()
g = fdry.CUDAGraph()
with fdry.graph(g):
    x = torch.zeros(1, device="cuda")
    x.fill_(1.0)
fdry.resume_allocation_region()
# Try to undo cursor shift:
fdry.set_current_alloc_offset(cursor_before)
```

Problem: `set_current_alloc_offset` rejects backward moves. The `torch.zeros(1)`
inside the graph allocated through the bump allocator (even though we called
`stop_allocation_region`, the allocation happened during capture which may
route differently). The cursor advanced by ~2 MB and couldn't be reset:
```
[HOOK] WARNING: New offset 0x200000 is less than current offset 0x400000, skipping
```
All subsequent allocations shifted by 2 MB. When graph replay referenced the
original addresses, they pointed to wrong data. Inference crashed.

**Attempt 2 — `stop_allocation_region` / `resume_allocation_region`:**

Used Foundry's APIs to truly disable the bump allocator:
```python
fdry.stop_allocation_region()    # tls_storage.enabled = false in hook.cpp:2924
g = fdry.CUDAGraph()
with fdry.graph(g):
    x = torch.zeros(1, device="cuda")
    x.fill_(1.0)
g.replay()  # instantiate the graph
fdry.resume_allocation_region()  # tls_storage.enabled = true
```

The warmup graph (1 node: `fill_` kernel) instantiated in 27 ms. The
bump cursor was unchanged (allocations went through default CUDA allocator).

But the real first-template penalty (480-760 ms) was **unchanged**. The
423-node template with FlashAttention kernels, layer norm kernels, and memcpy
nodes still took 480-760 ms.

**Root cause:** The penalty is **graph-complexity-dependent**, not a
process-level one-time JIT. The CUDA driver's `cuGraphInstantiate`
implementation likely does internal analysis/optimization proportional to the
number and type of nodes. A 1-node graph exercises a completely different code
path than a 423-node graph.

**Conclusion:** Cannot be warmed up. The correct approach is overlap (Patch 3),
not warmup.

---

### Abandoned: Early `start_graph_builds` before `LLM()` construction

Tried calling `start_graph_builds` in `cached_vllm_init_with_foundry`, before
`LLM()` is constructed. This would give Phase 2a the entire LLM() construction
time (~114 s) to complete.

**Attempt 1 — Without preallocation:**
```python
# In cached_vllm_init_with_foundry, before LLM():
state.load_all()
state._preparse_pending = fdry.CUDAGraph.start_graph_builds(paths)
llm = LLM(model, **kwargs)
```

Graph memcpy nodes reference model weight addresses (e.g., `0x10000200000`).
Before model loading, these addresses aren't mapped — the bump allocator
hasn't advanced past `0x10000000000`. `cuGraphAddMemcpyNode` validates
addresses at BUILD time:
```
[CGE LOAD ERROR] cuGraphAddMemcpyNode FAILED for node 9 with error 1
RuntimeError: CUDA driver error: invalid argument
```

**Attempt 2 — With full preallocation (36 GB):**
```python
# In cached_vllm_init_with_foundry, before LLM():
fdry.preallocate_region(fdry.parse_size("36GB"))  # map entire 36 GB
state._preparse_pending = fdry.CUDAGraph.start_graph_builds(paths)
llm = LLM(model, **kwargs)
```

Graph builds succeeded (all 5 templates built in 660 ms). But inside
`LLM()`, `init_device()` calls `request_memory()` which checks
`torch.cuda.mem_get_info()`. With 36 GB preallocated, only 2.97 GB is free:
```
ValueError: Free memory on device cuda:0 (2.97/39.49 GiB) on startup is less
than desired GPU memory utilization (0.8, 31.59 GiB)
```

**Why we can't free and re-preallocate:** Freeing the preallocated region
would require resetting the bump cursor to 0, which is rejected (cursor only
moves forward). And even if we could, the freed virtual addresses would need
to be re-mapped at exactly the same positions, which isn't guaranteed.

**Conclusion:** Full preallocation is incompatible with vLLM's startup memory
check. The working solution (Patch 3) fires after `init_device()` (memory
check passed) but before `load_weights()` (66 s of overlap available).

---

### Analyzed and decided against

**Adding `_dummy_run` back to the load-mode fast path:**
- Cost: ~50 ms added to init_engine
- Benefit: ~50 ms saved from first inference (non-graph forward pass kernel warm)
- Net: zero. init_engine is the optimization target, not first inference.

**Adding `_dummy_sampler_run` back to the load-mode fast path:**
- Cost: ~15 s added to init_engine (Triton JIT compiles ALL possible batch size
  variants of the top-k sampling kernel)
- Benefit: ~1.4 s saved from first inference (lazy JIT only compiles the one
  variant actually needed for the first request)
- Net: -13.6 s. Terrible tradeoff.

**Parallelizing template building in Foundry C++:**
From `CUDAGraphParallel.cpp:2306-2307`:
```cpp
// Phase 2a: Build template graphs sequentially (on this thread).
// CUDA driver API calls serialize on per-device mutex anyway.
```
Even if we built templates on separate threads, they'd serialize on NVIDIA's
internal per-device mutex for `cuGraphInstantiate`. We'd add threading overhead
with no parallelism gain.

---

## 20. Timeline visualizations

### Before Foundry (baseline)

```
 ── init_device + request_memory ─────────────────────────────────────────────┐
 ── load_model (download 66s + copy 18s) ────────────────────────────────────┤
 ── determine_available_memory ──────────────────────────────────────────────┤
 │     ├── profile_run() — full forward pass (16s)                           │
 │     └── profile_cudagraph_memory() (0.5s)                                │
 ── compile_or_warm_up_model ────────────────────────────────────────────────┤
 │     ├── kernel_warmup                                                     │
 │     ├── capture_model (35 graphs x ~80ms each = 3s) ─────────────────────┤
 │     ├── _dummy_run (~50ms) ──────────────────────────────────────────────┤
 │     └── _dummy_sampler_run (~15s Triton JIT) ────────────────────────────┤
 ── init_engine total: ~30s ────────────────────────────────────────────────┘
 ── LLM() total: ~137s ────────────────────────────────────────────────────┘
```

### Historical: After Optimizations 1+2 only (no early builds)

*(This timeline reflects a prior state when `start_graph_builds`/`finish_graph_loads`
was used. Retained for context on the optimization progression.)*

```
 ── load_model (download 66s + copy 18s) ────────────────────────────────────┐
 ── determine_available_memory → SKIPPED (Patch 0b), returns saved ─────────┤
 │     └── start_graph_builds fires here (in _skip_determine)               │
 ── compile_or_warm_up_model (Patch 1b) ────────────────────────────────────┤
 │     └── capture_model (Patch 1) ─────────────────────────────────────────┤
 │           ├── finish_graph_loads()                                        │
 │           │     └── Phase 2a: 0.7s needed, only 0.5s elapsed → wait 313ms│
 │           └── direct-populate (~0.5ms) ──────────────────────────────────┤
 ── init_engine: 0.63s ─────────────────────────────────────────────────────┘
```

### Current architecture (per-graph loading)

```
 ── init_device + request_memory ─────────────────────────────────────────────┐
 ── load_model (download 66s + copy 18s) ────────────────────────────────────┤
 │     └── Patch 3 wraps load_weights (passthrough, debug logging only)      │
 ── determine_available_memory ← Patch 0b ──────────────────────────────────┤
 │     ├── fdry.stop_allocation_region()                                     │
 │     ├── Run original (cuBLAS workspace init, etc.)                        │
 │     ├── fdry.resume_allocation_region()                                   │
 │     └── Return saved kv_cache_memory value                                │
 ── compile_or_warm_up_model ← Patch 1b (timing) ──────────────────────────┤
 │     └── capture_model ← Patch 1 ────────────────────────────────────────┤
 │           ├── End extended passthrough recording (multi-GPU only)          │
 │           ├── _build_passthrough_addr_map()                               │
 │           ├── _patch_graph_json_addresses() × N graphs                    │
 │           ├── _premap_non_foundry_addresses()                             │
 │           ├── fdry.CUDAGraph.load(path, pool) × 35 graphs                │
 │           └── direct-populate into CUDAGraphWrapper entries               │
 ── init_engine: ~few seconds ──────────────────────────────────────────────┘
 ── LLM() total: ~114s ────────────────────────────────────────────────────┘
```

---

## 21. Final results

Benchmarked on A100-SXM4-40GB, Qwen/Qwen2.5-7B-Instruct, Modal:

| Metric | Baseline (no Foundry) | Cached (current architecture) |
|--------|-----------------------|-------------------------------|
| `init_engine` | ~30 s | Eliminated (graph loading via `fdry.CUDAGraph.load`) |
| `compile_or_warm_up_model` | ~10 s | Skips warmup + Triton JIT (Patches 1b2, 1b3) |
| `determine_available_memory` | ~16 s | Runs with Foundry disabled, returns saved value |
| Graph loading | N/A (captured fresh) | Per-graph `fdry.CUDAGraph.load()` + direct-populate |
| Total `LLM()` init | 137 s | **~114 s** (17% faster) |
| First inference | 1.14 s | 2.58 s (+1.44 s kernel JIT from deferred Triton compilation) |

### Historical progression

*(These numbers reflect earlier architecture iterations using `start_graph_builds`/
`finish_graph_loads`. Retained for context on how the design evolved.)*

| State | `init_engine` | `compile_or_warm_up` | `finish_graph_loads` wait |
|-------|---------------|----------------------|---------------------------|
| Before session (graph loading, no fast path) | 0.72 s | 0.44 s | 230 ms |
| + Skip sync/empty_cache (Opt 1) | 0.72 s | 0.44 s | 230 ms |
| + compile_or_warm_up fast path (Opt 2) | 0.63 s | 0.32 s | 313 ms |
| + Early graph builds / Patch 3 (Opt 3) | **0.01 s** | **0.00 s** | **0.6 ms** |

### What the remaining 114s consists of

| Component | Time | Optimizable? |
|-----------|------|--------------|
| Model weight download (HuggingFace → GPU host) | ~66 s | No (network I/O) |
| Weight loading (`param.data.copy_` → GPU) | ~18 s | No (PCIe bandwidth) |
| Model architecture init (PyTorch module creation) | ~5 s | No (CPU-bound, one-time) |
| init_device + request_memory | ~2 s | No (CUDA context creation) |
| Other vLLM setup (config, tokenizer, scheduler) | ~23 s | Not in scope |

The serialization path (graph capture → graph load) is fully optimized at
0.01 s. The remaining 114 s is vLLM's own model loading pipeline, which is
outside the scope of the CUDA graph serialization integration.

---

## 22. Key files modified

### `src/vllm_profile_cache/foundry_graphs.py` (1608 lines)

All monkey-patches, address patching, and multi-GPU support live here.

**Monkey-patches:**

| Patch | Target | Purpose |
|-------|--------|---------|
| SHM pre-copy | (thread at init) | Copy graph files to tmpfs |
| Patch 0 | `GPUModelRunner.profile_cudagraph_memory` | Skip/instrument graph memory profiling |
| Patch 0b | `GPUWorker.determine_available_memory` | Run with Foundry disabled, return saved value |
| Patch 1 | `GPUModelRunner.capture_model` | Passthrough matching + address patching + graph loading + direct-populate (load) / phase flag + finalize (save) |
| Patch 1b | `GPUWorker.compile_or_warm_up_model` | Timing instrumentation |
| Patch 1b2 | `GPUModelRunner._warmup_and_capture` | Skip warmup forward passes in load mode |
| Patch 1b3 | `GPUModelRunner._dummy_sampler_run` | Skip sampling kernel compilation in load mode |
| Patch 2 | `CUDAGraphWrapper.__call__` | Intercept FULL mode: Foundry capture (save) / preloaded return (load) |
| Patch 3 | `BaseModelLoader.load_model` | Hook load_weights for early builds (currently passthrough) |
| `_finalize_save` | (internal) | Save all graphs, pack fatbins, write metadata + passthrough events |
| `_install_sigabrt_handler` | (C-level handler) | Survive CGE BUILD thread abort() |

**Address patching functions:**

| Function | Purpose |
|----------|---------|
| `_build_passthrough_addr_map` | Size-based greedy matching of save/load passthrough events |
| `_remap_addr` | Binary search for address in sorted range table |
| `_patch_graph_json_addresses` | Patch kernel params, memset/memcpy addresses in graph JSONs |
| `_premap_non_foundry_addresses` | Pre-map physical memory at non-Foundry addresses via CUDA VMM (ctypes) |
| `_diagnose_unpatched_addresses` | Log uncovered non-Foundry addresses for debugging |

**Multi-GPU:**

| Function | Purpose |
|----------|---------|
| `_cached_vllm_init_multi_gpu` | Multi-GPU entry point: per-rank caching, global mode, patched init_device |

### `profiling/modal_foundry_benchmark.py` (807 lines)

A/B/C benchmark on Modal (single GPU): baseline vs save vs load.

### `profiling/modal_foundry_multi_gpu_benchmark.py` (370 lines)

A/B/C benchmark on Modal (multi-GPU, tp=2): baseline vs save vs load with NCCL.

### Foundry C++ files (read-only, not modified)

| File | What we learned |
|------|-----------------|
| `csrc/CUDAGraphParallel.cpp:2280-2360` | Phase 2a template building is sequential because CUDA driver serializes on per-device mutex |
| `csrc/CUDAGraphParallel.cpp:814-960` | `build_graph_from_binary`: how kernel nodes are rebuilt from binary format |
| `csrc/CUDAGraphParallel.cpp:2402-2461` | `finish_graph_loads_impl`: wait on future, replay allocator events, reconstruct tensors |
| `csrc/hook.cpp:2924-2939` | `stop_allocation_region` / `resume_allocation_region`: sets `tls_storage.enabled` |
| `csrc/hook.cpp:3259-3294` | `query_function_handle`: O(1) hash table lookup for kernel function pointers |
| `csrc/hook.cpp` (passthrough) | `start_passthrough_record`, `end_passthrough_record`, `get_passthrough_events`, `pause/resume_passthrough_record` |

### vLLM files (read-only, not modified)

| File | What we learned |
|------|-----------------|
| `vllm/model_executor/model_loader/base_loader.py:43-82` | `load_model()` calls `initialize_model()` then `load_weights()` — hook point for Patch 3 |
| `vllm/v1/worker/gpu_worker.py:668-685` | Original `compile_or_warm_up_model` body — what Patch 1b wraps |
| `vllm/v1/worker/gpu_worker.py` | `Worker.init_device` — hook point for multi-GPU Foundry setup |
