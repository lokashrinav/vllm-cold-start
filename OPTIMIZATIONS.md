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
17. [Abandoned optimizations](#17-abandoned-optimizations)
18. [Timeline visualizations](#18-timeline-visualizations)
19. [Final results](#19-final-results)
20. [Key files modified](#20-key-files-modified)

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
    load_seconds: float            # Wall time for finish_graph_loads()
    _preloaded_graphs: Optional[list]  # All (graph, output) from finish_graph_loads

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

    # ---- Guard flags (prevent duplicate work across patches) ----
    _preallocated_early: bool              # True after preallocate_region() called
    _preparse_pending: Optional[Any]       # PendingGraphLoads handle from start_graph_builds
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
immediately, and writes metadata. Called during the save path for each captured
graph. Eagerly flushes everything to survive potential SIGABRT.

**`finalize()`** — Final metadata write + logging. Registered as an `atexit`
handler via `atexit.register(state.finalize)` in the entry point.

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

**Location:** ~line 523

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

**Location:** ~line 557

**What the original does:** Runs a full forward pass (`profile_run()`, ~16 s),
then calls `profile_cudagraph_memory()`, measures free VRAM, and returns bytes
available for KV cache.

**Save mode:** Wraps the original to record:
- `available_kv_cache_memory` — bytes available for KV cache (int)
- `determine_cursor_delta` — total cursor movement from profiling (int, bytes)

**Load mode:** Replaces entirely:
```python
def _skip_determine(self):
    cur = fdry.get_current_alloc_offset()

    # Fallback preallocation (if Patch 3 didn't fire)
    if not state._preallocated_early:
        remaining = fdry.parse_size(region_size) - cur
        if remaining > 0:
            fdry.preallocate_region(remaining)
            state._preallocated_early = True

    # Advance cursor to match save-run position
    fdry.set_current_alloc_offset(cur + _saved_det_delta)

    # Fallback graph build start (if Patch 3 didn't fire)
    if not state._preparse_pending:
        precopy_thread.join()
        state.load_all()
        if state._graph_files:
            state._preparse_pending = fdry.CUDAGraph.start_graph_builds(paths)

    return _saved_kv_mem
```

This skips the **16-second forward pass** (`profile_run()`) that vLLM uses to
measure peak VRAM. The saved value is exact — same model, same GPU, same config.

The fallback `preallocate_region` and `start_graph_builds` calls ensure
correctness even if Patch 3 didn't fire (e.g., a different model loader code
path that doesn't go through `BaseModelLoader.load_model`).

---

### Patch 1: `GPUModelRunner.capture_model`

**Location:** ~line 628

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
1. Call `load_all()` — ensure graph file list is populated
2. **Fallback preallocation** (if Patch 3 didn't fire):
   ```python
   if not state._preallocated_early:
       remaining = fdry.parse_size(region_size) - fdry.get_current_alloc_offset()
       fdry.preallocate_region(remaining)
   ```
3. **Fallback start_graph_builds** (if Patch 3 didn't fire):
   ```python
   if not state._preparse_pending:
       precopy_thread.join()
       state._preparse_pending = fdry.CUDAGraph.start_graph_builds(paths)
   ```
4. **Skip `torch.cuda.synchronize()` and `empty_cache()`** — in load mode, no
   forward passes have run. The CUDA caching allocator is empty. These would
   be no-op CUDA driver round-trips (~2-5 ms wasted).
5. **`finish_graph_loads()`** — blocks until Phase 2a completes:
   ```python
   all_loaded = list(fdry.CUDAGraph.finish_graph_loads(state._preparse_pending))
   ```
   If Patch 3 gave enough lead time (66 s of weight loading), Phase 2a (0.7 s)
   is already done. Wait time: **0.6 ms**.
6. **Direct-populate** — bypass the entire capture loop:
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
7. Return 0 (no memory used for capture)

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

**Location:** ~line 743

**What the original does** (vLLM `gpu_worker.py:668-685`):
1. Compile-size warmup — Triton JIT for various batch sizes
2. `maybe_remove_all_loras()` — remove LoRA adapters during capture
3. `kernel_warmup()` — warm up attention kernels (no-op on A100 + FlashAttention)
4. `capture_model()` — capture CUDA graphs for all batch sizes
5. `_dummy_run(NONE mode)` — one forward pass without graph capture (~50 ms)
6. `_dummy_sampler_run()` — compile sampling Triton kernels

**Save mode:** Wraps original with timing instrumentation only.

**Load mode:** Replaces entire function body:
```python
def _timed_compile_warmup(self):
    t0 = time.perf_counter()
    cuda_graph_memory_bytes = 0
    if not self.model_config.enforce_eager:
        cuda_graph_memory_bytes = self.model_runner.capture_model()
    set_random_seed(self.model_config.seed)
    elapsed = time.perf_counter() - t0
    return CompilationTimes(
        language_model=self.compilation_config.compilation_time,
        encoder=self.compilation_config.encoder_compilation_time,
    )
```

Only `capture_model()` (which does `finish_graph_loads` + direct-populate) and
`set_random_seed()` are kept. Everything else is skipped.

**Why each skip is safe:**
| Skipped step | Why it's safe to skip |
|------|------|
| Compile-size warmup | Triton kernels for graph capture aren't needed (no capture happens) |
| `maybe_remove_all_loras()` | No LoRA in this workload; and even with LoRA, loaded graphs don't need adapter removal |
| `kernel_warmup()` | No-op on A100 + FlashAttention (only needed for xFormers attention) |
| `_dummy_run(NONE)` | Warms up non-graph forward pass. Net zero: saves ~50 ms init, costs ~50 ms first inference |
| `_dummy_sampler_run()` | See Patch 1b3 below |

---

### Patch 1b2: Skip warmup passes in capture loop

**Location:** ~line 776

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

**Location:** ~line 796

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

**Location:** ~line 810

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

### Patch 3: Early graph builds before weight loading

**Location:** ~line 967

This is the highest-impact optimization. It hooks `BaseModelLoader.load_model`
to fire `preallocate_region()` + `start_graph_builds()` **before** `load_weights()`
is called.

```python
if _is_load_mode:
    from vllm.model_executor.model_loader.base_loader import BaseModelLoader
    _original_base_load_weights = BaseModelLoader.load_model

    def _load_model_with_early_builds(self_loader, **kwargs):
        _orig_load_weights = self_loader.load_weights

        def _hooked_load_weights(model, model_config):
            # Step 1: Map remaining VA space (one cuMemCreate + cuMemMap)
            if not state._preallocated_early:
                cur = fdry.get_current_alloc_offset()
                remaining = fdry.parse_size(region_size) - cur
                if remaining > 0:
                    fdry.preallocate_region(remaining)
                    state._preallocated_early = True

            # Step 2: Start Phase 1+2a in background C++ thread
            if not state._preparse_pending:
                precopy_thread = getattr(state, '_precopy_thread', None)
                if precopy_thread is not None:
                    precopy_thread.join()  # ensure SHM copy done
                state.load_all()
                if state._graph_files:
                    shm_dir = getattr(state, '_shm_graph_dir', None)
                    if shm_dir is not None and shm_dir.exists():
                        all_paths = [str(shm_dir / p.name) for p in state._graph_files]
                    else:
                        all_paths = [str(p) for p in state._graph_files]
                    state._preparse_pending = fdry.CUDAGraph.start_graph_builds(
                        all_paths, num_threads=min(len(all_paths), 16),
                    )

            # Step 3: Original load_weights (66s download + 18s copy)
            return _orig_load_weights(model, model_config)

        self_loader.load_weights = _hooked_load_weights
        return _original_base_load_weights(self_loader, **kwargs)

    BaseModelLoader.load_model = _load_model_with_early_builds
```

**Why `BaseModelLoader.load_model` is the right hook point:**

`BaseModelLoader.load_model()` (in `vllm/model_executor/model_loader/base_loader.py:43`)
does:
```python
def load_model(self, **kwargs):
    model = self.initialize_model(**kwargs)  # creates architecture, allocates tensors
    self.load_weights(model, model_config)    # downloads + copies weights
    return model
```

After `initialize_model()`:
- All parameter tensors exist at deterministic GPU addresses
- The bump cursor is at the "post-model-init" position
- But weight data hasn't been downloaded yet (66 s of network I/O ahead)

We intercept `self.load_weights` to inject our work between `initialize_model()`
and the actual weight download. This gives Phase 2a (~0.7 s) a 66-second window
to complete — more than enough.

**Why `preallocate_region` is needed at this point:**
After model init, the bump cursor is at (say) offset 15 GB. The region is 36 GB.
There are 21 GB of unmapped virtual addresses above the cursor. Graph memcpy
nodes reference addresses throughout the region (model weights, KV cache
placeholders, etc.). `cuGraphAddMemcpyNode` (called in Phase 2a) validates
that src/dst addresses are **mapped** GPU memory. Without preallocation, Phase
2a fails with:
```
[CGE LOAD ERROR] cuGraphAddMemcpyNode FAILED for node 9 with error 1
RuntimeError: CUDA driver error: invalid argument
```

`preallocate_region(remaining)` does a single `cuMemCreate + cuMemMap` call
to back the entire remaining 21 GB with physical memory. This is fast (~1 ms)
and makes all addresses valid.

**Why this is safe:**
1. **Model tensors at final addresses:** `initialize_model()` ran, bump
   allocator assigned deterministic addresses.
2. **Weight loading doesn't allocate:** `param.data.copy_(loaded_weight)` does
   `cudaMemcpy` to existing addresses. No `cuMemAlloc` calls. Cursor unchanged.
3. **`request_memory()` already passed:** Runs in `init_device()` before
   `load_model()`. Preallocating after model init doesn't trigger memory checks.
4. **Preallocation is idempotent:** Guard flag `_preallocated_early` prevents
   double preallocation.

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

## 13. Guard flag deduplication

Three locations can trigger `preallocate_region()` and `start_graph_builds()`:

| Location | When it fires |
|----------|---------------|
| **Patch 3** (`_hooked_load_weights`) | Before weight loading (66 s before finish_graph_loads) |
| **Patch 0b** (`_skip_determine`) | After weight loading, when vLLM calls determine_available_memory |
| **Patch 1** (`_flagged_capture_model`) | During capture_model, just before finish_graph_loads |

Two guard flags on `_FoundryPatchState` prevent duplicate work:

```python
state._preallocated_early: bool         # Guards preallocate_region()
state._preparse_pending: Optional[Any]  # Guards start_graph_builds()
```

**Flow in the happy case (Patch 3 fires):**
1. Patch 3 sets `_preallocated_early = True` after `preallocate_region()`
2. Patch 3 sets `_preparse_pending = <PendingGraphLoads>` after `start_graph_builds()`
3. Patch 0b checks `if not state._preallocated_early:` → skips (already done)
4. Patch 0b checks `if not state._preparse_pending:` → skips (already done)
5. Patch 1 checks the same flags → skips (already done)
6. Patch 1 calls `finish_graph_loads(state._preparse_pending)` → uses the handle

**Flow in the fallback case (Patch 3 doesn't fire):**
This happens if the model loader doesn't go through `BaseModelLoader.load_model`
(e.g., a custom loader). Patch 0b or Patch 1 fires instead:
1. Patch 0b sets `_preallocated_early = True` and `_preparse_pending = <handle>`
2. Patch 1 checks flags → skips (already done)
3. Patch 1 calls `finish_graph_loads(state._preparse_pending)` → works correctly

The triple-redundancy ensures graphs are always loaded regardless of which
vLLM code path executes.

---

## 14. Architecture: full save path call tree

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
  │     ├── Patch 0: instrument profile_cudagraph_memory (record result)
  │     ├── Patch 0b: instrument determine_available_memory (record result)
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
        │     ├── profile_run() — full forward pass (~16s)
        │     └── profile_cudagraph_memory() ← Patch 0 instruments
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
        │     │     ├── cache.save_metadata(all cursor deltas, profiling data)
        │     │     └── cache.write_save_complete_marker()
        │     ├── _dummy_run(NONE) — forward pass without graphs
        │     └── _dummy_sampler_run() — compile sampling kernels
        └── Ready to serve
```

---

## 15. Architecture: full load path call tree

```
cached_vllm_init_with_foundry(model="Qwen/Qwen2.5-7B-Instruct")
  │
  ├── is_foundry_available() → True
  ├── resolve_graph_cache_dir() → cache_key="a1b2c3d4"
  ├── FoundryGraphCache(dir) → has_cached=True, is_save_complete=True
  ├── load_metadata() → {region_base, kv_mem, cursor deltas, ...}
  ├── setup_foundry_regions("36GB", force_base=0x10000000000)
  │     ├── (same steps as save, but only one base address tried)
  │
  ├── patch_vllm_for_foundry(load_mode=True)
  │     ├── SHM pre-copy thread starts → copies graph files to /dev/shm
  │     ├── Patch 0: skip profile_cudagraph_memory (return saved estimate)
  │     ├── Patch 0b: skip determine_available_memory (return saved value)
  │     ├── Patch 1: capture_model → finish_graph_loads + direct-populate
  │     ├── Patch 1b: compile_or_warm_up_model → only capture_model + seed
  │     ├── Patch 1b2: skip warmup passes (safety net)
  │     ├── Patch 1b3: skip _dummy_sampler_run
  │     ├── Patch 2: CUDAGraphWrapper.__call__ → return preloaded graph
  │     └── Patch 3: hook BaseModelLoader.load_model
  │
  ├── Restore cursor metadata from saved_meta
  ├── atexit.register(state.finalize)
  │
  └── LLM(model, **kwargs)
        ├── init_device()
        │     └── request_memory() — checks free GPU memory (passes: region not preallocated yet)
        │
        ├── load_model() ← Patch 3 wraps this
        │     ├── initialize_model() — creates model architecture (~5s)
        │     │     └── Bump cursor advances to post-model-init position
        │     │
        │     └── load_weights() ← Patch 3 intercepts here
        │           ├── [Patch 3 Step 1] preallocate_region(remaining)
        │           │     └── One cuMemCreate + cuMemMap for ~21 GB
        │           │     └── Set _preallocated_early = True
        │           │
        │           ├── [Patch 3 Step 2] precopy_thread.join()
        │           │     └── SHM copy already done (started seconds ago)
        │           │
        │           ├── [Patch 3 Step 3] start_graph_builds(shm_paths, 16 threads)
        │           │     ├── Phase 1: parallel file I/O + parse (~13 ms)
        │           │     ├── Spawns detached C++ thread for Phase 2a
        │           │     ├── Returns PendingGraphLoads handle
        │           │     └── Set _preparse_pending = handle
        │           │
        │           └── [Patch 3 Step 4] _orig_load_weights(model, model_config)
        │                 ├── Download weights from HuggingFace (~66 s) ─────────┐
        │                 └── Copy weights to GPU tensors (~18 s)                │
        │                       └── param.data.copy_(loaded_weight)              │
        │                       └── No cuMemAlloc — cursor unchanged             │
        │                                                                        │
        │                 Meanwhile, on C++ background thread:                   │
        │                   Phase 2a: build_graph_from_binary x 5 templates      │
        │                     ├── Template 0 (423 nodes): 480-760 ms ←───────────┤
        │                     ├── Template 1: 30-45 ms                           │
        │                     ├── Template 2: 30-45 ms                           │
        │                     ├── Template 3: 30-45 ms                           │
        │                     └── Template 4: 30-45 ms                           │
        │                   Phase 2c: link_on_demand_shared_exec x 30            │
        │                   build_promise->set_value() ← Phase 2a done (~0.7 s)  │
        │                                                                        │
        │                 Weight download continues for remaining ~65 s ──────────┘
        │
        ├── determine_available_memory() ← Patch 0b
        │     ├── Check _preallocated_early → True (skip prealloc)
        │     ├── fdry.set_current_alloc_offset(cur + saved_delta) — advance cursor
        │     ├── Check _preparse_pending → not None (skip start_graph_builds)
        │     └── Return saved_kv_cache_memory (0 ms)
        │
        ├── compile_or_warm_up_model() ← Patch 1b
        │     └── capture_model() ← Patch 1
        │           ├── load_all() → already done (no-op)
        │           ├── Check _preallocated_early → True (skip prealloc)
        │           ├── Check _preparse_pending → not None (skip start_graph_builds)
        │           ├── finish_graph_loads(state._preparse_pending)
        │           │     ├── pending->build_complete_.get()
        │           │     │     └── Phase 2a done 65 seconds ago → returns in 0.6 ms
        │           │     ├── Phase 2b: replay_hook_events_from_json x 35 → advance cursor
        │           │     └── Phase 2c: reconstruct output tensors x 35
        │           │     └── Returns [(graph_0, output_0), ..., (graph_34, output_34)]
        │           │
        │           └── Direct-populate:
        │                 for each (desc, graph, output):
        │                   entry = CUDAGraphEntry(batch_descriptor=desc)
        │                   entry.cudagraph = graph
        │                   entry.output = output
        │                   wrapper.concrete_cudagraph_entries[desc] = entry
        │                 # 35 dict writes, ~0.5 ms total
        │
        └── Ready to serve
              First inference: +1.4s kernel JIT (sampling Triton kernels compile lazily)
```

---

## 16. Optimization deep-dives

### Optimization 1 — Skip `torch.cuda.synchronize()` and `empty_cache()` in load mode

**Location:** ~line 635 (inside Patch 1)

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

### Optimization 2 — `compile_or_warm_up_model` fast path

See [Patch 1b](#patch-1b-gpuworkercompile_or_warm_up_model).

**Impact:** `compile_or_warm_up_model` dropped from **0.44 s → 0.32 s**
(0.12 s saved, mostly from skipping `_dummy_run` and setup overhead).

---

### Optimization 3 — Early graph builds before weight loading

See [Patch 3](#patch-3-early-graph-builds-before-weight-loading).

**Impact:** `finish_graph_loads` wait dropped from **313 ms → 0.6 ms**.
`init_engine` dropped from **0.63 s → 0.01 s**. This one optimization
accounts for >95% of the total speedup.

**Why 313 ms before, 0.6 ms after:**

Before Patch 3, `start_graph_builds` fired in `_skip_determine` (Patch 0b),
which runs AFTER `load_model()` completes. The timeline:

```
load_model finishes → _skip_determine fires start_graph_builds
                      0.5s gap
                      compile_or_warm_up → finish_graph_loads
```

Phase 2a needs 600-950 ms, but only has ~0.5 s of lead time (between
`_skip_determine` and `finish_graph_loads`). So `finish_graph_loads` blocks
for 313 ms.

After Patch 3, `start_graph_builds` fires BEFORE `load_weights()`:

```
start_graph_builds → load_weights (66s) → _skip_determine → finish_graph_loads
                     Phase 2a (0.7s) done here, 65s before finish_graph_loads
```

Phase 2a has 66 seconds of lead time. It finishes in 0.7 s. When
`finish_graph_loads` runs 65 seconds later, the future is already resolved.
Wait time: 0.6 ms (just the future.get() overhead).

---

## 17. Abandoned optimizations

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

## 18. Timeline visualizations

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

### After Optimizations 1+2 only (no early builds)

```
 ── load_model (download 66s + copy 18s) ────────────────────────────────────┐
 ── determine_available_memory → SKIPPED (Patch 0b), returns saved ─────────┤
 │     └── start_graph_builds fires here (in _skip_determine)               │
 ── compile_or_warm_up_model (Patch 1b fast path) ──────────────────────────┤
 │     └── capture_model (Patch 1) ─────────────────────────────────────────┤
 │           ├── start_graph_builds already called → skip                    │
 │           ├── finish_graph_loads()                                        │
 │           │     └── Phase 2a: 0.7s needed, only 0.5s elapsed → wait 313ms│
 │           └── direct-populate (~0.5ms) ──────────────────────────────────┤
 ── init_engine: 0.63s ─────────────────────────────────────────────────────┘
```

### After all optimizations (including Patch 3)

```
 ── init_device + request_memory ─────────────────────────────────────────────┐
 ── load_model() → Patch 3 fires: ──────────────────────────────────────────┤
 │     ├── initialize_model() (~5s) ────────────────────────────────────────┤
 │     ├── preallocate_region() ── 1ms ─────────────────────────────────────┤
 │     ├── start_graph_builds() ── 13ms ─────────────────┐                  │
 │     │                                                  │ Phase 2a (0.7s) │
 │     └── load_weights() ── 66s download + 18s copy ────┤ completes here  │
 │                                                        └─────────────────┤
 ── determine_available_memory → SKIPPED, 0ms (Patch 0b) ──────────────────┤
 │     └── Guards: _preallocated_early=True, _preparse_pending=set → skip  │
 ── compile_or_warm_up_model (fast path, Patch 1b) ─────────────────────────┤
 │     └── capture_model (Patch 1) ─────────────────────────────────────────┤
 │           ├── Guards: _preallocated_early=True → skip prealloc           │
 │           ├── Guards: _preparse_pending=set → skip start_graph_builds    │
 │           ├── finish_graph_loads() → 0.6ms (Phase 2a done 65s ago!) ─────┤
 │           └── direct-populate (~0.5ms) ──────────────────────────────────┤
 ── init_engine: 0.01s ─────────────────────────────────────────────────────┘
 ── LLM() total: ~114s ────────────────────────────────────────────────────┘
```

---

## 19. Final results

Benchmarked on A100-SXM4-40GB, Qwen/Qwen2.5-7B-Instruct, Modal:

| Metric | Baseline (no Foundry) | Cached (all optimizations) |
|--------|-----------------------|----------------------------|
| `init_engine` | ~30 s | **0.01 s** (3000x faster) |
| `compile_or_warm_up_model` | ~10 s | **0.00 s** |
| `determine_available_memory` | ~16 s | **0.00 s** |
| `finish_graph_loads` wait | N/A | **0.6 ms** |
| Phase 2a overlap | N/A | 100% (during 66 s download) |
| Total `LLM()` init | 137 s | **114 s** (17% faster) |
| First inference | 1.14 s | 2.58 s (+1.44 s kernel JIT) |

### Progression across optimizations

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

## 20. Key files modified

### `src/vllm_profile_cache/foundry_graphs.py`

All monkey-patches live here. Complete list:

| Patch | Target | Line | Purpose |
|-------|--------|------|---------|
| SHM pre-copy | (thread at init) | ~493 | Copy graph files to tmpfs |
| Patch 0 | `GPUModelRunner.profile_cudagraph_memory` | ~523 | Skip/instrument graph memory profiling |
| Patch 0b | `GPUWorker.determine_available_memory` | ~557 | Skip/instrument full profiling phase |
| Patch 1 | `GPUModelRunner.capture_model` | ~628 | Phase flag + load path (finish_graph_loads + direct-populate) |
| Patch 1b | `GPUWorker.compile_or_warm_up_model` | ~743 | Fast path: only capture_model + set_random_seed |
| Patch 1b2 | `GPUModelRunner._warmup_and_capture` | ~776 | Skip warmup forward passes in load mode |
| Patch 1b3 | `GPUModelRunner._dummy_sampler_run` | ~796 | Skip sampling kernel compilation in load mode |
| Patch 2 | `CUDAGraphWrapper.__call__` | ~810 | Intercept FULL mode: Foundry capture (save) / preloaded return (load) |
| Patch 3 | `BaseModelLoader.load_model` | ~967 | Early preallocate + start_graph_builds before weight loading |
| `_finalize_save` | (internal) | ~914 | Save all graphs, pack fatbins, write metadata |
| `_install_sigabrt_handler` | (C-level handler) | ~342 | Survive CGE BUILD thread abort() |

### `profiling/modal_foundry_benchmark.py`

- Increased baseline subprocess timeout from 500 s to 900 s for cold containers.

### Foundry C++ files (read-only, not modified)

| File | What we learned |
|------|-----------------|
| `csrc/CUDAGraphParallel.cpp:2280-2360` | Phase 2a template building is sequential because CUDA driver serializes on per-device mutex |
| `csrc/CUDAGraphParallel.cpp:814-960` | `build_graph_from_binary`: how kernel nodes are rebuilt from binary format |
| `csrc/CUDAGraphParallel.cpp:2402-2461` | `finish_graph_loads_impl`: wait on future, replay allocator events, reconstruct tensors |
| `csrc/hook.cpp:2924-2939` | `stop_allocation_region` / `resume_allocation_region`: just sets `tls_storage.enabled` |
| `csrc/hook.cpp:3259-3294` | `query_function_handle`: O(1) hash table lookup for kernel function pointers |

### vLLM files (read-only, not modified)

| File | What we learned |
|------|-----------------|
| `vllm/model_executor/model_loader/base_loader.py:43-82` | `load_model()` calls `initialize_model()` then `load_weights()` — hook point for Patch 3 |
| `vllm/v1/worker/gpu_worker.py:668-685` | Original `compile_or_warm_up_model` body — what Patch 1b replaces |
