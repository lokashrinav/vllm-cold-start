# Foundry x vLLM: CUDA Graph Serialization for Cold Start Elimination

## Table of Contents

1. [The Problem](#the-problem)
2. [Architecture Overview](#architecture-overview)
3. [Foundry's Deterministic VMM](#foundrys-deterministic-vmm)
4. [The Bump Allocator & LD_PRELOAD Hook](#the-bump-allocator--ld_preload-hook)
5. [CUDA Graph Capture Flow (Save Mode)](#cuda-graph-capture-flow-save-mode)
6. [CUDA Graph Engine (CGE) — Serialization Format](#cuda-graph-engine-cge--serialization-format)
7. [Three-Phase Graph Loading (Load Mode)](#three-phase-graph-loading-load-mode)
8. [Template Sharing via Topology Grouping](#template-sharing-via-topology-grouping)
9. [vLLM Integration Layer](#vllm-integration-layer)
10. [Profile Skip — Eliminating Memory Profiling](#profile-skip--eliminating-memory-profiling)
11. [File I/O Optimization — /dev/shm Pre-Copy](#file-io-optimization--devshm-pre-copy)
12. [Cursor Mismatch Problem & Solution](#cursor-mismatch-problem--solution)
13. [Preallocation for Graph-Private Memory](#preallocation-for-graph-private-memory)
14. [Sampler Warmup Skip](#sampler-warmup-skip)
15. [End-to-End Timeline Comparison](#end-to-end-timeline-comparison)
16. [Key Files Reference](#key-files-reference)

---

## The Problem

Every vLLM cold start on a fresh container runs through a fixed initialization pipeline:

```
Container boot → Python imports → Model download → Weight loading → Memory profiling
→ KV cache allocation → CUDA graph capture → First inference ready
```

For Qwen2.5-7B-Instruct on an A100-40GB, the breakdown is approximately:

| Phase | Time | Controllable? |
|-------|------|---------------|
| Model download (HuggingFace CDN) | 60-90s | No (network-bound) |
| Weight loading to GPU | 20-28s | Partially |
| Memory profiling (`profile_run` + `profile_cudagraph_memory`) | 16-37s | **Yes** |
| KV cache allocation | <1s | No |
| CUDA graph capture (35 graphs for decode) | 3-9s | **Yes** |
| First inference | 1-2s | No |
| **Total** | **130-220s** | |

The **memory profiling** and **CUDA graph capture** phases are pure waste on repeat cold starts — the results are deterministic for the same model + GPU + config. Foundry eliminates the graph capture phase by serializing captured CUDA graphs to disk. Our additional optimizations eliminate the memory profiling phase entirely.

---

## Architecture Overview

The integration has two modes:

### Save Mode (First Cold Start)
```
                    LD_PRELOAD=libcuda_hook.so
                              │
                              ▼
    ┌─────────────────────────────────────────────┐
    │         Foundry CUDA Driver Hook            │
    │  Intercepts: cuMemAlloc, cuModuleLoad, etc. │
    │  Records: allocation addresses, fatbins     │
    └──────────────┬──────────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────────┐
    │      Deterministic VMM (Bump Allocator)     │
    │  set_allocation_region(0x10000000000, 36GB) │
    │  All GPU allocs get sequential addresses    │
    └──────────────┬──────────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────────┐
    │           vLLM Engine Init                  │
    │  1. Load model weights (deterministic addr) │
    │  2. Profile memory (record cursor delta)    │
    │  3. Allocate KV cache (deterministic addr)  │
    │  4. capture_model() → Foundry CUDAGraph     │
    │     - capture_begin: start_hook_record()    │
    │     - forward pass: record alloc events     │
    │     - capture_end: save_hook_events_to_json │
    │  5. graph.save() → .json + .cugraph files   │
    │  6. pack_fatbins_to_folder() → fatbin archive│
    └─────────────────────────────────────────────┘
```

### Load Mode (Subsequent Cold Starts)
```
    ┌─────────────────────────────────────────────┐
    │         Same LD_PRELOAD hook active          │
    │  CGE_MODE=load → skip fatbin processing     │
    │  load_cuda_modules_and_libraries(archive)   │
    └──────────────┬──────────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────────┐
    │      Same Deterministic VMM region          │
    │  Same base address → same model addresses   │
    └──────────────┬──────────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────────┐
    │           vLLM Engine Init                  │
    │  1. Load model weights (same addresses!)    │
    │  2. Skip profiling (return saved value)     │
    │  3. Allocate KV cache (same addresses!)     │
    │  4. capture_model() → load from disk        │
    │     - start_graph_builds() → Phase 1+2a     │
    │     - finish_graph_loads() → Phase 2b+2c    │
    │     - replay_hook_events → cursor adjust    │
    │     - cuGraphInstantiate → graph ready      │
    │  5. First inference uses loaded graphs      │
    └─────────────────────────────────────────────┘
```

---

## Foundry's Deterministic VMM

CUDA graphs contain hardcoded GPU memory addresses. When a graph is captured, every `cuMemAlloc`, `cudaMemcpy`, and kernel launch references specific pointers. If those pointers are different on the next cold start, the graph is invalid.

Foundry solves this by **forcing deterministic GPU virtual memory addresses** via the CUDA Virtual Memory Management (VMM) API.

### How `set_allocation_region` works

```
hook.cpp:2875-2917 — set_allocation_region()

GPU Virtual Address Space (A100-40GB):
┌────────────────────────────────────────────────────┐
│ 0x0                                         ~128TB │
│                                                    │
│         ┌──────────────────────────────┐           │
│         │ Reserved Region (36 GB)      │           │
│         │ Base: 0x10000000000 (64GB)   │           │
│         │                              │           │
│         │ cuMemAddressReserve(base,36G)│           │
│         │                              │           │
│         │ All subsequent cuMemAlloc    │           │
│         │ calls return addresses from  │           │
│         │ this region sequentially     │           │
│         └──────────────────────────────┘           │
│                                                    │
└────────────────────────────────────────────────────┘
```

The function:
1. Calls `cuMemAddressReserve(base=0x10000000000, size=36GB)` to claim a contiguous VA range
2. Initializes `tls_storage` (thread-local) with the region bounds
3. Sets `current_alloc_base_addr = base` — the bump allocator cursor
4. All future `cuMemAlloc` calls are intercepted and served from this region

### Why 36GB on a 40GB GPU?

The region must hold: model weights (~14.25 GiB) + KV cache (~15.9 GiB) + profiling overhead (~4 GiB) + graph-private memory (~1.1 GiB) ≈ 35.25 GiB. We use 36GB with some headroom.

---

## The Bump Allocator & LD_PRELOAD Hook

Foundry's `libcuda_hook.so` is loaded via `LD_PRELOAD` before any CUDA library. It intercepts CUDA driver calls at the lowest level — before they reach `libcuda.so`.

### Thread-Local Storage (TLS)

```cpp
// hook.cpp:223-248
struct ThreadLocalStorage {
    AllocRegion region;
    size_t current_alloc_base_addr;    // The bump cursor
    size_t current_vmm_reserve_addr;
    bool enabled;
    bool region_initialized;

    // Preallocation state
    CUmemGenericAllocationHandle preallocated_handle;
    size_t preallocated_start_addr;
    size_t preallocated_end_addr;
    bool has_preallocation;             // Fast path flag

    CUdevice cached_device;
    size_t cached_granularity;
    bool device_cached;
};

static thread_local ThreadLocalStorage tls_storage;
```

**Critical detail**: `tls_storage` is `thread_local`. Each thread has its own independent bump allocator state. This means background threads (like CGE's Phase 2a template builder) have their **own** `tls_storage` with `enabled = false` by default. Allocations on background threads fall through to the real CUDA driver, not the bump allocator.

### The cuMemAlloc Hook — Two Paths

```cpp
// hook.cpp:2098-2157 — cuMemAlloc_v2 hook

CUresult hooked_cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize) {
    if (!tls_storage.enabled) {
        // Thread doesn't have region enabled → real driver call
        return real_cuMemAlloc_v2(dptr, bytesize);
    }

    size_t aligned_size = align_to(bytesize, kAllocAlignment);
    CUdeviceptr target_addr = tls_storage.current_alloc_base_addr;

    // ═══ FAST PATH ═══
    // If region is preallocated and allocation fits: just return the address.
    // No cuMemCreate, no cuMemMap, no cuMemSetAccess. Zero driver calls.
    if (tls_storage.has_preallocation &&
        (target_addr + aligned_size) <= tls_storage.preallocated_end_addr) {

        *dptr = target_addr;
        tls_storage.current_alloc_base_addr = align_to(
            target_addr + bytesize, kAllocAlignment);

        // Record metadata for graph serialization
        global_alloc_metadata.emplace(*dptr, AllocMetadata{...});
        return CUDA_SUCCESS;
    }

    // ═══ SLOW PATH ═══
    // Allocate physical memory and map it to the target VA
    CUmemGenericAllocationHandle handle;
    cuMemCreate(&handle, aligned_size, &alloc_prop, 0);
    cuMemMap(target_addr, aligned_size, 0, handle, 0);
    cuMemSetAccess(target_addr, aligned_size, &access_desc, 1);

    *dptr = target_addr;
    tls_storage.current_alloc_base_addr = align_to(
        target_addr + bytesize, kAllocAlignment);

    return CUDA_SUCCESS;
}
```

### Bump Allocator Properties

1. **One-directional**: The cursor only moves forward. It never reclaims freed memory.
2. **Deterministic**: Same allocation sequence → same addresses. Model weights at `0x10005000000`, KV cache at `0x10xxxxx`, always.
3. **Never frees physical pages** for preallocated memory: `cuMemFree` is intercepted and the memory stays mapped.
4. **Thread-local**: Background threads don't affect the main thread's cursor.

---

## CUDA Graph Capture Flow (Save Mode)

When vLLM captures CUDA graphs, our monkey-patched `CUDAGraphWrapper.__call__` routes through Foundry's `fdry.CUDAGraph` instead of `torch.cuda.CUDAGraph`.

### What happens during `fdry.graph(graph)`:

```python
# foundry_graphs.py — _foundry_wrapper_call (save path)

_pre_cursor = fdry.get_current_alloc_offset()

graph = fdry.CUDAGraph()
with fdry.graph(graph):           # calls capture_begin()
    output = self.runnable(*args) # actual model forward pass
                                  # all cuMemAlloc calls are recorded
                                  # all kernel launches are captured
# context exit calls capture_end()
```

### Inside `capture_begin` (CUDAGraph.cpp:201-247)

```cpp
void CUDAGraph::capture_begin(...) {
    // Tell PyTorch's caching allocator to route to our pool
    c10::cuda::CUDACachingAllocator::beginAllocateToPool(
        capture_dev_, mempool_id_, ...);

    // Resume the hook's allocation tracking
    foundry::resume_allocation_region();

    // START RECORDING: every cuMemAlloc from this point
    // is recorded as an "allocator event"
    foundry::start_hook_record();

    // Begin CUDA stream capture
    cudaStreamBeginCapture(capture_stream_, capture_mode);
}
```

### Inside `capture_end` (CUDAGraph.cpp:249-290)

```cpp
void CUDAGraph::capture_end() {
    // End CUDA stream capture — produces a CUgraph
    cudaStreamEndCapture(capture_stream_, &graph_);

    // STOP RECORDING allocator events
    foundry::end_hook_record();
    foundry::stop_allocation_region();

    // Serialize all recorded alloc/free events to JSON
    allocator_events_ = foundry::save_hook_events_to_json();
    foundry::clear_hook_events();

    // Analyze the captured graph structure
    analyze_captured_graph();
}
```

### What gets recorded

The hook records every memory operation that happened **during capture** (between `start_hook_record` and `end_hook_record`):

```json
{
  "start_base_addr": 34034679808,
  "events": [
    {"type": "alloc", "ptr": "0x107ecc00000", "size": 62914560},
    {"type": "alloc", "ptr": "0x107f0600000", "size": 4194304},
    {"type": "free",  "ptr": "0x107f0600000"}
  ]
}
```

These are **graph-private allocations** — temporary buffers that CUDA creates for internal graph execution (attention scratch space, workspace buffers, etc.). They are NOT the model weights or KV cache — those exist before capture begins.

**Important**: `cuGraphInstantiate` (which compiles the graph into an executable) also allocates memory, but this happens AFTER `capture_end`. Those allocations are NOT in the allocator events. They're handled separately by the template building phase.

---

## CUDA Graph Engine (CGE) — Serialization Format

When `graph.save(path)` is called, Foundry serializes the CUDA graph in two formats:

### JSON Format (`.json`) — Full Fidelity

The JSON file contains the complete graph structure: nodes, dependencies, kernel parameters (hex-encoded), allocator events, output tensor metadata, and generator states. It's human-readable but large (~551 KB per graph for Qwen2.5-7B).

### Binary Format (`.cugraph`) — Optimized for Speed

The `.cugraph` file uses a custom binary format defined in `BinaryGraphFormat.h`:

```
File Layout:
┌─────────────────────────────┐
│ FileHeader (64 bytes)       │  magic: "CUGRAPH\0"
│  - version, flags           │  version: 1
│  - num_nodes, num_deps      │  num_nodes, num_sections, etc.
│  - num_generators           │
│  - num_sections             │
├─────────────────────────────┤
│ SectionDirectory            │  Array of {type, offset, size}
│  (num_sections * 24 bytes)  │  entries pointing to section data
├─────────────────────────────┤
│ STRING_TABLE                │  Deduplicated null-terminated strings
│                             │  (kernel names, module paths)
├─────────────────────────────┤
│ NODE_TABLE                  │  Fixed-size BinNodeEntry structs
│  (num_nodes * 168 bytes)    │  Direct struct read — no parsing
├─────────────────────────────┤
│ DEPENDENCY_TABLE            │  (from_id, to_id) pairs
├─────────────────────────────┤
│ KERNEL_PARAM_INDEX          │  Per-kernel param descriptors
├─────────────────────────────┤
│ KERNEL_PARAM_DATA           │  Raw binary kernel parameters
│                             │  (no hex encoding/decoding)
├─────────────────────────────┤
│ ALLOCATOR_EVENTS            │  JSON text (kept for simplicity)
├─────────────────────────────┤
│ OUTPUT_TENSORS              │  JSON text
├─────────────────────────────┤
│ TOPOLOGY_KEY                │  Raw string for template matching
└─────────────────────────────┘
```

### BinNodeEntry — Fixed-Size Node Descriptor

```cpp
// BinaryGraphFormat.h:167-181
struct BinNodeEntry {
    uint32_t node_id;
    BinNodeType type;       // KernelNode, MemcpyNode, MemsetNode, etc.
    uint8_t  _pad[3];
    union {
        BinKernelNode kernel;   // func_name_idx, grid/block dims, shared_mem
        BinMemsetNode memset;   // dst_addr, value, element_size, width, height
        BinMemcpyNode memcpy;   // src_addr, dst_addr, widths, kind
        BinEventNode  event;    // event_id
    };
};
static_assert(sizeof(BinNodeEntry) == 168, "BinNodeEntry must be 168 bytes");
```

Every node is exactly 168 bytes. The loader reads `num_nodes * 168` bytes in a single I/O call and casts directly to an array of `BinNodeEntry`. No parsing, no string scanning, no hex decoding.

### Performance: Binary vs JSON

| Metric | JSON | Binary (.cugraph) |
|--------|------|-------------------|
| File size per graph | ~551 KB | ~188 KB |
| Parse time (Phase 1) | ~300ms/graph | ~0.8ms/graph |
| Kernel params | Hex-encoded strings | Raw binary bytes |
| Node access | Hash table lookup | Direct array index |
| Total for 35 graphs | ~10.5s | ~28ms |

The binary format achieves **~10x speedup** in Phase 1 parsing and **~3x reduction** in file size.

---

## Three-Phase Graph Loading (Load Mode)

Graph loading is implemented in `CUDAGraphParallel.cpp` with a pipeline designed to maximize parallelism:

```
Timeline:
─────────────────────────────────────────────────────────────►
│ Phase 1 │     Phase 2a      │ Phase 2b │ Phase 2c │
│  Parse  │  Template Build   │ On-Demand│  Link    │
│ 30ms    │  512-632ms        │ parallel │  <1ms    │
│         │                   │ with 2a  │          │
│ 16 threads │ sequential    │ 16 thds  │ sequential│
│         │  (cuGraphInstant) │ (CPU only)│         │
└─────────┴───────────────────┴──────────┴──────────┘
```

### Phase 1: Parallel File I/O + Parse (30ms)

```cpp
// CUDAGraphParallel.cpp:1901-1913

// Phase 1a: Read and parse all graph files in parallel
//   - Binary path (.cugraph): mmap + direct struct reads
//   - JSON fallback: boost::json parse
// Phase 1b: Sequential shell creation + metadata extraction
// Phase 1c: Extract allocator_events + output_tensors
// Phase 1d: Compute topology groups (from graph_manifest.json)
```

With the `/dev/shm` pre-copy optimization, files are read from local tmpfs instead of a network volume, reducing Phase 1 from 11.4s to 30ms.

### Phase 2a: Sequential Template Building (512-632ms)

For each unique topology, ONE template graph is fully built:

```cpp
// CUDAGraphParallel.cpp:2306-2360

for (const auto& [key, indices] : topology_groups) {
    size_t tmpl_idx = indices[0];  // First graph is the template

    // Build from binary format (10x faster than JSON)
    CUDAGraph::build_graph_from_binary(
        bin_files[tmpl_idx],
        all_parsed[tmpl_idx].graph,
        main_ctx,
        &tmpl
    );
    // Creates SharedGraphExec with the template's cuGraphExec
    shared_execs[tmpl_idx] = SharedGraphExec(tmpl);
}
```

**Why sequential?** `cuGraphInstantiate` (called inside `build_graph_from_binary`) invokes the CUDA driver to compile the graph into an executable. This modifies driver-global state and must run on a single thread to avoid race conditions.

For Qwen2.5-7B, there are only **5 unique topologies** among 35 graphs (the graphs differ only in batch size — the kernel structure is the same). So only 5 templates are built. The first template takes ~512ms (CUDA JIT compilation), subsequent ones take ~24-31ms each.

### Phase 2b: Parallel On-Demand Preparation

The remaining 30 graphs (those sharing a template) have their metadata prepared on worker threads while Phase 2a is still running:

```cpp
// CUDAGraphParallel.cpp:2277-2340

// On-demand prep: parse kernel params, node data from binary
// Pure CPU work — no CUDA driver calls — safe for parallel execution
for (graph : non_template_graphs) {
    thread_pool.submit([&]() {
        prepare_on_demand_data(graph, bin_files[graph]);
    });
}
```

### Phase 2c: Quick Linking (<1ms)

After templates are built and on-demand data is prepared, link each graph to its template's `SharedGraphExec`:

```cpp
// CUDAGraphParallel.cpp:2367-2373

for (size_t i = 0; i < num; ++i) {
    if (template_for[i] == SIZE_MAX) continue;  // Skip templates
    CUDAGraph::link_on_demand_shared_exec(
        *all_parsed[i].graph,
        shared_execs.at(template_for[i])
    );
}
```

Linked graphs share the template's `cuGraphExec` but apply their own kernel parameters at replay time. This avoids 30 redundant `cuGraphInstantiate` calls.

### finish_graph_loads: Allocator Replay

After the pipeline completes, `finish_graph_loads_impl` (called per-graph from Python) replays allocator events:

```cpp
// CUDAGraphParallel.cpp:2391-2405

// For each graph:
// 1. Register CUDA generators (allocates small GPU tensors)
// 2. Replay allocator events (set cursor, replay alloc/free)
// 3. Reconstruct output tensors from saved metadata
```

The allocator replay calls `replay_hook_events_from_json` in `hook.cpp`, which adjusts the bump allocator cursor and replays each allocation event to ensure graph-private memory is at the expected addresses.

---

## Template Sharing via Topology Grouping

CUDA graphs that have the **same structure** (same node types, same kernel signatures, same dependency edges) but **different parameters** (different batch sizes, different tensor dimensions) share a single compiled template.

### How topology keys are computed

```cpp
// CUDAGraphParallel.cpp:2096-2220

// For each graph, build a topology key string:
// "KernelNode:C1_1_1,MemcpyNode,KernelNode:C1_1_1,MemsetNode,..."
//
// The key encodes:
//   - Node type (KernelNode, MemcpyNode, MemsetNode, EventNode)
//   - For KernelNodes: cluster dimensions (Cx_y_z)
//
// Graphs with identical topology keys share a template.
```

### Topology distribution for Qwen2.5-7B

| Topology | Graphs | Node count | Description |
|----------|--------|------------|-------------|
| A | 5 | 423 | Batch sizes 1-8 (small decode) |
| B | 4 | 423 | Batch sizes 16-56 |
| C | 4 | 451 | Batch sizes 64-104 |
| D | 5 | 395 | Batch sizes 112-152 |
| E | 17 | 423 | Batch sizes 160-512 |
| **Total** | **35** | | **5 templates, 30 on-demand** |

Only 5 `cuGraphInstantiate` calls instead of 35 — a **7x reduction** in template build time.

### Manifest-Based Template Assignment

At save time, topology groups are computed and written to `graph_manifest.json`:

```json
{
  "topology_groups": [
    {
      "topology_key": "KernelNode:C1_1_1,MemcpyNode,...",
      "template": "graph_0.json",
      "members": ["graph_0.json", "graph_1.json", "graph_2.json", ...]
    },
    ...
  ]
}
```

At load time, the manifest is read first, avoiding the need to re-compute topology keys from graph data.

---

## vLLM Integration Layer

The integration monkey-patches three vLLM methods to intercept the graph lifecycle:

### Patch 0: `GPUModelRunner.profile_cudagraph_memory`

vLLM runs warmup captures to estimate CUDA graph memory usage. In load mode, we return the saved estimate and advance the bump allocator cursor by the recorded delta:

```python
def _skip_profile_cudagraph(self):
    fdry.set_current_alloc_offset(pre + saved_cursor_delta)
    return saved_estimate  # e.g., 230686720 bytes (0.21 GiB)
```

### Patch 0b: `Worker.determine_available_memory`

This is the biggest optimization. vLLM's `determine_available_memory` runs a full model forward pass (`profile_run`) just to measure peak GPU memory. For a 7B model, this takes **~16 seconds**. In load mode, we skip it entirely:

```python
def _skip_determine(self):
    # Preallocate remaining region for fast-path cuMemAlloc
    fdry.preallocate_region(remaining)
    # Advance cursor to match save-mode position
    fdry.set_current_alloc_offset(cursor + saved_delta)
    # Return saved value directly
    return saved_kv_cache_memory  # e.g., 17,091,788,390 bytes
```

This replaces ~16s with ~0.5s.

### Patch 1: `GPUModelRunner.capture_model`

The `capture_model` patch handles the overall orchestration:

**Save mode:**
1. Records pre-capture allocator offset
2. Flushes PyTorch caching allocator (`torch.cuda.empty_cache()`)
3. Calls original `capture_model()` (which iterates batch sizes)
4. Each batch size triggers `CUDAGraphWrapper.__call__` (Patch 2)
5. After all graphs captured, `_finalize_save()` writes to disk

**Load mode:**
1. Preallocates remaining GPU VA region (if not already done)
2. Copies graph files to `/dev/shm` (background pre-copy)
3. Calls `start_graph_builds()` to begin three-phase loading pipeline
4. Calls original `capture_model()` which triggers `__call__` per batch size
5. Each `__call__` returns a pre-loaded graph from `finish_graph_loads()`

### Patch 2: `CUDAGraphWrapper.__call__`

**Save mode:** Replaces `torch.cuda.CUDAGraph` with `fdry.CUDAGraph`, capturing with Foundry's hook-aware implementation.

**Load mode:** Instead of capturing, returns the next pre-loaded graph from the three-phase pipeline:

```python
def _foundry_wrapper_call(self, *args, **kwargs):
    if _is_load_mode:
        result = state.next_loaded_graph()
        if result is not None:
            graph, output = result
            entry.cudagraph = graph
            entry.output = output
            return entry.output
    # ... save mode capture logic ...
```

---

## Profile Skip — Eliminating Memory Profiling

### What `determine_available_memory` does

vLLM's `Worker.determine_available_memory()` runs inside engine initialization:

```python
# vllm/v1/worker/gpu_worker.py

def determine_available_memory(self):
    baseline = MemorySnapshot.measure()       # ~instant
    self.model_runner.profile_run()            # FULL FORWARD PASS (~16s!)
    torch.cuda.synchronize()
    peak = MemorySnapshot.measure()
    estimate = self.model_runner.profile_cudagraph_memory()  # ~4s
    available = total_gpu * utilization - peak - estimate
    return available
```

The `profile_run()` runs a complete forward pass with max sequence length (32768 tokens) through all 28 transformer layers. For a 7B model on A100, this takes ~16 seconds and is the **single largest controllable cost** in the cold start pipeline.

### Save mode: Instrument and record

During save mode, we wrap `determine_available_memory` to capture:
- The return value (`available_kv_cache_memory`)
- The total cursor advancement (`determine_cursor_delta`)

These are saved in the graph cache metadata alongside the graph files.

### Load mode: Skip entirely

We replace `determine_available_memory` with a function that:
1. **Preallocates** the remaining GPU region (so subsequent KV cache allocation uses the fast path)
2. **Advances** the bump cursor by `determine_cursor_delta` (replicating the cursor effect of profile_run + profile_cudagraph_memory without actually running them)
3. **Returns** the saved `available_kv_cache_memory` value

This ensures KV cache allocation happens at the **exact same addresses** as during the save run, because:
- Same base address (deterministic VMM)
- Same model weights (same addresses from bump allocator)
- Same cursor advancement (replayed via `set_current_alloc_offset`)
- Same available memory value → same number of KV cache blocks → same KV cache size

---

## File I/O Optimization — /dev/shm Pre-Copy

On Modal (and similar serverless platforms), graph cache files are stored on a network volume. Reading 35 graph files (each ~188KB binary + ~551KB JSON) over the network takes **~11.4 seconds**.

### The optimization

During `patch_vllm_for_foundry()`, we start a background thread that copies all graph files from the network volume to `/dev/shm` (Linux tmpfs — RAM-backed filesystem):

```python
# Runs in background during model download
def _precopy():
    shm_dir = Path("/dev/shm/foundry_graphs")
    shm_dir.mkdir(parents=True, exist_ok=True)
    for src in state._graph_files:
        shutil.copy2(str(src.with_suffix(".cugraph")), str(shm_dir / ...))
        shutil.copy2(str(src), str(shm_dir / src.name))
```

When `capture_model` runs, it waits for the pre-copy to complete and uses `/dev/shm` paths:

```python
precopy_thread.join()
if shm_dir.exists():
    all_paths = [str(shm_dir / p.name) for p in state._graph_files]
```

### Result

| Source | Phase 1 parse time | Speedup |
|--------|-------------------|---------|
| Network volume | 11,418 ms | 1x |
| /dev/shm (tmpfs) | 29.9 ms | **382x** |

The pre-copy completes during model download (which takes 60-90s), so it adds zero latency to the critical path.

---

## Cursor Mismatch Problem & Solution

### The problem

In save mode, the bump allocator cursor follows this path:

```
Start → model weights → profile_run → profile_cudagraph → KV cache → [capture]
```

In load mode (before our fixes), the path was different:

```
Start → model weights → [skip profile_cudagraph] → [profile_run still runs] → KV cache → [load]
```

Because fatbin loading and module initialization take different code paths in save vs load mode, the cursor diverges by ~54 MB by the time graph loading begins. This caused `replay_hook_events_from_json` to abort:

```
[HOOK] ERROR: Memory offset mismatch during replay
  Current cursor: 0x107f0200000
  Expected:       0x107ecc00000
  Difference:     +54.00 MB
```

### The fix (hook.cpp:3154-3203)

We changed `replay_hook_events_from_json` to **adjust** the cursor instead of aborting:

```cpp
if (events_obj.contains("start_base_addr")) {
    uint64_t start_base_addr = ...;
    if (start_base_addr != tls_storage.current_alloc_base_addr) {
        // Log the adjustment for diagnostics
        fprintf(stderr, "[HOOK] Adjusting cursor for replay: "
                "0x%llx -> 0x%llx (%+.2f MB)\n", ...);
    }
    // Force cursor to the expected position
    tls_storage.current_alloc_base_addr = start_base_addr;
}
```

### Why this is safe

1. The region is **preallocated**: all physical pages from the current cursor to the region end are already mapped via `cuMemCreate + cuMemMap + cuMemSetAccess`.

2. The cursor adjustment happens **per-graph**: each graph's `allocator_events` JSON includes a `start_base_addr` that is the expected cursor position. The replay function sets the cursor to that exact position, regardless of where it was before.

3. Graph-private memory addresses are **absolute**: the CUDA graph references specific GPU pointers. As long as those pointers have mapped physical memory (ensured by preallocation), the graph replay works correctly.

---

## Preallocation for Graph-Private Memory

### Why we need it

During graph capture, CUDA allocates temporary buffers (graph-private memory) that the graph reads/writes at replay time. In save mode, these are allocated naturally via `cuMemAlloc`. In load mode, we must ensure these addresses have physical memory mapped before graph instantiation.

Without preallocation, `cuGraphAddMemcpyNode` fails with `CUDA_ERROR_INVALID_VALUE` because the target addresses aren't mapped.

### How it works

```python
# In _flagged_capture_model (or _skip_determine for the early path):

cur = fdry.get_current_alloc_offset()
region_end = fdry.parse_size("36GB")
remaining = region_end - cur

fdry.preallocate_region(remaining)
```

This calls into `hook.cpp:preallocate_region()`:

```cpp
bool preallocate_region(size_t size) {
    // Single cuMemCreate for the entire remaining region
    cuMemCreate(&handle, size, &alloc_prop, 0);

    // Map it into the VA space at the current cursor
    cuMemMap(tls_storage.current_alloc_base_addr, size, 0, handle, 0);

    // Make it accessible
    cuMemSetAccess(tls_storage.current_alloc_base_addr, size, &desc, 1);

    // Mark fast path available
    tls_storage.has_preallocation = true;
    tls_storage.preallocated_start_addr = tls_storage.current_alloc_base_addr;
    tls_storage.preallocated_end_addr = tls_storage.current_alloc_base_addr + size;

    // NOTE: does NOT advance current_alloc_base_addr
}
```

After preallocation, all subsequent `cuMemAlloc` calls within the region use the **fast path** — zero driver calls, just pointer arithmetic.

### When preallocation happens

| Phase | Without optimization | With determine_available_memory skip |
|-------|---------------------|--------------------------------------|
| Model loading | Slow path (cuMemCreate per alloc) | Slow path |
| Profile run | Slow path | **SKIPPED** |
| KV cache | Slow path | **Fast path** (preallocated early) |
| Graph loading | Fast path (preallocated in capture_model) | Fast path (already preallocated) |

By preallocating inside the skipped `determine_available_memory`, KV cache allocation also benefits from the fast path.

---

## Sampler Warmup Skip

### The Problem

vLLM's `compile_or_warm_up_model()` calls `_dummy_sampler_run()` after CUDA graph capture. This method runs 256 dummy requests through the sampler to JIT-compile Triton sampling kernels (top-k sort on `vocab_size * batch_size` logits). For Qwen2.5-7B (vocab=152,064), this means sorting ~39M elements.

In normal mode, `profile_run()` already compiled these kernels as a side effect, so `_dummy_sampler_run` runs in ~0.02s (cache hit). But in load mode, we skip `profile_run()`, so `_dummy_sampler_run` triggers **first-time Triton JIT compilation (~15s)**.

### The Fix

Monkey-patch `GPUModelRunner._dummy_sampler_run` to a no-op in load mode:

```python
if _is_load_mode:
    def _skip_dummy_sampler_run(self, *args, **kwargs):
        return None
    GPUModelRunner._dummy_sampler_run = _skip_dummy_sampler_run
```

Sampling kernels compile lazily on the first real inference instead, adding ~1s to first request latency — an acceptable tradeoff for 15s cold start savings.

### Impact

| Metric | Before | After |
|--------|--------|-------|
| `compile_or_warm_up_model` | 12-18s | 1.78s |
| `init_engine` total | 12-18s | 1.79s |
| First inference latency | 1.22s | 2.22s (+1s) |

---

## End-to-End Timeline Comparison

### Baseline vLLM (no Foundry)

```
Time  0s ──────────────────────────────────────────────────── 140s
      │ Model download (60-88s)                │
      │                                         │ Weight loading (20s)
      │                                         │              │ Profile run (16s)
      │                                         │              │         │ Profile CG (4s)
      │                                         │              │         │    │ KV cache
      │                                         │              │         │    │  │ Graph capture (3-9s)
      │                                         │              │         │    │  │       │ Inf
      └─────────────────────────────────────────┴──────────────┴─────────┴────┴──┴───────┴──►

Init engine: ~54s
Total: ~140s
```

### Foundry Load Mode (all optimizations)

```
Time  0s ──────────────────────────────────────────────────── ~110s
      │ Model download (60-88s)                │
      │  [pre-copy to /dev/shm runs in bg]      │ Weight loading (16-20s)
      │                                         │              │ Profiling SKIPPED (0s)
      │                                         │              │ Sampler warmup SKIPPED
      │                                         │              │ KV cache (fast path)
      │                                         │              │  │ Graph load (1.8s)
      │                                         │              │  │ │ Inf (2.2s)
      └─────────────────────────────────────────┴──────────────┴──┴─┴──►

Init engine: 1.79s (measured)
Total: ~110s (measured, Qwen2.5-7B on A100-40GB)
```

### What each optimization saves

| Optimization | Saves | Mechanism |
|-------------|-------|-----------|
| Skip `profile_run` | ~16s | Return saved `available_kv_cache_memory` |
| Skip `profile_cudagraph_memory` | ~4s | Return saved estimate + advance cursor |
| Skip `_dummy_sampler_run` | ~15s | Defer sampling kernel JIT to first inference |
| Graph loading (vs capture) | ~3-9s | Load from .cugraph files instead of capturing |
| `/dev/shm` pre-copy | ~11s | Overlap file copy with model download |
| Binary .cugraph format | ~10s | 168-byte fixed nodes, no hex decode |
| Template sharing | ~5s | 5 templates instead of 35 instantiations |
| Fast-path cuMemAlloc | ~1s | Preallocated region → zero driver calls |
| `CGE_MODE=load` + fatbin skip | ~2-3s | Skip fatbin extraction during module loading |

---

## Key Files Reference

### Foundry Core (C++)

| File | Lines | Purpose |
|------|-------|---------|
| `foundry/csrc/hook.cpp` | ~3258 | LD_PRELOAD hook: bump allocator, cuMemAlloc interception, allocator event recording/replay, `set_allocation_region`, `preallocate_region`, `replay_hook_events_from_json` |
| `foundry/csrc/CUDAGraph.cpp` | ~600 | `capture_begin`, `capture_end`, `analyze_captured_graph`, `save()`, `load()` |
| `foundry/csrc/CUDAGraphParallel.cpp` | ~2464 | Three-phase loading pipeline: `start_graph_builds_impl`, `finish_graph_loads_impl`, template sharing, topology grouping |
| `foundry/include/BinaryGraphFormat.h` | ~200 | Binary `.cugraph` format: `FileHeader`, `BinNodeEntry` (168 bytes), section types |
| `foundry/include/hook.h` | ~54 | Public API: `set_allocation_region`, `preallocate_region`, `get_current_alloc_offset`, etc. |

### vLLM Integration (Python)

| File | Lines | Purpose |
|------|-------|---------|
| `src/vllm_profile_cache/foundry_graphs.py` | ~1080 | Main integration: `patch_vllm_for_foundry`, `cached_vllm_init_with_foundry`, `_FoundryPatchState`, profile skip, `/dev/shm` pre-copy, graph save/load orchestration |
| `profiling/modal_foundry_benchmark.py` | ~800 | A/B benchmark on Modal: baseline vs save vs load, subprocess with proper LD_PRELOAD, timing comparison |

### Cache Directory Layout

```
/root/.cache/foundry-graphs/<cache_key>/
├── .save_complete                    # Marker file
├── metadata.json                     # Model ID, region info, cursor positions,
│                                     # saved profile estimates, KV cache memory
├── hook_archive/
│   ├── fatbin_entrypoint_packed.txt  # Module entrypoint → fatbin mapping
│   └── fatbin_image_packed.img       # All CUDA fatbin images (kernels)
└── wrapper_graphs/
    ├── graph_manifest.json           # Topology groups, template assignments
    ├── graph_0.json                  # Full JSON graph (551 KB)
    ├── graph_0.cugraph               # Binary graph (188 KB)
    ├── graph_1.json
    ├── graph_1.cugraph
    ├── ...
    ├── graph_34.json
    └── graph_34.cugraph
```
