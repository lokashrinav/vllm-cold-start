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
    │  Passthrough Recording (non-Foundry allocs) │
    │  fdry.stop_allocation_region()              │
    │  fdry.start_passthrough_record()            │
    └──────────────┬──────────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────────┐
    │           vLLM Engine Init                  │
    │  1. init_device() → NCCL init (recorded)    │
    │  2. fdry.resume_allocation_region()          │
    │  3. Load model weights (same addresses!)    │
    │  4. determine_available_memory() → run with  │
    │     Foundry disabled, return saved value     │
    │  5. Allocate KV cache (same addresses!)     │
    │  6. capture_model():                        │
    │     a. NCCL warmup allreduce (if multi-GPU) │
    │     b. End passthrough recording            │
    │     c. Match passthrough events (save↔load) │
    │     d. Patch graph JSON addresses           │
    │     e. Pre-map non-Foundry addresses        │
    │     f. fdry.CUDAGraph.load() per graph      │
    │     g. Direct-populate graph entries         │
    │  7. First inference uses loaded graphs      │
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

The integration monkey-patches vLLM's initialization pipeline via `patch_vllm_for_foundry()`. Eight patches intercept the graph lifecycle, profiling, and initialization:

### Patch 0: `GPUModelRunner.profile_cudagraph_memory`

vLLM runs warmup captures to estimate CUDA graph memory usage. In load mode, we return the saved estimate and advance the bump allocator cursor by the recorded delta:

```python
def _skip_profile_cudagraph(self):
    fdry.set_current_alloc_offset(pre + saved_cursor_delta)
    return saved_estimate  # e.g., 230686720 bytes (0.21 GiB)
```

In save mode, passthrough recording is paused during profiling (`fdry.pause_passthrough_record()`) to avoid capturing profiling-only allocations, then resumed after.

### Patch 0b: `Worker.determine_available_memory`

In load mode, we run the original `determine_available_memory` with the Foundry allocator **temporarily disabled** (`fdry.stop_allocation_region()`). This lets cuBLAS workspace initialization and other runtime side effects happen normally with the default CUDA allocator. But we return the **saved** KV cache memory value instead of the freshly-computed one, keeping cursor alignment correct:

```python
def _skip_determine(self):
    fdry.stop_allocation_region()
    try:
        result = _original_determine(self)  # side effects happen
    finally:
        fdry.resume_allocation_region()
    return _saved_kv_mem  # use saved value for cursor alignment
```

This ensures runtime libraries initialize their internal state (which matters for graph replay) while still returning the deterministic KV cache value from the save run.

In save mode, the function is instrumented to record `available_kv_cache_memory` and `determine_cursor_delta`.

### Patch 1: `GPUModelRunner.capture_model`

The core orchestration patch.

**Save mode:**
1. Stops extended passthrough recording and captures all events
2. Records pre-capture allocator offset
3. Flushes PyTorch caching allocator (`torch.cuda.empty_cache()`)
4. Calls original `capture_model()` (which iterates batch sizes)
5. Each batch size triggers `CUDAGraphWrapper.__call__` (Patch 2)
6. After all graphs captured, `_finalize_save()` writes to disk (graphs, fatbins, metadata, passthrough events)

**Load mode:**
1. NCCL warmup allreduce (if `torch.distributed` is initialized and extended recording is active) — captures NCCL's internal allocations as passthrough events
2. Stops extended passthrough recording, captures events
3. Builds address remap table from save/load passthrough events (`_build_passthrough_addr_map`)
4. Patches graph JSON kernel parameters with remapped addresses (`_patch_graph_json_addresses`)
5. Pre-maps physical memory at remaining non-Foundry addresses (`_premap_non_foundry_addresses`)
6. Loads each graph via `fdry.CUDAGraph.load(path, pool)`
7. Direct-populates `CUDAGraphWrapper.concrete_cudagraph_entries` — bypasses entire capture loop

### Patch 1b: `GPUWorker.compile_or_warm_up_model`

Timing instrumentation in both modes. In load mode, the original function still runs (it calls `capture_model` which is already patched).

### Patch 1b2: Skip warmup passes in capture loop

Safety net for load mode. If any code path still calls `_warmup_and_capture`, reduces work to a single `_dummy_run` pass instead of N warmup passes + capture.

### Patch 1b3: Skip `_dummy_sampler_run`

Skips Triton JIT compilation of sampling kernels (~15s) in load mode. Kernels compile lazily on first inference instead, adding ~1.4s to first request.

### Patch 2: `CUDAGraphWrapper.__call__`

**Save mode:** Replaces `torch.cuda.CUDAGraph` with `fdry.CUDAGraph`, capturing with Foundry's hook-aware implementation. Skips `set_graph_pool_id` because graph pools use `cudaMallocAsync` which bypasses Foundry's `cuMemAlloc_v2` hook. After `fdry.graph().__exit__`, calls `fdry.resume_allocation_region()` because `capture_end()` disables it.

**Load mode:** Returns the next pre-loaded graph from `state._preloaded_graphs` (populated by Patch 1's loading sequence). Falls back to original capture if cache is exhausted.

### Patch 3: Hook `BaseModelLoader.load_model`

Intercepts `load_weights` to inject work before the weight download. Currently used as a passthrough (early preallocation + graph builds disabled for debugging), but the hook point remains for overlapping graph loading with the 66s weight download.

---

## Profile Skip — Memory Profiling Optimization

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

The Foundry allocator is temporarily stopped during profiling so cuBLAS and other runtime allocations happen through the default CUDA allocator. This is important because `profile_run` triggers cuBLAS workspace initialization as a side effect.

### Load mode: Run with Foundry disabled, return saved value

We run the **original** `determine_available_memory` with the Foundry allocator temporarily disabled (`fdry.stop_allocation_region()`). This is necessary because:
- cuBLAS workspace initialization happens as a side effect of `profile_run`
- These workspaces get referenced by CUDA graph kernel parameters
- If they don't exist at graph load time, replay faults (XID 31)

But we return the **saved** KV cache memory value, not the freshly-computed one. The saved value ensures KV cache allocation produces the same number of blocks and same addresses as the save run.

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

| Phase | Without optimization | With optimization |
|-------|---------------------|-------------------|
| Model loading | Slow path (cuMemCreate per alloc) | Slow path |
| Profile run | Slow path | Runs with Foundry disabled (for side effects) |
| KV cache | Slow path | **Fast path** (preallocated early, if applicable) |
| Graph loading | Fast path (preallocated in capture_model) | Fast path |

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

## Passthrough Recording & Address Patching

### The Problem

Not all GPU allocations go through Foundry's bump allocator. NCCL workspace buffers, cuBLAS handles, and CUDA runtime scratch memory are allocated by libraries that call `cuMemAlloc` on threads where `tls_storage.enabled = false` (Foundry's bump allocator is thread-local). These allocations land at non-deterministic addresses controlled by the CUDA driver.

During CUDA graph capture, kernel parameters encode these addresses. On reload, the addresses are different, and graph replay faults with XID 31 (GPU memory violation).

### Solution: Passthrough Event Recording

Foundry provides passthrough recording APIs that log non-bump-allocator allocations:

```python
fdry.stop_allocation_region()       # disable bump allocator
fdry.start_passthrough_record()     # start recording
# ... init_device, NCCL init, cuBLAS init ...
fdry.resume_allocation_region()     # re-enable bump allocator
# extended recording continues through capture_model
fdry.end_passthrough_record()       # stop recording
events = fdry.get_passthrough_events()  # list of {source, ptr, size}
```

### Save Path

1. Before `init_device()`: stop bump allocator, start passthrough recording
2. `init_device()` runs: NCCL initializes, cuBLAS creates workspace handles
3. Resume bump allocator, but keep recording active (`_extended_recording = True`)
4. Recording continues through `capture_model()` to catch post-init allocations
5. In `_flagged_capture_model`, `fdry.end_passthrough_record()` captures all events
6. Events saved to `metadata.json` alongside graph files

### Load Path

1. Same recording captures new addresses
2. `_build_passthrough_addr_map()`: size-based greedy matching
   - Group load events by size
   - For each save event, find first unmatched load event with same size
   - If addresses differ, create `(old_base, old_end, new_base, delta)` tuple
   - Sort by `old_base` for binary search
3. `_patch_graph_json_addresses()`: for each graph JSON
   - Iterate all kernel parameter hex values
   - Extract 8-byte little-endian integers
   - Binary search in range table → apply delta if match
   - Also patches MemsetNode.dst, MemcpyNode.srcDevice/dstDevice
   - Writes patched JSON to temp file
4. `_premap_non_foundry_addresses()`: for remaining non-Foundry pointers
   - Scan all graph JSONs for pointer-sized values outside Foundry region
   - Filter: must be page-aligned, in plausible GPU address range
   - Merge into 2MB-aligned ranges
   - Pre-map via raw CUDA VMM: `cuMemAddressReserve` + `cuMemCreate` + `cuMemMap` + `cuMemSetAccess` (ctypes → libcuda.so)
5. `_diagnose_unpatched_addresses()`: log any remaining uncovered pointers

### NCCL Warmup

In multi-GPU load mode, NCCL's internal buffers must exist before graph loading. The integration does a warmup allreduce during passthrough recording:

```python
if state._extended_recording and _is_load_mode:
    if torch.distributed.is_initialized():
        warmup = torch.zeros(1024, device=f"cuda:{dev}")
        torch.distributed.all_reduce(warmup)
        torch.cuda.synchronize()
```

This forces NCCL to allocate its workspace buffers while passthrough recording is active, so the addresses are captured and can be matched against save-run events.

---

## Multi-GPU Support (Tensor Parallelism)

### Architecture

Multi-GPU mode (`tensor_parallel_size > 1`) uses `_cached_vllm_init_multi_gpu()`:

1. **Parent process** (no CUDA init):
   - Computes cache key from model + versions + kwargs
   - Checks ALL per-rank caches to determine global save/load mode
   - Patches `GPUWorker.init_device` → `_foundry_init_device`
   - Sets `disable_custom_all_reduce=True` (forces NCCL path)
   - Calls `LLM(model, ...)`

2. **Worker subprocesses** (via fork):
   - `_foundry_init_device()` runs per-rank:
     - Per-rank cache directory: `{base_dir}/rank_{rank}/`
     - Loads CUDA modules from hook_archive (load mode)
     - `setup_foundry_regions()` per-GPU
     - `patch_vllm_for_foundry()` with rank-specific cache
     - Starts passthrough recording
     - Runs original `init_device()` (NCCL init, model architecture creation)
     - Resumes Foundry allocator, keeps extended recording active
   - Rest of init uses patched methods (same as single-GPU)

### Global Mode Decision

All ranks must use the same mode (save or load) to avoid NCCL deadlocks. If rank 0 is in save mode (running `profile_run`, which involves NCCL collectives) but rank 1 is in load mode (skipping profiling), the ranks hang waiting for each other.

```python
def _rank_cache_ready(r: int) -> bool:
    rank_dir = base_dir / f"rank_{r}"
    if not (rank_dir / ".save_complete").exists():
        return False
    wg_dir = rank_dir / "wrapper_graphs"
    return wg_dir.exists() and any(wg_dir.glob("graph_*.json"))

global_load_mode = all(_rank_cache_ready(r) for r in range(tp_size))
```

### Cache Structure (Multi-GPU)

```
/root/.cache/foundry-graphs/<cache_key>/
├── rank_0/
│   ├── .save_complete
│   ├── metadata.json          # Includes passthrough_events for rank 0
│   ├── hook_archive/
│   └── wrapper_graphs/
│       ├── graph_0.json
│       ├── graph_0.cugraph
│       └── ...
├── rank_1/
│   ├── .save_complete
│   ├── metadata.json          # Includes passthrough_events for rank 1
│   ├── hook_archive/
│   └── wrapper_graphs/
│       └── ...
└── ...
```

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
| Profile run (run with Foundry disabled) | ~16s | Run for side effects, return saved `available_kv_cache_memory` |
| Skip `profile_cudagraph_memory` | ~4s | Return saved estimate + advance cursor |
| Skip `_dummy_sampler_run` | ~15s | Defer sampling kernel JIT to first inference |
| Graph loading (vs capture) | ~3-9s | Load from .cugraph files instead of capturing |
| Passthrough addr patching | N/A | Enables correct graph replay with non-deterministic NCCL/cuBLAS addresses |
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
| `foundry/csrc/hook.cpp` | ~3258 | LD_PRELOAD hook: bump allocator, cuMemAlloc interception, allocator event recording/replay, passthrough recording, `set_allocation_region`, `preallocate_region` |
| `foundry/csrc/CUDAGraph.cpp` | ~600 | `capture_begin`, `capture_end`, `analyze_captured_graph`, `save()`, `load()` |
| `foundry/csrc/CUDAGraphParallel.cpp` | ~2464 | Three-phase loading pipeline: `start_graph_builds_impl`, `finish_graph_loads_impl`, template sharing, topology grouping |
| `foundry/include/BinaryGraphFormat.h` | ~200 | Binary `.cugraph` format: `FileHeader`, `BinNodeEntry` (168 bytes), section types |
| `foundry/include/hook.h` | ~54 | Public API: `set_allocation_region`, `preallocate_region`, `get_current_alloc_offset`, passthrough APIs |

### vLLM Integration (Python)

| File | Lines | Purpose |
|------|-------|---------|
| `src/vllm_profile_cache/foundry_graphs.py` | ~1608 | Main integration: 8 monkey-patches, passthrough recording, address patching (binary search + graph JSON rewrite), non-Foundry address pre-mapping (ctypes VMM), multi-GPU support, NCCL warmup, SIGABRT handler, `/dev/shm` pre-copy |
| `profiling/modal_foundry_benchmark.py` | ~807 | A/B/C benchmark on Modal (single GPU): baseline vs save vs load |
| `profiling/modal_foundry_multi_gpu_benchmark.py` | ~422 | A/B/C benchmark on Modal (multi-GPU, tp=2) |

### Cache Directory Layout (Single GPU)

```
/root/.cache/foundry-graphs/<cache_key>/
├── .save_complete                    # Marker file
├── metadata.json                     # Model ID, region info, cursor positions,
│                                     # saved profile estimates, KV cache memory,
│                                     # passthrough_events (non-Foundry allocs)
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

### Cache Directory Layout (Multi-GPU)

```
/root/.cache/foundry-graphs/<cache_key>/
├── rank_0/
│   ├── .save_complete
│   ├── metadata.json                 # Per-rank passthrough_events
│   ├── hook_archive/
│   └── wrapper_graphs/
│       ├── graph_0.json
│       ├── graph_0.cugraph
│       └── ...
├── rank_1/
│   ├── .save_complete
│   ├── metadata.json
│   ├── hook_archive/
│   └── wrapper_graphs/
│       └── ...
└── ...
```
