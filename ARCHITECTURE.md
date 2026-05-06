# vllm-profile-cache — Architecture

## Problem

Every vLLM cold start runs `determine_available_memory()` in `gpu_worker.py`, which executes a full forward pass with dummy data and profiles CUDA graph memory just to measure how much VRAM is available for KV cache. This takes ~14-18s and happens on **every** cold start, even when the model, GPU, and config are identical to the previous run.

vLLM already has a `--kv-cache-memory-bytes` flag (PR #21489) that skips profiling entirely, and even prints the recommended value in its logs — but never persists it.

## Solution

Cache the profiling result to disk (keyed by a deterministic hash of every parameter that affects memory profiling) and inject `kv_cache_memory_bytes` on subsequent launches.

```
First cold start:
  vLLM init → profile_run() → profile_cudagraph_memory() → compute available_kv_cache_memory
  → extract value from engine state → save to cache file

Subsequent cold starts:
  cache hit → check free memory → inject kv_cache_memory_bytes → vLLM skips profiling
  (if OOM: delete cache entry, retry with normal profiling)
```

## Benchmark Results

**A100-SXM4-40GB, vLLM 0.20.1, Qwen2.5-0.5B-Instruct:**

| Run | LLM Init | Total |
|-----|----------|-------|
| Baseline (no cache) | 126.38s | 135.19s |
| First run (saves cache) | 92.19s | 98.27s |
| Cache hit (skips profiling) | 58.70s | 59.64s |
| **Savings vs baseline** | **67.69s (53.6%)** | **75.55s (55.9%)** |

The profiling skip itself saves ~14s. Additional savings come from the torch.compile cache being warm on repeat launches. On a real production deployment where the compile cache is already warm, the profile cache saves ~14s per cold start.

## Why This Is Dangerous

When you pass `kv_cache_memory_bytes`, you tell vLLM: "I already know the safe amount. Do not calculate it again." If that number is wrong, vLLM will try to allocate more memory than is available, causing OOM at startup, during CUDA graph capture, or on the first real request.

The cached number can become wrong when:

1. `max_num_batched_tokens` changed → different peak activation memory
2. `max_num_seqs` changed → different scheduling memory pressure
3. `max_model_len` changed → different KV cache token coverage
4. CUDA graph capture sizes changed → different graph memory reservation
5. Attention backend changed → different memory usage patterns
6. vLLM/PyTorch/CUDA/driver version changed → different allocator behavior
7. Quantization or KV cache dtype changed → different per-token memory
8. TP/PP size changed → different per-GPU memory distribution
9. GPU type or MIG slice changed → different total memory
10. Another process is using GPU memory → less free memory available
11. MoE model → data-dependent expert routing makes profiling non-deterministic

## How We Make It Safe

### 1. Exhaustive cache key

The cache key is a SHA-256 hash of **every** parameter that affects memory profiling:

```
model_id | gpu_name | gpu_identity | dtype | tp_size | pp_size |
max_model_len | max_num_batched_tokens | max_num_seqs |
vllm_version | cuda_version | torch_version | driver_version |
gpu_total_memory_bytes | gpu_memory_utilization |
quantization | kv_cache_dtype | enforce_eager
```

Any change to any of these invalidates the cache automatically. By default,
`gpu_identity` is scoped to GPU type: `gpu_name` plus
`gpu_total_memory_bytes`. This lets cloud pools such as Modal reuse a profile
when a later cold start lands on a different physical A100 with the same memory
shape. For strict per-device caching, set `gpu_identity_scope="device"` or
`VLLM_PROFILE_CACHE_GPU_SCOPE=device`; this includes the physical `gpu_uuid` in
the key. The actual `gpu_uuid` is always stored in cache metadata for audit/debug
purposes.

### 2. Safety margin

Cached values are reduced by 5% before injection. If vLLM profiled 40 GB, we inject 38 GB. This absorbs minor memory variance between runs (driver allocations, CUDA context size differences, etc.).

### 3. Free memory check

Before using a cached value, we check current free GPU memory against what was free when the cache was created. If free memory dropped significantly (e.g., another process is using the GPU), we skip the cache and fall back to normal profiling.

### 4. OOM fallback

If vLLM OOMs with a cached value, we catch it, delete the cache entry, and retry with normal profiling. The user never sees a crash — just a slower startup.

### 5. TTL expiration

Cache entries expire after 7 days by default. This catches cases where the environment changed in ways the cache key doesn't cover (e.g., a driver update that didn't change the version string, or a system-level memory configuration change).

### 6. MoE exclusion

MoE models (DeepSeek, Mixtral, Qwen-MoE, DBRX, Jamba, Arctic, Grok) are excluded entirely. Per vLLM RFC #27951, their data-dependent expert routing makes memory profiling non-deterministic — the same model can use different amounts of memory on different inputs.

## Architecture Details

### Cache Entry

JSON file stored at `~/.cache/vllm-profile-cache/{hash}.json`:

```json
{
  "kv_cache_memory_bytes": 31851479040,
  "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
  "gpu_name": "NVIDIA A100-SXM4-40GB",
  "gpu_uuid": "GPU-12345678-abcd-...",
  "dtype": "auto",
  "tp_size": 1,
  "pp_size": 1,
  "max_model_len": 32768,
  "max_num_batched_tokens": 8192,
  "max_num_seqs": 256,
  "vllm_version": "0.20.1",
  "cuda_version": "12.8",
  "torch_version": "2.11.0",
  "driver_version": "570.86.15",
  "gpu_total_memory_bytes": 42949672960,
  "gpu_memory_utilization": 0.8,
  "safety_margin_pct": 5.0,
  "free_memory_at_cache_time": 41523456000,
  "created_at": 1777856667.0
}
```

### Extracting kv_cache_memory_bytes from vLLM V1

vLLM V1 runs the engine core in a **separate process** (via `spawn`). The raw `available_kv_cache_memory_bytes` computed in `gpu_worker.py` is not accessible from the parent process. Only `num_gpu_blocks` is sent back via IPC (`EngineCoreReadyResponse`).

We reconstruct the value:

```
kv_cache_memory = num_gpu_blocks × page_size_per_layer × num_layers
where page_size_per_layer = 2 × block_size × kv_heads_per_gpu × head_size × dtype_size
```

The model's KV cache geometry (layers, kv_heads, head_size) comes from `transformers.AutoConfig`.

### torch.compile Cache Hash Fix

vLLM's `CacheConfig.compute_hash()` includes `kv_cache_memory_bytes` in the factors used to compute the torch.compile cache directory. This means injecting a cached value changes the hash, which forces a **full recompilation** (~25-45s) — defeating the purpose of the profile cache.

`kv_cache_memory_bytes` controls memory allocation, not computation graph structure. We monkey-patch `CacheConfig.compute_hash()` to exclude it from the hash, preserving the compile cache across cached/uncached launches.

## File Structure

```
src/vllm_profile_cache/
├── cache.py           — Cache key computation, read/write, safety margins
├── wrapper.py         — CLI wrapper for `vllm serve` (parses logs for kv_cache value)
├── modal_plugin.py    — Modal integration (extracts value from engine state)
├── foundry_graphs.py  — Foundry integration for CUDA graph persistence
└── __init__.py

profiling/
├── modal_cache_benchmark.py    — A/B benchmark for profile caching
└── modal_foundry_benchmark.py  — A/B benchmark for Foundry graph caching
```

### cache.py

- `compute_cache_key()` — SHA-256 hash of 18 config/environment parameters
- `ProfileCache` — get (with TTL)/put/invalidate/list_entries on JSON files
- `CacheEntry` — dataclass with kv_cache_memory_bytes + full metadata
- `build_cache_key_from_vllm_config()` — detects GPU UUID, driver version, PyTorch version, etc.
- `check_free_memory()` — compares current free GPU memory against cache-time snapshot
- `apply_safety_margin()` — reduces cached value by configurable percentage
- `_is_moe_model()` — checks against MoE denylist
- `_get_gpu_uuid()` — nvidia-smi query for GPU/MIG UUID
- `_get_driver_version()` — nvidia-smi query for driver version
- `_get_free_gpu_memory()` — torch.cuda.mem_get_info for current free memory

### modal_plugin.py

- `cached_vllm_init()` — drop-in replacement for `vllm.LLM()` with caching, free memory check, and OOM fallback
- `_extract_kv_cache_bytes_from_engine()` — reconstructs kv_cache_memory from num_gpu_blocks + model geometry
- `_patch_cache_config_hash()` — excludes kv_cache_memory_bytes from compile cache hash

### wrapper.py

- `launch_vllm()` — wraps `vllm serve`, injects `--kv-cache-memory-bytes` from cache
- `parse_kv_cache_from_logs()` — regex extraction from vLLM log output
- `_parse_vllm_args()` — extracts key vLLM arguments from command line

### foundry_graphs.py

Integrates [Foundry](https://github.com/foundry-org/foundry) to serialize and restore CUDA graphs across cold starts, eliminating the graph capture phase (~3-60s depending on model size).

**How Foundry works**: Foundry uses an `LD_PRELOAD` hook (`libcuda_hook.so`) to intercept all CUDA driver calls and force GPU memory allocations into a fixed virtual address range. This means GPU pointers are identical between save and load runs — no pointer relocation needed. Graphs are serialized with their kernel binaries, parameters, and topology.

**Key functions**:

- `is_foundry_available()` — checks for `foundry` import + `LD_PRELOAD=libcuda_hook.so`
- `setup_foundry_regions()` — calls `fdry.set_allocation_region()` to fix the VMM base address before any GPU allocation
- `patch_vllm_for_foundry(graph_cache_dir)` — monkey-patches `CudaGraphManager.capture()` to use Foundry's `fdry.CUDAGraph` and `fdry.graph()` (drop-in replacements for PyTorch equivalents)
- `cached_vllm_init_with_foundry(model, ...)` — drop-in replacement for `vllm.LLM()` that sets up regions, patches capture, and falls back to standard vLLM if Foundry is unavailable
- `FoundryGraphCache` — manages serialized graph files on disk (save metadata, check for cached graphs, clear cache)

**Capture flow (patched)**:

1. PIECEWISE mode graphs: captured normally (not serialized — they use torch.compile internals)
2. FULL mode graphs: captured with `fdry.CUDAGraph()` + `fdry.graph()`, then saved via `graph.save()` + `fdry.save_graph_manifest()`

**Load flow**:

1. Warmup forward pass for each desc (initializes buffers at deterministic addresses)
2. PIECEWISE graphs: captured normally
3. FULL graphs: loaded from disk via `fdry.CUDAGraph.start_graph_builds()` + `finish_graph_loads()` (multi-threaded, ~100ms for 512 graphs)
4. Falls back to full capture if cache is corrupt or missing

**Constraints**:

- Requires `LD_PRELOAD=libcuda_hook.so` set before process start (before `libcuda.so` is loaded)
- Currently supports `tp_size=1` only (in-process engine core, no subprocess)
- Linux only

### modal_foundry_benchmark.py

Three-phase A/B/C benchmark on Modal:

1. **Baseline** — standard vLLM, no Foundry, measures normal cold start
2. **Foundry first run** — capture + save graphs to a Modal Volume
3. **Foundry cache hit** — load graphs from Volume, skip FULL capture

Uses `run_with_foundry_subprocess()` to spawn a child process with `LD_PRELOAD` properly set (since `LD_PRELOAD` must be active before `libcuda.so` loads, which happens before Modal function code runs).

The graph cache persists across runs via `modal.Volume`.

## Bugs Fixed During Development

1. **`total_mem` → `total_memory`** (`cache.py`): PyTorch's `cuda.get_device_properties()` uses `total_memory`, not `total_mem`.

2. **`kv_cache_dtype` → `cache_dtype`** (`modal_plugin.py`): vLLM 0.20.1 renamed the CacheConfig field from `kv_cache_dtype` to `cache_dtype`.

3. **Compile cache invalidation** (`modal_plugin.py`): Injecting `kv_cache_memory_bytes` changed vLLM's torch.compile cache hash, forcing full recompilation on every cached launch. Fixed by patching the hash to exclude this field.

## What This Project Addresses

| Cold start phase | Time | Solution |
|------------------|------|----------|
| Memory profiling | ~14s | **Profile cache** (`cache.py`, `modal_plugin.py`) — caches `kv_cache_memory_bytes` keyed by config hash |
| CUDA graph capture | 3-60s | **Foundry integration** (`foundry_graphs.py`) — serializes graphs to disk, loads on subsequent starts |

## What This Does NOT Cache

- torch.compile/Inductor compilation (~25-45s) — vLLM handles this with its own compile cache
- Model weight loading (~1-2s) — file I/O, not profiling
- Python imports (~10s) — interpreter startup
