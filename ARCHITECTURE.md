# vllm-cold-start — Architecture

## Problem

Every vLLM cold start runs expensive GPU initialization that produces the same results for identical model + GPU + config:

| Phase | Time (A100, 7B model) | Cacheable? |
|-------|----------------------|------------|
| Memory profiling (`determine_available_memory`) | ~16s | Yes |
| CUDA graph capture (35 graphs for decode) | ~3-9s | Yes |
| Triton kernel compilation (`_dummy_sampler_run`) | ~10s | Yes (defer to lazy JIT) |
| Model weight download + loading | ~85s | No (network/PCIe bound) |

## Solution

Two complementary caching layers:

1. **Profile cache** (`cache.py`, `modal_plugin.py`) — caches `kv_cache_memory_bytes` keyed by a SHA-256 hash of 18 config/environment parameters. Standalone, no Foundry needed.

2. **Foundry integration** (`foundry_graphs.py`) — serializes CUDA graphs to disk using Foundry's deterministic VMM. Handles memory profiling, graph capture, kernel compilation deferral, address patching for non-Foundry allocations, and multi-GPU support.

## File Structure

```
src/vllm_profile_cache/
├── foundry_graphs.py  — Foundry integration: monkey-patches, passthrough recording,
│                         address patching, multi-GPU, SIGABRT handler (1608 lines)
├── cache.py           — Profile cache: key computation, read/write, safety margins
├── modal_plugin.py    — Modal integration: Volume-backed cache, OOM fallback
├── wrapper.py         — CLI wrapper for `vllm serve`
└── __init__.py

profiling/
├── modal_foundry_benchmark.py            — A/B benchmark on Modal A100s (single GPU)
├── modal_foundry_multi_gpu_benchmark.py  — A/B benchmark on Modal A100s (tp=2)
├── modal_foundry_matrix.py               — Parameter sweep benchmarks
├── modal_foundry_introspect.py           — Foundry internals debug introspection
├── modal_cache_benchmark.py              — A/B benchmark for profile caching
├── modal_cache_smoke.py                  — Quick smoke test for profile cache
├── modal_benchmark.py                    — General vLLM cold start measurement
├── modal_speed_proof.py                  — Speed proof benchmark
├── measure_cold_start.py                 — Local cold start measurement
└── trace_imports.py                      — Import time tracing
```

## foundry_graphs.py — Core Architecture

### Entry Points

- `cached_vllm_init_with_foundry(model, ...)` — Drop-in replacement for `vllm.LLM()`. Routes to single-GPU or multi-GPU path based on `tensor_parallel_size`.
- `_cached_vllm_init_multi_gpu(...)` — Multi-GPU path: defers Foundry setup to each worker subprocess via patched `Worker.init_device`.

### Key Classes

- `FoundryGraphCache` — Manages serialized graph files on disk (metadata, graph JSONs, fatbin archive, save-complete marker).
- `_FoundryPatchState` — Shared mutable state across all monkey-patches. Tracks cursor positions, profiling data, loaded graphs, passthrough events, guard flags.

### Monkey-Patches (applied by `patch_vllm_for_foundry`)

| Patch | Target | Purpose |
|-------|--------|---------|
| 0 | `GPUModelRunner.profile_cudagraph_memory` | Skip (load) or instrument (save) graph memory profiling |
| 0b | `GPUWorker.determine_available_memory` | Run with Foundry disabled for side effects, return saved value |
| 1 | `GPUModelRunner.capture_model` | Graph loading + direct-populate (load) or phase flag + finalize (save) |
| 1b | `GPUWorker.compile_or_warm_up_model` | Timing instrumentation |
| 1b2 | `GPUModelRunner._warmup_and_capture` | Skip warmup forward passes (load mode safety net) |
| 1b3 | `GPUModelRunner._dummy_sampler_run` | Skip Triton JIT compilation (load mode) |
| 2 | `CUDAGraphWrapper.__call__` | Intercept FULL mode: Foundry capture (save) / preloaded return (load) |
| 3 | `BaseModelLoader.load_model` | Hook load_weights for early graph builds (currently passthrough only) |

### Passthrough Recording System

Non-Foundry allocations (NCCL workspaces, cuBLAS handles, CUDA runtime buffers) bypass the bump allocator and get non-deterministic addresses. These get baked into CUDA graph kernel parameters during capture. On reload, those addresses point to unmapped memory.

**Save path:**
1. `fdry.stop_allocation_region()` → disable bump allocator
2. `fdry.start_passthrough_record()` → start recording non-bump allocations
3. `Worker.init_device()` runs → NCCL init, cuBLAS workspace, etc. get recorded
4. `fdry.resume_allocation_region()` → re-enable bump allocator
5. Extended recording continues through `capture_model` (captures post-init allocations)
6. `fdry.end_passthrough_record()` → events saved to metadata

**Load path:**
1. Same passthrough recording captures new addresses
2. `_build_passthrough_addr_map()` — size-based greedy matching: each save event matched to first unmatched load event with same size
3. `_patch_graph_json_addresses()` — binary-search patch of kernel parameters in every graph JSON
4. `_premap_non_foundry_addresses()` — scan graphs for remaining non-Foundry pointers, pre-map physical memory via raw CUDA VMM APIs (ctypes → `cuMemAddressReserve` + `cuMemCreate` + `cuMemMap` + `cuMemSetAccess`)
5. `_diagnose_unpatched_addresses()` — log any pointers still not covered

### Graph Loading (Load Mode)

Graphs are loaded one at a time via `fdry.CUDAGraph.load(path, pool)`:

1. Passthrough events matched and addresses patched in graph JSONs
2. Non-Foundry addresses pre-mapped
3. Each graph loaded sequentially
4. Loaded graphs directly populated into `CUDAGraphWrapper.concrete_cudagraph_entries` — bypasses entire capture loop

### Multi-GPU Architecture

```
Parent Process:
  ├── resolve_graph_cache_dir() → cache_key
  ├── Check all rank caches → global_load_mode (prevents NCCL deadlock)
  ├── Patch GPUWorker.init_device → _foundry_init_device
  ├── Set disable_custom_all_reduce=True (force NCCL path)
  └── LLM(model, tensor_parallel_size=N, ...)

Worker Subprocess (per rank):
  ├── _foundry_init_device():
  │   ├── Per-rank cache dir: {base_dir}/rank_{rank}/
  │   ├── Load CUDA modules from hook_archive (if load mode)
  │   ├── setup_foundry_regions() per-GPU
  │   ├── patch_vllm_for_foundry() with rank-specific cache
  │   ├── Start passthrough recording
  │   ├── Original init_device() → NCCL init, model architecture
  │   ├── Resume Foundry allocator, keep recording active
  │   └── NCCL warmup allreduce (load mode, during recording)
  └── Rest of vLLM init uses patched methods
```

### SIGABRT Handler

Foundry's `[CGE BUILD]` background thread can `abort()` when it encounters unrecognized kernel binary hashes (JIT-compiled Triton kernels). A C-level signal handler (via ctypes → `libc.signal()`) intercepts SIGABRT and calls `_exit(0)` to survive cleanly. C-level because Python's `signal.signal()` only catches signals on the main thread.

## cache.py — Profile Cache

### Cache Key

SHA-256 hash of 18 parameters:
```
model_id | gpu_name | gpu_identity | dtype | tp_size | pp_size |
max_model_len | max_num_batched_tokens | max_num_seqs |
vllm_version | cuda_version | torch_version | driver_version |
gpu_total_memory_bytes | gpu_memory_utilization |
quantization | kv_cache_dtype | enforce_eager
```

### Safety Mechanisms

1. **5% safety margin** on cached values
2. **Free memory check** before using cache
3. **OOM fallback** — catches OOM, deletes cache, retries with profiling
4. **7-day TTL** on cache entries
5. **MoE exclusion** — non-deterministic expert routing

## What This Does NOT Cache

- torch.compile/Inductor compilation (~25-45s) — vLLM handles this with its own compile cache
- Model weight download/loading (~85s) — network/PCIe bound
- Python imports (~10s) — interpreter startup
