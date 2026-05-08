# vllm-cold-start

Eliminate ~30s of vLLM cold start overhead by caching GPU initialization work
(memory profiling, CUDA graph capture, kernel compilation) across container boots.
Supports single-GPU and multi-GPU (tensor parallelism) deployments.

## Project structure

- `src/vllm_profile_cache/foundry_graphs.py` — All monkey-patches, passthrough recording, address patching, multi-GPU support (1608 lines)
- `src/vllm_profile_cache/cache.py` — Profile cache key computation, read/write, safety margins
- `src/vllm_profile_cache/wrapper.py` — CLI wrapper for `vllm serve`
- `src/vllm_profile_cache/modal_plugin.py` — Modal integration with Volume-backed cache
- `profiling/modal_foundry_benchmark.py` — A/B benchmark for Foundry (single GPU)
- `profiling/modal_foundry_multi_gpu_benchmark.py` — A/B benchmark for Foundry (multi-GPU, tp=2)
- `profiling/modal_cache_benchmark.py` — A/B benchmark for profile caching (no Foundry)

## Foundry CUDA graph persistence

Integrates [Foundry](https://github.com/foundry-org/foundry) to serialize CUDA graphs
to disk and restore them on subsequent cold starts, eliminating the graph capture phase
(~3-60s depending on model size).

- Requires `LD_PRELOAD=libcuda_hook.so` (Foundry's driver interception hook)
- Monkey-patches `CUDAGraphWrapper.__call__` for FULL mode graph capture/load
- Only FULL mode graphs are serialized; PIECEWISE mode uses torch.compile internals
- Supports single-GPU (tp=1) and multi-GPU (tp>1) via per-rank caching
- Falls back to standard vLLM if Foundry is not available

## Key architecture details

- **Passthrough recording**: Non-Foundry allocations (NCCL, cuBLAS) are recorded during save and load. Size-based greedy matching builds old→new address maps.
- **Address patching**: Graph JSON kernel params are binary-search patched with remapped addresses. Remaining non-Foundry addresses get physical memory pre-mapped via raw CUDA VMM APIs (ctypes → libcuda.so).
- **Multi-GPU**: `_cached_vllm_init_multi_gpu` defers Foundry setup to each worker subprocess. Per-rank cache dirs (`rank_0/`, `rank_1/`). Global save/load mode decided in parent to avoid NCCL deadlocks.
- **NCCL warmup**: In load mode, an allreduce warmup runs during passthrough recording to capture NCCL's internal allocations before graph loading.

## Safety constraints

- MoE models are excluded from profile caching (per vLLM RFC #27951)
- 5% safety margin on cached profile values
- Cache auto-invalidates on vLLM version, CUDA version, PyTorch version, or config change
- SIGABRT handler survives Foundry's CGE BUILD thread abort()
