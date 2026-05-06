# vllm-profile-cache

Cache vLLM's memory profiling results to skip the ~18s `determine_available_memory`
forward pass on subsequent cold starts.

## The Problem

Every vLLM cold start runs `determine_available_memory()` in `gpu_worker.py`,
which executes a full forward pass with dummy data just to measure peak VRAM usage.
This takes 18-20s and happens EVERY cold start, even with warm torch.compile and
CUDA graph caches.

## The Solution

vLLM PR #21489 added `--kv-cache-memory-bytes` which skips profiling entirely.
vLLM even prints the recommended value in its logs. We cache that value to disk
(keyed by model + GPU + config hash) and inject it on subsequent launches.

## Key vLLM code locations

- `vllm/v1/worker/gpu_worker.py:325-477` — `determine_available_memory()`
- `vllm/v1/worker/gpu_worker.py:601-656` — log line printing recommended kv_cache_memory
- `vllm/config/cache.py:158` — `kv_cache_memory_bytes` field
- `vllm/v1/worker/gpu_model_runner.py:5811-5884` — `profile_run()` forward pass

## Project structure

- `src/vllm_profile_cache/cache.py` — cache key computation, read/write, safety margins
- `src/vllm_profile_cache/wrapper.py` — CLI wrapper for `vllm serve`
- `src/vllm_profile_cache/modal_plugin.py` — Modal integration with Volume-backed cache
- `src/vllm_profile_cache/foundry_graphs.py` — Foundry integration for CUDA graph persistence
- `profiling/modal_cache_benchmark.py` — A/B benchmark on Modal (cached vs uncached)
- `profiling/modal_foundry_benchmark.py` — A/B benchmark for Foundry graph caching

## Foundry CUDA graph persistence

Integrates [Foundry](https://github.com/foundry-org/foundry) to serialize CUDA graphs
to disk and restore them on subsequent cold starts, eliminating the graph capture phase
(~3-60s depending on model size).

- Requires `LD_PRELOAD=libcuda_hook.so` (Foundry's driver interception hook)
- Monkey-patches `CudaGraphManager.capture()` to use Foundry's `fdry.CUDAGraph`
- Only FULL mode graphs are serialized; PIECEWISE mode uses torch.compile internals
- Currently supports `tp_size=1` (single GPU)
- Falls back to standard vLLM if Foundry is not available

## Safety constraints

- MoE models are excluded (per vLLM RFC #27951 — data-dependent expert routing)
- 5% safety margin applied to cached values
- Cache auto-invalidates on vLLM version, CUDA version, or GPU change
