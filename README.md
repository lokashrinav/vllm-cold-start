# vllm-cold-start

Cuts ~30s off vLLM cold starts by caching GPU initialization work across container boots.

## The problem

Every time vLLM starts on a fresh container, the same expensive GPU initialization runs again.

- Memory profiling runs a full forward pass (~16s) to measure available VRAM.
- CUDA graph capture builds graphs from scratch for every batch size (~3s).
- Kernel compilation compiles Triton kernels for all possible batch shapes upfront (~10s).

30 seconds of work. Same result every time. All on top of ~85 seconds spent downloading and loading model weights.

## What this does

Caches all three across cold starts so they only happen once.

- Memory profiling: saves results and replays allocator state on subsequent boots. No more running a fresh forward pass every time.
- CUDA graph capture: serializes graphs to disk using Foundry's deterministic GPU addressing. Restores them instead of recapturing.
- Kernel compilation: defers to lazy JIT. Only the kernels needed on first inference compile, not every possible variant.

## The hard part

Getting graph restoration to run in the background during the 66-second weight download took the most work. The integration hooks into vLLM's init pipeline where GPU addresses are finalized but weights haven't started downloading. The graph builder gets 66 seconds of free overlap time. By the time vLLM needs the graphs, they're already ready.

## Results

Tested on A100-SXM4-40GB with Qwen/Qwen2.5-7B-Instruct on Modal.

| Metric | Baseline | Cached |
|---|---|---|
| GPU initialization | ~30s | <1s |
| Total cold start | 137s | 114s |
| First inference | 1.14s | 2.58s |

The 23s improvement comes from eliminating GPU initialization overhead. The remaining 114s is model weight downloading and loading.

First inference is ~1.4s slower because kernel compilation is deferred. Only the kernels needed for the first request compile, instead of all possible variants compiling upfront.

## How the integration works

The integration monkey-patches vLLM's initialization pipeline. On the first cold start (save mode), everything runs normally and gets written to disk. On subsequent cold starts (load mode), everything restores from cache.

Save path (first boot):
1. Foundry's LD_PRELOAD hook forces all GPU allocations to deterministic addresses via a bump allocator.
2. vLLM initializes normally. Model loads, memory gets profiled, CUDA graphs get captured.
3. All profiling results, cursor positions, and CUDA graphs save to disk.

Load path (subsequent boots):
1. Same deterministic addressing. Every tensor lands at the same GPU address as the save run.
2. Profiling is skipped. Saved results return directly, allocator cursor advances to match.
3. CUDA graphs restore from disk instead of recapturing.
4. Graph restoration runs in the background during weight download. Zero added latency.

See OPTIMIZATIONS.md for a detailed walkthrough of every optimization, including failed attempts.

## Usage

```python
from vllm_profile_cache.foundry_graphs import cached_vllm_init_with_foundry

llm = cached_vllm_init_with_foundry(
    model="Qwen/Qwen2.5-7B-Instruct",
    graph_cache_dir="/root/.cache/foundry-graphs",
)
```

First call captures and saves everything. Subsequent calls restore from cache.

## Requirements

- Linux + NVIDIA GPU (CUDA 12+)
- Foundry installed with LD_PRELOAD=libcuda_hook.so
- vLLM 0.20+
- tp_size=1 (single GPU)

## Project structure

```
src/vllm_profile_cache/
    foundry_graphs.py    # All monkey-patches and main entry point
    cache.py             # Profile cache key computation and read/write
    wrapper.py           # CLI wrapper for vllm serve
    modal_plugin.py      # Modal integration with Volume-backed cache

profiling/
    modal_foundry_benchmark.py   # A/B benchmark on Modal A100s
```

## Benchmarking

```bash
python -m modal run profiling/modal_foundry_benchmark.py
```

Runs a baseline (standard vLLM) and a cached run (with Foundry graph persistence) on Modal A100s, then prints a comparison.

## GPU memory snapshots

GPU memory snapshots (like Modal's) checkpoint the entire GPU state and restore the whole thing. A more complete approach for single-GPU setups. This project is complementary: works with standard vLLM out of the box and covers workloads running without snapshots.
