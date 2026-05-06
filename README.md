# vllm-cold-start

Cuts ~30s off vLLM cold starts by caching GPU initialization work that gets repeated on every container boot.

## The problem

Every time vLLM starts up on a fresh container, it repeats the same expensive GPU initialization:

- **Memory profiling** — runs a full forward pass (~16s) just to measure how much VRAM is available
- **CUDA graph capture** — builds CUDA graphs from scratch for every batch size (~3s)
- **Kernel compilation** — compiles Triton kernels for all possible batch shapes upfront (~10s)

That's ~30 seconds of work that produces the same result every time, on top of the ~85 seconds already spent downloading and loading model weights.

## What this does

Caches all three across cold starts so they only happen once:

- **Memory profiling** — saves the profiling results and replays allocator state on subsequent boots, instead of running a fresh forward pass each time
- **CUDA graph capture** — serializes CUDA graphs to disk using [Foundry's](https://github.com/FoundryAI/foundry) deterministic GPU addressing, then restores them instead of recapturing
- **Kernel compilation** — defers compilation to lazy JIT, so only the kernels actually needed on first inference get compiled, not every possible variant

The hardest part was getting graph restoration to happen in the background during the 66-second weight download. The integration hooks into vLLM's init pipeline at the point where GPU addresses are finalized but weights haven't started downloading — giving the graph builder 66 seconds of free overlap time to work in the background. By the time vLLM asks for the graphs, they're already ready.

## Results

Tested on A100-SXM4-40GB with Qwen/Qwen2.5-7B-Instruct on Modal:

| | Baseline | Cached |
|---|---|---|
| GPU initialization | ~30s | <1s |
| Total cold start | 137s | 114s |
| First inference | 1.14s | 2.58s |

The 23s total improvement comes entirely from eliminating GPU initialization overhead. The remaining 114s is model weight downloading and loading, which this project doesn't touch.

First inference is ~1.4s slower because kernel compilation is deferred — only the kernels needed for the actual first request compile, instead of all possible variants compiling upfront during init.

## How it works

The integration monkey-patches vLLM's initialization pipeline. On the first cold start (save mode), it captures everything normally and writes results to disk. On subsequent cold starts (load mode), it restores everything from cache.

**Save path (first boot):**
1. Foundry's LD_PRELOAD hook forces all GPU allocations to deterministic addresses via a bump allocator
2. vLLM initializes normally — model loads, memory gets profiled, CUDA graphs get captured
3. All profiling results, cursor positions, and CUDA graphs are saved to disk

**Load path (subsequent boots):**
1. Same deterministic addressing — every tensor ends up at the same GPU address as the save run
2. Profiling is skipped — saved results are returned, allocator cursor is advanced to match
3. CUDA graphs are restored from disk instead of recaptured
4. Graph restoration runs in the background during weight download (zero added latency)

See [OPTIMIZATIONS.md](OPTIMIZATIONS.md) for a detailed technical walkthrough of every optimization, including the ones that didn't work.

## Usage

```python
from vllm_profile_cache.foundry_graphs import cached_vllm_init_with_foundry

# Drop-in replacement for vllm.LLM()
llm = cached_vllm_init_with_foundry(
    model="Qwen/Qwen2.5-7B-Instruct",
    graph_cache_dir="/root/.cache/foundry-graphs",
)
```

First call captures and saves everything. Subsequent calls restore from cache.

## Requirements

- Linux + NVIDIA GPU (CUDA 12+)
- [Foundry](https://github.com/FoundryAI/foundry) installed with `LD_PRELOAD=libcuda_hook.so`
- vLLM 0.20+
- `tp_size=1` (single GPU)

## Project structure

```
src/vllm_profile_cache/
    foundry_graphs.py    # All monkey-patches and the main entry point
    cache.py             # Profile cache key computation and read/write
    wrapper.py           # CLI wrapper for vllm serve
    modal_plugin.py      # Modal integration with Volume-backed cache

profiling/
    modal_foundry_benchmark.py   # A/B benchmark: baseline vs cached on Modal
    modal_cache_benchmark.py     # Profile cache benchmark
```

## Benchmarking

```bash
python -m modal run profiling/modal_foundry_benchmark.py
```

Runs both a baseline (standard vLLM) and a cached run (with Foundry graph persistence) on Modal A100s, then prints a comparison.

## Relationship to GPU memory snapshots

GPU memory snapshots (like Modal's) checkpoint the entire GPU state and restore it wholesale — a more powerful approach that can skip weight loading entirely. This project is complementary: it works with standard vLLM out of the box and covers workloads running without snapshots.
