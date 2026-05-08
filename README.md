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

## Why CUDA graphs are hard to save

A CUDA graph records a sequence of GPU operations at specific memory addresses. "Read weights from address 0xA, multiply with input at address 0xB, write output to address 0xC." Replaying the graph runs the same operations without CPU overhead.

The problem: GPU memory addresses change every time a process starts. Your model's weights land at different addresses on each boot. A saved graph still points to the old addresses. Replay breaks.

## How Foundry solves the address problem

Foundry is a library with an LD_PRELOAD hook intercepting all GPU memory allocations. Foundry forces every allocation through a bump allocator starting at a fixed base address (e.g., 0x10000000000).

First allocation gets the base address. Second gets base + size_of_first. Third gets base + size_of_first + size_of_second. And so on.

PyTorch model initialization is deterministic. Same model, same config, same allocation order. So every tensor lands at the same address on every boot. Saved CUDA graphs point to the right addresses. Replay works.

## How the integration works

The integration monkey-patches vLLM's initialization pipeline in two modes.

First boot (save mode):

1. Foundry's hook activates. All GPU allocations go through the bump allocator at fixed addresses.
2. vLLM initializes normally. The model loads, memory gets profiled, CUDA graphs get captured.
3. The integration records everything to disk: profiling results, allocator positions, and all CUDA graphs with their kernel binaries.

Subsequent boots (load mode):

1. Foundry's hook activates at the same base address. Every tensor lands at the same spot as the save run.
2. Memory profiling is skipped. The saved result returns directly. The allocator cursor advances forward to match where profiling would have left off.
3. CUDA graphs restore from disk instead of recapturing. Foundry rebuilds the graph topology and links each node back to the correct GPU addresses.
4. The integration starts graph restoration in the background before model weights start downloading. Weight download takes ~66 seconds. Graph restoration takes <1 second. By the time vLLM needs the graphs, they're already done.

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

### Multi-GPU (tensor parallelism)

```python
llm = cached_vllm_init_with_foundry(
    model="Qwen/Qwen2.5-7B-Instruct",
    graph_cache_dir="/root/.cache/foundry-graphs",
    tensor_parallel_size=2,
)
```

For `tensor_parallel_size > 1`, Foundry setup is deferred to each worker subprocess. Each GPU gets its own allocation region and per-rank graph cache. Requires fork-based multiprocessing (Linux default for vLLM).

## Requirements

- Linux + NVIDIA GPU (CUDA 12+)
- Foundry installed with LD_PRELOAD=libcuda_hook.so
- vLLM 0.20+

## Project structure

```
src/vllm_profile_cache/
    foundry_graphs.py    # All monkey-patches and main entry point
    cache.py             # Profile cache key computation and read/write
    wrapper.py           # CLI wrapper for vllm serve
    modal_plugin.py      # Modal integration with Volume-backed cache

profiling/
    modal_foundry_benchmark.py             # A/B benchmark on Modal A100s (single GPU)
    modal_foundry_multi_gpu_benchmark.py   # A/B benchmark on Modal A100s (multi-GPU)
```

## Benchmarking

```bash
# Single GPU
python -m modal run profiling/modal_foundry_benchmark.py

# Multi-GPU (2x A100)
python -m modal run profiling/modal_foundry_multi_gpu_benchmark.py
```

Runs a baseline (standard vLLM) and a cached run (with Foundry graph persistence) on Modal A100s, then prints a comparison.

## GPU memory snapshots

GPU memory snapshots (like Modal's) checkpoint the entire GPU state and restore the whole thing. A more complete approach for single-GPU setups. This project is complementary: works with standard vLLM out of the box and covers workloads running without snapshots.
