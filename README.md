# vllm-cold-start

Cuts ~30s off vLLM cold starts by caching GPU initialization work across container boots. Supports single-GPU and multi-GPU (tensor parallelism) deployments.

## The problem

Every time vLLM starts on a fresh container, the same expensive GPU initialization runs again.

- Memory profiling runs a full forward pass (~16s) to measure available VRAM.
- CUDA graph capture builds graphs from scratch for every batch size (~3s).
- Kernel compilation compiles Triton kernels for all possible batch shapes upfront (~10s).

30 seconds of work. Same result every time. All on top of ~85 seconds spent downloading and loading model weights.

## What this does

Caches all three across cold starts so they only happen once.

- Memory profiling: runs the profiling pass with Foundry's allocator disabled (so side effects like cuBLAS workspace init still happen), but returns the saved KV cache memory value from the first run. Cursor alignment stays correct.
- CUDA graph capture: serializes graphs to disk using Foundry's deterministic GPU addressing. Restores them instead of recapturing.
- Kernel compilation: defers to lazy JIT. Only the kernels needed on first inference compile, not every possible variant.

## The hard parts

**Address determinism.** Foundry's bump allocator makes model weights and KV cache land at the same GPU addresses every boot. But not everything goes through Foundry. NCCL workspaces, cuBLAS handles, and other runtime allocations bypass the bump allocator and land at unpredictable addresses. These get baked into CUDA graph kernel parameters during capture. On reload, those addresses point to unmapped memory.

The integration solves this with a passthrough recording system. During the save run, Foundry records every non-bump-allocator allocation (source, address, size). On load, the same recording captures the new addresses. A size-based greedy matcher pairs save/load events, then binary-searches through every kernel parameter in every graph JSON to patch old addresses to new ones. For any remaining non-Foundry addresses not covered by passthrough matching, raw CUDA VMM APIs pre-map physical memory at those locations.

**Multi-GPU.** With tensor parallelism, each GPU rank runs in a separate subprocess. Foundry setup is deferred to each worker via a patched `init_device`. All ranks must agree on save vs. load mode before any NCCL collective fires, or the ranks deadlock. The parent process checks all per-rank caches before spawning workers and passes a global mode flag.

## Results

Tested on A100-SXM4-40GB with Qwen/Qwen2.5-7B-Instruct on Modal.

### Multi-GPU (tp=2, same-container comparison)

Baseline and cached runs on the **same container** to eliminate Modal volume I/O variance:

| Metric | Baseline | Cached | Savings |
|---|---|---|---|
| LLM init | 206.49s | 55.25s | 151.24s (73.2%) |
| Total (init + first inference) | 216.14s | 62.01s | 154.13s (71.3%) |
| CUDA graphs loaded | 0 (captured fresh) | 35/35 per rank | — |

### Single-GPU

| Metric | Baseline | Cached |
|---|---|---|
| GPU initialization | ~30s | <1s |
| Total cold start | ~137s | ~114s |
| First inference | 1.14s | 2.58s |

First inference is ~1.4s slower because kernel compilation is deferred. Only the kernels needed for the first request compile, instead of all possible variants compiling upfront.

## Why CUDA graphs are hard to save

A CUDA graph records a sequence of GPU operations at specific memory addresses. "Read weights from address 0xA, multiply with input at address 0xB, write output to address 0xC." Replaying the graph runs the same operations without CPU overhead.

The problem: GPU memory addresses change every time a process starts. Your model's weights land at different addresses on each boot. A saved graph still points to the old addresses. Replay breaks.

## How Foundry solves the address problem

Foundry is a library with an LD_PRELOAD hook intercepting all GPU memory allocations. Foundry forces every allocation through a bump allocator starting at a fixed base address (e.g., 0x10000000000).

First allocation gets the base address. Second gets base + size_of_first. Third gets base + size_of_first + size_of_second. And so on.

PyTorch model initialization is deterministic. Same model, same config, same allocation order. So every tensor lands at the same address on every boot. Saved CUDA graphs point to the right addresses. Replay works.

But not all allocations go through Foundry's bump allocator. NCCL, cuBLAS, and other CUDA runtime libraries allocate their own workspace buffers through the default CUDA allocator. These addresses are non-deterministic. The integration handles them with passthrough event recording and address patching (see FOUNDRY_INTEGRATION.md for details).

## How the integration works

The integration monkey-patches vLLM's initialization pipeline in two modes.

First boot (save mode):

1. Foundry's hook activates. All GPU allocations go through the bump allocator at fixed addresses.
2. Passthrough recording starts: non-bump allocations (NCCL, cuBLAS) are logged with their source, address, and size.
3. vLLM initializes normally. The model loads, memory gets profiled, CUDA graphs get captured.
4. The integration records everything to disk: profiling results, allocator positions, passthrough events, and all CUDA graphs with their kernel binaries.

Subsequent boots (load mode):

1. Foundry's hook activates at the same base address. Every model tensor lands at the same spot as the save run.
2. Passthrough recording captures the new non-Foundry addresses. A size-based matcher builds an old-to-new address map.
3. Memory profiling runs with the Foundry allocator temporarily disabled (for runtime side effects), but returns the saved KV cache memory value.
4. Each graph JSON is patched: kernel parameters containing old non-Foundry addresses get remapped to new ones. Any remaining unmapped addresses get physical memory pre-mapped via raw CUDA VMM calls.
5. CUDA graphs load from disk one at a time via `fdry.CUDAGraph.load()`, then get directly populated into vLLM's graph wrapper entries — skipping the entire capture loop.

See FOUNDRY_INTEGRATION.md for the full architecture and OPTIMIZATIONS.md for a detailed walkthrough of every optimization, including failed attempts.

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

For `tensor_parallel_size > 1`, Foundry setup is deferred to each worker subprocess via a patched `Worker.init_device`. Each GPU gets its own allocation region and per-rank graph cache (`rank_0/`, `rank_1/`, etc.). All ranks agree on save vs. load mode before any NCCL collective fires. Requires fork-based multiprocessing (Linux default for vLLM).

In multi-GPU mode, `disable_custom_all_reduce=True` is set to force the standard NCCL path, and an NCCL warmup allreduce runs during passthrough recording to capture NCCL's internal allocations.

## Requirements

- Linux + NVIDIA GPU (CUDA 12+)
- [Foundry fork](https://github.com/lokashrinav/foundry) with passthrough recording support (not upstream foundry-org/foundry)
- `LD_PRELOAD=libcuda_hook.so`
- vLLM 0.20+

### Foundry installation

This project requires a [fork of Foundry](https://github.com/lokashrinav/foundry) that adds passthrough recording APIs for tracking non-bump-allocator GPU allocations (NCCL, cuBLAS, PyTorch caching allocator). Upstream Foundry does not have these APIs.

```bash
git clone https://github.com/lokashrinav/foundry
cd foundry

# Patch for torch 2.10+ (c10::cuda::MemPool -> at::cuda::MemPool)
sed -i 's|c10::cuda::MemPool|at::cuda::MemPool|g' csrc/CUDAGraph.cpp csrc/CUDAGraphParallel.cpp
sed -i '/#include <c10\/cuda\/CUDACachingAllocator.h>/a #include <ATen/cuda/MemPool.h>' csrc/CUDAGraph.cpp csrc/CUDAGraphParallel.cpp

pip install -e . --no-build-isolation
```

## Project structure

```
src/vllm_profile_cache/
    foundry_graphs.py    # All monkey-patches, address patching, and main entry point (1608 lines)
    cache.py             # Profile cache key computation and read/write
    wrapper.py           # CLI wrapper for vllm serve
    modal_plugin.py      # Modal integration with Volume-backed cache

profiling/
    modal_foundry_benchmark.py             # A/B benchmark on Modal A100s (single GPU)
    modal_foundry_multi_gpu_benchmark.py   # A/B benchmark on Modal A100s (multi-GPU, tp=2)
    modal_foundry_matrix.py                # Parameter sweep benchmarks
    modal_foundry_introspect.py            # Debug introspection for Foundry internals
    modal_cache_benchmark.py               # A/B benchmark for profile caching (no Foundry)
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
