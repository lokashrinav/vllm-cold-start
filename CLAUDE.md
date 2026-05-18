# vllm-cold-start

Eliminate ~30s of vLLM cold start overhead by caching GPU initialization work
(memory profiling, CUDA graph capture, kernel compilation) across container boots.
Supports single-GPU and multi-GPU (tensor parallelism) deployments.

## Project structure

- `src/vllm_cuda_ckpt/` — **CUDA checkpoint/restore package** (pip installable as `vllm-ckpt` CLI)
  - `api.py` — CudaCheckpointAPI (ctypes bindings) and VLLMCheckpointer (Python API orchestrator)
  - `discover.py` — CUDA PID discovery for vLLM process trees
  - `cli.py` — CLI entrypoint: `vllm-ckpt discover|checkpoint|restore|cycle|benchmark|watch|recommend`
- `src/vllm_profile_cache/foundry_graphs.py` — All monkey-patches, passthrough recording, address patching, multi-GPU support (1608 lines)
- `src/vllm_profile_cache/cache.py` — Profile cache key computation, read/write, safety margins
- `src/vllm_profile_cache/wrapper.py` — CLI wrapper for `vllm serve`
- `src/vllm_profile_cache/modal_plugin.py` — Modal integration with Volume-backed cache
- `profiling/modal_foundry_benchmark.py` — A/B benchmark for Foundry (single GPU)
- `profiling/modal_foundry_multi_gpu_benchmark.py` — A/B benchmark for Foundry (multi-GPU, tp=2)
- `profiling/modal_cache_benchmark.py` — A/B benchmark for profile caching (no Foundry)

## cuda_serializer — Independent CUDA graph serialization (in development)

- `cuda_serializer/ARCHITECTURE.md` — Complete 16-section design document. Read this first.
- `cuda_serializer/RESEARCH.md` — Working knowledge base: empirical findings, dead ends, open questions, optimization hypotheses, competitive landscape. Read this to recover full context from prior conversations.
- `cuda_serializer/intercept.c` — Phase 1 LD_PRELOAD hook (working, tested on Modal T4)
- `cuda_serializer/modal_phase2.py` — Phase 2-6b: full pipeline (hooks, VMM, serialize, restore, launch). All C code embedded as INTERCEPT_PHASE2_C string.
- `cuda_serializer/modal_vllm_benchmark.py` — Main A/B benchmark (LD_PRELOAD approach, Phase 34)
- `cuda_serializer/modal_snapshot_bench.py` — Phase 37: Modal native GPU snapshot benchmark (deploy + invoke)
- `cuda_serializer/invoke_snapshot.py` — Invoke deployed GPU snapshot benchmark with multi-run support
- `cuda_serializer/modal_cuda_checkpoint_bench.py` — Phase 38: NVIDIA cuda-checkpoint benchmark (in-process cycle, multi-cycle support)
- `cuda_serializer/modal_cuda_api_bench.py` — Phase 40: Direct CUDA Driver API vs CLI benchmark (ctypes → libcuda.so.1)
- `cuda_serializer/modal_suspend_bench.py` — Phase 40b: 2-step vs 4-step API comparison (Suspend/Resume NOT available on driver 580)
- `cuda_serializer/modal_kv_free_bench.py` — Phase 41: KV cache freeing before checkpoint
- `cuda_serializer/cuda_checkpoint.py` — Reusable Python module for CUDA checkpoint APIs (ctypes bindings)
- `cuda_serializer/criu_checkpoint_design.md` — Phase 39 design: CRIU + cuda-checkpoint integration
- `cuda_serializer/vllm_checkpoint.sh` — Phase 39: complete CRIU + cuda-checkpoint orchestrator CLI (save/restore/benchmark)
- `cuda_serializer/rfc_comment_draft.md` — Draft comment for vLLM RFC #34303 with empirical results
- `cuda_serializer/test_graph.cu` — Test CUDA program with 3-kernel graph
- `cuda_serializer/modal_phase32_diag.py` — Phase 32 v6: XID 13 root cause (sm90 WGMMA confirmation)
- `cuda_serializer/modal_phase32_v7.py` — Phase 32 v7: cuGraphExecUpdate workaround (G1-G4 ALL PASS)
- `cuda_serializer/modal_phase32_v8.py` — Phase 32 v8: Full serialization path (AddKernelNode → ExecUpdate)
- `cuda_serializer/modal_phase32_v8b.py` — Phase 32 v8b: Stripped H1+H4 (FAIL: kernel-only rebuild → topology mismatch)
- `cuda_serializer/modal_phase32_v8c.py` — Phase 32 v8c: 4 approaches (ALL PASS: rebuild all nodes, clone+SetParams, clone→instantiate, 2nd capture)
- `cuda_serializer/modal_phase32_v8d.py` — Phase 32 v8d: Full pipeline + timing + cross-batch (H1+H2 PASS, 2.1x speedup, cross-batch TOPOLOGY_CHANGED)
- `cuda_serializer/modal_test.py` — Modal script to build and test on T4 GPU

This is NOT a fork of Foundry. Independent implementation. Key divergences from Foundry: xxh3-128 (not CRC64), 512B alignment (not 256B), chunked VA reservation (not single range), driver triple pinning in archive headers.

Current status (Phase 42 v35 COMPLETE + unit tests + K8s deploy, 2026-05-17): **CUDA graphs + multi-GPU production-ready.** `pip install vllm-cold-start[serve]` → `vllm-ckpt` CLI + `from vllm_cuda_ckpt import CudaCheckpointAPI`. TP scaling: TP=1 3.6s (92%), TP=2 4.9s (95%), TP=4 6.5s (93%), **CUDA graphs TP=2 5.2s (97%, best)**. Post-restore inference 3x faster with graphs (0.20s vs 0.63s). **3.1s multi-GPU restore (89% reduction)** via parallel PID + sleep. V0 sleep at 0.85 util: frees ALL GPU memory (65.7→0.77 GiB), checkpoint 1.7s, restore 3.3s, cold start 5.4s. Tests PASSED: 7B model, AWQ quantized, CUDA graphs, 10-cycle stability (zero leaks), concurrent load, error recovery, **V1 engine (no NCCL cleanup needed)**. V1 optimizations: sleep() frees weights (-20%), parallel ThreadPoolExecutor ckpt/restore (-43%), combined = 73% faster than sequential CLI. Manual KV cache freeing FAILS (custom allocator). Four paths: (1) Modal GPU snapshot: 97.5% best. (2) **NVIDIA cuda-checkpoint: 89% multi-GPU** with parallel+sleep (V1) or NCCL reinit (V0), ANY Linux driver 570+. (3) **Foundry portable: 76% end-to-end, 96% engine init, cross-machine proven**. (4) LD_PRELOAD: 85-87%.

**Phase 32 COMPLETE (v8d)**: CUDA graph serialization via cuGraphExecUpdate fully validated on H100. XID 13 root cause: sm90 WGMMA constant banks not initialized by cuGraphAddKernelNode. Fix: rebuild ALL node types (KERNEL+MEMCPY+MEMSET) → ExecUpdate template exec preserves constant banks. Full save/load pipeline PASS (H1+H2). Per-graph speedup: 2.1x (0.09ms vs 0.18ms). Cross-batch fails (topology changes per batch size). **Conclusion: cuda-checkpoint remains best approach** — graph serialization saves ~10% at most since forward pass during capture dominates (~460ms/graph).

## Foundry CUDA graph persistence

Integrates [Foundry](https://github.com/lokashrinav/foundry) (fork with passthrough recording) to serialize CUDA graphs
to disk and restore them on subsequent cold starts, eliminating the graph capture phase
(~3-60s depending on model size).

- Requires `LD_PRELOAD=libcuda_hook.so` (Foundry's driver interception hook)
- Monkey-patches `CUDAGraphWrapper.__call__` for FULL mode graph capture/load
- Only FULL mode graphs are serialized; PIECEWISE mode uses torch.compile internals
- Supports single-GPU (tp=1) and multi-GPU (tp>1) via per-rank caching
- Falls back to standard vLLM if Foundry is not available
- **Force-capture mode** (`FOUNDRY_SKIP_GRAPH_LOAD=1`, default): On load machine, skips loading serialized graphs (address conflicts) and re-captures using Foundry's fast `fdry.CUDAGraph()` path (3-5s for 51 graphs vs 98s with torch.compile). Profile cache still saves the 90s+ profiling phase.
- `_force_foundry_capture` mutable list avoids Python closure scoping issue (can't reassign nonlocal vars from nested functions)

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
