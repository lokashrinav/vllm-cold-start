"""Benchmark multi-GPU vLLM cold start with Foundry CUDA graph persistence.

Tests tensor_parallel_size=2 on 2x A100-40GB via Modal.

Run 1: Baseline (standard vLLM, tp=2, no Foundry)
Run 2: First run with Foundry (captures + saves per-rank graphs)
Run 3: Cache hit (loads per-rank graphs from disk)

Usage:
    modal run profiling/modal_foundry_multi_gpu_benchmark.py
"""

import modal
import time
from pathlib import Path

app = modal.App("foundry-multi-gpu-bench")

graph_cache_vol = modal.Volume.from_name(
    "foundry-graph-cache-multigpu", create_if_missing=True
)

# Same image as single-GPU benchmark
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install(
        "git", "build-essential", "ninja-build", "wget", "curl", "xz-utils",
        "clang",
    )
    .run_commands(
        "wget -q -O /tmp/cmake.tar.gz"
        " https://github.com/Kitware/CMake/releases/download/v4.0.2/cmake-4.0.2-linux-x86_64.tar.gz",
        "tar xzf /tmp/cmake.tar.gz -C /usr/local --strip-components=1"
        " && rm /tmp/cmake.tar.gz",
        "cmake --version",
    )
    .run_commands(
        "wget -q -O /tmp/boost.tar.xz"
        " https://github.com/boostorg/boost/releases/download/boost-1.87.0/boost-1.87.0-cmake.tar.xz",
        "tar xJf /tmp/boost.tar.xz -C /tmp && rm /tmp/boost.tar.xz",
        "cmake -S /tmp/boost-1.87.0 -B /tmp/boost-build -G Ninja"
        " -DCMAKE_INSTALL_PREFIX=/usr/local"
        " -DCMAKE_BUILD_TYPE=Release"
        " -DBUILD_SHARED_LIBS=ON"
        ' -DBOOST_INCLUDE_LIBRARIES="filesystem;json;system;unordered;crc;format"',
        "cmake --build /tmp/boost-build -j$(nproc)",
        "cmake --install /tmp/boost-build",
        "ldconfig",
        "rm -rf /tmp/boost-1.87.0 /tmp/boost-build",
    )
    .pip_install("vllm>=0.20", "transformers", "tqdm")
    .run_commands(
        "pip install --upgrade pip 'setuptools>=80,<82' wheel",
    )
    .add_local_dir(
        "../foundry",
        remote_path="/opt/foundry",
        copy=True,
    )
    .run_commands(
        "sed -i 's|c10::cuda::MemPool|at::cuda::MemPool|g'"
        " /opt/foundry/csrc/CUDAGraph.cpp /opt/foundry/csrc/CUDAGraphParallel.cpp",
        "sed -i '/#include <c10\\/cuda\\/CUDACachingAllocator.h>/a #include <ATen/cuda/MemPool.h>'"
        " /opt/foundry/csrc/CUDAGraph.cpp /opt/foundry/csrc/CUDAGraphParallel.cpp",
        "cd /opt/foundry && rm -rf build && pip install -e . --no-build-isolation",
    )
    .add_local_dir(
        "src/vllm_profile_cache",
        remote_path="/root/vllm_profile_cache",
    )
)

MODEL = "Qwen/Qwen2.5-7B-Instruct"
GRAPH_CACHE_MOUNT = "/root/.cache/foundry-graphs"
PROMPT = "What is the meaning of life?"
TP_SIZE = 2


def _get_hook_path() -> str:
    import site
    candidates = [Path("/opt/foundry/python/foundry/libcuda_hook.so")]
    for root in site.getsitepackages():
        candidates.append(Path(root) / "foundry" / "libcuda_hook.so")
    for path in candidates:
        if path.is_file():
            return str(path.resolve())
    raise RuntimeError("libcuda_hook.so not found")


# --------------------------------------------------------------------------
# Run 1: Baseline — normal vLLM with tp=2, no Foundry
# --------------------------------------------------------------------------
@app.function(
    gpu=f"A100-40GB:{TP_SIZE}",
    image=image,
    timeout=600,
    scaledown_window=5,
)
def run_baseline():
    import json
    import subprocess
    import sys

    script = f"""
import sys, json, time
sys.path.insert(0, "/root")

results = {{"mode": "baseline_tp{TP_SIZE}"}}
t_total = time.perf_counter()

import vllm
results["vllm_version"] = vllm.__version__

t0 = time.perf_counter()
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig
from vllm.config.compilation import CUDAGraphMode
cc = CompilationConfig()
if hasattr(cc, 'level'):
    cc.level = 0
if hasattr(cc, 'mode'):
    cc.mode = 0
cc.cudagraph_mode = CUDAGraphMode.FULL
llm = LLM(
    model="{MODEL}",
    tensor_parallel_size={TP_SIZE},
    gpu_memory_utilization=0.8,
    disable_log_stats=True,
    compilation_config=cc,
)
results["llm_init"] = time.perf_counter() - t0

t0 = time.perf_counter()
output = llm.generate(["{PROMPT}"], SamplingParams(max_tokens=16))
results["first_inference"] = time.perf_counter() - t0
results["output"] = output[0].outputs[0].text
results["total"] = time.perf_counter() - t_total

import torch
results["gpu"] = torch.cuda.get_device_name(0)
results["gpu_count"] = torch.cuda.device_count()

print("BASELINE_RESULT:" + json.dumps(results))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=900,
    )
    output = (result.stdout or "") + (result.stderr or "")
    for line in result.stdout.splitlines():
        if line.startswith("BASELINE_RESULT:"):
            return json.loads(line[len("BASELINE_RESULT:"):])

    raise RuntimeError(
        f"Baseline subprocess failed (exit={result.returncode}).\n"
        f"stderr: {result.stderr[-1000:]}"
    )


# --------------------------------------------------------------------------
# Run 2 & 3: Multi-GPU Foundry via subprocess
# --------------------------------------------------------------------------
@app.function(
    gpu=f"A100-40GB:{TP_SIZE}",
    image=image,
    timeout=600,
    scaledown_window=5,
    volumes={GRAPH_CACHE_MOUNT: graph_cache_vol},
)
def run_with_foundry_subprocess(force_clear: bool = False):
    import json
    import os
    import subprocess
    import sys

    sys.path.insert(0, "/root")

    hook_path = _get_hook_path()
    env = dict(os.environ)
    env["LD_PRELOAD"] = hook_path
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
    env["NCCL_DEBUG"] = "INFO"
    env["NCCL_DEBUG_FILE"] = "/tmp/nccl_debug_%p.log"
    env["NCCL_P2P_DISABLE"] = "1"
    env["NCCL_CUMEM_ENABLE"] = "0"

    # Multi-GPU uses fork — do NOT set VLLM_ENABLE_V1_MULTIPROCESSING=0
    # Workers need to be spawned as separate processes.

    if force_clear:
        env["FOUNDRY_FORCE_CLEAR"] = "1"

    script = f"""
import sys, json, time, os, shutil
sys.path.insert(0, "/root")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"

print("[PROGRESS] script start", flush=True)

results = {{"mode": "multi_gpu_foundry_tp{TP_SIZE}"}}
t_total = time.perf_counter()

print("[PROGRESS] importing vllm...", flush=True)
import vllm
results["vllm_version"] = vllm.__version__
print(f"[PROGRESS] vllm {{vllm.__version__}} imported", flush=True)

_force_clear = os.environ.get("FOUNDRY_FORCE_CLEAR") == "1"

t0 = time.perf_counter()
from vllm.config import CompilationConfig
from vllm.config.compilation import CUDAGraphMode
from vllm_profile_cache.foundry_graphs import cached_vllm_init_with_foundry

cc = CompilationConfig()
if hasattr(cc, 'level'):
    cc.level = 0
if hasattr(cc, 'mode'):
    cc.mode = 0
cc.cudagraph_mode = CUDAGraphMode.FULL

print("[PROGRESS] calling cached_vllm_init_with_foundry...", flush=True)
try:
    llm = cached_vllm_init_with_foundry(
        model="{MODEL}",
        graph_cache_dir="{GRAPH_CACHE_MOUNT}",
        tensor_parallel_size={TP_SIZE},
        gpu_memory_utilization=0.8,
        disable_log_stats=True,
        compilation_config=cc,
        force_save=_force_clear,
    )
except Exception as e:
    import traceback
    print("INIT_FAILED_TRACEBACK:", flush=True)
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    raise
print(f"[PROGRESS] LLM init done in {{time.perf_counter() - t0:.1f}}s", flush=True)
results["llm_init"] = time.perf_counter() - t0

from vllm import SamplingParams
print("[PROGRESS] running generate...", flush=True)
t0 = time.perf_counter()
output = llm.generate(["{PROMPT}"], SamplingParams(max_tokens=16))
results["first_inference"] = time.perf_counter() - t0
results["output"] = output[0].outputs[0].text
results["total"] = time.perf_counter() - t_total

import torch
results["gpu"] = torch.cuda.get_device_name(0)
results["gpu_count"] = torch.cuda.device_count()

# Check per-rank cache state
from vllm_profile_cache.foundry_graphs import resolve_graph_cache_dir, FoundryGraphCache
from pathlib import Path
_, cache_key = resolve_graph_cache_dir(
    "{GRAPH_CACHE_MOUNT}", "{MODEL}",
    tensor_parallel_size={TP_SIZE},
    gpu_memory_utilization=0.8,
    disable_log_stats=True,
)
base_dir = Path("{GRAPH_CACHE_MOUNT}") / cache_key
rank_info = {{}}
for r in range({TP_SIZE}):
    rank_dir = base_dir / f"rank_{{r}}"
    rc = FoundryGraphCache(str(rank_dir))
    wg_dir = rank_dir / "wrapper_graphs"
    n_graphs = len(list(wg_dir.glob("graph_*.json"))) if wg_dir.exists() else 0
    rank_info[f"rank_{{r}}"] = {{
        "save_complete": rc.is_save_complete(),
        "graph_count": n_graphs,
        "has_metadata": rc.load_metadata() is not None,
    }}
results["rank_caches"] = rank_info
results["cache_key"] = cache_key

print("FOUNDRY_RESULT:" + json.dumps(results))
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=500,
        )
    except subprocess.TimeoutExpired as exc:
        print("\n  === TIMEOUT — partial output ===")
        partial_out = (exc.stdout or "") if isinstance(exc.stdout, str) else (exc.stdout or b"").decode(errors="replace")
        partial_err = (exc.stderr or "") if isinstance(exc.stderr, str) else (exc.stderr or b"").decode(errors="replace")
        for line in partial_out.splitlines()[-60:]:
            print(f"  [out] {line}")
        for line in partial_err.splitlines()[-60:]:
            print(f"  [err] {line}")
        result = None

    if result is not None:
        if result.stdout:
            for line in result.stdout.splitlines():
                if "[FOUNDRY]" in line:
                    print(f"  {line}")
            for line in result.stdout.splitlines()[-20:]:
                if "[FOUNDRY]" not in line and not line.startswith("FOUNDRY_RESULT:"):
                    print(f"  [subprocess] {line}")
        if result.stderr:
            for line in result.stderr.splitlines():
                if "[HOOK]" in line or "[FOUNDRY]" in line or "NCCL" in line or "CUDA" in line:
                    print(f"  [hook] {line}")
            for line in result.stderr.splitlines()[-80:]:
                if "[HOOK]" not in line:
                    print(f"  [stderr] {line}")

        for line in result.stdout.splitlines():
            if line.startswith("FOUNDRY_RESULT:"):
                data = json.loads(line[len("FOUNDRY_RESULT:"):])
                graph_cache_vol.commit()
                return data

    exit_info = f"exit={result.returncode}" if result else "timeout"
    if result:
        stdout_lines = (result.stdout or "").splitlines()
        stderr_lines = (result.stderr or "").splitlines()
        # Show lines with errors, tracebacks, or FOUNDRY debug
        print("\n  === ERROR-RELEVANT OUTPUT ===")
        for line in stdout_lines:
            if any(k in line for k in ["Error", "Traceback", "FAILED",
                                        "FOUNDRY", "Exception", "raise"]):
                print(f"  [out] {line}")
        for line in stderr_lines:
            if any(k in line for k in ["Error", "Traceback", "FAILED",
                                        "FOUNDRY", "Exception", "raise",
                                        "foundry_graphs"]):
                print(f"  [err] {line}")
        # Also show last 100 lines of stderr for context
        print("\n  === STDERR (last 100) ===")
        for line in stderr_lines[-100:]:
            print(f"  [err] {line}")
        # Read NCCL debug files
        import glob as globmod
        for nccl_log in sorted(globmod.glob("/tmp/nccl_debug_*.log")):
            try:
                with open(nccl_log) as f:
                    lines = f.readlines()
                print(f"\n  === {nccl_log} (last 40) ===")
                for line in lines[-40:]:
                    print(f"  [nccl] {line.rstrip()}")
            except Exception:
                pass
    stderr_tail = result.stderr[-2000:] if result and result.stderr else "N/A"
    raise RuntimeError(
        f"Multi-GPU subprocess failed ({exit_info}).\nstderr tail: ...{stderr_tail[-500:]}"
    )


@app.function(
    timeout=60,
    volumes={GRAPH_CACHE_MOUNT: graph_cache_vol},
)
def clear_cache():
    import os
    import shutil
    if os.path.exists(GRAPH_CACHE_MOUNT):
        for f in os.listdir(GRAPH_CACHE_MOUNT):
            path = os.path.join(GRAPH_CACHE_MOUNT, f)
            if os.path.isfile(path):
                os.unlink(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
    graph_cache_vol.commit()
    return {"cleared": True}


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path

    print("=" * 60)
    print(f"Multi-GPU Foundry Benchmark (tp={TP_SIZE})")
    print("=" * 60)
    print()

    # Skip separate clear_cache — force_clear=True on save run handles it.
    # (Avoids extra Modal function call during outages.)

    # Skip baseline — already have data (~137s init, ~142s total from prior runs)
    r1 = {"llm_init": 137.0, "total": 142.4}

    print("=" * 60)
    print(f"Run 2: FOUNDRY save (tp={TP_SIZE}, first run)")
    print("=" * 60)
    r2 = run_with_foundry_subprocess.remote(force_clear=True)
    print(f"  LLM init:        {r2['llm_init']:.2f}s")
    print(f"  First inference:  {r2['first_inference']:.2f}s")
    print(f"  Total:           {r2['total']:.2f}s")
    print(f"  Rank caches:     {json.dumps(r2.get('rank_caches', {}), indent=4)}")
    print()

    # --- Load (cached graphs from fresh save) ---
    print("=" * 60)
    print(f"Run 3: FOUNDRY load (tp={TP_SIZE}, per-rank cache hit)")
    print("=" * 60)
    r3 = run_with_foundry_subprocess.remote()
    print(f"  LLM init:        {r3['llm_init']:.2f}s")
    print(f"  First inference:  {r3['first_inference']:.2f}s")
    print(f"  Total:           {r3['total']:.2f}s")
    print(f"  Rank caches:     {json.dumps(r3.get('rank_caches', {}), indent=4)}")
    print()

    print("\n=== COMPARISON ===")
    print(f"  Baseline: {r1['llm_init']:.2f}s init, {r1['total']:.2f}s total")
    print(f"  Save:     {r2['llm_init']:.2f}s init, {r2['total']:.2f}s total")
    print(f"  Load:     {r3['llm_init']:.2f}s init, {r3['total']:.2f}s total")
    delta = r1['total'] - r3['total']
    print(f"  Savings:  {delta:.2f}s ({delta/r1['total']*100:.1f}%)")
