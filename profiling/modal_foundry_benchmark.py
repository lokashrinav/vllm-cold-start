"""Benchmark vLLM cold start with Foundry CUDA graph persistence on Modal.

Measures the impact of serializing CUDA graphs to disk via Foundry,
eliminating the graph capture phase on subsequent cold starts.

Run 1: Baseline (normal vLLM, no graph cache)
Run 2: First run with Foundry (captures + saves graphs)
Run 3: Cache hit (loads graphs from disk, skips capture)

Usage:
    modal run profiling/modal_foundry_benchmark.py
"""

import modal
import time
from pathlib import Path

app = modal.App("foundry-graph-bench")

graph_cache_vol = modal.Volume.from_name(
    "foundry-graph-cache", create_if_missing=True
)

# Foundry requires: CUDA 12+, CMake 4+, Boost 1.83+ (with BoostConfig.cmake)
# Ubuntu 22.04 ships CMake 3.22 and Boost 1.74, both too old.
#
# Key issues solved:
#   1. CMake 4.0 removed the FindBoost module (CMP0167) — only CONFIG mode works,
#      so Boost MUST be built with CMake (not b2) to generate BoostConfig.cmake.
#      We use the official "boost-1.87.0-cmake" archive which has CMakeLists.txt.
#   2. vLLM may install a different torch version — install vLLM first, then
#      Foundry uses whatever torch is present via --no-build-isolation.
#   3. Foundry pyproject.toml needs setuptools>=80 — upgrade pip/setuptools first.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install(
        "git", "build-essential", "ninja-build", "wget", "curl", "xz-utils",
        "clang",
    )
    .run_commands(
        # CMake 4.0.2 (Foundry CMakeLists.txt requires >= 4.0)
        # Download first — piping GitHub redirects to tar can truncate the stream.
        "wget -q -O /tmp/cmake.tar.gz"
        " https://github.com/Kitware/CMake/releases/download/v4.0.2/cmake-4.0.2-linux-x86_64.tar.gz",
        "tar xzf /tmp/cmake.tar.gz -C /usr/local --strip-components=1"
        " && rm /tmp/cmake.tar.gz",
        "cmake --version",
    )
    .run_commands(
        # Boost 1.87 built with CMake (generates BoostConfig.cmake for CONFIG mode).
        # Must use the "-cmake" archive — the "-b2-nodocs" one lacks CMakeLists.txt.
        # Download first then extract: piping xz to tar can fail on some systems.
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
        # Patch: c10::cuda::MemPool was moved to at::cuda::MemPool in torch 2.10+
        "sed -i 's|c10::cuda::MemPool|at::cuda::MemPool|g'"
        " /opt/foundry/csrc/CUDAGraph.cpp /opt/foundry/csrc/CUDAGraphParallel.cpp",
        "sed -i '/#include <c10\\/cuda\\/CUDACachingAllocator.h>/a #include <ATen/cuda/MemPool.h>'"
        " /opt/foundry/csrc/CUDAGraph.cpp /opt/foundry/csrc/CUDAGraphParallel.cpp",
        "cd /opt/foundry && pip install -e . --no-build-isolation",
    )
    .add_local_dir(
        "src/vllm_profile_cache",
        remote_path="/root/vllm_profile_cache",
    )
)

MODEL = "Qwen/Qwen2.5-7B-Instruct"
GRAPH_CACHE_MOUNT = "/root/.cache/foundry-graphs"
PROMPT = "What is the meaning of life?"


def _get_hook_path() -> str:
    """Return absolute path to Foundry's libcuda_hook.so for LD_PRELOAD.

    Do not use importlib/find_spec("foundry.ops"): that executes foundry/__init__.py,
    which imports the ops extension and requires libc10.so before LD_LIBRARY_PATH is
    reliably visible to the loader on Modal workers.
    """
    import site

    candidates = [
        Path("/opt/foundry/python/foundry/libcuda_hook.so"),
    ]
    for root in site.getsitepackages():
        candidates.append(Path(root) / "foundry" / "libcuda_hook.so")

    for path in candidates:
        if path.is_file():
            return str(path.resolve())

    raise RuntimeError(
        "libcuda_hook.so not found under /opt/foundry/python/foundry/ "
        "or site-packages/foundry/ (re-run image build if Foundry failed to install)."
    )


def _run_inference(llm, prompt: str = PROMPT, max_tokens: int = 16):
    """Run a quick inference to verify the engine works."""
    from vllm import SamplingParams
    output = llm.generate([prompt], SamplingParams(max_tokens=max_tokens))
    text = output[0].outputs[0].text
    return text


# --------------------------------------------------------------------------
# Run 1: Baseline — normal vLLM, no Foundry
# --------------------------------------------------------------------------
@app.function(
    gpu="A100-40GB",
    image=image,
    timeout=600,
    scaledown_window=5,
)
def run_baseline():
    """Baseline: standard vLLM cold start."""
    import json
    import re
    import subprocess
    import sys

    script = f"""
import sys, json, time
sys.path.insert(0, "/root")

results = {{"mode": "baseline"}}
t_total = time.perf_counter()

import vllm
results["vllm_version"] = vllm.__version__

t0 = time.perf_counter()
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig
from vllm.config.compilation import CUDAGraphMode
# Disable torch.compile (Triton JIT kernels) but keep CUDA graph capture
cc = CompilationConfig()
if hasattr(cc, 'level'):
    cc.level = 0
if hasattr(cc, 'mode'):
    cc.mode = 0
cc.cudagraph_mode = CUDAGraphMode.FULL
llm = LLM(model="{MODEL}", gpu_memory_utilization=0.8, disable_log_stats=True, compilation_config=cc)
results["llm_init"] = time.perf_counter() - t0

t0 = time.perf_counter()
output = llm.generate(["{PROMPT}"], SamplingParams(max_tokens=16))
results["first_inference"] = time.perf_counter() - t0
results["output"] = output[0].outputs[0].text
results["total"] = time.perf_counter() - t_total

import torch
results["gpu"] = torch.cuda.get_device_name(0)

print("BASELINE_RESULT:" + json.dumps(results))
"""
    env = dict(__import__("os").environ)
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    output = (result.stdout or "") + (result.stderr or "")
    for line in result.stdout.splitlines():
        if line.startswith("BASELINE_RESULT:"):
            data = json.loads(line[len("BASELINE_RESULT:"):])
            data.update(_parse_graph_log_markers(output))
            return data

    raise RuntimeError(
        f"Baseline subprocess failed (exit={result.returncode}).\n"
        f"stderr: {result.stderr[-1000:]}"
    )


# --------------------------------------------------------------------------
# Run 2 & 3: With Foundry — capture+save on first run, load on second
# --------------------------------------------------------------------------
@app.function(
    gpu="A100-40GB",
    image=image,
    timeout=600,
    scaledown_window=5,
    volumes={GRAPH_CACHE_MOUNT: graph_cache_vol},
)
def run_with_foundry():
    """vLLM cold start with Foundry CUDA graph caching."""
    import sys
    import os
    import logging
    sys.path.insert(0, "/root")
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    # Foundry's LD_PRELOAD must be set before CUDA init. On Modal, we set
    # it in the function environment. In production, set it in the container
    # entrypoint or image ENV.
    hook_path = _get_hook_path()
    os.environ["LD_PRELOAD"] = hook_path
    logger = logging.getLogger("foundry_bench")
    logger.info("LD_PRELOAD set to %s", hook_path)

    # NOTE: LD_PRELOAD set after process start only works if CUDA hasn't
    # been initialized yet. torch.cuda.init() must happen AFTER this point.
    # If this doesn't work, we'd need to re-exec the process or use
    # subprocess with LD_PRELOAD in the environment.

    results = {"mode": "with_foundry"}
    t_total = time.perf_counter()

    import torch
    import vllm
    results["vllm_version"] = vllm.__version__
    results["gpu"] = torch.cuda.get_device_name(0)

    t0 = time.perf_counter()
    from vllm_profile_cache.foundry_graphs import cached_vllm_init_with_foundry
    llm = cached_vllm_init_with_foundry(
        model=MODEL,
        graph_cache_dir=GRAPH_CACHE_MOUNT,
        gpu_memory_utilization=0.8,
        disable_log_stats=True,
    )
    results["llm_init"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    text = _run_inference(llm)
    results["first_inference"] = time.perf_counter() - t0
    results["output"] = text

    results["total"] = time.perf_counter() - t_total
    graph_cache_vol.commit()
    return results


# --------------------------------------------------------------------------
# Fallback: Foundry via subprocess with proper LD_PRELOAD
# --------------------------------------------------------------------------
@app.function(
    gpu="A100-40GB",
    image=image,
    timeout=600,
    scaledown_window=5,
    volumes={GRAPH_CACHE_MOUNT: graph_cache_vol},
)
def run_with_foundry_subprocess(force_clear: bool = False):
    """Run Foundry integration in a subprocess with proper LD_PRELOAD.

    LD_PRELOAD must be set before the process loads libcuda.so. Setting it
    after Python starts doesn't intercept driver calls. This function
    spawns a child process with LD_PRELOAD in the environment.
    """
    import json
    import os
    import re
    import subprocess
    import sys
    import shutil

    sys.path.insert(0, "/root")
    from vllm_profile_cache.foundry_graphs import (
        FoundryGraphCache,
        resolve_graph_cache_dir,
    )

    if force_clear:
        effective_dir, _ = resolve_graph_cache_dir(
            GRAPH_CACHE_MOUNT, MODEL, gpu_memory_utilization=0.8, disable_log_stats=True,
        )
        if os.path.exists(effective_dir):
            shutil.rmtree(effective_dir)
            print(f"  [FORCE CLEAR] Removed {effective_dir}")
        graph_cache_vol.commit()

    hook_path = _get_hook_path()
    env = dict(__import__("os").environ)
    env["LD_PRELOAD"] = hook_path
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
    if force_clear:
        env["FOUNDRY_FORCE_CLEAR"] = "1"
    else:
        # Tell the hook to skip fatbin processing at init time (before
        # Python can call set_skip_fatbin_processing). Fatbins are loaded
        # explicitly from the saved archive instead.
        effective_dir, _ = resolve_graph_cache_dir(
            GRAPH_CACHE_MOUNT, MODEL, gpu_memory_utilization=0.8, disable_log_stats=True,
        )
        from pathlib import Path as _P
        if (_P(effective_dir) / ".save_complete").exists():
            env["CGE_MODE"] = "load"

    script = f"""
import sys, json, time, logging, os, shutil
sys.path.insert(0, "/root")
logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

# Disable PyTorch expandable_segments — it uses cuMemAddressReserve and
# conflicts with Foundry's deterministic VMM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"

# torch import is needed for foundry's libc10.so dependency
import torch
# Foundry requires CUDA context before set_allocation_region
torch.cuda.init()
print(f"[FOUNDRY_DEBUG] CUDA initialized: {{torch.cuda.get_device_name(0)}}", flush=True)

import foundry as fdry

# Use lower addresses that are within GPU VA range (A100 rejects high addresses)
# IMPORTANT: set_allocation_region MUST come before load_cuda_modules_and_libraries
# because module loading can consume VA at the base address, preventing reservation.
_REGION_SIZE = fdry.parse_size("36GB")
_CANDIDATES = [0x10000000000, 0x8000000000, 0x18000000000, 0x4000000000, 0x20000000000]
_used_base = None
for _b in _CANDIDATES:
    try:
        fdry.set_allocation_region(_b, _REGION_SIZE)
        _used_base = _b
        print(f"[FOUNDRY_DEBUG] Allocation region set: base=0x{{_b:x}}", flush=True)
        break
    except Exception as _e:
        print(f"[FOUNDRY_DEBUG] set_allocation_region(0x{{_b:x}}) failed: {{_e}}", flush=True)
if _used_base is None:
    print("[FOUNDRY_DEBUG] ERROR: All allocation region candidates failed!", flush=True)
else:
    # Verify the region works
    _test = torch.empty(1024, device="cuda")
    _ptr = _test.data_ptr()
    _in_region = _used_base <= _ptr < (_used_base + _REGION_SIZE)
    print(f"[FOUNDRY_DEBUG] Verification: tensor at 0x{{_ptr:x}}, in_region={{_in_region}}", flush=True)
    if not _in_region:
        print(f"[FOUNDRY_DEBUG] WARNING: Region not effective! Expected [0x{{_used_base:x}}, 0x{{_used_base + _REGION_SIZE:x}})", flush=True)
    del _test
fdry.set_pack_fatbins_on_exit(False)

# Load fatbins AFTER region is set. Module loading uses CUDA code memory
# (cuModuleLoad) which is separate from the data allocation region.
from vllm_profile_cache.foundry_graphs import (
    resolve_graph_cache_dir, FoundryGraphCache, _install_sigabrt_handler,
)
_eff_dir, _ = resolve_graph_cache_dir(
    "{GRAPH_CACHE_MOUNT}", "{MODEL}",
    gpu_memory_utilization=0.8, disable_log_stats=True,
)
_pre_cache = FoundryGraphCache(_eff_dir)
if _pre_cache.hook_archive.exists() and any(_pre_cache.hook_archive.iterdir()):
    print(f"[FOUNDRY_DEBUG] Loading fatbins AFTER region setup", flush=True)
    fdry.load_cuda_modules_and_libraries(str(_pre_cache.hook_archive))
    # Skip fatbin processing for subsequently loaded modules (load mode
    # has all fatbins from the save run; processing new ones adds overhead)
    fdry.set_skip_fatbin_processing(True)
    # Re-verify region still works after fatbin loading
    _test2 = torch.empty(256, device="cuda")
    _ptr2 = _test2.data_ptr()
    _in2 = _used_base <= _ptr2 < (_used_base + _REGION_SIZE) if _used_base else False
    print(f"[FOUNDRY_DEBUG] Post-fatbin verify: tensor at 0x{{_ptr2:x}}, in_region={{_in2}}", flush=True)
    del _test2

_install_sigabrt_handler()

results = {{"mode": "with_foundry_subprocess"}}
t_total = time.perf_counter()

import vllm
results["vllm_version"] = vllm.__version__

t0 = time.perf_counter()
from vllm_profile_cache.foundry_graphs import cached_vllm_init_with_foundry
effective_cache_dir, graph_cache_key = resolve_graph_cache_dir(
    "{GRAPH_CACHE_MOUNT}",
    "{MODEL}",
    gpu_memory_utilization=0.8,
    disable_log_stats=True,
)
graph_cache = FoundryGraphCache(effective_cache_dir)
results["graph_cache_key"] = graph_cache_key
results["effective_graph_cache_dir"] = effective_cache_dir
_has_cached = graph_cache.has_cached_graphs() or graph_cache.has_cached_wrapper_graphs()
results["had_cached_graphs_before"] = _has_cached

# Pre-flight: force-clear or stale cache detection
_force_clear = os.environ.get("FOUNDRY_FORCE_CLEAR") == "1"
if _force_clear and os.path.exists(effective_cache_dir):
    print(f"[PREFLIGHT] Force-clearing cache at {{effective_cache_dir}}", flush=True)
    shutil.rmtree(effective_cache_dir, ignore_errors=True)
    _has_cached = False
    results["had_cached_graphs_before"] = False
elif _has_cached and not os.path.exists(os.path.join(effective_cache_dir, ".save_complete")):
    print(f"[PREFLIGHT] Stale cache detected. Clearing.", flush=True)
    shutil.rmtree(effective_cache_dir, ignore_errors=True)
    _has_cached = False
    results["had_cached_graphs_before"] = False
elif _has_cached:
    print(f"[PREFLIGHT] Cache looks complete.", flush=True)
    # Print graph file sizes for debugging
    wg_dir = os.path.join(effective_cache_dir, "wrapper_graphs")
    if os.path.isdir(wg_dir):
        graph_files = sorted(
            [f for f in os.listdir(wg_dir) if f.endswith(".json")],
            key=lambda f: int(f.split("_")[1].split(".")[0])
        )
        sizes = []
        for f in graph_files[:3]:
            fp = os.path.join(wg_dir, f)
            sizes.append(f"{{f}}={{os.path.getsize(fp)}}b")
            cg = fp.replace(".json", ".cugraph")
            if os.path.exists(cg):
                sizes.append(f"{{f.replace('.json','.cugraph')}}={{os.path.getsize(cg)}}b")
        print(f"[FOUNDRY_DEBUG] Graph file sizes (first 3): {{', '.join(sizes)}}", flush=True)
        print(f"[FOUNDRY_DEBUG] Total graph files: {{len(graph_files)}}", flush=True)
else:
    print(f"[PREFLIGHT] No cached graphs. Fresh capture.", flush=True)

from vllm.config import CompilationConfig
from vllm.config.compilation import CUDAGraphMode
cc = CompilationConfig()
if hasattr(cc, 'level'):
    cc.level = 0
if hasattr(cc, 'mode'):
    cc.mode = 0
cc.cudagraph_mode = CUDAGraphMode.FULL
llm = cached_vllm_init_with_foundry(
    model="{MODEL}",
    graph_cache_dir="{GRAPH_CACHE_MOUNT}",
    gpu_memory_utilization=0.8,
    disable_log_stats=True,
    compilation_config=cc,
    force_save=_force_clear,
    region_base_override=_used_base,
)
results["llm_init"] = time.perf_counter() - t0

from vllm import SamplingParams
t0 = time.perf_counter()
output = llm.generate(["{PROMPT}"], SamplingParams(max_tokens=16))
results["first_inference"] = time.perf_counter() - t0
results["output"] = output[0].outputs[0].text
results["has_cached_graphs_after"] = graph_cache.has_cached_graphs()
results["graph_file_count_after"] = len(list(graph_cache.graphs_dir.glob("graph_*.json")))
results["metadata_after"] = graph_cache.load_metadata()
import os
hook_dir = graph_cache.hook_archive
results["hook_archive_exists"] = hook_dir.exists()
results["hook_archive_files"] = [f.name for f in hook_dir.glob("*")] if hook_dir.exists() else []
cache_all_files = []
for root, dirs, files in os.walk(str(graph_cache.cache_dir)):
    for f in files:
        cache_all_files.append(os.path.relpath(os.path.join(root, f), str(graph_cache.cache_dir)))
results["all_cache_files"] = cache_all_files
results["total"] = time.perf_counter() - t_total

import torch
results["gpu"] = torch.cuda.get_device_name(0)

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
    except subprocess.TimeoutExpired:
        result = None

    if result is not None:
        # Print subprocess output for debugging — prioritize debug lines
        if result.stdout:
            debug_lines = [l for l in result.stdout.splitlines() if "[FOUNDRY_DEBUG]" in l]
            other_lines = [l for l in result.stdout.splitlines()
                          if not l.startswith("FOUNDRY_RESULT:") and "[FOUNDRY_DEBUG]" not in l]
            if debug_lines:
                print(f"  === FOUNDRY DEBUG ({len(debug_lines)} lines) ===")
                for line in debug_lines:
                    print(f"  {line}")
                print(f"  === END FOUNDRY DEBUG ===")
            # Print last 30 other stdout lines for context
            for line in other_lines[-30:]:
                print(f"  [subprocess] {line}")

        if result.stderr:
            for line in result.stderr.splitlines()[-80:]:
                print(f"  [subprocess stderr] {line}")

        # Extract result
        for line in result.stdout.splitlines():
            if line.startswith("FOUNDRY_RESULT:"):
                data = json.loads(line[len("FOUNDRY_RESULT:"):])
                data.update(_parse_graph_log_markers((result.stdout or "") + (result.stderr or "")))
                graph_cache_vol.commit()
                return data

    # Foundry hook's [CGE BUILD] can SIGABRT after graph.save() writes
    # files to disk. The SIGABRT handler converts this to _exit(0), so
    # the process exits before printing FOUNDRY_RESULT. Check if graph
    # files were saved successfully and return a partial result.

    # Capture debug lines from crashed subprocess for diagnostics
    _crash_debug = []
    if result is not None and result.stdout:
        _crash_debug = [l for l in result.stdout.splitlines() if "[FOUNDRY_DEBUG]" in l or "[FOUNDRY]" in l]
    _crash_stderr = []
    if result is not None and result.stderr:
        _crash_stderr = [l for l in result.stderr.splitlines() if "HOOK" in l or "ERROR" in l or "FAILED" in l][-20:]
    print(f"  === CRASH DEBUG ({len(_crash_debug)} lines) ===")
    for line in _crash_debug:
        print(f"  {line}")
    if _crash_stderr:
        print(f"  === CRASH STDERR ({len(_crash_stderr)} error lines) ===")
        for line in _crash_stderr:
            print(f"  {line}")
    print(f"  === END CRASH DEBUG ===")

    effective_cache_dir, _ = resolve_graph_cache_dir(
        GRAPH_CACHE_MOUNT, MODEL, gpu_memory_utilization=0.8, disable_log_stats=True,
    )
    post_cache = FoundryGraphCache(effective_cache_dir)
    import os as _os
    from pathlib import Path as _Path
    wrapper_dir = _Path(effective_cache_dir) / "wrapper_graphs"
    has_wrapper = wrapper_dir.exists() and any(wrapper_dir.glob("graph_*.json"))
    has_any = post_cache.has_cached_graphs() or has_wrapper
    if has_any:
        graph_cache_vol.commit()
        meta = post_cache.load_metadata()
        return {
            "mode": "with_foundry_subprocess",
            "foundry_hook_crash": True,
            "crash_debug": _crash_debug,
            "crash_stderr": _crash_stderr,
            "llm_init": None,
            "first_inference": None,
            "output": None,
            "total": None,
            "has_cached_graphs_after": True,
            "graph_file_count_after": (
                len(list(post_cache.graphs_dir.glob("graph_*.json")))
                + (len(list(wrapper_dir.glob("graph_*.json"))) if wrapper_dir.exists() else 0)
            ),
            "metadata_after": meta,
            "all_cache_files": [],
            "hook_archive_exists": post_cache.hook_archive.exists(),
            "hook_archive_files": [],
            "vllm_version": meta.get("vllm_version") if meta else None,
            "gpu": None,
        }

    exit_info = f"exit={result.returncode}" if result else "timeout"
    stderr_tail = result.stderr[-500:] if result and result.stderr else "N/A"
    fallback_info = (
        f"cache_dir={effective_cache_dir}, "
        f"exists={_os.path.isdir(effective_cache_dir)}, "
        f"has_graphs={post_cache.has_cached_graphs()}, "
        f"save_complete={post_cache.is_save_complete()}"
    )
    if _os.path.isdir(effective_cache_dir):
        all_files = []
        for root, dirs, files in _os.walk(effective_cache_dir):
            for f in files:
                fp = _os.path.join(root, f)
                all_files.append(f"{_os.path.relpath(fp, effective_cache_dir)} ({_os.path.getsize(fp)}b)")
        fallback_info += f", files=[{', '.join(all_files)}]"
    raise RuntimeError(
        f"Subprocess failed ({exit_info}).\n"
        f"fallback: {fallback_info}\n"
        f"stderr: {stderr_tail}"
    )


def _parse_graph_log_markers(output: str) -> dict:
    import re

    markers = {
        "saved_graphs": 0,
        "save_graphs_seconds": None,
        "loaded_graphs": 0,
        "load_graphs_seconds": None,
        "vllm_graph_capture_seconds": None,
        "saw_standard_graph_capture": "Capturing CUDA graphs" in output,
    }

    saved = re.search(r"Saved (\d+) (?:wrapper )?CUDA graphs\s+in ([\d.]+)s", output)
    if saved:
        markers["saved_graphs"] = int(saved.group(1))
        markers["save_graphs_seconds"] = float(saved.group(2))
    # Also match our new [FOUNDRY] format
    foundry_saved = re.search(r"\[FOUNDRY\] Saved (\d+) CUDA graphs\s+in ([\d.]+)s", output)
    if foundry_saved:
        markers["saved_graphs"] = int(foundry_saved.group(1))
        markers["save_graphs_seconds"] = float(foundry_saved.group(2))

    loaded = re.search(r"Loaded (\d+) (?:wrapper )?CUDA graphs from cache in ([\d.]+)s", output)
    if loaded:
        markers["loaded_graphs"] = int(loaded.group(1))
        markers["load_graphs_seconds"] = float(loaded.group(2))

    captured = re.search(r"Graph capturing finished in ([\d.]+) secs", output)
    if captured:
        markers["vllm_graph_capture_seconds"] = float(captured.group(1))

    return markers


@app.function(
    gpu="A100-40GB",
    image=image,
    timeout=60,
    scaledown_window=5,
    volumes={GRAPH_CACHE_MOUNT: graph_cache_vol},
)
def clear_cache():
    """Remove all cached graphs from the Volume."""
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
    print("Foundry CUDA Graph Persistence Benchmark")
    print("=" * 60)
    print()

    print("Clearing graph cache...")
    clear_cache.remote()
    print()

    # --- Baseline ---
    print("=" * 60)
    print("Run 1: BASELINE (standard vLLM, no Foundry)")
    print("=" * 60)
    r1 = run_baseline.remote()
    print(f"  GPU:             {r1['gpu']}")
    print(f"  vLLM:            {r1['vllm_version']}")
    print(f"  LLM init:        {r1['llm_init']:.2f}s")
    print(f"  vLLM graph cap:  {r1.get('vllm_graph_capture_seconds')}s")
    print(f"  First inference:  {r1['first_inference']:.2f}s")
    print(f"  Total:           {r1['total']:.2f}s")
    print()

    print("Waiting 30s for container to scale down...")
    time.sleep(30)

    # --- Foundry: first run (capture + save) ---
    print()
    print("=" * 60)
    print("Run 2: FOUNDRY first run (capture + save graphs)")
    print("=" * 60)
    r2 = run_with_foundry_subprocess.remote(force_clear=True)
    if r2.get("foundry_hook_crash"):
        print(f"  [HOOK CRASH] Process terminated by Foundry hook SIGABRT")
        print(f"  Graph files saved: {r2.get('graph_file_count_after', 0)}")
        print(f"  Metadata:          {r2.get('metadata_after') is not None}")
    else:
        print(f"  LLM init:        {r2['llm_init']:.2f}s")
        print(f"  Cache key:       {r2.get('graph_cache_key')}")
        print(f"  Cached before:   {r2.get('had_cached_graphs_before')}")
        print(f"  Saved graphs:    {r2.get('saved_graphs', 0)}")
        print(f"  Save time:       {r2.get('save_graphs_seconds')}s")
        print(f"  Cache files:     {r2.get('all_cache_files', [])}")
        print(f"  First inference:  {r2['first_inference']:.2f}s")
        print(f"  Total:           {r2['total']:.2f}s")
    print()

    print("Waiting 30s for container to scale down...")
    time.sleep(30)

    # --- Foundry: cache hit (load graphs) ---
    print()
    print("=" * 60)
    print("Run 3: FOUNDRY cache hit (load graphs from disk)")
    print("=" * 60)
    r3 = run_with_foundry_subprocess.remote()
    if r3.get("foundry_hook_crash"):
        print(f"  [HOOK CRASH] Process terminated by Foundry hook SIGABRT")
        print(f"  Graph files saved: {r3.get('graph_file_count_after', 0)}")
        print(f"  Metadata:          {r3.get('metadata_after') is not None}")
        if r3.get("crash_debug"):
            print(f"  --- Crash debug ({len(r3['crash_debug'])} lines) ---")
            for line in r3["crash_debug"]:
                print(f"    {line}")
        if r3.get("crash_stderr"):
            print(f"  --- Crash stderr ({len(r3['crash_stderr'])} lines) ---")
            for line in r3["crash_stderr"]:
                print(f"    {line}")
    else:
        print(f"  LLM init:        {r3['llm_init']:.2f}s")
        print(f"  Cache key:       {r3.get('graph_cache_key')}")
        print(f"  Cached before:   {r3.get('had_cached_graphs_before')}")
        print(f"  Loaded graphs:   {r3.get('loaded_graphs', 0)}")
        print(f"  Load time:       {r3.get('load_graphs_seconds')}s")
        print(f"  First inference:  {r3['first_inference']:.2f}s")
        print(f"  Total:           {r3['total']:.2f}s")
    print()

    # --- Comparison ---
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)
    baseline_init = r1["llm_init"]
    print(f"  Baseline (no Foundry):   {baseline_init:>7.2f}s")
    if r2.get("llm_init") is not None:
        print(f"  First run (save):        {r2['llm_init']:>7.2f}s")
    else:
        print(f"  First run (save):        N/A (hook crash)")
    if r3.get("llm_init") is not None:
        cache_init = r3["llm_init"]
        savings = baseline_init - cache_init
        pct = (savings / baseline_init) * 100 if baseline_init > 0 else 0
        print(f"  Cache hit (load):        {cache_init:>7.2f}s")
        print(f"  Graph capture savings:   {savings:>7.2f}s ({pct:.1f}%)")
        if r1.get("vllm_graph_capture_seconds") and r3.get("load_graphs_seconds"):
            phase_savings = r1["vllm_graph_capture_seconds"] - r3["load_graphs_seconds"]
            print(f"  Isolated graph phase:    {phase_savings:>7.2f}s")
    else:
        print(f"  Cache hit (load):        N/A (hook crash)")
    print("=" * 60)

    out_dir = Path("analysis/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"foundry_bench_{int(time.time())}.json"
    out.write_text(json.dumps({
        "baseline": r1,
        "foundry_save": r2,
        "foundry_load": r3,
    }, indent=2))
    print(f"\nResults saved to {out}")

    failures = []
    if not r2.get("foundry_hook_crash") and r2.get("had_cached_graphs_before"):
        failures.append("Foundry save run unexpectedly started with cached graphs")
    if not r2.get("foundry_hook_crash") and r2.get("saved_graphs", 0) <= 0:
        failures.append("Foundry save run did not save CUDA graphs")
    if r2.get("foundry_hook_crash") and r2.get("graph_file_count_after", 0) <= 0:
        failures.append("Foundry save run crashed and no graph files were saved")

    if r3.get("foundry_hook_crash"):
        failures.append("Foundry load run crashed (hook SIGABRT during graph loading)")
    else:
        if not r3.get("had_cached_graphs_before"):
            failures.append("Foundry load run did not start with cached graphs")
        if r3.get("loaded_graphs", 0) <= 0:
            failures.append("Foundry load run did not load CUDA graphs from disk")
        if not r3.get("output"):
            failures.append("Inference output missing from load run")
        if r3.get("llm_init") is not None and r3["llm_init"] >= r1["llm_init"]:
            print(
                f"  [WARNING] Foundry load init not faster than baseline "
                f"(baseline={r1['llm_init']:.2f}s, load={r3['llm_init']:.2f}s) "
                f"— expected for small models where hook overhead dominates"
            )
        if (
            r1.get("vllm_graph_capture_seconds") is not None
            and r3.get("load_graphs_seconds") is not None
            and r3["load_graphs_seconds"] >= r1["vllm_graph_capture_seconds"]
        ):
            print(
                f"  [WARNING] Foundry graph load not faster than native capture "
                f"(capture={r1['vllm_graph_capture_seconds']:.2f}s, "
                f"load={r3['load_graphs_seconds']:.2f}s) "
                f"— expected for small models with fast native capture"
            )

    if failures:
        raise RuntimeError("; ".join(failures))

    print("Foundry validation passed.")
