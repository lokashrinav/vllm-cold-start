"""Inspect the vLLM CUDA graph capture path inside the Modal Foundry image.

This does not allocate a model or run GPU inference. It prints the exact classes,
methods, modules, and environment values we need before patching Foundry support.

Usage:
    modal run profiling/modal_foundry_introspect.py
"""

from __future__ import annotations

import inspect
from pathlib import Path

import modal


app = modal.App("foundry-graph-introspect")

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
    .run_commands(
        "git clone https://github.com/lokashrinav/foundry /opt/foundry",
        "cd /opt/foundry && pip install -e . --no-build-isolation",
    )
    .add_local_dir(
        "src/vllm_profile_cache",
        remote_path="/root/vllm_profile_cache",
    )
)


@app.function(gpu="A100", image=image, timeout=300, scaledown_window=2)
def inspect_capture_path():
    import json
    import os
    import sys

    sys.path.insert(0, "/root")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import vllm
    import vllm.envs as envs
    from vllm.v1.worker.gpu import cudagraph_utils
    from vllm_profile_cache.foundry_graphs import patch_vllm_for_foundry

    CudaGraphManager = cudagraph_utils.CudaGraphManager
    ModelCudaGraphManager = cudagraph_utils.ModelCudaGraphManager

    before_base = CudaGraphManager.capture
    before_model = ModelCudaGraphManager.capture

    patch_error = None
    try:
        patch_vllm_for_foundry("/tmp/foundry-introspect", model_id="inspect")
    except Exception as exc:
        patch_error = repr(exc)

    after_base = CudaGraphManager.capture
    after_model = ModelCudaGraphManager.capture

    def method_info(fn):
        try:
            source = inspect.getsource(fn)
        except Exception as exc:
            source = f"<source unavailable: {exc!r}>"
        return {
            "module": getattr(fn, "__module__", None),
            "qualname": getattr(fn, "__qualname__", None),
            "signature": str(inspect.signature(fn)),
            "source_file": inspect.getsourcefile(fn),
            "source_head": "\n".join(source.splitlines()[:80]),
        }

    return {
        "python": sys.version,
        "vllm_version": vllm.__version__,
        "env_var": os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING"),
        "envs_value": envs.VLLM_ENABLE_V1_MULTIPROCESSING,
        "module_file": str(Path(cudagraph_utils.__file__)),
        "patch_error": patch_error,
        "base_before": method_info(before_base),
        "base_after": method_info(after_base),
        "model_before": method_info(before_model),
        "model_after": method_info(after_model),
        "base_changed": before_base is not after_base,
        "model_changed": before_model is not after_model,
    }


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path
    import time

    result = inspect_capture_path.remote()

    out_dir = Path("analysis/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"foundry_introspect_{int(time.time())}.json"
    out.write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))
    print(f"Results saved to {out}")
