"""CEO-facing Foundry validation matrix on Modal.

This tests CUDA graph persistence across more than one model/GPU shape:
- A100 + Qwen2.5-0.5B
- A100 + Qwen2.5-1.5B
- L4 + Qwen2.5-0.5B

Each case runs:
1. baseline vLLM
2. Foundry first run that must save CUDA graphs
3. Foundry second run that must load CUDA graphs

The command fails unless every case saves graphs, loads graphs, runs inference,
and has faster cached Foundry init than baseline init.

Usage:
    modal run profiling/modal_foundry_matrix.py
"""

from __future__ import annotations

import json
import re
import textwrap
import time
from pathlib import Path

import modal


app = modal.App("foundry-graph-matrix")

graph_cache_vol = modal.Volume.from_name(
    "foundry-graph-cache-matrix", create_if_missing=True
)

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
    .run_commands(
        "git clone https://github.com/lokashrinav/foundry /opt/foundry",
        "cd /opt/foundry && pip install -e . --no-build-isolation",
    )
    .add_local_dir(
        "src/vllm_profile_cache",
        remote_path="/root/vllm_profile_cache",
    )
)

GRAPH_CACHE_MOUNT = "/root/.cache/foundry-graphs"
PROMPT = "What is the meaning of life?"

CASES = [
    {
        "name": "a100_qwen_0_5b",
        "gpu": "A100",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "gpu_memory_utilization": 0.8,
        "region_size": "32GB",
    },
    {
        "name": "a100_qwen_1_5b",
        "gpu": "A100",
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "gpu_memory_utilization": 0.8,
        "region_size": "32GB",
    },
    {
        "name": "l4_qwen_0_5b",
        "gpu": "L4",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "gpu_memory_utilization": 0.7,
        "region_size": "24GB",
    },
]


def _get_hook_path() -> str:
    import site

    candidates = [Path("/opt/foundry/python/foundry/libcuda_hook.so")]
    for root in site.getsitepackages():
        candidates.append(Path(root) / "foundry" / "libcuda_hook.so")

    for path in candidates:
        if path.is_file():
            return str(path.resolve())

    raise RuntimeError("libcuda_hook.so not found")


def _parse_graph_log_markers(output: str) -> dict:
    markers = {
        "saved_graphs": 0,
        "save_graphs_seconds": None,
        "loaded_graphs": 0,
        "load_graphs_seconds": None,
        "vllm_graph_capture_seconds": None,
        "saw_standard_graph_capture": "Capturing CUDA graphs" in output,
    }

    saved = re.search(r"Saved (\d+) CUDA graphs in ([\d.]+)s", output)
    if saved:
        markers["saved_graphs"] = int(saved.group(1))
        markers["save_graphs_seconds"] = float(saved.group(2))

    loaded = re.search(r"Loaded (\d+) CUDA graphs from cache in ([\d.]+)s", output)
    if loaded:
        markers["loaded_graphs"] = int(loaded.group(1))
        markers["load_graphs_seconds"] = float(loaded.group(2))

    captured = re.search(r"Graph capturing finished in ([\d.]+) secs", output)
    if captured:
        markers["vllm_graph_capture_seconds"] = float(captured.group(1))

    return markers


def _child_script(case: dict, mode: str) -> str:
    use_foundry = mode != "baseline"
    return textwrap.dedent(
        f"""
        import json
        import logging
        import sys
        import time

        sys.path.insert(0, "/root")
        logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

        model = {case["model"]!r}
        gpu_memory_utilization = {case["gpu_memory_utilization"]!r}
        region_size = {case["region_size"]!r}

        results = {{
            "case": {case["name"]!r},
            "mode": {mode!r},
            "model": model,
        }}
        t_total = time.perf_counter()

        import vllm
        from vllm import LLM, SamplingParams

        results["vllm_version"] = vllm.__version__

        if {use_foundry!r}:
            from vllm_profile_cache.foundry_graphs import (
                FoundryGraphCache,
                cached_vllm_init_with_foundry,
                is_foundry_available,
                resolve_graph_cache_dir,
            )

            effective_cache_dir, graph_cache_key = resolve_graph_cache_dir(
                {GRAPH_CACHE_MOUNT!r},
                model,
                region_size,
                gpu_memory_utilization=gpu_memory_utilization,
                disable_log_stats=True,
            )
            graph_cache = FoundryGraphCache(effective_cache_dir)
            results["graph_cache_key"] = graph_cache_key
            results["effective_graph_cache_dir"] = effective_cache_dir
            results["had_cached_graphs_before"] = graph_cache.has_cached_graphs()
            results["foundry_available_before_init"] = is_foundry_available()

            t0 = time.perf_counter()
            llm = cached_vllm_init_with_foundry(
                model=model,
                graph_cache_dir={GRAPH_CACHE_MOUNT!r},
                region_size=region_size,
                gpu_memory_utilization=gpu_memory_utilization,
                disable_log_stats=True,
            )
            results["llm_init"] = time.perf_counter() - t0
            results["has_cached_graphs_after"] = graph_cache.has_cached_graphs()
            results["graph_file_count_after"] = len(list(graph_cache.graphs_dir.glob("graph_*.json")))
            results["metadata_after"] = graph_cache.load_metadata()
            results["foundry_available_after_init"] = is_foundry_available()
        else:
            t0 = time.perf_counter()
            llm = LLM(
                model=model,
                gpu_memory_utilization=gpu_memory_utilization,
                disable_log_stats=True,
            )
            results["llm_init"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        output = llm.generate([{PROMPT!r}], SamplingParams(max_tokens=16))
        results["first_inference"] = time.perf_counter() - t0
        results["output"] = output[0].outputs[0].text
        results["total"] = time.perf_counter() - t_total

        import torch
        results["gpu"] = torch.cuda.get_device_name(0)

        print("CASE_RESULT:" + json.dumps(results))
        """
    )


def _run_child(case: dict, mode: str) -> dict:
    import os
    import subprocess
    import sys

    env = dict(os.environ)
    # Foundry patches Python vLLM objects. Keep vLLM's V1 engine core in this
    # process so the patched CudaGraphManager.capture() is the one that runs.
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    if mode != "baseline":
        env["LD_PRELOAD"] = _get_hook_path()

    result = subprocess.run(
        [sys.executable, "-c", _child_script(case, mode)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    output = (result.stdout or "") + (result.stderr or "")

    for line in result.stdout.splitlines():
        if line.startswith("CASE_RESULT:"):
            data = json.loads(line[len("CASE_RESULT:"):])
            data.update(_parse_graph_log_markers(output))
            data["returncode"] = result.returncode
            data["log_tail"] = "\n".join(output.splitlines()[-80:])
            return data

    raise RuntimeError(
        f"{case['name']} {mode} failed (exit={result.returncode}).\n"
        f"Output tail:\n" + "\n".join(output.splitlines()[-120:])
    )


def _clear_graph_cache_root():
    import os
    import shutil

    if os.path.exists(GRAPH_CACHE_MOUNT):
        for name in os.listdir(GRAPH_CACHE_MOUNT):
            path = os.path.join(GRAPH_CACHE_MOUNT, name)
            if os.path.isfile(path):
                os.unlink(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)


def _validate_case(result: dict) -> list[str]:
    failures = []
    baseline = result["baseline"]
    save = result["foundry_save"]
    load = result["foundry_load"]

    if save["had_cached_graphs_before"]:
        failures.append("save run unexpectedly started with cached graphs")
    if save["saved_graphs"] <= 0:
        failures.append("save run did not save CUDA graphs")
    if save["graph_file_count_after"] <= 0:
        failures.append("save run did not leave graph files on disk")
    if not load["had_cached_graphs_before"]:
        failures.append("load run did not start with cached graphs")
    if load["loaded_graphs"] <= 0:
        failures.append("load run did not load CUDA graphs")
    if not baseline["output"] or not save["output"] or not load["output"]:
        failures.append("inference output missing")
    if load["llm_init"] >= baseline["llm_init"]:
        failures.append(
            "Foundry load init was not faster than baseline "
            f"(baseline={baseline['llm_init']:.2f}s, load={load['llm_init']:.2f}s)"
        )
    if (
        baseline.get("vllm_graph_capture_seconds") is not None
        and load.get("load_graphs_seconds") is not None
        and load["load_graphs_seconds"] >= baseline["vllm_graph_capture_seconds"]
    ):
        failures.append(
            "Foundry load phase was not faster than standard graph capture "
            f"(capture={baseline['vllm_graph_capture_seconds']:.2f}s, "
            f"load={load['load_graphs_seconds']:.2f}s)"
        )

    return failures


@app.function(
    gpu="A100",
    image=image,
    timeout=1800,
    scaledown_window=5,
    volumes={GRAPH_CACHE_MOUNT: graph_cache_vol},
)
def run_case_a100(case: dict):
    return _run_case(case)


@app.function(
    gpu="L4",
    image=image,
    timeout=1800,
    scaledown_window=5,
    volumes={GRAPH_CACHE_MOUNT: graph_cache_vol},
)
def run_case_l4(case: dict):
    return _run_case(case)


def _run_case(case: dict):
    _clear_graph_cache_root()
    graph_cache_vol.commit()

    baseline = _run_child(case, "baseline")
    save = _run_child(case, "foundry_save")
    graph_cache_vol.commit()
    load = _run_child(case, "foundry_load")
    graph_cache_vol.commit()

    result = {
        "case": case,
        "baseline": baseline,
        "foundry_save": save,
        "foundry_load": load,
        "init_speedup_seconds": baseline["llm_init"] - load["llm_init"],
        "init_speedup_pct": (
            (baseline["llm_init"] - load["llm_init"]) / baseline["llm_init"] * 100
        ),
    }
    result["failures"] = _validate_case(result)
    if result["failures"]:
        detail = [
            f"{case['name']}: " + "; ".join(result["failures"]),
            "",
            "Foundry save result:",
            json.dumps(
                {
                    key: save.get(key)
                    for key in (
                        "foundry_available_before_init",
                        "foundry_available_after_init",
                        "had_cached_graphs_before",
                        "has_cached_graphs_after",
                        "saved_graphs",
                        "graph_file_count_after",
                        "save_graphs_seconds",
                        "vllm_graph_capture_seconds",
                        "saw_standard_graph_capture",
                    )
                },
                indent=2,
            ),
            "",
            "Foundry save log tail:",
            save.get("log_tail", ""),
            "",
            "Foundry load result:",
            json.dumps(
                {
                    key: load.get(key)
                    for key in (
                        "foundry_available_before_init",
                        "foundry_available_after_init",
                        "had_cached_graphs_before",
                        "has_cached_graphs_after",
                        "loaded_graphs",
                        "graph_file_count_after",
                        "load_graphs_seconds",
                        "vllm_graph_capture_seconds",
                        "saw_standard_graph_capture",
                    )
                },
                indent=2,
            ),
            "",
            "Foundry load log tail:",
            load.get("log_tail", ""),
        ]
        raise RuntimeError("\n".join(detail))
    return result


@app.local_entrypoint()
def main():
    results = []

    print("=" * 70)
    print("Foundry CUDA graph persistence validation matrix")
    print("=" * 70)

    for case in CASES:
        print()
        print(f"Running {case['name']} on {case['gpu']} ({case['model']})")
        if case["gpu"] == "A100":
            result = run_case_a100.remote(case)
        elif case["gpu"] == "L4":
            result = run_case_l4.remote(case)
        else:
            raise ValueError(f"Unsupported GPU in matrix: {case['gpu']}")
        results.append(result)

        baseline = result["baseline"]
        save = result["foundry_save"]
        load = result["foundry_load"]
        print(f"  Baseline init:   {baseline['llm_init']:.2f}s")
        print(f"  Foundry save:    {save['llm_init']:.2f}s")
        print(f"  Foundry load:    {load['llm_init']:.2f}s")
        print(
            f"  Init speedup:    {result['init_speedup_seconds']:.2f}s "
            f"({result['init_speedup_pct']:.1f}%)"
        )
        print(f"  Saved graphs:    {save['saved_graphs']}")
        print(f"  Loaded graphs:   {load['loaded_graphs']}")
        print(f"  Load time:       {load['load_graphs_seconds']}s")

    out_dir = Path("analysis/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"foundry_matrix_{int(time.time())}.json"
    out.write_text(json.dumps({"results": results}, indent=2))

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for result in results:
        case = result["case"]
        print(
            f"{case['name']}: {result['init_speedup_seconds']:.2f}s "
            f"({result['init_speedup_pct']:.1f}%) init speedup"
        )
    print(f"Results saved to {out}")
    print("Foundry matrix validation passed.")
