"""Modal speed proof for vllm-profile-cache.

This is not a broad benchmark. It isolates the profile-cache effect:
1. populate the profile cache once
2. run a normal vLLM init after warmup
3. run a cached vLLM init after warmup

Both measured starts run in child processes in the same Modal container so model
files and compile artifacts are already local. The test fails if the cached
start does not hit the cache, does not inject kv_cache_memory_bytes, or is not
faster than the normal start.

Usage:
    modal run profiling/modal_speed_proof.py
"""

from __future__ import annotations

import json
import textwrap
import time

import modal


app = modal.App("vllm-profile-cache-speed-proof")

profile_cache_vol = modal.Volume.from_name(
    "vllm-profile-cache-speed-proof", create_if_missing=True
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm", "torch", "transformers")
    .add_local_dir(
        "src/vllm_profile_cache",
        remote_path="/root/vllm_profile_cache",
    )
)

GPU_TYPE = "L4"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
CACHE_MOUNT = "/root/.cache/vllm-profile-cache"


@app.function(
    gpu=GPU_TYPE,
    image=image,
    timeout=1200,
    scaledown_window=2,
    volumes={CACHE_MOUNT: profile_cache_vol},
)
def run_speed_proof():
    import os
    import shutil
    import subprocess
    import sys

    def clear_cache_dir():
        if os.path.exists(CACHE_MOUNT):
            for name in os.listdir(CACHE_MOUNT):
                path = os.path.join(CACHE_MOUNT, name)
                if os.path.isfile(path):
                    os.unlink(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)

    def child_script(mode: str) -> str:
        use_cache = mode != "uncached"
        return textwrap.dedent(
            f"""
            import json
            import logging
            import sys
            import time

            sys.path.insert(0, "/root")
            logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

            from vllm import LLM, SamplingParams
            from vllm_profile_cache.cache import ProfileCache, build_cache_key_from_vllm_config
            from vllm_profile_cache.modal_plugin import cached_vllm_init

            model = {MODEL!r}
            cache_dir = {CACHE_MOUNT!r}
            cache_key, _ = build_cache_key_from_vllm_config(
                model=model,
                gpu_memory_utilization=0.7,
                enforce_eager=True,
            )
            cache = ProfileCache(cache_dir)
            hit_before = cache.get(cache_key) is not None

            t0 = time.perf_counter()
            if {use_cache!r}:
                llm = cached_vllm_init(
                    model=model,
                    profile_cache_dir=cache_dir,
                    gpu_memory_utilization=0.7,
                    enforce_eager=True,
                    disable_log_stats=True,
                    safety_margin_pct=5.0,
                )
            else:
                llm = LLM(
                    model=model,
                    gpu_memory_utilization=0.7,
                    enforce_eager=True,
                    disable_log_stats=True,
                )
            init_seconds = time.perf_counter() - t0

            cache_config = llm.llm_engine.vllm_config.cache_config
            injected = getattr(cache_config, "kv_cache_memory_bytes", None)

            t0 = time.perf_counter()
            output = llm.generate(["Hello!"], SamplingParams(max_tokens=8))
            inference_seconds = time.perf_counter() - t0
            text = output[0].outputs[0].text if output else ""

            result = {{
                "mode": {mode!r},
                "cache_key": cache_key,
                "cache_hit_before_init": hit_before,
                "kv_cache_memory_bytes_in_config": injected,
                "llm_init": init_seconds,
                "first_inference": inference_seconds,
                "generated_text": text,
                "cache_entry_count_after": len(cache.list_entries()),
            }}
            print("RESULT:" + json.dumps(result))
            """
        )

    def run_child(mode: str) -> dict:
        proc = subprocess.run(
            [sys.executable, "-c", child_script(mode)],
            capture_output=True,
            text=True,
            timeout=900,
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        for line in output.splitlines():
            if line.startswith("RESULT:"):
                data = json.loads(line[len("RESULT:"):])
                data["returncode"] = proc.returncode
                data["saw_profile_markers"] = any(
                    marker in output
                    for marker in (
                        "Initial profiling/warmup run",
                        "Profiling CUDA graph memory",
                        "Available KV cache memory",
                    )
                )
                data["log_tail"] = "\n".join(output.splitlines()[-40:])
                return data
        raise RuntimeError(
            f"{mode} child failed with exit {proc.returncode}. "
            f"Output tail:\n" + "\n".join(output.splitlines()[-80:])
        )

    clear_cache_dir()

    populate = run_child("populate")
    profile_cache_vol.commit()

    uncached = run_child("uncached")
    cached = run_child("cached")
    profile_cache_vol.commit()

    failures = []
    if populate["cache_hit_before_init"]:
        failures.append("populate run unexpectedly hit cache")
    if populate["cache_entry_count_after"] < 1:
        failures.append("populate run did not save a cache entry")
    if not cached["cache_hit_before_init"]:
        failures.append("cached run did not hit cache")
    if cached["kv_cache_memory_bytes_in_config"] in (None, 0):
        failures.append("cached run did not inject kv_cache_memory_bytes")
    if not cached["generated_text"]:
        failures.append("cached run did not generate text")
    if cached["llm_init"] >= uncached["llm_init"]:
        failures.append(
            "cached init was not faster "
            f"(uncached={uncached['llm_init']:.2f}s, cached={cached['llm_init']:.2f}s)"
        )

    return {
        "gpu_type": GPU_TYPE,
        "model": MODEL,
        "populate": populate,
        "uncached": uncached,
        "cached": cached,
        "speedup_seconds": uncached["llm_init"] - cached["llm_init"],
        "speedup_pct": (
            (uncached["llm_init"] - cached["llm_init"]) / uncached["llm_init"] * 100
        ),
        "failures": failures,
    }


@app.local_entrypoint()
def main():
    from pathlib import Path

    result = run_speed_proof.remote()

    print("=" * 60)
    print("vLLM profile-cache speed proof")
    print("=" * 60)
    print(f"GPU: {result['gpu_type']}")
    print(f"Model: {result['model']}")
    print(f"Populate init: {result['populate']['llm_init']:.2f}s")
    print(f"Uncached init: {result['uncached']['llm_init']:.2f}s")
    print(f"Cached init:   {result['cached']['llm_init']:.2f}s")
    print(f"Speedup:       {result['speedup_seconds']:.2f}s ({result['speedup_pct']:.1f}%)")
    print(f"Cache hit:     {result['cached']['cache_hit_before_init']}")
    print(
        "Injected:      "
        f"{result['cached']['kv_cache_memory_bytes_in_config']}"
    )

    out_dir = Path("analysis/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"speed_proof_{int(time.time())}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"Results saved to {out}")

    if result["failures"]:
        raise RuntimeError("; ".join(result["failures"]))

    print("Speed proof passed.")
