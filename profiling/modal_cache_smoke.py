"""Minimal Modal smoke test for vllm-profile-cache.

This verifies correctness rather than benchmark timing:
1. clear an isolated Modal Volume
2. run cached_vllm_init once to populate the profile
3. run cached_vllm_init again and require a cache hit before init
4. verify vLLM can still generate after the cached launch

Usage:
    modal run profiling/modal_cache_smoke.py
"""

from __future__ import annotations

import time

import modal


app = modal.App("vllm-profile-cache-smoke")

profile_cache_vol = modal.Volume.from_name(
    "vllm-profile-cache-smoke", create_if_missing=True
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
    timeout=600,
    scaledown_window=2,
    volumes={CACHE_MOUNT: profile_cache_vol},
)
def clear_cache():
    import os
    import shutil

    if os.path.exists(CACHE_MOUNT):
        for name in os.listdir(CACHE_MOUNT):
            path = os.path.join(CACHE_MOUNT, name)
            if os.path.isfile(path):
                os.unlink(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)

    profile_cache_vol.commit()
    return {"cleared": True}


@app.function(
    gpu=GPU_TYPE,
    image=image,
    timeout=900,
    scaledown_window=2,
    volumes={CACHE_MOUNT: profile_cache_vol},
)
def run_cached(label: str):
    import logging
    import sys
    import time

    sys.path.insert(0, "/root")
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    import torch
    import vllm
    from vllm import SamplingParams
    from vllm_profile_cache.cache import (
        ProfileCache,
        build_cache_key_from_vllm_config,
    )
    from vllm_profile_cache.modal_plugin import cached_vllm_init

    cache_key, _ = build_cache_key_from_vllm_config(
        model=MODEL,
        gpu_memory_utilization=0.7,
    )
    cache = ProfileCache(CACHE_MOUNT)
    hit_before = cache.get(cache_key) is not None

    t0 = time.perf_counter()
    llm = cached_vllm_init(
        model=MODEL,
        profile_cache_dir=CACHE_MOUNT,
        gpu_memory_utilization=0.7,
        disable_log_stats=True,
        safety_margin_pct=5.0,
    )
    init_seconds = time.perf_counter() - t0

    cache_config = llm.llm_engine.vllm_config.cache_config
    injected_bytes = getattr(cache_config, "kv_cache_memory_bytes", None)

    t0 = time.perf_counter()
    output = llm.generate(["Hello!"], SamplingParams(max_tokens=8))
    inference_seconds = time.perf_counter() - t0
    text = output[0].outputs[0].text if output else ""

    entries_after = cache.list_entries()
    profile_cache_vol.commit()

    return {
        "label": label,
        "gpu": torch.cuda.get_device_name(0),
        "vllm_version": vllm.__version__,
        "cache_key": cache_key,
        "cache_hit_before_init": hit_before,
        "cache_entries_after": [entry.cache_key for entry in entries_after],
        "cache_entry_count_after": len(entries_after),
        "kv_cache_memory_bytes_in_config": injected_bytes,
        "generated_text": text,
        "llm_init": init_seconds,
        "first_inference": inference_seconds,
    }


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path

    print("Clearing isolated smoke-test profile cache...")
    clear_cache.remote()

    print("Run 1: populate profile cache")
    first = run_cached.remote("populate")
    print(f"  GPU: {first['gpu']}")
    print(f"  Cache key: {first['cache_key']}")
    print(f"  Cache hit before init: {first['cache_hit_before_init']}")
    print(f"  Cache entries after: {first['cache_entry_count_after']}")
    print(f"  LLM init: {first['llm_init']:.2f}s")

    print("Waiting for scale-down window...")
    time.sleep(5)

    print("Run 2: require profile cache hit")
    second = run_cached.remote("hit")
    print(f"  GPU: {second['gpu']}")
    print(f"  Cache key: {second['cache_key']}")
    print(f"  Cache hit before init: {second['cache_hit_before_init']}")
    print(f"  kv_cache_memory_bytes in config: {second['kv_cache_memory_bytes_in_config']}")
    print(f"  LLM init: {second['llm_init']:.2f}s")
    print(f"  First inference: {second['first_inference']:.2f}s")

    failures = []
    if first["cache_hit_before_init"]:
        failures.append("first run unexpectedly found a cache entry")
    if first["cache_entry_count_after"] < 1:
        failures.append("first run did not save a cache entry")
    if not second["cache_hit_before_init"]:
        failures.append("second run did not find the saved cache entry")
    if second["kv_cache_memory_bytes_in_config"] in (None, 0):
        failures.append("cached run did not inject kv_cache_memory_bytes")
    if not second["generated_text"]:
        failures.append("cached run did not produce generated text")

    out_dir = Path("analysis/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"cache_smoke_{int(time.time())}.json"
    out.write_text(json.dumps({"first": first, "second": second}, indent=2))
    print(f"Results saved to {out}")

    if failures:
        raise RuntimeError("; ".join(failures))

    print("Smoke test passed.")
