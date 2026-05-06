"""Benchmark vLLM cold start WITH vs WITHOUT profile caching on Modal.

Run 1: Baseline (no cache)
Run 2: First cached run (saves profile to Volume)
Run 3: Second cached run (should hit cache, skip profiling)

Usage:
    modal run profiling/modal_cache_benchmark.py
"""

import modal
import time

app = modal.App("vllm-profile-cache-bench")

profile_cache_vol = modal.Volume.from_name(
    "vllm-profile-cache", create_if_missing=True
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm", "torch", "transformers")
    .add_local_dir(
        "src/vllm_profile_cache",
        remote_path="/root/vllm_profile_cache",
    )
)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
CACHE_MOUNT = "/root/.cache/vllm-profile-cache"


@app.function(
    gpu="A100",
    image=image,
    timeout=600,
    scaledown_window=5,
    volumes={CACHE_MOUNT: profile_cache_vol},
)
def clear_cache():
    """Remove all cached profiles from the Volume."""
    import os
    import shutil
    if os.path.exists(CACHE_MOUNT):
        for f in os.listdir(CACHE_MOUNT):
            path = os.path.join(CACHE_MOUNT, f)
            if os.path.isfile(path):
                os.unlink(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
    profile_cache_vol.commit()
    return {"cleared": True}


@app.function(
    gpu="A100",
    image=image,
    timeout=600,
    scaledown_window=5,
    volumes={CACHE_MOUNT: profile_cache_vol},
)
def run_without_cache():
    """Baseline: normal vLLM cold start without profile caching."""
    import sys
    sys.path.insert(0, "/root")

    results = {"mode": "no_cache"}
    t_total = time.perf_counter()

    import torch
    import vllm
    results["vllm_version"] = vllm.__version__
    results["gpu"] = torch.cuda.get_device_name(0)

    t0 = time.perf_counter()
    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL, gpu_memory_utilization=0.8, disable_log_stats=True)
    results["llm_init"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    output = llm.generate(["Hello!"], SamplingParams(max_tokens=8))
    results["first_inference"] = time.perf_counter() - t0

    results["total"] = time.perf_counter() - t_total
    return results


@app.function(
    gpu="A100",
    image=image,
    timeout=600,
    scaledown_window=5,
    volumes={CACHE_MOUNT: profile_cache_vol},
)
def run_with_cache():
    """Cached: use vllm-profile-cache to skip memory profiling."""
    import sys
    import logging
    sys.path.insert(0, "/root")
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    results = {"mode": "with_cache"}
    t_total = time.perf_counter()

    import torch
    import vllm
    results["vllm_version"] = vllm.__version__
    results["gpu"] = torch.cuda.get_device_name(0)

    t0 = time.perf_counter()
    from vllm import SamplingParams
    from vllm_profile_cache.modal_plugin import cached_vllm_init
    from vllm_profile_cache.cache import ProfileCache, build_cache_key_from_vllm_config

    cache_key, _ = build_cache_key_from_vllm_config(
        model=MODEL,
        gpu_memory_utilization=0.8,
    )
    results["profile_cache_key"] = cache_key
    results["profile_cache_hit_before"] = (
        ProfileCache(CACHE_MOUNT).get(cache_key) is not None
    )

    llm = cached_vllm_init(
        model=MODEL,
        profile_cache_dir=CACHE_MOUNT,
        gpu_memory_utilization=0.8,
        disable_log_stats=True,
        safety_margin_pct=5.0,
    )
    results["llm_init"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    output = llm.generate(["Hello!"], SamplingParams(max_tokens=8))
    results["first_inference"] = time.perf_counter() - t0

    results["total"] = time.perf_counter() - t_total
    results["profile_cache_keys_after"] = [
        entry.cache_key for entry in ProfileCache(CACHE_MOUNT).list_entries()
    ]

    profile_cache_vol.commit()
    return results


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path

    print("Clearing profile cache volume...")
    clear_cache.remote()
    print("Cache cleared.\n")

    print("=" * 60)
    print("Run 1: BASELINE (no profile cache)")
    print("=" * 60)
    r1 = run_without_cache.remote()
    print(f"  Mode: {r1['mode']}")
    print(f"  LLM init: {r1['llm_init']:.2f}s")
    print(f"  First inference: {r1['first_inference']:.2f}s")
    print(f"  Total: {r1['total']:.2f}s")
    print()

    print("Waiting 10s for container to scale down...")
    time.sleep(10)
    print()

    print("=" * 60)
    print("Run 2: WITH cache (first run — saves profile)")
    print("=" * 60)
    r2 = run_with_cache.remote()
    print(f"  Mode: {r2['mode']}")
    print(f"  Cache key: {r2['profile_cache_key']}")
    print(f"  Cache hit before init: {r2['profile_cache_hit_before']}")
    print(f"  LLM init: {r2['llm_init']:.2f}s")
    print(f"  First inference: {r2['first_inference']:.2f}s")
    print(f"  Total: {r2['total']:.2f}s")
    print()

    print("Waiting 10s for container to scale down...")
    time.sleep(10)
    print()

    print("=" * 60)
    print("Run 3: WITH cache (second run — should skip profiling)")
    print("=" * 60)
    r3 = run_with_cache.remote()
    print(f"  Mode: {r3['mode']}")
    print(f"  Cache key: {r3['profile_cache_key']}")
    print(f"  Cache hit before init: {r3['profile_cache_hit_before']}")
    print(f"  LLM init: {r3['llm_init']:.2f}s")
    print(f"  First inference: {r3['first_inference']:.2f}s")
    print(f"  Total: {r3['total']:.2f}s")
    print()

    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)
    if r1["llm_init"] > 0 and r3["llm_init"] > 0:
        savings = r1["llm_init"] - r3["llm_init"]
        pct = (savings / r1["llm_init"]) * 100 if r1["llm_init"] > 0 else 0
        print(f"  Baseline (no cache):     {r1['llm_init']:>7.2f}s")
        print(f"  First run (saves cache): {r2['llm_init']:>7.2f}s")
        print(f"  Cache hit (skips prof):  {r3['llm_init']:>7.2f}s")
        print(f"  Savings vs baseline:     {savings:>7.2f}s ({pct:.1f}%)")
        print(f"  Verified cache hit:      {r3['profile_cache_hit_before']}")
    print("=" * 60)

    out_dir = Path("analysis/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"cache_bench_{int(time.time())}.json"
    out.write_text(json.dumps({
        "baseline": r1,
        "cache_save": r2,
        "cache_hit": r3,
    }, indent=2))
    print(f"\nResults saved to {out}")
