"""Wrapper that launches vLLM with cached memory profiling results.

On first run: starts vLLM normally, parses the recommended
kv_cache_memory_bytes from logs, and saves it to the profile cache.

On subsequent runs: loads the cached value and injects
--kv-cache-memory-bytes to skip the ~18s profiling forward pass.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
from typing import Optional

from vllm_profile_cache.cache import (
    CacheEntry,
    ProfileCache,
    _get_free_gpu_memory,
    _is_moe_model,
    apply_safety_margin,
    build_cache_key_from_vllm_config,
    check_free_memory,
)

logger = logging.getLogger(__name__)

KV_CACHE_REGEX = re.compile(
    r"(?:--kv-cache-memory(?:-bytes)?(?:=|\s+)|kv_cache_memory_bytes[=:]\s*)"
    r"[`\"]?(\d+)[`\"]?",
    re.IGNORECASE,
)

OOM_PATTERNS = (
    "out of memory",
    "cuda out of memory",
    "torch.cuda.outofmemoryerror",
)


def parse_kv_cache_from_logs(log_output: str) -> Optional[int]:
    """Extract the recommended kv_cache_memory_bytes from vLLM log output."""
    match = KV_CACHE_REGEX.search(log_output)
    if match:
        return int(match.group(1))
    return None


def launch_vllm(
    args: list[str],
    cache_dir: str = "~/.cache/vllm-profile-cache",
    safety_margin_pct: float = 5.0,
    force_reprofile: bool = False,
    gpu_identity_scope: Optional[str] = None,
) -> subprocess.CompletedProcess:
    """Launch vLLM with profile caching.

    Parameters
    ----------
    args:
        Arguments for ``vllm serve`` (e.g. ["--model", "meta-llama/..."]).
    cache_dir:
        Directory for cached profiles.
    safety_margin_pct:
        Percentage to reduce cached kv_cache_memory by (OOM protection).
    force_reprofile:
        If True, ignore cache and re-run profiling.
    gpu_identity_scope:
        ``"gpu_type"`` (default) reuses profiles across interchangeable GPUs
        with the same name and total memory. ``"device"`` ties the key to the
        physical GPU UUID.
    """
    import os
    cache_dir = os.path.expanduser(cache_dir)
    cache = ProfileCache(cache_dir)

    parsed = _parse_vllm_args(args)
    model = parsed.get("model", "")

    if _is_moe_model(model):
        logger.warning(
            "MoE model detected (%s). Skipping profile cache per vLLM RFC #27951 "
            "(data-dependent expert routing makes cached values unreliable).",
            model,
        )
        return _run_vllm(args)

    cache_key, metadata = build_cache_key_from_vllm_config(
        model=model,
        gpu_memory_utilization=parsed.get("gpu_memory_utilization", 0.9),
        dtype=parsed.get("dtype", "auto"),
        tensor_parallel_size=parsed.get("tensor_parallel_size", 1),
        pipeline_parallel_size=parsed.get("pipeline_parallel_size", 1),
        max_model_len=parsed.get("max_model_len"),
        max_num_batched_tokens=parsed.get("max_num_batched_tokens"),
        max_num_seqs=parsed.get("max_num_seqs"),
        quantization=parsed.get("quantization"),
        kv_cache_dtype=parsed.get("kv_cache_dtype"),
        enforce_eager=parsed.get("enforce_eager", False),
        gpu_identity_scope=gpu_identity_scope,
    )

    if not force_reprofile:
        entry = cache.get(cache_key)
        if entry is not None:
            if not check_free_memory(entry):
                logger.info("Free memory check failed, falling back to profiling.")
            else:
                cached_bytes = apply_safety_margin(
                    entry.kv_cache_memory_bytes, safety_margin_pct
                )
                logger.info(
                    "Using cached profile: kv_cache_memory=%d (with %.1f%% margin: %d)",
                    entry.kv_cache_memory_bytes, safety_margin_pct, cached_bytes,
                )
                injected_args = args + [f"--kv-cache-memory-bytes={cached_bytes}"]
                result = _run_vllm(injected_args, capture_output=True)

                if result.returncode != 0 and _looks_like_oom(result):
                    logger.warning(
                        "OOM with cached kv_cache_memory_bytes=%d. "
                        "Deleting cache entry and retrying with profiling.",
                        cached_bytes,
                    )
                    cache.invalidate(cache_key)
                    retry = _run_vllm(args, capture_output=True)
                    _save_profile_from_result(
                        retry, cache, cache_key, metadata, safety_margin_pct
                    )
                    return retry

                return result

    logger.info("No cached profile found. Running vLLM with profiling to capture baseline.")
    result = _run_vllm(args, capture_output=True)
    _save_profile_from_result(result, cache, cache_key, metadata, safety_margin_pct)
    return result


def _save_profile_from_result(
    result: subprocess.CompletedProcess,
    cache: ProfileCache,
    cache_key: str,
    metadata: dict,
    safety_margin_pct: float,
) -> bool:
    combined_output = _result_output(result)
    kv_bytes = parse_kv_cache_from_logs(combined_output)

    if kv_bytes is not None:
        free_mem = _get_free_gpu_memory()
        entry = CacheEntry(
            kv_cache_memory_bytes=kv_bytes,
            cache_key=cache_key,
            safety_margin_pct=safety_margin_pct,
            free_memory_at_cache_time=free_mem,
            **{k: v for k, v in metadata.items()
               if k not in ("quantization", "kv_cache_dtype", "enforce_eager")},
        )
        cache.put(entry)
        logger.info("Saved profile: kv_cache_memory_bytes=%d", kv_bytes)
        return True
    else:
        logger.warning(
            "Could not parse kv_cache_memory_bytes from vLLM output. "
            "The profile was not cached. This can happen if vLLM version "
            "changed the log format or if --kv-cache-memory-bytes was already set."
        )
        return False


def _result_output(result: subprocess.CompletedProcess) -> str:
    return (result.stdout or "") + (result.stderr or "")


def _looks_like_oom(result: subprocess.CompletedProcess) -> bool:
    output = _result_output(result).lower()
    return any(pattern in output for pattern in OOM_PATTERNS)


def _parse_vllm_args(args: list[str]) -> dict:
    """Extract key vLLM arguments from the command line."""
    parsed = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--enforce-eager":
            parsed["enforce_eager"] = True
            i += 1
        elif arg.startswith("--enforce-eager="):
            value = arg.split("=", 1)[1].lower()
            parsed["enforce_eager"] = value not in ("0", "false", "no")
            i += 1
        elif arg.startswith("--"):
            consumed = _parse_value_arg(args, i, parsed)
            i += consumed
        elif i == 0 and "model" not in parsed:
            parsed["model"] = arg
            i += 1
        else:
            i += 1
    return parsed


_VALUE_ARGS = {
    "--model": ("model", str),
    "--dtype": ("dtype", str),
    "--tensor-parallel-size": ("tensor_parallel_size", int),
    "--pipeline-parallel-size": ("pipeline_parallel_size", int),
    "--gpu-memory-utilization": ("gpu_memory_utilization", float),
    "--max-model-len": ("max_model_len", int),
    "--max-num-batched-tokens": ("max_num_batched_tokens", int),
    "--max-num-seqs": ("max_num_seqs", int),
    "--quantization": ("quantization", str),
    "--kv-cache-dtype": ("kv_cache_dtype", str),
}


def _parse_value_arg(args: list[str], index: int, parsed: dict) -> int:
    arg = args[index]
    if "=" in arg:
        name, value = arg.split("=", 1)
        consumed = 1
    else:
        name = arg
        if index + 1 >= len(args) or args[index + 1].startswith("--"):
            return 1
        value = args[index + 1]
        consumed = 2

    spec = _VALUE_ARGS.get(name)
    if spec is None:
        return consumed

    key, cast = spec
    parsed[key] = cast(value)
    return consumed


def _run_vllm(
    args: list[str],
    capture_output: bool = False,
) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"] + args
    logger.debug("Running: %s", " ".join(cmd))

    if capture_output:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=600
        )
    else:
        return subprocess.run(cmd, timeout=600)
