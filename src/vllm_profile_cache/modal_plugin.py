"""Modal integration for vllm-profile-cache.

Provides a drop-in replacement for vLLM on Modal that automatically
caches memory profiling results in a Modal Volume, skipping the ~14s
profiling forward pass on subsequent cold starts.

Usage:
    from vllm_profile_cache.modal_plugin import cached_vllm_init

    llm = cached_vllm_init(
        model="meta-llama/Llama-3.1-8B-Instruct",
        gpu_memory_utilization=0.9,
    )
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

PROFILE_CACHE_MOUNT = "/root/.cache/vllm-profile-cache"


def _patch_cache_config_hash():
    """Exclude kv_cache_memory_bytes from CacheConfig's compile cache hash.

    kv_cache_memory_bytes controls memory allocation, not the computation
    graph structure. Including it in the hash forces a full torch.compile
    recompilation when the value changes, which defeats the purpose of
    caching profiling results.
    """
    from vllm.config.cache import CacheConfig
    if getattr(CacheConfig, "_profile_cache_patched", False):
        return
    _original = CacheConfig.compute_hash

    def _patched(self):
        from vllm.config.utils import get_hash_factors, hash_factors
        ignored = {
            "is_attention_free", "num_gpu_blocks_override",
            "enable_prefix_caching", "hash_block_size",
            "mamba_page_size_padded", "user_specified_block_size",
            "user_specified_mamba_block_size", "_block_size_resolved",
            "num_gpu_blocks", "num_cpu_blocks", "kv_sharing_fast_prefill",
            "kv_cache_memory_bytes",
        }
        factors = get_hash_factors(self, ignored)
        return hash_factors(factors)

    CacheConfig.compute_hash = _patched
    CacheConfig._profile_cache_patched = True


def cached_vllm_init(
    model: str,
    profile_cache_dir: str = PROFILE_CACHE_MOUNT,
    safety_margin_pct: float = 5.0,
    gpu_identity_scope: Optional[str] = None,
    **vllm_kwargs,
):
    """Initialize vLLM LLM engine with profile caching.

    On first cold start: runs normal initialization (including profiling),
    extracts the kv_cache_memory_bytes, saves to profile_cache_dir.

    On subsequent cold starts: loads cached value, passes it as
    kv_cache_memory_bytes to skip profiling entirely.

    If the cached value causes an OOM, the cache entry is deleted
    and vLLM is restarted with normal profiling.

    gpu_identity_scope defaults to "gpu_type", so Modal cold starts can reuse a
    profile across physical GPUs with the same model name and total memory. Use
    "device" to include the physical GPU UUID in the key.
    """
    from vllm_profile_cache.cache import (
        CacheEntry,
        ProfileCache,
        _get_free_gpu_memory,
        _is_moe_model,
        apply_safety_margin,
        build_cache_key_from_vllm_config,
        check_free_memory,
    )

    _patch_cache_config_hash()

    cache = ProfileCache(profile_cache_dir)

    if _is_moe_model(model):
        logger.warning("MoE model detected, skipping profile cache.")
        from vllm import LLM
        return LLM(model=model, **vllm_kwargs)

    cache_key, metadata = build_cache_key_from_vllm_config(
        model=model,
        gpu_memory_utilization=vllm_kwargs.get("gpu_memory_utilization", 0.9),
        dtype=vllm_kwargs.get("dtype", "auto"),
        tensor_parallel_size=vllm_kwargs.get("tensor_parallel_size", 1),
        pipeline_parallel_size=vllm_kwargs.get("pipeline_parallel_size", 1),
        max_model_len=vllm_kwargs.get("max_model_len"),
        max_num_batched_tokens=vllm_kwargs.get("max_num_batched_tokens"),
        max_num_seqs=vllm_kwargs.get("max_num_seqs"),
        quantization=vllm_kwargs.get("quantization"),
        kv_cache_dtype=vllm_kwargs.get("kv_cache_dtype"),
        enforce_eager=vllm_kwargs.get("enforce_eager", False),
        gpu_identity_scope=gpu_identity_scope,
    )

    entry = cache.get(cache_key)
    using_cache = False

    if entry is not None:
        if check_free_memory(entry):
            cached_bytes = apply_safety_margin(
                entry.kv_cache_memory_bytes, safety_margin_pct
            )
            logger.info(
                "Profile cache hit! Injecting kv_cache_memory_bytes=%d "
                "(original=%d, margin=%.1f%%).",
                cached_bytes, entry.kv_cache_memory_bytes, safety_margin_pct,
            )
            vllm_kwargs["kv_cache_memory_bytes"] = cached_bytes
            using_cache = True
        else:
            logger.info("Free memory check failed, falling back to profiling.")
            entry = None

    t0 = time.perf_counter()
    from vllm import LLM
    try:
        llm = LLM(model=model, **vllm_kwargs)
    except (RuntimeError, torch_cuda_oom_error()) as e:
        if not using_cache:
            raise
        logger.warning(
            "OOM with cached kv_cache_memory_bytes=%d. "
            "Deleting cache entry and retrying with profiling.",
            vllm_kwargs["kv_cache_memory_bytes"],
        )
        cache.invalidate(cache_key)
        vllm_kwargs.pop("kv_cache_memory_bytes", None)
        llm = LLM(model=model, **vllm_kwargs)
        using_cache = False

    init_time = time.perf_counter() - t0
    logger.info("LLM initialization took %.2fs", init_time)

    if not using_cache and entry is None:
        kv_bytes = _extract_kv_cache_bytes_from_engine(llm, model=model)
        if kv_bytes is not None:
            free_mem = _get_free_gpu_memory()
            entry = CacheEntry(
                kv_cache_memory_bytes=kv_bytes,
                cache_key=cache_key,
                safety_margin_pct=safety_margin_pct,
                model_id=metadata["model_id"],
                gpu_name=metadata["gpu_name"],
                gpu_uuid=metadata["gpu_uuid"],
                dtype=metadata["dtype"],
                tp_size=metadata["tp_size"],
                pp_size=metadata["pp_size"],
                max_model_len=metadata["max_model_len"],
                max_num_batched_tokens=metadata["max_num_batched_tokens"],
                max_num_seqs=metadata["max_num_seqs"],
                vllm_version=metadata["vllm_version"],
                cuda_version=metadata["cuda_version"],
                torch_version=metadata["torch_version"],
                driver_version=metadata["driver_version"],
                gpu_total_memory_bytes=metadata["gpu_total_memory_bytes"],
                gpu_memory_utilization=metadata["gpu_memory_utilization"],
                free_memory_at_cache_time=free_mem,
            )
            cache.put(entry)

    return llm


def torch_cuda_oom_error():
    """Return the CUDA OOM exception class."""
    try:
        import torch
        return torch.cuda.OutOfMemoryError
    except (ImportError, AttributeError):
        return type(None)


def _extract_kv_cache_bytes_from_engine(llm, model: Optional[str] = None) -> Optional[int]:
    """Compute kv_cache_memory_bytes from engine state after initialization.

    In vLLM V1, the engine core runs in a separate process, so the raw
    available_kv_cache_memory_bytes is not directly accessible. Instead,
    we reconstruct it from num_gpu_blocks (sent back via IPC) and the
    model's KV cache geometry.

    Formula: kv_cache_memory = num_blocks * page_size_per_layer * num_layers
    Where page_size_per_layer = 2 * block_size * kv_heads_per_gpu * head_size * dtype_size
    """
    try:
        vllm_config = llm.llm_engine.vllm_config
        cache_config = vllm_config.cache_config
        num_blocks = cache_config.num_gpu_blocks
        block_size = cache_config.block_size

        if not num_blocks or num_blocks <= 0 or not block_size:
            logger.debug("num_gpu_blocks=%s, block_size=%s — cannot compute",
                         num_blocks, block_size)
            return None

        model_id = model or vllm_config.model_config.model
        tp_size = vllm_config.parallel_config.tensor_parallel_size

        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

        num_layers = config.num_hidden_layers
        num_kv_heads = getattr(config, "num_key_value_heads",
                               config.num_attention_heads)
        head_size = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        kv_heads_per_gpu = max(1, num_kv_heads // tp_size)

        kv_cache_dtype = getattr(cache_config, "cache_dtype", "auto")
        if kv_cache_dtype and "fp8" in str(kv_cache_dtype):
            dtype_size = 1
        else:
            import torch
            dtype_size = torch.tensor([], dtype=vllm_config.model_config.dtype).element_size()

        page_size_per_layer = 2 * block_size * kv_heads_per_gpu * head_size * dtype_size
        kv_cache_memory = num_blocks * page_size_per_layer * num_layers

        logger.info(
            "Computed kv_cache_memory_bytes=%d from num_blocks=%d, "
            "layers=%d, kv_heads/gpu=%d, head=%d, block=%d, dtype=%dB",
            kv_cache_memory, num_blocks, num_layers, kv_heads_per_gpu,
            head_size, block_size, dtype_size,
        )
        return kv_cache_memory
    except Exception as e:
        logger.warning("Could not compute kv_cache_memory_bytes: %s", e)
        return None
