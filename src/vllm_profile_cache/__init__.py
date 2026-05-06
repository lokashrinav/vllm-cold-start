"""vllm-profile-cache: Skip vLLM's memory profiling and CUDA graph capture on warm starts.

Two caching layers:
1. Profile cache — caches kv_cache_memory_bytes to skip ~14s profiling forward pass
2. Foundry graph cache — serializes CUDA graphs to skip 3-60s capture phase
"""

__version__ = "0.2.0"

from vllm_profile_cache.cache import ProfileCache, CacheEntry
from vllm_profile_cache.wrapper import launch_vllm

__all__ = ["ProfileCache", "CacheEntry", "launch_vllm"]
