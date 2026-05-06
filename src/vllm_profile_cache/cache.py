"""Persistent cache for vLLM memory profiling results.

Stores kv_cache_memory_bytes keyed by a deterministic hash of
every parameter that affects memory profiling, so that
determine_available_memory() can be bypassed on subsequent cold starts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/vllm-profile-cache")
DEFAULT_TTL_SECONDS = 7 * 24 * 3600  # 7 days
DEFAULT_GPU_IDENTITY_SCOPE = "gpu_type"
GPU_IDENTITY_SCOPES = frozenset(["gpu_type", "device"])

MOE_MODEL_DENYLIST = frozenset([
    "deepseek-v2",
    "deepseek-v3",
    "deepseek-r1",
    "mixtral",
    "qwen-moe",
    "qwen2-moe",
    "qwen3-moe",
    "dbrx",
    "jamba",
    "arctic",
    "grok-1",
])


@dataclass
class CacheEntry:
    kv_cache_memory_bytes: int
    model_id: str
    gpu_name: str
    gpu_uuid: str
    dtype: str
    tp_size: int
    pp_size: int
    max_model_len: int
    max_num_batched_tokens: int
    max_num_seqs: int
    vllm_version: str
    cuda_version: str
    torch_version: str
    driver_version: str
    gpu_total_memory_bytes: int
    gpu_memory_utilization: float
    safety_margin_pct: float
    free_memory_at_cache_time: int
    created_at: float = field(default_factory=time.time)
    cache_key: str = ""


def compute_cache_key(
    model_id: str,
    gpu_name: str,
    gpu_uuid: str,
    dtype: str,
    tp_size: int,
    pp_size: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    vllm_version: str,
    cuda_version: str,
    torch_version: str,
    driver_version: str,
    gpu_total_memory_bytes: int,
    gpu_memory_utilization: float,
    quantization: Optional[str] = None,
    kv_cache_dtype: Optional[str] = None,
    enforce_eager: bool = False,
) -> str:
    """Deterministic hash of all parameters that affect memory profiling."""
    parts = [
        f"model={model_id}",
        f"gpu={gpu_name}",
        f"gpu_uuid={gpu_uuid}",
        f"dtype={dtype}",
        f"tp={tp_size}",
        f"pp={pp_size}",
        f"max_model_len={max_model_len}",
        f"max_batched={max_num_batched_tokens}",
        f"max_seqs={max_num_seqs}",
        f"vllm={vllm_version}",
        f"cuda={cuda_version}",
        f"torch={torch_version}",
        f"driver={driver_version}",
        f"gpu_mem={gpu_total_memory_bytes}",
        f"util={gpu_memory_utilization}",
        f"quant={quantization or 'none'}",
        f"kv_dtype={kv_cache_dtype or 'auto'}",
        f"eager={enforce_eager}",
    ]
    key_str = "|".join(parts)
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


def _is_moe_model(model_id: str) -> bool:
    model_lower = model_id.lower().replace("/", "-").replace("_", "-")
    return any(moe in model_lower for moe in MOE_MODEL_DENYLIST)


class ProfileCache:
    """Read/write memory profiling cache entries from disk."""

    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _entry_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"

    def get(
        self,
        cache_key: str,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
    ) -> Optional[CacheEntry]:
        path = self._entry_path(cache_key)
        if not path.exists():
            logger.info("Profile cache miss: %s", cache_key)
            return None
        try:
            data = json.loads(path.read_text())
            entry = CacheEntry(**data)

            age = time.time() - entry.created_at
            if age > ttl_seconds:
                logger.info(
                    "Profile cache expired: %s (age=%.0fs, ttl=%.0fs)",
                    cache_key, age, ttl_seconds,
                )
                path.unlink(missing_ok=True)
                return None

            logger.info(
                "Profile cache hit: %s (kv_cache_memory=%d, age=%.0fh)",
                cache_key, entry.kv_cache_memory_bytes, age / 3600,
            )
            return entry
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.warning("Corrupt cache entry %s: %s", cache_key, e)
            path.unlink(missing_ok=True)
            return None

    def put(self, entry: CacheEntry) -> None:
        path = self._entry_path(entry.cache_key)
        path.write_text(json.dumps(asdict(entry), indent=2))
        logger.info(
            "Profile cache saved: %s (kv_cache_memory=%d)",
            entry.cache_key, entry.kv_cache_memory_bytes
        )

    def invalidate(self, cache_key: str) -> bool:
        path = self._entry_path(cache_key)
        if path.exists():
            path.unlink()
            logger.info("Profile cache invalidated: %s", cache_key)
            return True
        return False

    def list_entries(self) -> list[CacheEntry]:
        entries = []
        for path in self.cache_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                entries.append(CacheEntry(**data))
            except Exception:
                continue
        return entries


def _get_gpu_uuid() -> str:
    """Get GPU UUID to detect MIG slices vs full GPUs."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=gpu_uuid", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _get_driver_version() -> str:
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _get_free_gpu_memory() -> int:
    """Get current free GPU memory in bytes."""
    try:
        import torch
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info(0)
            return free
    except Exception:
        pass
    return 0


def build_cache_key_from_vllm_config(
    model: str,
    gpu_memory_utilization: float = 0.9,
    dtype: str = "auto",
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    max_model_len: Optional[int] = None,
    max_num_batched_tokens: Optional[int] = None,
    max_num_seqs: Optional[int] = None,
    quantization: Optional[str] = None,
    kv_cache_dtype: Optional[str] = None,
    enforce_eager: bool = False,
    gpu_identity_scope: Optional[str] = None,
) -> tuple[str, dict]:
    """Build cache key from vLLM launch args + detected GPU/runtime info.

    Returns (cache_key, metadata_dict) where metadata_dict contains all
    the values used for key computation (useful for creating CacheEntry later).

    ``gpu_identity_scope`` controls whether the cache key is tied to a physical
    GPU UUID (``"device"``) or to interchangeable GPUs with the same model name
    and total memory (``"gpu_type"``, the default). The default works better on
    cloud pools such as Modal, where each cold start can land on a different
    physical GPU.
    """
    import torch
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"
    gpu_total = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
    cuda_version = torch.version.cuda or "unknown"
    torch_version = torch.__version__
    gpu_uuid = _get_gpu_uuid()
    gpu_uuid_for_key = _gpu_uuid_for_cache_key(gpu_uuid, gpu_identity_scope)
    driver_version = _get_driver_version()

    try:
        import vllm
        vllm_version = vllm.__version__
    except ImportError:
        vllm_version = "unknown"

    metadata = dict(
        model_id=model,
        gpu_name=gpu_name,
        gpu_uuid=gpu_uuid,
        dtype=dtype,
        tp_size=tensor_parallel_size,
        pp_size=pipeline_parallel_size,
        max_model_len=max_model_len or 0,
        max_num_batched_tokens=max_num_batched_tokens or 0,
        max_num_seqs=max_num_seqs or 0,
        vllm_version=vllm_version,
        cuda_version=cuda_version,
        torch_version=torch_version,
        driver_version=driver_version,
        gpu_total_memory_bytes=gpu_total,
        gpu_memory_utilization=gpu_memory_utilization,
        quantization=quantization,
        kv_cache_dtype=kv_cache_dtype,
        enforce_eager=enforce_eager,
    )

    key_metadata = dict(metadata)
    key_metadata["gpu_uuid"] = gpu_uuid_for_key
    key = compute_cache_key(**key_metadata)
    return key, metadata


def _resolve_gpu_identity_scope(gpu_identity_scope: Optional[str]) -> str:
    scope = gpu_identity_scope or os.environ.get(
        "VLLM_PROFILE_CACHE_GPU_SCOPE",
        DEFAULT_GPU_IDENTITY_SCOPE,
    )
    scope = scope.strip().lower().replace("-", "_")
    if scope not in GPU_IDENTITY_SCOPES:
        raise ValueError(
            "gpu_identity_scope must be 'gpu_type' or 'device' "
            f"(got {gpu_identity_scope!r})"
        )
    return scope


def _gpu_uuid_for_cache_key(gpu_uuid: str, gpu_identity_scope: Optional[str]) -> str:
    scope = _resolve_gpu_identity_scope(gpu_identity_scope)
    if scope == "device":
        return gpu_uuid
    return "gpu_type_scope"


def check_free_memory(
    entry: CacheEntry,
    margin_bytes: int = 512 * 1024 * 1024,
) -> bool:
    """Check if current free GPU memory is close enough to when the cache was created.

    If free memory dropped significantly (e.g. another process is using GPU),
    the cached value may be unsafe. Returns True if safe to use.
    """
    current_free = _get_free_gpu_memory()
    if current_free == 0:
        return True  # can't check, proceed with cache

    cached_free = entry.free_memory_at_cache_time
    if cached_free == 0:
        return True  # old entry without free memory info

    diff = cached_free - current_free
    if diff > margin_bytes:
        logger.warning(
            "Free GPU memory dropped by %d MB since cache was created "
            "(was %d MB, now %d MB). Skipping cache to avoid OOM.",
            diff // (1024 * 1024),
            cached_free // (1024 * 1024),
            current_free // (1024 * 1024),
        )
        return False

    return True


def apply_safety_margin(kv_cache_memory_bytes: int, margin_pct: float = 5.0) -> int:
    """Reduce cached value by a safety margin to prevent OOM."""
    factor = 1.0 - (margin_pct / 100.0)
    return int(kv_cache_memory_bytes * factor)
