import json
import sys
import time
import types
from types import SimpleNamespace

import vllm_profile_cache.cache as cache_module
from vllm_profile_cache.cache import (
    CacheEntry,
    ProfileCache,
    _is_moe_model,
    apply_safety_margin,
    build_cache_key_from_vllm_config,
    check_free_memory,
    compute_cache_key,
)


def make_entry(**overrides):
    data = {
        "kv_cache_memory_bytes": 1_000_000,
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "gpu_name": "NVIDIA A100-SXM4-40GB",
        "gpu_uuid": "GPU-test",
        "dtype": "auto",
        "tp_size": 1,
        "pp_size": 1,
        "max_model_len": 32768,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 256,
        "vllm_version": "0.20.1",
        "cuda_version": "12.8",
        "torch_version": "2.11.0",
        "driver_version": "570.86.15",
        "gpu_total_memory_bytes": 40 * 1024**3,
        "gpu_memory_utilization": 0.9,
        "safety_margin_pct": 5.0,
        "free_memory_at_cache_time": 30 * 1024**3,
        "created_at": time.time(),
        "cache_key": "test-key",
    }
    data.update(overrides)
    return CacheEntry(**data)


def cache_key_args(**overrides):
    data = {
        "model_id": "model-a",
        "gpu_name": "gpu-a",
        "gpu_uuid": "uuid-a",
        "dtype": "auto",
        "tp_size": 1,
        "pp_size": 1,
        "max_model_len": 1024,
        "max_num_batched_tokens": 2048,
        "max_num_seqs": 16,
        "vllm_version": "0.20.1",
        "cuda_version": "12.8",
        "torch_version": "2.11.0",
        "driver_version": "570.86.15",
        "gpu_total_memory_bytes": 40 * 1024**3,
        "gpu_memory_utilization": 0.9,
        "quantization": None,
        "kv_cache_dtype": None,
        "enforce_eager": False,
    }
    data.update(overrides)
    return data


def test_compute_cache_key_is_stable_and_sensitive():
    base = compute_cache_key(**cache_key_args())

    assert base == compute_cache_key(**cache_key_args())
    assert base != compute_cache_key(**cache_key_args(model_id="model-b"))
    assert base != compute_cache_key(**cache_key_args(gpu_uuid="uuid-b"))
    assert base != compute_cache_key(**cache_key_args(kv_cache_dtype="fp8"))
    assert base != compute_cache_key(**cache_key_args(enforce_eager=True))


def install_fake_runtime(monkeypatch, gpu_uuid):
    torch = types.ModuleType("torch")
    torch.__version__ = "2.11.0"
    torch.version = SimpleNamespace(cuda="12.8")
    torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda index: "NVIDIA A100-SXM4-40GB",
        get_device_properties=lambda index: SimpleNamespace(total_memory=40 * 1024**3),
    )
    vllm = types.ModuleType("vllm")
    vllm.__version__ = "0.20.1"

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setattr(cache_module, "_get_gpu_uuid", lambda: gpu_uuid)
    monkeypatch.setattr(cache_module, "_get_driver_version", lambda: "570.86.15")


def test_build_cache_key_defaults_to_gpu_type_scope(monkeypatch):
    install_fake_runtime(monkeypatch, "GPU-a")
    key_a, meta_a = build_cache_key_from_vllm_config(model="demo")
    install_fake_runtime(monkeypatch, "GPU-b")
    key_b, meta_b = build_cache_key_from_vllm_config(model="demo")

    assert key_a == key_b
    assert meta_a["gpu_uuid"] == "GPU-a"
    assert meta_b["gpu_uuid"] == "GPU-b"


def test_build_cache_key_device_scope_includes_gpu_uuid(monkeypatch):
    install_fake_runtime(monkeypatch, "GPU-a")
    key_a, _ = build_cache_key_from_vllm_config(
        model="demo",
        gpu_identity_scope="device",
    )
    install_fake_runtime(monkeypatch, "GPU-b")
    key_b, _ = build_cache_key_from_vllm_config(
        model="demo",
        gpu_identity_scope="device",
    )

    assert key_a != key_b


def test_profile_cache_roundtrip(tmp_path):
    cache = ProfileCache(str(tmp_path))
    entry = make_entry(cache_key="abc123")

    cache.put(entry)
    loaded = cache.get("abc123")

    assert loaded == entry


def test_profile_cache_expires_and_deletes_entry(tmp_path):
    cache = ProfileCache(str(tmp_path))
    entry = make_entry(cache_key="expired", created_at=time.time() - 100)
    cache.put(entry)

    assert cache.get("expired", ttl_seconds=1) is None
    assert not (tmp_path / "expired.json").exists()


def test_profile_cache_drops_corrupt_entry(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not valid json")
    cache = ProfileCache(str(tmp_path))

    assert cache.get("broken") is None
    assert not path.exists()


def test_list_entries_skips_invalid_json(tmp_path):
    cache = ProfileCache(str(tmp_path))
    cache.put(make_entry(cache_key="good"))
    (tmp_path / "bad.json").write_text(json.dumps({"missing": "fields"}))

    assert [entry.cache_key for entry in cache.list_entries()] == ["good"]


def test_check_free_memory_rejects_large_drop(monkeypatch):
    entry = make_entry(free_memory_at_cache_time=10 * 1024**3)
    monkeypatch.setattr(cache_module, "_get_free_gpu_memory", lambda: 9 * 1024**3)

    assert not check_free_memory(entry, margin_bytes=512 * 1024**2)


def test_check_free_memory_allows_unknown_current_memory(monkeypatch):
    entry = make_entry(free_memory_at_cache_time=10 * 1024**3)
    monkeypatch.setattr(cache_module, "_get_free_gpu_memory", lambda: 0)

    assert check_free_memory(entry)


def test_apply_safety_margin():
    assert apply_safety_margin(1_000, margin_pct=5.0) == 950


def test_moe_model_detection():
    assert _is_moe_model("mistralai/Mixtral-8x7B-Instruct")
    assert _is_moe_model("Qwen/Qwen3-MoE-A2B")
    assert not _is_moe_model("Qwen/Qwen2.5-0.5B-Instruct")
