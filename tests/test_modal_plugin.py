import sys
import types
from types import SimpleNamespace

import vllm_profile_cache.modal_plugin as modal_plugin


def install_fake_transformers(monkeypatch, config):
    module = types.ModuleType("transformers")

    class AutoConfig:
        @staticmethod
        def from_pretrained(model_id, trust_remote_code=True):
            return config

    module.AutoConfig = AutoConfig
    monkeypatch.setitem(sys.modules, "transformers", module)


def fake_llm(num_blocks=10, block_size=16, tp_size=2, cache_dtype="fp8"):
    cache_config = SimpleNamespace(
        num_gpu_blocks=num_blocks,
        block_size=block_size,
        cache_dtype=cache_dtype,
    )
    vllm_config = SimpleNamespace(
        cache_config=cache_config,
        model_config=SimpleNamespace(model="demo-model", dtype=None),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp_size),
    )
    return SimpleNamespace(llm_engine=SimpleNamespace(vllm_config=vllm_config))


def test_extract_kv_cache_bytes_from_engine_uses_model_geometry(monkeypatch):
    install_fake_transformers(
        monkeypatch,
        SimpleNamespace(
            num_hidden_layers=4,
            num_key_value_heads=8,
            num_attention_heads=16,
            head_dim=64,
        ),
    )

    result = modal_plugin._extract_kv_cache_bytes_from_engine(fake_llm())

    page_size_per_layer = 2 * 16 * 4 * 64 * 1
    assert result == 10 * page_size_per_layer * 4


def test_extract_kv_cache_bytes_returns_none_without_blocks(monkeypatch):
    install_fake_transformers(monkeypatch, SimpleNamespace())

    assert modal_plugin._extract_kv_cache_bytes_from_engine(fake_llm(num_blocks=0)) is None


def test_patch_cache_config_hash_excludes_kv_cache_memory_bytes(monkeypatch):
    vllm = types.ModuleType("vllm")
    config_pkg = types.ModuleType("vllm.config")
    cache_module = types.ModuleType("vllm.config.cache")
    utils_module = types.ModuleType("vllm.config.utils")

    class CacheConfig:
        def __init__(self):
            self.kv_cache_memory_bytes = 123
            self.keep = "value"

        def compute_hash(self):
            return "original"

    def get_hash_factors(obj, ignored):
        return sorted(key for key in obj.__dict__ if key not in ignored)

    def hash_factors(factors):
        return "|".join(factors)

    cache_module.CacheConfig = CacheConfig
    utils_module.get_hash_factors = get_hash_factors
    utils_module.hash_factors = hash_factors

    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.config", config_pkg)
    monkeypatch.setitem(sys.modules, "vllm.config.cache", cache_module)
    monkeypatch.setitem(sys.modules, "vllm.config.utils", utils_module)

    modal_plugin._patch_cache_config_hash()

    assert CacheConfig().compute_hash() == "keep"
    assert CacheConfig._profile_cache_patched is True
