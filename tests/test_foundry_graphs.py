from types import SimpleNamespace

from vllm_profile_cache.foundry_graphs import (
    FoundryGraphCache,
    _FoundryPatchState,
    build_graph_cache_key,
    is_foundry_available,
    resolve_graph_cache_dir,
)


def test_foundry_availability_is_false_without_preload(monkeypatch):
    monkeypatch.delenv("LD_PRELOAD", raising=False)

    assert not is_foundry_available()


def test_foundry_graph_cache_detects_graph_files(tmp_path):
    cache = FoundryGraphCache(str(tmp_path))

    assert not cache.has_cached_graphs()

    cache.graphs_dir.mkdir(parents=True)
    (cache.graphs_dir / "graph_0.json").write_text("{}")

    assert cache.has_cached_graphs()


def test_graph_cache_key_is_stable_and_config_sensitive():
    base = build_graph_cache_key(
        "Qwen/Qwen2.5-0.5B-Instruct",
        gpu_memory_utilization=0.8,
        max_model_len=32768,
    )

    assert base == build_graph_cache_key(
        "Qwen/Qwen2.5-0.5B-Instruct",
        gpu_memory_utilization=0.8,
        max_model_len=32768,
    )
    assert base != build_graph_cache_key(
        "Qwen/Qwen2.5-1.5B-Instruct",
        gpu_memory_utilization=0.8,
        max_model_len=32768,
    )
    assert base != build_graph_cache_key(
        "Qwen/Qwen2.5-0.5B-Instruct",
        gpu_memory_utilization=0.7,
        max_model_len=32768,
    )


def test_resolve_graph_cache_dir_places_key_under_root(tmp_path):
    cache_dir, cache_key = resolve_graph_cache_dir(
        str(tmp_path),
        "Qwen/Qwen2.5-0.5B-Instruct",
        gpu_memory_utilization=0.8,
    )

    assert cache_dir == str(tmp_path / cache_key)


def test_foundry_graph_cache_metadata_roundtrip(tmp_path):
    cache = FoundryGraphCache(str(tmp_path))
    desc = SimpleNamespace(
        cg_mode=SimpleNamespace(name="FULL"),
        num_tokens=128,
        num_reqs=4,
        uniform_token_count=True,
    )

    cache.save_metadata([desc], model_id="demo-model", vllm_version="0.20.1")
    loaded = cache.load_metadata()

    assert loaded["model_id"] == "demo-model"
    assert loaded["num_graphs"] == 1
    assert loaded["vllm_version"] == "0.20.1"
    assert len(loaded["descs"]) == 1
    d = loaded["descs"][0]
    assert d["cg_mode"] == "FULL"
    assert d["num_tokens"] == 128
    assert d["num_reqs"] == 4
    assert d["uniform_token_count"] is True


def test_foundry_patch_state_no_cached_wrapper_graphs(tmp_path):
    cache = FoundryGraphCache(str(tmp_path))
    state = _FoundryPatchState(
        cache=cache,
        model_id="test-model",
        cache_key="abc123",
        region_size="32GB",
    )

    assert not state.has_cached_wrapper_graphs
    assert state.next_loaded_graph() is None
    assert state.load_index == 0
    assert not state.finalized
    assert not state.manager_handled


def test_foundry_patch_state_finalize_noop_without_saved(tmp_path):
    cache = FoundryGraphCache(str(tmp_path))
    state = _FoundryPatchState(
        cache=cache,
        model_id="test-model",
        cache_key="abc123",
        region_size="32GB",
    )

    state.finalize()
    assert state.finalized
    assert not cache.metadata_path.exists()

    state.finalize()
    assert state.finalized


def test_foundry_patch_state_manager_handled_skips_wrapper(tmp_path):
    cache = FoundryGraphCache(str(tmp_path))
    state = _FoundryPatchState(
        cache=cache,
        model_id="test-model",
        cache_key="abc123",
        region_size="32GB",
    )
    state.manager_handled = True

    assert state.next_loaded_graph() is None
    assert not state.has_cached_wrapper_graphs


def test_foundry_patch_state_wrapper_graphs_dir_is_separate(tmp_path):
    cache = FoundryGraphCache(str(tmp_path))
    state = _FoundryPatchState(
        cache=cache,
        model_id="test-model",
        cache_key="abc123",
        region_size="32GB",
    )

    assert state.wrapper_graphs_dir == tmp_path / "wrapper_graphs"
    assert state.wrapper_graphs_dir != cache.graphs_dir


def test_foundry_patch_state_detects_cached_wrapper_graphs(tmp_path):
    cache = FoundryGraphCache(str(tmp_path))
    state = _FoundryPatchState(
        cache=cache,
        model_id="test-model",
        cache_key="abc123",
        region_size="32GB",
    )

    assert not state.has_cached_wrapper_graphs

    state.wrapper_graphs_dir.mkdir(parents=True)
    (state.wrapper_graphs_dir / "graph_0.json").write_text("{}")

    assert state.has_cached_wrapper_graphs


def test_foundry_graph_cache_ignores_corrupt_metadata(tmp_path):
    cache = FoundryGraphCache(str(tmp_path))
    cache.cache_dir.mkdir(parents=True, exist_ok=True)
    cache.metadata_path.write_text("{not valid json")

    assert cache.load_metadata() is None
