import subprocess

import pytest

from vllm_profile_cache.cache import CacheEntry, ProfileCache
import vllm_profile_cache.wrapper as wrapper


def metadata():
    return {
        "model_id": "demo-model",
        "gpu_name": "gpu",
        "gpu_uuid": "uuid",
        "dtype": "auto",
        "tp_size": 1,
        "pp_size": 1,
        "max_model_len": 0,
        "max_num_batched_tokens": 0,
        "max_num_seqs": 0,
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


def entry(cache_key="cache-key", kv_bytes=1_000):
    data = metadata()
    for key in ("quantization", "kv_cache_dtype", "enforce_eager"):
        data.pop(key)
    return CacheEntry(
        kv_cache_memory_bytes=kv_bytes,
        cache_key=cache_key,
        safety_margin_pct=5.0,
        free_memory_at_cache_time=30 * 1024**3,
        **data,
    )


@pytest.mark.parametrize(
    ("log", "expected"),
    [
        ("Use --kv-cache-memory-bytes=12345 (0.01 GiB) to fit into requested memory", 12345),
        ("Use --kv-cache-memory-bytes `12345` to skip profiling", 12345),
        ("Use --kv-cache-memory=12345 (0.01 GiB) to fit into requested memory", 12345),
        ("kv_cache_memory_bytes: 12345", 12345),
        ("no recommendation here", None),
    ],
)
def test_parse_kv_cache_from_logs(log, expected):
    assert wrapper.parse_kv_cache_from_logs(log) == expected


def test_parse_vllm_args_supports_space_and_equals_forms():
    parsed = wrapper._parse_vllm_args([
        "--model=demo",
        "--dtype", "float16",
        "--tensor-parallel-size=2",
        "--pipeline-parallel-size", "1",
        "--gpu-memory-utilization=0.85",
        "--max-model-len", "4096",
        "--max-num-batched-tokens=8192",
        "--max-num-seqs", "128",
        "--quantization", "awq",
        "--kv-cache-dtype=fp8",
        "--enforce-eager=false",
    ])

    assert parsed == {
        "model": "demo",
        "dtype": "float16",
        "tensor_parallel_size": 2,
        "pipeline_parallel_size": 1,
        "gpu_memory_utilization": 0.85,
        "max_model_len": 4096,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 128,
        "quantization": "awq",
        "kv_cache_dtype": "fp8",
        "enforce_eager": False,
    }


def test_parse_vllm_args_supports_initial_positional_model():
    assert wrapper._parse_vllm_args(["demo-model", "--dtype", "auto"]) == {
        "model": "demo-model",
        "dtype": "auto",
    }


def test_launch_vllm_cache_hit_injects_margin(tmp_path, monkeypatch):
    cache = ProfileCache(str(tmp_path))
    cache.put(entry(kv_bytes=1_000))
    calls = []

    monkeypatch.setattr(
        wrapper,
        "build_cache_key_from_vllm_config",
        lambda **kwargs: ("cache-key", metadata()),
    )
    monkeypatch.setattr(wrapper, "check_free_memory", lambda cache_entry: True)

    def fake_run(args, capture_output=False):
        calls.append((args, capture_output))
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(wrapper, "_run_vllm", fake_run)

    result = wrapper.launch_vllm(
        ["--model", "demo-model"],
        cache_dir=str(tmp_path),
        safety_margin_pct=10.0,
    )

    assert result.returncode == 0
    assert calls == [(["--model", "demo-model", "--kv-cache-memory-bytes=900"], True)]


def test_launch_vllm_free_memory_failure_reprofiles_and_saves(tmp_path, monkeypatch):
    cache = ProfileCache(str(tmp_path))
    cache.put(entry(kv_bytes=1_000))
    calls = []

    monkeypatch.setattr(
        wrapper,
        "build_cache_key_from_vllm_config",
        lambda **kwargs: ("cache-key", metadata()),
    )
    monkeypatch.setattr(wrapper, "check_free_memory", lambda cache_entry: False)
    monkeypatch.setattr(wrapper, "_get_free_gpu_memory", lambda: 123)

    def fake_run(args, capture_output=False):
        calls.append((args, capture_output))
        return subprocess.CompletedProcess(
            args,
            0,
            stdout="Recommended --kv-cache-memory-bytes=2000 (0.01 GiB)",
            stderr="",
        )

    monkeypatch.setattr(wrapper, "_run_vllm", fake_run)

    wrapper.launch_vllm(["--model", "demo-model"], cache_dir=str(tmp_path))

    assert calls == [(["--model", "demo-model"], True)]
    assert ProfileCache(str(tmp_path)).get("cache-key").kv_cache_memory_bytes == 2_000


def test_launch_vllm_cached_oom_invalidates_and_retries(tmp_path, monkeypatch):
    cache = ProfileCache(str(tmp_path))
    cache.put(entry(kv_bytes=1_000))
    calls = []

    monkeypatch.setattr(
        wrapper,
        "build_cache_key_from_vllm_config",
        lambda **kwargs: ("cache-key", metadata()),
    )
    monkeypatch.setattr(wrapper, "check_free_memory", lambda cache_entry: True)
    monkeypatch.setattr(wrapper, "_get_free_gpu_memory", lambda: 456)

    def fake_run(args, capture_output=False):
        calls.append((args, capture_output))
        if "--kv-cache-memory-bytes=950" in args:
            return subprocess.CompletedProcess(
                args,
                1,
                stdout="",
                stderr="torch.cuda.OutOfMemoryError: CUDA out of memory",
            )
        return subprocess.CompletedProcess(
            args,
            0,
            stdout="Recommended --kv-cache-memory-bytes=2200 (0.01 GiB)",
            stderr="",
        )

    monkeypatch.setattr(wrapper, "_run_vllm", fake_run)

    result = wrapper.launch_vllm(["--model", "demo-model"], cache_dir=str(tmp_path))

    assert result.returncode == 0
    assert calls == [
        (["--model", "demo-model", "--kv-cache-memory-bytes=950"], True),
        (["--model", "demo-model"], True),
    ]
    assert ProfileCache(str(tmp_path)).get("cache-key").kv_cache_memory_bytes == 2_200
