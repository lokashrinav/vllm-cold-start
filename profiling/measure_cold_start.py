"""Measure vLLM cold start phases with fine-grained timing.

Run on a GPU machine (Modal, Vast.ai, RunPod) to reproduce the ~17s
import/plugin/subprocess overhead from vLLM issue #21051.

Usage:
    python measure_cold_start.py --model Qwen/Qwen1.5-4B
    python measure_cold_start.py --model meta-llama/Llama-3.1-8B-Instruct
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class PhaseResult:
    name: str
    duration_s: float
    detail: str = ""


@dataclass
class ColdStartProfile:
    model: str
    phases: list[PhaseResult] = field(default_factory=list)
    total_s: float = 0.0
    gpu_name: str = ""
    vllm_version: str = ""
    python_version: str = ""

    def add(self, name: str, duration: float, detail: str = ""):
        self.phases.append(PhaseResult(name, duration, detail))

    def summary(self) -> str:
        lines = [
            f"\n{'='*70}",
            f"vLLM Cold Start Profile: {self.model}",
            f"GPU: {self.gpu_name}  |  vLLM: {self.vllm_version}  |  Python: {self.python_version}",
            f"{'='*70}",
        ]
        for p in self.phases:
            bar = "#" * int(p.duration_s * 4)
            detail = f"  ({p.detail})" if p.detail else ""
            lines.append(f"  {p.name:<45} {p.duration_s:>7.3f}s  {bar}{detail}")
        lines.append(f"{'─'*70}")
        lines.append(f"  {'TOTAL':<45} {self.total_s:>7.3f}s")
        lines.append(f"{'='*70}\n")
        return "\n".join(lines)


def time_import(module_name: str) -> float:
    if module_name in sys.modules:
        return 0.0
    t0 = time.perf_counter()
    importlib.import_module(module_name)
    return time.perf_counter() - t0


def measure_phase(name: str):
    """Context manager that yields a PhaseResult after timing."""
    class _Timer:
        def __init__(self):
            self.start = 0.0
            self.duration = 0.0
        def __enter__(self):
            self.start = time.perf_counter()
            return self
        def __exit__(self, *_):
            self.duration = time.perf_counter() - self.start
    return _Timer()


def profile_cold_start(model: str) -> ColdStartProfile:
    profile = ColdStartProfile(model=model)
    profile.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    t_total_start = time.perf_counter()

    # Phase 1: torch import
    with measure_phase("torch import") as t:
        import torch
    profile.add("1. import torch", t.duration)

    # Phase 2: transformers import
    with measure_phase("transformers import") as t:
        import transformers
    profile.add("2. import transformers", t.duration)

    # Phase 3: vllm import
    with measure_phase("vllm import") as t:
        import vllm
    profile.add("3. import vllm", t.duration, f"v{vllm.__version__}")
    profile.vllm_version = vllm.__version__

    # GPU info
    if torch.cuda.is_available():
        profile.gpu_name = torch.cuda.get_device_name(0)
    else:
        profile.gpu_name = "no-gpu"

    # Phase 4: Platform detection
    with measure_phase("platform detection") as t:
        from vllm.platforms import current_platform
        _ = current_platform.device_type
    profile.add("4. platform detection", t.duration, f"detected: {current_platform.device_type}")

    # Phase 5: Plugin loading
    with measure_phase("plugin loading") as t:
        from vllm.plugins import load_general_plugins
        load_general_plugins()
    profile.add("5. load_general_plugins()", t.duration)

    # Phase 6: EngineArgs creation
    with measure_phase("engine args") as t:
        from vllm import EngineArgs
        engine_args = EngineArgs(model=model, enforce_eager=True)
    profile.add("6. EngineArgs creation", t.duration)

    # Phase 7: Engine config creation (triggers inspect_model_cls subprocess)
    with measure_phase("engine config") as t:
        engine_config = engine_args.create_engine_config()
    profile.add("7. create_engine_config (subprocess)", t.duration,
                "includes inspect_model_cls subprocess")

    # Phase 8: Full LLM initialization (includes subprocess spawn, weight load, etc.)
    with measure_phase("LLM init") as t:
        from vllm import LLM
        llm = LLM(model=model, enforce_eager=True, gpu_memory_utilization=0.8)
    profile.add("8. LLM(...) full init", t.duration,
                "subprocess spawn + weight load + warmup")

    # Phase 9: First inference (may trigger PTX/FA compilation)
    with measure_phase("first inference") as t:
        from vllm import SamplingParams
        output = llm.generate(["Hello, world!"], SamplingParams(max_tokens=16))
    profile.add("9. first inference", t.duration,
                "may include PTX/FA JIT")

    profile.total_s = time.perf_counter() - t_total_start
    return profile


def profile_import_only() -> ColdStartProfile:
    """Lighter profile that only measures imports + plugin loading.

    Useful when you don't have a GPU or want to isolate the Python overhead.
    """
    profile = ColdStartProfile(model="(import-only)")
    profile.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    t_total_start = time.perf_counter()

    with measure_phase("torch") as t:
        import torch
    profile.add("1. import torch", t.duration)

    with measure_phase("transformers") as t:
        import transformers
    profile.add("2. import transformers", t.duration)

    with measure_phase("vllm") as t:
        import vllm
    profile.add("3. import vllm", t.duration, f"v{vllm.__version__}")
    profile.vllm_version = vllm.__version__

    with measure_phase("platform") as t:
        from vllm.platforms import current_platform
        _ = current_platform.device_type
    profile.add("4. platform detection", t.duration)

    with measure_phase("plugins") as t:
        from vllm.plugins import load_general_plugins
        load_general_plugins()
    profile.add("5. load_general_plugins()", t.duration)

    # Measure what a subprocess reimport costs
    with measure_phase("subprocess reimport") as t:
        result = subprocess.run(
            [sys.executable, "-c",
             "import time; t0=time.perf_counter(); "
             "import vllm; "
             "from vllm.plugins import load_general_plugins; "
             "load_general_plugins(); "
             "print(f'{time.perf_counter()-t0:.3f}')"],
            capture_output=True, text=True, timeout=120
        )
    subprocess_time = result.stdout.strip() if result.returncode == 0 else "failed"
    profile.add("6. subprocess reimport cost", t.duration,
                f"child self-reported: {subprocess_time}s")

    profile.total_s = time.perf_counter() - t_total_start

    if torch.cuda.is_available():
        profile.gpu_name = torch.cuda.get_device_name(0)
    else:
        profile.gpu_name = "no-gpu"

    return profile


def main():
    parser = argparse.ArgumentParser(description="Profile vLLM cold start")
    parser.add_argument("--model", default=None,
                        help="Model to load (e.g. Qwen/Qwen1.5-4B). "
                             "If omitted, runs import-only profiling.")
    parser.add_argument("--output", default=None,
                        help="Write JSON results to this file")
    args = parser.parse_args()

    if args.model:
        profile = profile_cold_start(args.model)
    else:
        profile = profile_import_only()

    print(profile.summary())

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(asdict(profile), indent=2))
        print(f"Results written to {out}")


if __name__ == "__main__":
    main()
