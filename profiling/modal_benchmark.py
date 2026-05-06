"""Run vLLM cold start profiling on Modal's free tier.

Deploys the measurement harness to a GPU container, waits for scale-to-zero
between runs, and collects timing data for real cold starts.

Usage:
    modal run profiling/modal_benchmark.py
    modal run profiling/modal_benchmark.py --model meta-llama/Llama-3.1-8B-Instruct
"""

import modal
import time

app = modal.App("vllm-cold-start-profiler")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm",
        "torch",
        "transformers",
    )
)


@app.function(
    gpu="A100",
    image=image,
    timeout=900,
    scaledown_window=5,
)
def profile_full(model: str = "Qwen/Qwen2.5-0.5B-Instruct", eager: bool = False):
    """Full cold start profile including model load and first inference.

    Set eager=True to skip torch.compile and CUDA graphs (faster but incomplete).
    Default is eager=False to measure the REAL cold start with all phases.
    """
    import sys
    import subprocess

    results = {}
    results["enforce_eager"] = eager
    t_total = time.perf_counter()

    # Phase 1: Python imports
    t0 = time.perf_counter()
    import torch
    results["import_torch"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    import transformers
    results["import_transformers"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    import vllm
    results["import_vllm"] = time.perf_counter() - t0

    results["vllm_version"] = vllm.__version__
    results["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"

    # Phase 2: Subprocess reimport cost (standalone measurement)
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-c",
         "import time; t0=time.perf_counter(); "
         "import vllm; "
         "from vllm.plugins import load_general_plugins; "
         "load_general_plugins(); "
         "print(f'{time.perf_counter()-t0:.3f}')"],
        capture_output=True, text=True, timeout=120
    )
    results["subprocess_reimport_wall"] = time.perf_counter() - t0
    results["subprocess_reimport_self"] = proc.stdout.strip() if proc.returncode == 0 else "failed"

    # Phase 3: Full LLM init (the real cold start — everything happens inside here)
    t0 = time.perf_counter()
    from vllm import LLM, SamplingParams
    llm = LLM(model=model, enforce_eager=eager, gpu_memory_utilization=0.8)
    results["llm_init"] = time.perf_counter() - t0

    # Phase 4: First inference (may trigger PTX/FA JIT on non-SM80/SM90)
    t0 = time.perf_counter()
    output = llm.generate(["Hello, world!"], SamplingParams(max_tokens=16))
    results["first_inference"] = time.perf_counter() - t0
    results["first_output"] = output[0].outputs[0].text if output else ""

    results["total"] = time.perf_counter() - t_total
    results["model"] = model

    print(f"\n{'='*70}")
    print(f"vLLM Cold Start Profile: {model}")
    print(f"GPU: {results['gpu']}  |  vLLM: {results['vllm_version']}")
    print(f"enforce_eager: {eager}")
    print(f"{'='*70}")
    for key in ["import_torch", "import_transformers", "import_vllm",
                "subprocess_reimport_wall",
                "llm_init", "first_inference"]:
        val = results.get(key, 0)
        if isinstance(val, (int, float)):
            bar = "#" * int(val * 2)
            print(f"  {key:<40} {val:>7.3f}s  {bar}")
    print(f"{'─'*70}")
    print(f"  {'TOTAL':<40} {results['total']:>7.3f}s")
    print(f"{'='*70}\n")

    return results


@app.function(
    gpu="A100",
    image=image,
    timeout=300,
    scaledown_window=5,
)
def profile_imports_only():
    """Lighter profile — just imports and plugin loading, no model."""
    import sys
    import subprocess

    results = {}
    t_total = time.perf_counter()

    t0 = time.perf_counter()
    import torch
    results["import_torch"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    import transformers
    results["import_transformers"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    import vllm
    results["import_vllm"] = time.perf_counter() - t0
    results["vllm_version"] = vllm.__version__

    t0 = time.perf_counter()
    from vllm.platforms import current_platform
    _ = current_platform.device_type
    results["platform_detection"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    from vllm.plugins import load_general_plugins
    load_general_plugins()
    results["plugin_loading"] = time.perf_counter() - t0

    # Subprocess reimport cost
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-c",
         "import time; t0=time.perf_counter(); "
         "import vllm; "
         "from vllm.plugins import load_general_plugins; "
         "load_general_plugins(); "
         "print(f'{time.perf_counter()-t0:.3f}')"],
        capture_output=True, text=True, timeout=120
    )
    results["subprocess_reimport_wall"] = time.perf_counter() - t0
    results["subprocess_reimport_self"] = proc.stdout.strip() if proc.returncode == 0 else "failed"

    results["total"] = time.perf_counter() - t_total
    results["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"

    print(f"\n{'='*70}")
    print(f"Import-Only Cold Start Profile")
    print(f"GPU: {results['gpu']}  |  vLLM: {results['vllm_version']}")
    print(f"{'='*70}")
    for key in ["import_torch", "import_transformers", "import_vllm",
                "platform_detection", "plugin_loading",
                "subprocess_reimport_wall"]:
        val = results.get(key, 0)
        if isinstance(val, (int, float)):
            bar = "#" * int(val * 4)
            print(f"  {key:<40} {val:>7.3f}s  {bar}")
    print(f"{'─'*70}")
    print(f"  {'TOTAL':<40} {results['total']:>7.3f}s")
    print(f"  subprocess self-reported: {results['subprocess_reimport_self']}s")
    print(f"{'='*70}\n")

    return results


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-0.5B-Instruct", imports_only: bool = False, eager: bool = False):
    if imports_only:
        result = profile_imports_only.remote()
    else:
        result = profile_full.remote(model, eager)

    import json
    from pathlib import Path

    out_dir = Path("analysis/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"profile_{int(time.time())}.json"
    out_file.write_text(json.dumps(result, indent=2))
    print(f"\nResults saved to {out_file}")
