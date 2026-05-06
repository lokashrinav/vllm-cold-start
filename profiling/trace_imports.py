"""Trace Python import times for vLLM's dependency tree.

Uses Python's built-in -X importtime flag to measure every import,
then parses the output to find the heaviest contributors.

Usage:
    python profiling/trace_imports.py
    python profiling/trace_imports.py --top 20
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class ImportEntry:
    module: str
    self_us: int
    cumulative_us: int
    depth: int


def run_import_trace(module: str = "vllm") -> list[ImportEntry]:
    """Run `python -X importtime -c 'import <module>'` and parse output."""
    result = subprocess.run(
        [sys.executable, "-X", "importtime", "-c", f"import {module}"],
        capture_output=True, text=True, timeout=120
    )
    entries = []
    for line in result.stderr.splitlines():
        match = re.match(
            r"^import time:\s+(\d+)\s+\|\s+(\d+)\s+\|\s+(\s*)(\S+)",
            line
        )
        if match:
            self_us = int(match.group(1))
            cumulative_us = int(match.group(2))
            indent = len(match.group(3))
            depth = indent // 2
            mod = match.group(4)
            entries.append(ImportEntry(mod, self_us, cumulative_us, depth))
    return entries


def measure_subprocess_reimport_cost() -> dict:
    """Measure how long a fresh subprocess takes to import vllm + load plugins."""
    result = subprocess.run(
        [sys.executable, "-c",
         "import time; t0=time.perf_counter(); "
         "import vllm; "
         "from vllm.plugins import load_general_plugins; "
         "load_general_plugins(); "
         "t1=time.perf_counter(); "
         "import torch; "
         "print(f'vllm_import={t1-t0:.3f}'); "
         "print(f'cuda_available={torch.cuda.is_available()}')"],
        capture_output=True, text=True, timeout=120
    )
    info = {}
    for line in result.stdout.strip().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            info[k] = v
    info["wall_time"] = None
    return info


def main():
    parser = argparse.ArgumentParser(description="Trace vLLM import times")
    parser.add_argument("--top", type=int, default=15,
                        help="Show top N heaviest imports")
    parser.add_argument("--module", default="vllm",
                        help="Module to trace (default: vllm)")
    args = parser.parse_args()

    print(f"Tracing imports for '{args.module}'...")
    entries = run_import_trace(args.module)

    if not entries:
        print("No import data captured. Make sure the module is installed.")
        return

    top_level = [e for e in entries if e.depth == 0]
    by_self = sorted(entries, key=lambda e: e.self_us, reverse=True)
    by_cumulative = sorted(top_level, key=lambda e: e.cumulative_us, reverse=True)

    total_us = sum(e.self_us for e in entries)

    print(f"\n{'='*70}")
    print(f"Import trace: {args.module}")
    print(f"Total import time: {total_us/1e6:.3f}s")
    print(f"Total unique modules imported: {len(entries)}")
    print(f"{'='*70}")

    print(f"\nTop {args.top} by self time (time in the module itself):")
    print(f"  {'Module':<50} {'Self':>8} {'Cum':>8}")
    print(f"  {'─'*50} {'─'*8} {'─'*8}")
    for e in by_self[:args.top]:
        print(f"  {e.module:<50} {e.self_us/1e6:>7.3f}s {e.cumulative_us/1e6:>7.3f}s")

    print(f"\nTop {args.top} by cumulative time (top-level imports):")
    print(f"  {'Module':<50} {'Cum':>8}")
    print(f"  {'─'*50} {'─'*8}")
    for e in by_cumulative[:args.top]:
        print(f"  {e.module:<50} {e.cumulative_us/1e6:>7.3f}s")

    print(f"\nSubprocess reimport cost:")
    info = measure_subprocess_reimport_cost()
    for k, v in info.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
