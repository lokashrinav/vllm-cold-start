"""Wrapper to run modal benchmark with proper output capture."""
import subprocess
import sys
import os
import io

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUNBUFFERED"] = "1"
os.chdir(r"C:\Users\lokas\vllm-cold-start")

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

log_path = r"C:\Users\lokas\vllm-cold-start\benchmark_v24.log"

proc = subprocess.Popen(
    [r"C:\Users\lokas\AppData\Local\Microsoft\WindowsApps\python3.exe",
     "-u", "-m", "modal", "run",
     "profiling/modal_foundry_multi_gpu_benchmark.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    encoding="utf-8",
    errors="replace",
    bufsize=1,
)

with open(log_path, "w", encoding="utf-8") as f:
    for line in proc.stdout:
        f.write(line)
        f.flush()
        try:
            sys.stdout.write(line)
            sys.stdout.flush()
        except UnicodeEncodeError:
            sys.stdout.write(line.encode("ascii", "replace").decode("ascii"))
            sys.stdout.flush()

proc.wait()
print(f"\nProcess exited with code {proc.returncode}")
