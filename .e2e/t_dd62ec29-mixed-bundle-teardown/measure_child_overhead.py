"""Measure the per-child overhead: interpreter start + importing the CLI, the way a child pays it."""

import statistics
import subprocess
import sys
import time

runs = []
for _ in range(5):
    started = time.perf_counter()
    subprocess.run([sys.executable, "-m", "ggufone", "bench", "--help"], capture_output=True,
                   check=True)
    runs.append(time.perf_counter() - started)
print("bench --help (full CLI import), 5 runs:",
      ", ".join(f"{value * 1000:.0f} ms" for value in runs),
      f"· median {statistics.median(runs) * 1000:.0f} ms")
