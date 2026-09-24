### throughput — Spark-X2.5-4B-Q8_0.gguf

- generated: 2026-09-18T13:46:23Z
- host: Linux-7.2.4-ogc3.1.fc44.x86_64-x86_64-with-glibc2.41 · cpus 24 · cgroup quota 2.0
- config: backend=auto runs=1 threads=2
- reproduce: `uv run ggufone bench --suite throughput --model /var/home/rybens/.hermes/models/Spark-X2.5-4B-Q8_0.gguf --backend auto --quick --threads 2 --json`
- preset: --quick (runs=1 sizes=[256] candidates=[2, 4] waves=[1, 2] items=6 = 2/type, determinism repeats=2, backends<=1) — an iteration preset, never a published table
- wall: 122.8 s

**backends**

| backend | placement | prefill tok/s (p50) | decision ms (p50) | load ms (p50) | decision tok/s (p50) |
|---|---|---|---|---|---|
| cpu | n_gpu_layers=0 | 5.836 | 3,869.844 | 934.115 | 2.067 |

- quick preset (card t_f46cec41): runs=1, prefill sizes [256], candidates [2, 4], waves [1, 2], 6 dev items (2 per type), determinism 2 repeats, 1 backend. This is an iteration preset, not a published table: docs/BENCHMARKS.md is full-campaign only and is never produced with --quick.

report: /work/t_f46cec41/evidence/spark_quick2/quick_throughput.json
