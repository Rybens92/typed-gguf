### throughput — Qwen3.5-0.8B-UD-Q4_K_XL.gguf

- generated: 2026-09-18T12:52:40Z
- host: Linux-7.2.4-ogc3.1.fc44.x86_64-x86_64-with-glibc2.41 · cpus 24 · cgroup quota 2.0
- config: backend=auto runs=1 threads=2
- reproduce: `uv run ggufone bench --suite throughput --model /var/home/rybens/.cache/llama.cpp/Qwen3.5-0.8B-UD-Q4_K_XL.gguf --backend auto --quick --threads 2 --json`
- preset: --quick (runs=1 sizes=[256] candidates=[2, 4] waves=[1, 2] items=6 = 2/type, determinism repeats=2, backends<=1) — an iteration preset, never a published table
- wall: 70.3 s

**backends**

| backend | placement | prefill tok/s (p50) | decision ms (p50) | load ms (p50) | decision tok/s (p50) |
|---|---|---|---|---|---|
| cpu | n_gpu_layers=0 | 16.843 | 2,627.709 | 2,899.046 | 3.044 |

- quick preset (card t_f46cec41): runs=1, prefill sizes [256], candidates [2, 4], waves [1, 2], 6 dev items (2 per type), determinism 2 repeats, 1 backend. This is an iteration preset, not a published table: docs/BENCHMARKS.md is full-campaign only and is never produced with --quick.

report: /work/t_f46cec41/evidence/qwen_quick/quick_throughput.json
