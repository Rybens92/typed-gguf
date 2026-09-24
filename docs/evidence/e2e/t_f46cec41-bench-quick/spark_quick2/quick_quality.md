### quality — Spark-X2.5-4B-Q8_0.gguf

- generated: 2026-09-18T13:51:42Z
- host: Linux-7.2.4-ogc3.1.fc44.x86_64-x86_64-with-glibc2.41 · cpus 24 · cgroup quota 2.0
- config: backend=auto runs=1 threads=2
- reproduce: `uv run ggufone bench --suite quality --model /var/home/rybens/.hermes/models/Spark-X2.5-4B-Q8_0.gguf --backend auto --quick --threads 2 --json`
- preset: --quick (runs=1 sizes=[256] candidates=[2, 4] waves=[1, 2] items=6 = 2/type, determinism repeats=2, backends<=1) — an iteration preset, never a published table
- wall: 57.4 s

**exact-match agreement**

| type | n | correct | agreement | 95% CI |
|---|---|---|---|---|
| choice | 2 | 1 | 0.5000 | 0.0945 – 0.9055 |
| noul | 2 | 2 | 1.000 | 0.3424 – 1.000 |
| score | 2 | 0 | 0.0000 | 0.0000 – 0.6576 |
| overall | 6 | 3 | 0.5000 | 0.1876 – 0.8124 |

- quick preset (card t_f46cec41): runs=1, prefill sizes [256], candidates [2, 4], waves [1, 2], 6 dev items (2 per type), determinism 2 repeats, 1 backend. This is an iteration preset, not a published table: docs/BENCHMARKS.md is full-campaign only and is never produced with --quick.
- agreement = the highest-probability candidate equals the gold candidate (the discrete decision), measured per question type and overall with 95% Wilson intervals; report-only in v1 (SPEC S-11).

report: /work/t_f46cec41/evidence/spark_quick2/quick_quality.json
