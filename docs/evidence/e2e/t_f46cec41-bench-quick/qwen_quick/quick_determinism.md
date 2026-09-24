### determinism — Qwen3.5-0.8B-UD-Q4_K_XL.gguf

- generated: 2026-09-18T12:55:34Z
- host: Linux-7.2.4-ogc3.1.fc44.x86_64-x86_64-with-glibc2.41 · cpus 24 · cgroup quota 2.0
- config: backend=auto runs=1 threads=2
- reproduce: `uv run ggufone bench --suite determinism --model /var/home/rybens/.cache/llama.cpp/Qwen3.5-0.8B-UD-Q4_K_XL.gguf --backend auto --quick --threads 2 --json`
- preset: --quick (runs=1 sizes=[256] candidates=[2, 4] waves=[1, 2] items=6 = 2/type, determinism repeats=2, backends<=1) — an iteration preset, never a published table
- wall: 23.6 s

**byte identity (timings stripped, 3 repeats)**

| backend | identical | digest |
|---|---|---|
| cpu | yes | `sha256:354a671aff1e50df…` |

- repeats: 2 · ok: True

- quick preset (card t_f46cec41): runs=1, prefill sizes [256], candidates [2, 4], waves [1, 2], 6 dev items (2 per type), determinism 2 repeats, 1 backend. This is an iteration preset, not a published table: docs/BENCHMARKS.md is full-campaign only and is never produced with --quick.

report: /work/t_f46cec41/evidence/qwen_quick/quick_determinism.json
