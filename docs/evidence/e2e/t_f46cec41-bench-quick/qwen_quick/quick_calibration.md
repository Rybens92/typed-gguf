### calibration — Qwen3.5-0.8B-UD-Q4_K_XL.gguf

- generated: 2026-09-18T12:54:54Z
- host: Linux-7.2.4-ogc3.1.fc44.x86_64-x86_64-with-glibc2.41 · cpus 24 · cgroup quota 2.0
- config: backend=auto runs=1 threads=2
- reproduce: `uv run ggufone bench --suite calibration --model /var/home/rybens/.cache/llama.cpp/Qwen3.5-0.8B-UD-Q4_K_XL.gguf --backend auto --quick --threads 2 --json`
- preset: --quick (runs=1 sizes=[256] candidates=[2, 4] waves=[1, 2] items=6 = 2/type, determinism repeats=2, backends<=1) — an iteration preset, never a published table
- wall: 40.0 s

**exact-match agreement**

| type | n | correct | agreement | 95% CI |
|---|---|---|---|---|
| choice | 2 | 1 | 0.5000 | 0.0945 – 0.9055 |
| noul | 2 | 2 | 1.000 | 0.3424 – 1.000 |
| score | 2 | 0 | 0.0000 | 0.0000 – 0.6576 |
| overall | 6 | 3 | 0.5000 | 0.1876 – 0.8124 |

**reliability bins**

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| 0.5000–0.6667 | 1 | 0.5248 | 0.0000 | -0.5248 |
| 0.6667–0.8333 | 4 | 0.7555 | 0.7500 | -0.0055 |
| 0.8333–1.000 | 1 | 0.9025 | 0.0000 | -0.9025 |

**confidence modes**

| mode | n | ECE |
|---|---|---|
| entropy | 6 | 0.5891 |
| margin | 6 | 0.2275 |
| normalized_peak | 6 | 0.2995 |

- ECE (bins=6): 0.2416
- confidence/coverage correlation: -0.4388

- quick preset (card t_f46cec41): runs=1, prefill sizes [256], candidates [2, 4], waves [1, 2], 6 dev items (2 per type), determinism 2 repeats, 1 backend. This is an iteration preset, not a published table: docs/BENCHMARKS.md is full-campaign only and is never produced with --quick.
- confidence is the answer's own `confidence` (noul carries none, so the highest probability stands in); `coverage` is the full-vocabulary mass the engine reported.
- the three confidence modes are recomputed from the same stored distributions, so the table costs no extra model runs; the correlation is undefined (null) when every confidence is identical.
- quick calibration: 6 samples in 6 bins is a shape check (the presets, the modes and the bin machinery all exercised), never a calibration claim — the full campaign's 60 items are what an ECE is read from.

report: /work/t_f46cec41/evidence/qwen_quick/quick_calibration.json
