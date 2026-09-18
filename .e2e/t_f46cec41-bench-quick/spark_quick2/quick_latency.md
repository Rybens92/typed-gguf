### latency — Spark-X2.5-4B-Q8_0.gguf

- generated: 2026-09-18T13:41:16Z
- host: Linux-7.2.4-ogc3.1.fc44.x86_64-x86_64-with-glibc2.41 · cpus 24 · cgroup quota 2.0
- config: backend=auto runs=1 threads=2
- reproduce: `uv run ggufone bench --suite latency --model /var/home/rybens/.hermes/models/Spark-X2.5-4B-Q8_0.gguf --backend auto --quick --threads 2 --json`
- preset: --quick (runs=1 sizes=[256] candidates=[2, 4] waves=[1, 2] items=6 = 2/type, determinism repeats=2, backends<=1) — an iteration preset, never a published table
- wall: 306.0 s
- placement: requested n_gpu_layers=0, used n_gpu_layers=0 kv_type=auto

**model load (ms)**

| row | n | p50 | p95 | min | max |
|---|---|---|---|---|---|
| model_load_ms | 1 | 1,080.814 | 1,080.814 | 1,080.814 | 1,080.814 |

**prefill**

| tokens | n | p50 | p95 | min | max |
|---|---|---|---|---|---|
| 256 | 1 | 49,540.741 | 49,540.741 | 49,540.741 | 49,540.741 |

**prefill throughput (tok/s)**

| tokens | n | p50 | p95 | min | max |
|---|---|---|---|---|---|
| 256 | 1 | 5.167 | 5.167 | 5.167 | 5.167 |

**per question**

| candidates | waves | forks | n | p50 | p95 | min | max |
|---|---|---|---|---|---|---|---|
| 2 | 2 | 2 | 1 | 9,951.520 | 9,951.520 | 9,951.520 | 9,951.520 |
| 4 | 2 | 4 | 1 | 10,732.286 | 10,732.286 | 10,732.286 | 10,732.286 |

**wave scaling (N questions)**

| questions | waves | n | p50 | p95 | min | max |
|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 9,598.688 | 9,598.688 | 9,598.688 | 9,598.688 |
| 2 | 2 | 1 | 14,806.666 | 14,806.666 | 14,806.666 | 14,806.666 |

**warm cache (state reuse)**

| row | n | p50 | p95 | min | max |
|---|---|---|---|---|---|
| prefill_ms | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| questions_ms | 1 | 9,662.007 | 9,662.007 | 9,662.007 | 9,662.007 |

**load amortisation (serve vs one-shot)**

| path | ms per request |
|---|---|
| serve (model already loaded) | 8,869.860 |
| one-shot CLI (load each call) | 9,950.673 |
| model_load_ms | 1,080.814 |

- quick preset (card t_f46cec41): runs=1, prefill sizes [256], candidates [2, 4], waves [1, 2], 6 dev items (2 per type), determinism 2 repeats, 1 backend. This is an iteration preset, not a published table: docs/BENCHMARKS.md is full-campaign only and is never produced with --quick.
- `waves` counts the decode batches a decision takes after the prefill (here: 1 suffix decode per question group + 1 per extra candidate token); five single-token candidates in one question are 1 wave, not ceil(5 / n_seq_max): {'groups': 1, 'suffix_decodes': 1, 'step_decodes': 1, 'waves': 2}.
- every measured call reports `prefill_reused: true` once the prefix state cache is warm, which is why the decision tables isolate the question phase.

report: /work/t_f46cec41/evidence/spark_quick2/quick_latency.json
