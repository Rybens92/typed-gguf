### latency — Qwen3.5-0.8B-UD-Q4_K_XL.gguf

- generated: 2026-09-18T12:50:50Z
- host: Linux-7.2.4-ogc3.1.fc44.x86_64-x86_64-with-glibc2.41 · cpus 24 · cgroup quota 2.0
- config: backend=auto runs=1 threads=2
- reproduce: `uv run ggufone bench --suite latency --model /var/home/rybens/.cache/llama.cpp/Qwen3.5-0.8B-UD-Q4_K_XL.gguf --backend auto --quick --threads 2 --json`
- preset: --quick (runs=1 sizes=[256] candidates=[2, 4] waves=[1, 2] items=6 = 2/type, determinism repeats=2, backends<=1) — an iteration preset, never a published table
- wall: 107.6 s
- placement: requested n_gpu_layers=0, used n_gpu_layers=0 kv_type=auto

**model load (ms)**

| row | n | p50 | p95 | min | max |
|---|---|---|---|---|---|
| model_load_ms | 1 | 1,580.046 | 1,580.046 | 1,580.046 | 1,580.046 |

**prefill**

| tokens | n | p50 | p95 | min | max |
|---|---|---|---|---|---|
| 256 | 1 | 15,199.017 | 15,199.017 | 15,199.017 | 15,199.017 |

**prefill throughput (tok/s)**

| tokens | n | p50 | p95 | min | max |
|---|---|---|---|---|---|
| 256 | 1 | 16.843 | 16.843 | 16.843 | 16.843 |

**per question**

| candidates | waves | forks | n | p50 | p95 | min | max |
|---|---|---|---|---|---|---|---|
| 2 | 2 | 2 | 1 | 3,369.637 | 3,369.637 | 3,369.637 | 3,369.637 |
| 4 | 2 | 4 | 1 | 717.760 | 717.760 | 717.760 | 717.760 |

**wave scaling (N questions)**

| questions | waves | n | p50 | p95 | min | max |
|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 3,691.534 | 3,691.534 | 3,691.534 | 3,691.534 |
| 2 | 2 | 1 | 3,148.423 | 3,148.423 | 3,148.423 | 3,148.423 |

**warm cache (state reuse)**

| row | n | p50 | p95 | min | max |
|---|---|---|---|---|---|
| prefill_ms | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| questions_ms | 1 | 2,376.030 | 2,376.030 | 2,376.030 | 2,376.030 |

**load amortisation (serve vs one-shot)**

| path | ms per request |
|---|---|
| serve (model already loaded) | 5,573.707 |
| one-shot CLI (load each call) | 7,153.753 |
| model_load_ms | 1,580.046 |

- quick preset (card t_f46cec41): runs=1, prefill sizes [256], candidates [2, 4], waves [1, 2], 6 dev items (2 per type), determinism 2 repeats, 1 backend. This is an iteration preset, not a published table: docs/BENCHMARKS.md is full-campaign only and is never produced with --quick.
- `waves` counts the decode batches a decision takes after the prefill (here: 1 suffix decode per question group + 1 per extra candidate token); five single-token candidates in one question are 1 wave, not ceil(5 / n_seq_max): {'groups': 1, 'suffix_decodes': 1, 'step_decodes': 1, 'waves': 2}.
- every measured call reports `prefill_reused: true` once the prefix state cache is warm, which is why the decision tables isolate the question phase.

report: /work/t_f46cec41/evidence/qwen_quick/quick_latency.json
