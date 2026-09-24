   Building ggufone @ file:///work/t_f46cec41/ggufone
      Built ggufone @ file:///work/t_f46cec41/ggufone
Uninstalled 1 package in 1ms
Installed 1 package in 2ms
load_backend: loaded RPC backend from /var/home/rybens/.hermes/runtime/b11026-linux-x64-cpu/libggml-rpc.so
load_backend: loaded CPU backend from /var/home/rybens/.hermes/runtime/b11026-linux-x64-cpu/libggml-cpu-haswell.so
~llama_context:        CPU compute buffer size is 307.2086 MiB, matches expectation of 307.2086 MiB
~llama_context:        CPU compute buffer size is 307.2086 MiB, matches expectation of 307.2086 MiB
~llama_context:        CPU compute buffer size is 307.2086 MiB, matches expectation of 307.2086 MiB
~llama_context:        CPU compute buffer size is 307.2086 MiB, matches expectation of 307.2086 MiB
~llama_context:        CPU compute buffer size is 307.2086 MiB, matches expectation of 307.2086 MiB
~llama_context:        CPU compute buffer size is 491.2819 MiB, matches expectation of 491.2819 MiB
~llama_context:        CPU compute buffer size is 491.2819 MiB, matches expectation of 491.2819 MiB
~llama_context:        CPU compute buffer size is 491.2819 MiB, matches expectation of 491.2819 MiB
~llama_context:        CPU compute buffer size is 491.2819 MiB, matches expectation of 491.2819 MiB
~llama_context:        CPU compute buffer size is 491.2819 MiB, matches expectation of 491.2819 MiB
### latency — Qwen3.5-0.8B-UD-Q4_K_XL.gguf

- generated: 2026-09-18T13:18:19Z
- host: Linux-7.2.4-ogc3.1.fc44.x86_64-x86_64-with-glibc2.41 · cpus 24 · cgroup quota 2.0
- config: backend=auto runs=5 threads=2
- reproduce: `uv run ggufone bench --suite latency --model /var/home/rybens/.cache/llama.cpp/Qwen3.5-0.8B-UD-Q4_K_XL.gguf --backend auto --runs 5 --threads 2 --max-seconds 300 --json`
- wall: 506.3 s (soft cap 300.0 s)
- TRUNCATED at --max-seconds 300.0 s: 22 unmeasured row(s) — prefill/tokens=8192, per_question/candidates=2, per_question/candidates=4, per_question/candidates=10, wave_scaling/questions=1, wave_scaling/questions=2, wave_scaling/questions=3, wave_scaling/questions=4, wave_scaling/questions=5, wave_scaling/questions=6, wave_scaling/questions=7, wave_scaling/questions=8, wave_scaling/questions=9, wave_scaling/questions=10, wave_scaling/questions=11, wave_scaling/questions=12, wave_scaling/questions=13, wave_scaling/questions=14, wave_scaling/questions=15, wave_scaling/questions=16, warm_cache/state reuse, load_amortisation/serve vs one-shot
- placement: requested n_gpu_layers=0, used n_gpu_layers=0 kv_type=auto

**model load (ms)**

| row | n | p50 | p95 | min | max |
|---|---|---|---|---|---|
| model_load_ms | 5 | 2,003.525 | 2,355.840 | 1,052.284 | 2,385.940 |

**prefill**

| tokens | n | p50 | p95 | min | max |
|---|---|---|---|---|---|
| 256 | 5 | 15,698.645 | 16,659.684 | 13,647.258 | 16,694.902 |
| 2048 | 5 | 86,147.701 | 94,669.362 | 58,968.957 | 96,713.580 |

**prefill throughput (tok/s)**

| tokens | n | p50 | p95 | min | max |
|---|---|---|---|---|---|
| 256 | 5 | 16.307 | 18.300 | 15.334 | 18.758 |
| 2048 | 5 | 23.773 | 34.488 | 21.176 | 34.730 |

- `waves` counts the decode batches a decision takes after the prefill (here: 1 suffix decode per question group + 1 per extra candidate token); five single-token candidates in one question are 1 wave, not ceil(5 / n_seq_max): {'groups': 1, 'suffix_decodes': 1, 'step_decodes': 1, 'waves': 2}.

report: /work/t_f46cec41/evidence/qwen_full_latency_cap300.json
EXIT=0
