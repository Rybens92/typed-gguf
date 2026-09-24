==================== after_mixed.raw
### throughput — Qwen3.5-4B-Q4_0.gguf

- generated: 2026-09-18T14:28:15Z
- host: Linux-7.2.4-ogc3.1.fc44.x86_64-x86_64-with-glibc2.41 · cpus 24 · cgroup quota 2.0
- config: backend=all runs=1 threads=4
- reproduce: `uv run ggufone bench --suite throughput --model /var/home/rybens/.hermes/models/Qwen3.5-4B-Q4_0.gguf --backend all --runs 1 --threads 4 --json`
- wall: 265.3 s

**backends**

| backend | effective | device buffers | placement | prefill tok/s (p50) | decision ms (p50) | load ms (p50) | decision tok/s (p50) |
|---|---|---|---|---|---|---|---|
| cpu | cpu | CPU=3 | n_gpu_layers=0 | 7.117 | 7,035.657 | 5,776.118 | 1.137 |
| vulkan | unverified | — | n_gpu_layers=-1 | 5.082 | 8,584.742 | 4,034.095 | 0.9319 |
| cuda | — | — | not measured | — | — | — | no local llama.cpp bundle carries libggml-cuda.so (benchmarks never download one: run `ggufone init --backend cuda` or point GGUFONE_BENCH_RUNTIME_DIR at extracted bundles) |

- W_BACKEND_MISMATCH: the row claims backend `vulkan` but the engine's own log carries no compute-buffer line for that backend, so the row cannot be corroborated; re-run one backend per process (`--backend <one>`) before publishing it.

