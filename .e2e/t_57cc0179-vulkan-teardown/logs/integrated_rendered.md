### throughput — Qwen3.5-0.8B-UD-Q4_K_XL.gguf

- generated: 2026-09-19T14:13:22Z
- host: Linux-7.2.4-ogc3.1.fc44.x86_64-x86_64-with-glibc2.41 · cpus 24 · cgroup quota 2.0
- config: backend=all runs=1 threads=4
- reproduce: `uv run ggufone bench --suite throughput --model /var/home/rybens/.cache/llama.cpp/Qwen3.5-0.8B-UD-Q4_K_XL.gguf --backend all --runs 1 --threads 4 --gpu-layers 0 --json`
- wall: 133.3 s

**backends**

| backend | effective | device buffers | placement | prefill tok/s (p50) | decision ms (p50) | load ms (p50) | decision tok/s (p50) |
|---|---|---|---|---|---|---|---|
| cpu | cpu | CPU=3 | n_gpu_layers=0 | 19.436 | 6,207.043 | 5,810.008 | 1.289 |
| vulkan | vulkan | Vulkan0=3 · Vulkan_Host=3 | n_gpu_layers=0 | 53.648 | 5,803.405 | 4,026.912 | 1.379 |
| cuda | — | — | not measured | — | — | — | no local llama.cpp bundle carries libggml-cuda.so (benchmarks never download one: run `ggufone init --backend cuda` or point GGUFONE_BENCH_RUNTIME_DIR at extracted bundles) |

- one bundle per process: `--backend all` selected cpu, vulkan, cuda over 2 distinct local bundle directories, and two distinct local bundles cannot be dlopened into one process: the second bundle's libllama/libggml are shadowed by the first one's SONAMEs (RTLD_GLOBAL, identical names) and the process aborts at teardown with `double free or corruption` — exit 134 (card t_dd62ec29). Every row was measured by its own child process — the documented `--backend <one>` path — and carries the `process` block that produced it; a child whose answer the parent could not verify leaves no row at all.

