### throughput — Qwen3.5-4B-Q4_0.gguf

- generated: 2026-09-18T15:13:06Z
- host: Linux-7.2.4-ogc3.1.fc44.x86_64-x86_64-with-glibc2.41 · cpus 24 · cgroup quota 2.0
- config: backend=all runs=1 threads=4
- reproduce: `uv run ggufone bench --suite throughput --model /var/home/rybens/.hermes/models/Qwen3.5-4B-Q4_0.gguf --backend all --runs 1 --threads 4 --json`
- wall: 156.4 s

**backends**

| backend | effective | device buffers | placement | prefill tok/s (p50) | decision ms (p50) | load ms (p50) | decision tok/s (p50) |
|---|---|---|---|---|---|---|---|
| cpu | cpu | CPU=3 | n_gpu_layers=0 | 7.199 | 5,868.123 | 11,899.820 | 1.363 |
| vulkan | — | — | not measured | — | — | — | BackendOomError: E_BACKEND_OOM: llama.cpp could not allocate device memory for the fit plan (n_gpu_layers=-1, kv_type=auto, needed ~426 MiB); the driver reports unknown free; tried 3 placement(s) down to CPU-only, none fit: ctx kv_type=f16 -> oom; ctx kv_type=q8_0 -> oom; ctx kv_type=q4_0 -> oom; the backend asked for a 426 MiB allocation; backend log: 'ggml_gallocr_reserve_n_impl: failed to allocate Vulkan0 buffer of size 446799360'; fix: `--no-fit` runs on the CPU, `--fit-target <MiB>` leaves that much device memory free for the rest of the desktop, or use a smaller quant |
| cuda | — | — | not measured | — | — | — | no local llama.cpp bundle carries libggml-cuda.so (benchmarks never download one: run `ggufone init --backend cuda` or point GGUFONE_BENCH_RUNTIME_DIR at extracted bundles) |

- one bundle per process: `--backend all` selected cpu, vulkan, cuda over 2 distinct local bundle directories, and two distinct local bundles cannot be dlopened into one process: the second bundle's libllama/libggml are shadowed by the first one's SONAMEs (RTLD_GLOBAL, identical names) and the process aborts at teardown with `double free or corruption` — exit 134 (card t_dd62ec29). Every row was measured by its own child process — the documented `--backend <one>` path — and carries the `process` block that produced it; a child whose answer the parent could not verify leaves no row at all.

