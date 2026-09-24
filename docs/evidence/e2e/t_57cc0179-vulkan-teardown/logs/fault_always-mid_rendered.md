### throughput — Qwen3.5-0.8B-UD-Q4_K_XL.gguf

- generated: 2026-09-19T14:21:19Z
- host: Linux-7.2.4-ogc3.1.fc44.x86_64-x86_64-with-glibc2.41 · cpus 24 · cgroup quota 2.0
- config: backend=all runs=1 threads=4
- reproduce: `uv run ggufone bench --suite throughput --model /var/home/rybens/.cache/llama.cpp/Qwen3.5-0.8B-UD-Q4_K_XL.gguf --backend all --runs 1 --threads 4 --gpu-layers 0 --json`
- wall: 59.0 s

**backends**

| backend | effective | device buffers | placement | prefill tok/s (p50) | decision ms (p50) | load ms (p50) | decision tok/s (p50) |
|---|---|---|---|---|---|---|---|
| cpu | cpu | CPU=3 | n_gpu_layers=0 | 59.673 | 2,305.004 | 921.589 | 3.471 |
| vulkan | — | — | not measured | — | — | — | W_BACKEND_CRASHED_AT_TEARDOWN: the isolated child exited -11 (SIGSEGV; 139 in a shell) after writing a complete report (stderr tail: …mentation fault

Current thread 0x00007f6fba747380 (most recent call first):
  File "/work/t57cc-ggufone/.e2e/t_57cc0179-vulkan-teardown/repro/fault_inject/sitecustomize.py", line 46 in _raise_sigsegv); 1651 MiB free of 8192 MiB at the attempt's start (the weights would fit: the crash was not memory pressure); the retry with the placement it already had (n_gpu_layers=0, kv_type=q4_0 is already the bottom rung) exited -11 (SIGSEGV; 139 in a shell)  (stderr tail: …mentation fault

Current thread 0x00007f7e8545f380 (most recent call first):
  File "/work/t57cc-ggufone/.e2e/t_57cc0179-vulkan-teardown/repro/fault_inject/sitecustomize.py", line 46 in _raise_sigsegv) (1651 MiB free of 8192 MiB at its start); the row is withheld (`measured: false`) and the report is not ok (the child's own report and logs are kept at /tmp/ggufone-bench-isolated-j3ixpe14) |
| cuda | — | — | not measured | — | — | — | no local llama.cpp bundle carries libggml-cuda.so (benchmarks never download one: run `ggufone init --backend cuda` or point GGUFONE_BENCH_RUNTIME_DIR at extracted bundles) |

- one bundle per process: `--backend all` selected cpu, vulkan, cuda over 2 distinct local bundle directories, and two distinct local bundles cannot be dlopened into one process: the second bundle's libllama/libggml are shadowed by the first one's SONAMEs (RTLD_GLOBAL, identical names) and the process aborts at teardown with `double free or corruption` — exit 134 (card t_dd62ec29). Every row was measured by its own child process — the documented `--backend <one>` path — and carries the `process` block that produced it; a child whose answer the parent could not verify leaves no row at all.
- ISOLATED_CHILD_FAILED: the row for backend `vulkan` was measured in its own child process (one bundle per process — two distinct local bundles cannot be dlopened into one process: the second bundle's libllama/libggml are shadowed by the first one's SONAMEs (RTLD_GLOBAL, identical names) and the process aborts at teardown with `double free or corruption` — exit 134 (card t_dd62ec29)) and that child exited -11 (SIGSEGV; 139 in a shell): W_BACKEND_CRASHED_AT_TEARDOWN: the isolated child exited -11 (SIGSEGV; 139 in a shell) after writing a complete report (stderr tail: …mentation fault

Current thread 0x00007f6fba747380 (most recent call first):
  File "/work/t57cc-ggufone/.e2e/t_57cc0179-vulkan-teardown/repro/fault_inject/sitecustomize.py", line 46 in _raise_sigsegv); 1651 MiB free of 8192 MiB at the attempt's start (the weights would fit: the crash was not memory pressure); the retry with the placement it already had (n_gpu_layers=0, kv_type=q4_0 is already the bottom rung) exited -11 (SIGSEGV; 139 in a shell)  (stderr tail: …mentation fault

Current thread 0x00007f7e8545f380 (most recent call first):
  File "/work/t57cc-ggufone/.e2e/t_57cc0179-vulkan-teardown/repro/fault_inject/sitecustomize.py", line 46 in _raise_sigsegv) (1651 MiB free of 8192 MiB at its start); the row is withheld (`measured: false`) and the report is not ok (the child's own report and logs are kept at /tmp/ggufone-bench-isolated-j3ixpe14). The row is withheld (`measured: false`) and the report is not ok.

