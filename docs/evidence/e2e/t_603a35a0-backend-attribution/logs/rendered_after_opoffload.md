==================== after_opoffload.raw
### throughput — Qwen3.5-4B-Q4_0.gguf

- generated: 2026-09-18T14:32:48Z
- host: Linux-7.2.4-ogc3.1.fc44.x86_64-x86_64-with-glibc2.41 · cpus 24 · cgroup quota 2.0
- config: backend=cpu runs=1 threads=4
- reproduce: `uv run ggufone bench --suite throughput --model /var/home/rybens/.hermes/models/Qwen3.5-4B-Q4_0.gguf --backend cpu --runs 1 --threads 4 --json`
- wall: 222.5 s

**backends**

| backend | effective | device buffers | placement | prefill tok/s (p50) | decision ms (p50) | load ms (p50) | decision tok/s (p50) |
|---|---|---|---|---|---|---|---|
| cpu | vulkan | Vulkan0=3 · Vulkan_Host=3 | n_gpu_layers=0 | 1.147 | 11,136.193 | 3,082.003 | 0.7184 |

- W_BACKEND_MISMATCH: the row claims backend `cpu` but the engine's own log shows the compute on vulkan (compute buffers: Vulkan0=3 · Vulkan_Host=3); read this row as a vulkan measurement — re-run one backend per process (`--backend vulkan`) for a clean attribution.

