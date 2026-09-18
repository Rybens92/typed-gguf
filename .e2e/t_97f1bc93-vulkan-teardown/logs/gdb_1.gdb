warning: Error disabling address space randomization: Function not implemented
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
[Detaching after vfork from child process 294194]
[Detaching after vfork from child process 294262]
load_backend: loaded RPC backend from /var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan/libggml-rpc.so
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = NVIDIA GeForce RTX 3060 Ti (NVIDIA) | uma: 0 | fp16: 1 | bf16: 1 | fp4: 0 | warp size: 32 | shared memory: 49152 | int dot: 1 | matrix cores: NV_coopmat2
load_backend: loaded Vulkan backend from /var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan/libggml-vulkan.so
load_backend: loaded CPU backend from /var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan/libggml-cpu-haswell.so
[New Thread 0x7fc72c2e76c0 (LWP 294611)]
[New Thread 0x7fc72b27c6c0 (LWP 294614)]
[New Thread 0x7fc72aa7b6c0 (LWP 294615)]
[New Thread 0x7fc72a27a6c0 (LWP 294616)]
[New Thread 0x7fc729a796c0 (LWP 294637)]
[Detaching after vfork from child process 294840]
~llama_context:    Vulkan0 compute buffer size is 123.8272 MiB, matches expectation of 123.8272 MiB
~llama_context: Vulkan_Host compute buffer size is   2.5877 MiB, matches expectation of   2.5877 MiB
[Detaching after vfork from child process 296011]
~llama_context:    Vulkan0 compute buffer size is 425.8354 MiB, matches expectation of 425.8354 MiB
~llama_context: Vulkan_Host compute buffer size is   9.0404 MiB, matches expectation of   9.0404 MiB
[Detaching after vfork from child process 299140]
~llama_context:    Vulkan0 compute buffer size is 425.8354 MiB, matches expectation of 425.8354 MiB
~llama_context: Vulkan_Host compute buffer size is   9.0404 MiB, matches expectation of   9.0404 MiB
{
  "schema": "ggufone.bench/v1",
  "suite": "throughput",
  "quick": false,
  "generated_at": "2026-09-18T15:57:09Z",
  "host": {
    "platform": "Linux-7.2.4-ogc3.1.fc44.x86_64-x86_64-with-glibc2.41",
    "python": "3.11.15",
    "cpu_count": 24,
    "machine": "x86_64",
    "cgroup_cpu_max": 2.0,
    "cgroup_memory_bytes": 8589934592
  },
  "config": {
    "suite": "throughput",
    "model_path": "/var/home/rybens/.hermes/models/Qwen3.5-4B-Q4_0.gguf",
    "backend": "vulkan",
    "runs": 1,
    "threads": 4,
    "devset": null,
    "items": null,
    "n_seq_max": null,
    "kv_type": "auto",
    "gpu_layers": null,
    "n_bins": 10,
    "prefill_sizes": [
      64
    ],
    "candidate_counts": [
      2,
      4,
      10
    ],
    "wave_scaling": [
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16
    ],
    "parts": [],
    "home": null,
    "quick": false,
    "items_per_type": null,
    "determinism_repeats": 3,
    "backend_limit": null,
    "max_seconds": null
  },
  "model": {
    "path": "/var/home/rybens/.hermes/models/Qwen3.5-4B-Q4_0.gguf",
    "name": "Qwen3.5-4B-Q4_0.gguf",
    "bytes": 2583221408,
    "arch": "qwen35"
  },
  "notes": [],
  "commands": {
    "reproduce": "uv run ggufone bench --suite throughput --model /var/home/rybens/.hermes/models/Qwen3.5-4B-Q4_0.gguf --backend vulkan --runs 1 --threads 4 --json"
  },
  "backends": [
    {
      "backend": "vulkan",
      "measured": true,
      "runtime_dir": "/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan",
      "placement": "n_gpu_layers=-1",
      "threads": 4,
      "load_ms": {
        "n": 1,
        "min": 2531.7264340046677,
        "p50": 2531.7264340046677,
        "p95": 2531.7264340046677,
        "max": 2531.7264340046677,
        "mean": 2531.7264340046677
      },
      "placement_used": {
        "requested": "n_gpu_layers=-1",
        "used": {
          "note": "all layers requested: n_gpu_layers=-1 (kv_type=auto)",
          "n_gpu_layers": -1,
          "kv_type": "auto",
          "degraded": false,
          "attempts": [],
          "warnings": []
        }
      },
      "prefill": [
        {
          "tokens": 64,
          "ms": {
            "n": 1,
            "min": 11499.841921999177,
            "p50": 11499.841921999177,
            "p95": 11499.841921999177,
            "max": 11499.841921999177,
            "mean": 11499.841921999177
          },
          "tok_per_s": {
            "n": 1,
            "min": 5.565293891350638,
            "p50": 5.565293891350638,
            "p95": 5.565293891350638,
            "max": 5.565293891350638,
            "mean": 5.565293891350638
          },
          "threads": 4,
          "backend": "vulkan"
        }
      ],
      "prefill_tok_per_s": {
        "n": 1,
        "min": 5.565293891350638,
        "p50": 5.565293891350638,
        "p95": 5.565293891350638,
        "max": 5.565293891350638,
        "mean": 5.565293891350638
      },
      "decision_ms": {
        "n": 1,
        "min": 370.5653259967221,
        "p50": 370.5653259967221,
        "p95": 370.5653259967221,
        "max": 370.5653259967221,
        "mean": 370.5653259967221
      },
      "decision_tok_per_s": {
        "n": 1,
        "min": 21.588636169566403,
        "p50": 21.588636169566403,
        "p95": 21.588636169566403,
        "max": 21.588636169566403,
        "mean": 21.588636169566403
      },
      "devices": [
        "CPU_Mapped",
        "Vulkan0",
        "Vulkan_Host"
      ],
      "device_buffers": {
        "Vulkan0": 3,
        "Vulkan_Host": 3
      },
      "effective_backend": "vulkan",
      "warnings": []
    }
  ],
  "ok": true,
  "truncated": false,
  "skipped": [],
  "wall_ms": 57916.713,
  "budget": {
    "max_seconds": null,
    "expired": false,
    "skipped": 0
  }
}
[Thread 0x7fc729a796c0 (LWP 294637) exited]
[Thread 0x7fc72a27a6c0 (LWP 294616) exited]
[Thread 0x7fc72aa7b6c0 (LWP 294615) exited]
[Thread 0x7fc72c2e76c0 (LWP 294611) exited]
[Thread 0x7fc72b27c6c0 (LWP 294614) exited]
[Inferior 1 (process 294075) exited normally]

===== SIGNAL BACKTRACE =====
No stack.

===== SHARED LIBRARIES =====
From                To                  Syms Read   Shared Object Library
0x00007fc73ddc3000  0x00007fc73ddea3d1  Yes (*)     /lib64/ld-linux-x86-64.so.2
(*): Shared library is missing debugging information.

===== THREADS =====
