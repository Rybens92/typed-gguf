#!/usr/bin/env python3
"""Write <BASE>/speed_sample.json — the fully-resident generation-speed sample (AC5).

Reads the raw llama-bench JSON next to it and states the method, so the evidence doc can quote a
number that names its instrument instead of a bare tok/s.
"""
import json
import pathlib
import sys

BASE = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/work/t_d199e09c/mimo")
raw_path = BASE / "speed" / "llama_bench_tg128.json"
raw = json.loads(raw_path.read_text(encoding="utf-8"))[0]

sample = {
    "instrument": "llama-bench, llama.cpp b11026 (bundle linux-x64-vulkan)",
    "method": "llama-bench -m <MiMo-V2.6-Distill-Qwen-9B-Q4_K_M.gguf> -ngl 99 -p 0 -n 128 -r 3 "
              "-t 4 -ctk q4_0 -ctv q4_0 -fa 1 -o json",
    "what_it_measures": "tg128: 128 tokens generated after an empty prompt, with all 32 layers on "
                        "the Vulkan device (-ngl 99 = the fully-resident placement this arm measured), "
                        "KV cache q4_0 (the same KV type the run's fit plan chose) and flash attention "
                        "on (the engine's own ctypes binding sets LLAMA_FLASH_ATTN_TYPE_ENABLED, "
                        "src/typed_gguf/engine/session.py:731, so the bench matches the engine)",
    "why_not_a_chat_sample": "the run itself generates no free text (readout `sequence` scores fixed "
                             "label candidates), so tg tok/s cannot be read off the 30 items; a bench "
                             "completion is the smallest honest instrument for the 'is the full-GPU "
                             "speed real' question",
    "avg_ts": raw["avg_ts"],
    "stddev_ts": raw["stddev_ts"],
    "samples_ts": raw["samples_ts"],
    "avg_ns": raw["avg_ns"],
    "repetitions": raw["n_gen"],
    "test_time": raw["test_time"],
    "build_number": raw["build_number"],
    "build_commit": raw["build_commit"],
    "gpu_info": raw["gpu_info"],
    "cpu_info": raw["cpu_info"],
    "model_type": raw["model_type"],
    "model_size": raw["model_size"],
    "model_n_params": raw["model_n_params"],
    "n_gpu_layers": raw["n_gpu_layers"],
    "type_k": raw["type_k"],
    "type_v": raw["type_v"],
    "flash_attn": raw["flash_attn"],
    "n_threads": raw["n_threads"],
    "raw_file": str(raw_path.relative_to(BASE)),
    "raw": raw,
}
(BASE / "speed_sample.json").write_text(json.dumps(sample, indent=1) + "\n", encoding="utf-8")
print(f"wrote {BASE / 'speed_sample.json'}: tg128 {raw['avg_ts']:.2f} tok/s (sd {raw['stddev_ts']:.3f})")
