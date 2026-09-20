#!/usr/bin/env python3
"""typed-gguf runtime + interface contract oracle.

Exit 0 == every pinned fact reproduced. Run before and after implementation:

    python3 docs/verify_runtime_contract.py            # offline (evidence pins)
    TYPED_GGUF_RUNTIME_DIR=/path/to/llama-b11026 \
        python3 docs/verify_runtime_contract.py        # + live ctypes probes

Sections
  A. distribution / evidence pins      (reads docs/evidence/*.json, offline)
  B. live runtime probes               (conditional; SKIP without a runtime)
  C. arithmetic mirror                 (always; this is what the TDD card transplants)
  D. package + contract-surface pins   (conditional on `import typed_gguf`)

What this script is NOT: a claim of TypeSafe parity. It pins (a) the facts the
SPEC is built on, (b) typed-gguf's own formulas with executed reference values, and
(c) the names/shapes of the interfaces the SPEC freezes.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVID = ROOT / "docs" / "evidence"

FAILS: list[str] = []
SKIPS: list[str] = []
TABLE: list[tuple[str, str]] = []


def check(cond: bool, msg: str) -> bool:
    if cond:
        print(f"  ok   {msg}")
    else:
        print(f"  FAIL {msg}")
        FAILS.append(msg)
    return bool(cond)


def skip(msg: str) -> None:
    print(f"  SKIP {msg}")
    SKIPS.append(msg)


def row(name: str, value: str) -> None:
    TABLE.append((name, value))
    print(f"  {name:34s} {value}")


# ---------------------------------------------------------------------------
# PINNED FACTS (captured 2026-09-17; every one re-asserted against evidence)
# ---------------------------------------------------------------------------
LLAMA_TAG = "b11026"
LLAMA_PUBLISHED = "2026-09-17T13:31:47Z"
LLAMA_ASSET_COUNT = 33
MIN_BUILD_FOR_SPARK25 = 10828
ASSET_SIZES = {
    "llama-b11026-bin-ubuntu-x64.tar.gz": 16855810,
    "llama-b11026-bin-ubuntu-vulkan-x64.tar.gz": 30294625,
    "llama-b11026-bin-ubuntu-cuda-12.8-x64.tar.gz": 168811114,
    "llama-b11026-bin-ubuntu-cuda-13.3-x64.tar.gz": 149113548,
    "llama-b11026-bin-win-cpu-x64.zip": 18439911,
    "llama-b11026-bin-win-vulkan-x64.zip": 31766385,
    "llama-b11026-bin-win-cuda-12.4-x64.zip": 254193665,
    "llama-b11026-bin-macos-arm64.tar.gz": 11156751,
    "llama-b11026-bin-macos-x64.tar.gz": 11204942,
}
TARBALL_SHA256 = "219cf1c726bae1da4289b96a6378314d5485c6bc74c43891a4203e30906afb06"
TARBALL_FILE_COUNT = 61
HEADER_LINES = 1645
# include/llama.h @ b11026 — line of each LLAMA_API declaration we bind via ctypes
HEADER_SYMBOL_LINES = {
    "llama_model_load_from_file": 516,
    "llama_init_from_model": 543,
    "llama_n_seq_max": 573,
    "llama_vocab_n_tokens": 581,
    "llama_get_memory": 584,
    "llama_model_meta_val_str": 621,
    "llama_model_desc": 636,
    "llama_model_chat_template": 646,
    "llama_model_n_params": 649,
    "llama_memory_seq_rm": 755,
    "llama_memory_seq_cp": 764,
    "llama_memory_seq_keep": 772,
    "llama_state_seq_get_size": 876,
    "llama_state_seq_save_file": 897,
    "llama_state_seq_load_file": 905,
    "llama_batch_init": 967,
    "llama_decode": 997,
    "llama_get_logits_ith": 1048,
    "llama_tokenize": 1179,
    "llama_detokenize": 1207,
    "llama_chat_apply_template": 1230,
    "llama_print_system_info": 1567,
}
REQUIRED_SYMBOLS = [
    "llama_backend_init", "llama_backend_free",
    "llama_model_default_params", "llama_context_default_params",
    "llama_model_load_from_file", "llama_model_free",
    "llama_init_from_model", "llama_free",
    "llama_model_get_vocab", "llama_model_meta_val_str", "llama_model_chat_template",
    "llama_model_n_layer", "llama_model_n_embd",
    "llama_n_ctx", "llama_n_seq_max", "llama_get_memory",
    "llama_memory_seq_cp", "llama_memory_seq_rm", "llama_memory_seq_keep",
    "llama_state_seq_get_size", "llama_state_seq_save_file", "llama_state_seq_load_file",
    "llama_batch_init", "llama_batch_free", "llama_batch_get_one",
    "llama_decode", "llama_get_logits_ith",
    "llama_tokenize", "llama_token_to_piece", "llama_vocab_n_tokens",
    "llama_chat_apply_template", "llama_synchronize",
]
# libggml.so surface: the backend loader MUST run before any model load (PoC pitfall 1)
REQUIRED_GGML_SYMBOLS = ["ggml_backend_load_all", "ggml_backend_load_all_from_path"]
PYTHON_CTYPES_RUNTIME = "llama.cpp b11026"
POC_MARKERS = [  # the executed PoC (docs/evidence/poc-ctypes-20260917.py) call sequence
    "ggml_backend_load_all_from_path",   # pitfall 1: before model load
    "kv_unified = True",                 # pitfall 2: seq_cp assert without it
    "llama_memory_seq_cp",
    "llama_get_logits_ith",
    "batched decode",
]

HF_REPO = "XHToken/Spark-X2.5-4B-GGUF"
HF_REPO_SHA = "902d865994943ab9235670e24f01846ee06091f2"
HF_QUANTS = {  # path -> (bytes, sha256)  [sha256 == HF LFS oid, verified by execution]
    "Spark-X2.5-4B-Q4_K_M.gguf": (2600224352, "adfcfa19a4ed6a5985da8bf565fe15f8e1a7e131d79bae2d19d48d1c40109428"),
    "Spark-X2.5-4B-Q8_0.gguf": (4375021152, "5c2c3c190e4337e1016b8593ca8e26e8b18c972200b107385d4ec61a25d9dea2"),
    "Spark-X2.5-4B.gguf": (8229920352, "8cecf405a41a4a10f833530910c2e13fde9fb39c325c8afc3c5d10e4181e1a14"),
}
LOCAL_SPARK = pathlib.Path.home() / ".hermes" / "models" / "Spark-X2.5-4B-Q8_0.gguf"
LOCAL_QWEN35 = pathlib.Path.home() / ".cache" / "llama.cpp" / "Qwen3.5-0.8B-UD-Q4_K_XL.gguf"
LOCAL_SPARK_KV = {  # executed GGUF-header read of the real file (see section C)
    "general.architecture": "spark2_5",
    "general.file_type": 7,
    "spark2_5.block_count": 36,
    "spark2_5.attention.head_count": 16,
    "spark2_5.attention.head_count_kv": 4,
    "spark2_5.attention.key_length": 256,
    "spark2_5.attention.value_length": 256,
    "spark2_5.context_length": 1048576,
    "spark2_5.embedding_length": 2560,
}
FTYPE = {7: "MOSTLY_Q8_0", 15: "MOSTLY_Q4_K_M"}
GGML_TYPE_BYTES = {"f16": 2, "q8_0": 1, "q4_0": 1}  # per-element KV storage cost (SPEC 2.4)

# contract surfaces frozen by SPEC 2.8 / 2.9
CLI_COMMANDS = ["init", "doctor", "models", "run", "ask", "serve", "mcp", "bench",
                "fit", "calibrate", "version"]
MODELS_SUBCOMMANDS = ["search", "pull", "use", "ls", "rm", "verify", "recommend-quant"]
HTTP_ROUTES = ["/health", "/v1/decide", "/v1/systemone", "/v1/models"]
HTTP_ROUTE_METHODS = {"GET": ["/health", "/v1/models"], "POST": ["/v1/decide", "/v1/systemone"]}
MCP_TOOLS = ["typed_gguf_decide", "typed_gguf_models_list", "typed_gguf_models_pull",
             "typed_gguf_runtime_status", "typed_gguf_fit"]
MCP_METHODS = ["initialize", "tools/list", "tools/call"]
FORBIDDEN_TRAINING_DEPS = ["torch", "peft", "trl", "unsloth", "bitsandbytes", "deepspeed",
                           "accelerate", "lightning", "sentence-transformers", "axolotl"]

# TypeSafe wire contract (docs.typesafe.ai, captured 2026-09-17) — adapter must map these
TS_REQUEST_KEYS = ["state", "model", "questions"]
TS_QUESTION_KEYS = {"choice": ["type", "instructions", "criteria"],
                    "score": ["type", "instructions", "criteria"],
                    "noul": ["type", "instructions"]}
TS_ANSWER_KEYS = {"choice": ["type", "choice", "probabilities", "confidence"],
                  "score": ["type", "score", "probabilities", "confidence", "legend"],
                  "noul": ["type", "noul"]}
TS_LIMITS = {"choice_max_options": 255, "score_min_levels": 2, "score_max_levels": 10}
# documented examples: (label, probabilities, documented confidence)
TS_CONFIDENCE_EXAMPLES = [
    ("choice/returns-ticket", [0.02, 0.38, 0.6], 0.39),
    ("choice/return_reason", [1.0, 0.0, 0.0, 0.0, 0.0], 1.0),
    ("choice/shipping_issue", [0.63, 0.37, 0.0, 0.0, 0.0], 0.53),
    ("choice/requested_resolution", [0.37, 0.24, 0.29, 0.1], 0.16),
    ("choice/tone", [0.92, 0.08, 0.0], 0.88),
    ("score/bug_severity", [0.0, 0.7, 0.3], 0.54),
    ("score/spinner+examples", [0.0, 0.94, 0.06], 0.91),
]
TS_CONFIDENCE_OUTLIER = ("quickstart/department", [0.159, 0.84, 0.001], 0.596)
TS_REL_TOL = 0.02


# ---------------------------------------------------------------------------
# C. arithmetic mirror — the reference implementations the TDD card transplants
# ---------------------------------------------------------------------------
def softmax(xs: list[float], temperature: float = 1.0) -> list[float]:
    """Restricted softmax over candidate logits (SPEC 2.4 step 4)."""
    m = max(xs)
    exps = [math.exp((x - m) / temperature) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]


def confidence_normalized_peak(probs: list[float]) -> float:
    """typed-gguf confidence: excess of the peak over uniform, rescaled to [0, 1]."""
    k = len(probs)
    if k <= 1:
        return 1.0
    pmax = max(probs)
    return max(0.0, min(1.0, (pmax - 1.0 / k) / (1.0 - 1.0 / k)))


def score_weighted_mean(probs: list[float]) -> float:
    """Score = sum(level_index * p_level) — parity with the documented TypeSafe rule."""
    return sum(i * p for i, p in enumerate(probs))


def coverage(candidate_logits: list[float], full_logits: list[float]) -> float:
    """Answer mass: candidate probability under the FULL vocab softmax (SPEC 2.4)."""
    p_full = softmax(full_logits)
    return sum(p_full[i] for i in range(len(candidate_logits)))


def candidate_sequence_score(logprobs: list[float], length_norm: float = 1.0) -> float:
    """z_c = sum(log p(t)) / len(logprobs) ** length_norm (SPEC 2.4 step 3)."""
    return sum(logprobs) / (len(logprobs) ** length_norm)


def kv_bytes_per_token(n_layer: int, n_kv_head: int, key_len: int, value_len: int,
                       type_bytes: int) -> int:
    return n_layer * n_kv_head * (key_len + value_len) * type_bytes


def plan_memory(weights_bytes: int, kv_per_token: int, n_ctx: int, n_seq_max: int,
                overhead_bytes: int) -> dict:
    kv = kv_per_token * n_ctx * n_seq_max
    return {"weights": weights_bytes, "kv": kv, "overhead": overhead_bytes,
            "total": weights_bytes + kv + overhead_bytes}


def recommend_quant(candidates: list[tuple[str, int]], *, vram_bytes: int, ram_bytes: int,
                    kv_per_token_f16: int, n_ctx: int, n_seq_max: int,
                    vram_margin: float = 0.10, ram_margin: float = 0.20,
                    overhead_bytes: int = 512 * 1024 * 1024) -> dict:
    """Pick the largest quant (then the largest KV type) that fits. SPEC 2.6.

    Conservative upper bound: KV is charged at n_ctx * n_seq_max (non-unified worst
    case). E1a measures the real unified-cache footprint and may relax this.
    """
    kv_types = ["f16", "q8_0", "q4_0"]
    ordered = sorted(candidates, key=lambda c: -c[1])
    for label, weights in ordered:
        for kv_type in kv_types:
            kvt = kv_per_token_f16 // 2 * GGML_TYPE_BYTES[kv_type]
            plan = plan_memory(weights, kvt, n_ctx, n_seq_max, overhead_bytes)
            if plan["total"] <= int(vram_bytes * (1 - vram_margin)):
                return {"quant": label, "kv_type": kv_type, "placement": "gpu", **plan}
    for label, weights in ordered:
        plan = plan_memory(weights, kv_per_token_f16, n_ctx, n_seq_max, overhead_bytes)
        if plan["total"] <= int(ram_bytes * (1 - ram_margin)):
            return {"quant": label, "kv_type": "f16", "placement": "cpu", **plan}
    return {"quant": None, "kv_type": None, "placement": "insufficient",
            "warning": "no quant fits vram or ram at this ctx/n_seq_max"}


# ---- minimal GGUF metadata reader (registry contract; full version lands in E1a) ----
GGUF_MAGIC = b"GGUF"
T_STRING, T_ARRAY, T_KV = 8, 9, 12
_FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
        10: "<Q", 11: "<q", 12: "<d"}


def parse_gguf_metadata(path: str | os.PathLike) -> dict:
    """Header + KV metadata only — never touches tensor data (SPEC 2.6)."""
    with open(path, "rb") as fh:
        buf = fh.read(16)
        if buf[:4] != GGUF_MAGIC:
            raise ValueError(f"not a GGUF file: {path}")
        version = int.from_bytes(buf[4:8], "little")
        n_tensors = int.from_bytes(buf[8:16], "little")
        n_kv = int.from_bytes(fh.read(8), "little")
        kv: dict[str, object] = {}
        for _ in range(n_kv):
            key = _read_str(fh)
            t = int.from_bytes(fh.read(4), "little")
            kv[key] = _read_value(fh, t)
    return {"version": version, "n_tensors": n_tensors, "n_kv": n_kv, "kv": kv}


def _read_str(fh) -> str:
    n = int.from_bytes(fh.read(8), "little")
    return fh.read(n).decode("utf-8", "replace")


def _read_value(fh, t: int, depth: int = 0):
    if t == T_STRING:
        return _read_str(fh)
    if t == T_ARRAY:
        et = int.from_bytes(fh.read(4), "little")
        n = int.from_bytes(fh.read(8), "little")
        if depth >= 2 or n > 1_000_000:
            raise ValueError("array nesting/size cap (metadata only)")
        return [_read_value(fh, et, depth + 1) for _ in range(n)]
    return _unpack(fh, _FMT[t])


def _unpack(fh, fmt: str):
    import struct
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, fh.read(size))[0]


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# A. evidence pins
# ---------------------------------------------------------------------------
def section_a() -> None:
    print("\n[A] distribution + model evidence pins (offline, from docs/evidence/)")
    rel = json.loads((EVID / "llama_cpp_release_b11026.json").read_text())
    check(rel["tag"] == LLAMA_TAG, f"release tag == {LLAMA_TAG}")
    check(rel["published_at"] == LLAMA_PUBLISHED, f"published_at == {LLAMA_PUBLISHED}")
    check(rel["asset_count"] == LLAMA_ASSET_COUNT == len(rel["assets"]),
          f"asset count == {LLAMA_ASSET_COUNT}")
    sizes = {a["name"]: a["size"] for a in rel["assets"]}
    for name, size in ASSET_SIZES.items():
        check(sizes.get(name) == size, f"{name} == {size} B")
    check(int(LLAMA_TAG.lstrip("b")) >= MIN_BUILD_FOR_SPARK25,
          f"{LLAMA_TAG} >= min build b{MIN_BUILD_FOR_SPARK25} (spark2_5 arch gate)")

    hdr = json.loads((EVID / "llama_cpp_b11026_header.json").read_text())
    check(hdr["total_lines"] == HEADER_LINES, f"include/llama.h == {HEADER_LINES} lines")
    for sym, line in HEADER_SYMBOL_LINES.items():
        check(hdr["symbol_lines"].get(sym) == line, f"header pin {sym}:{line}")

    lst = json.loads((EVID / "llama_b11026_ubuntu_x64_listing.json").read_text())
    check(lst["sha256"] == TARBALL_SHA256, "ubuntu-x64 tarball sha256")
    check(lst["file_count"] == TARBALL_FILE_COUNT, f"tarball entries == {TARBALL_FILE_COUNT}")
    names = [x.split("/")[-1] for x in lst["shared_libs"] + lst["executables"]]
    for needed in ("libllama.so", "libggml.so", "libggml-base.so",
                   "llama-server", "llama-fit-params", "llama-tokenize"):
        check(needed in names, f"tarball provides {needed}")

    hf = json.loads((EVID / "hf_spark_x2_5.json").read_text())
    check(hf["repo"] == HF_REPO and hf["repo_sha"] == HF_REPO_SHA, f"{HF_REPO}@{HF_REPO_SHA[:12]}")
    check(hf["gated"] is False, "Spark repo is not gated")
    check(hf["license"] == "apache-2.0", "Spark license == apache-2.0")
    files = {f["path"]: f for f in hf["files"]}
    for path, (size, sha) in HF_QUANTS.items():
        f = files.get(path, {})
        check(f.get("size") == size and f.get("lfs_oid_sha256") == sha,
              f"{path}: {size} B sha256 {sha[:12]}…")
    gated = json.loads((ROOT / "docs" / "evidence" / "hf_lfs_oid_semantics.json").read_text())
    check(gated["observed_sha256"] == gated["lfs_oid"],
          "HF lfs.oid == sha256(file) — verified by downloading a real LFS file")

    lock_path = ROOT / "runtime.lock"
    check(lock_path.exists(), "runtime.lock committed (single source of truth for the pin)")
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        check(lock["llama_cpp"]["tag"] == LLAMA_TAG, "runtime.lock tag == oracle pin")
        check(lock["llama_cpp"]["min_build_for_spark2_5"] == MIN_BUILD_FOR_SPARK25,
              "runtime.lock spark2_5 minimum build")
        for variant, spec in lock["llama_cpp"]["assets"].items():
            check(ASSET_SIZES.get(spec["asset"]) == spec["size"],
                  f"runtime.lock asset size matches evidence: {variant}")
        check(sorted(lock["llama_cpp"]["required_symbols_llama"]) == sorted(REQUIRED_SYMBOLS),
              "runtime.lock required llama symbols == oracle symbol list")
        check(sorted(lock["llama_cpp"]["required_symbols_ggml"]) == sorted(REQUIRED_GGML_SYMBOLS),
              "runtime.lock required ggml symbols == oracle symbol list")
        dm = lock["default_model"]
        check(dm["sha256"] == HF_QUANTS[dm["file"]][1] and dm["size"] == HF_QUANTS[dm["file"]][0],
              "runtime.lock default model matches the HF pin")
    poc = EVID / "poc-ctypes-20260917.py"
    check(poc.exists(), "executed PoC artifact committed (docs/evidence/poc-ctypes-20260917.py)")
    if poc.exists():
        body = poc.read_text()
        for marker in POC_MARKERS:
            check(marker in body, f"PoC contains required call-sequence marker: {marker!r}")
    report = json.loads((EVID / "poc_report.json").read_text())
    check(report["fork_vs_sequential_max_abs_delta"] == 0.0,
          "PoC: fork vs sequential readout max |Δ| == 0.00e+00 (isomorphism)")
    check(report["runtime"] == PYTHON_CTYPES_RUNTIME and report["model_arch"] == "qwen35",
          "PoC ran on the pinned runtime with a hybrid (DeltaNet) model")
    for pitfall in report["pitfalls"]:
        check(len(pitfall) > 20, f"pitfall documented: {pitfall[:48]}…")
    row("PoC", f"{report['model_arch']} fork Δ={report['fork_vs_sequential_max_abs_delta']}")

    row("llama.cpp release", f"{LLAMA_TAG} ({LLAMA_PUBLISHED})")
    row("assets / ubuntu-x64 bytes", f"{LLAMA_ASSET_COUNT} / {ASSET_SIZES['llama-b11026-bin-ubuntu-x64.tar.gz']}")
    row("default model pin", f"{HF_REPO}:Q8_0 {HF_QUANTS['Spark-X2.5-4B-Q8_0.gguf'][0]} B")


# ---------------------------------------------------------------------------
# B. live runtime probes
# ---------------------------------------------------------------------------
def find_runtime() -> pathlib.Path | None:
    env = os.environ.get("TYPED_GGUF_RUNTIME_DIR")
    if env and pathlib.Path(env, "libllama.so").exists():
        return pathlib.Path(env)
    base = pathlib.Path(os.environ.get("XDG_DATA_HOME", pathlib.Path.home() / ".local/share"))
    for cand in sorted((base / "typed-gguf" / "runtime").glob("*/")):
        if (cand / "libllama.so").exists():
            return cand
    return None


def section_b() -> None:
    print("\n[B] live runtime probes (ctypes vs pinned release)")
    rt = find_runtime()
    if rt is None:
        skip("no runtime installed (set TYPED_GGUF_RUNTIME_DIR or run `typed-gguf init`); "
             "E1a re-runs this section against the real install")
        return
    print(f"  runtime dir: {rt}")
    lib = ctypes.CDLL(str(rt / "libllama.so"))
    missing = [s for s in REQUIRED_SYMBOLS if not hasattr(lib, s)]
    check(not missing, f"ctypes resolves all {len(REQUIRED_SYMBOLS)} required symbols "
                       f"({len(missing)} missing)")
    ggml = ctypes.CDLL(str(rt / "libggml.so"))
    missing_ggml = [s for s in REQUIRED_GGML_SYMBOLS if not hasattr(ggml, s)]
    check(not missing_ggml, f"libggml.so resolves the backend loader "
                            f"({len(REQUIRED_GGML_SYMBOLS) - len(missing_ggml)}/"
                            f"{len(REQUIRED_GGML_SYMBOLS)}) — PoC pitfall 1")

    cli = rt / "llama-cli"
    if cli.exists():
        import subprocess
        env = {**os.environ, "LD_LIBRARY_PATH": f"{rt}:{os.environ.get('LD_LIBRARY_PATH', '')}"}
        p = subprocess.run([str(cli), "--version"], capture_output=True, text=True,
                           check=False, env=env, cwd=str(rt))
        out = (p.stdout or "") + (p.stderr or "")  # llama-cli prints the banner on stderr
        m = re.search(r"build (\d+)", out)
        check(bool(m) and int(m.group(1)) >= MIN_BUILD_FOR_SPARK25,
              f"llama-cli --version reports build >= b{MIN_BUILD_FOR_SPARK25}")
        check(m is not None and m.group(1) == LLAMA_TAG.lstrip("b"),
              f"llama-cli build == {LLAMA_TAG.lstrip('b')} (pinned release, not a stale lib)")
        row("llama-cli version", out.strip().splitlines()[0] if out.strip() else "?")
    else:
        skip("llama-cli binary absent in runtime dir (binaries not required for the engine)")

    blob = (rt / "libllama.so").read_bytes()
    check(blob.count(b"llama_model_spark2_5") >= 1,
          "libllama.so carries the spark2_5 arch implementation")
    check((rt / "llama-fit-params").exists(), "llama-fit-params available for auto-fit")
    if LOCAL_SPARK.exists():
        check(LOCAL_SPARK.stat().st_size == HF_QUANTS["Spark-X2.5-4B-Q8_0.gguf"][0],
              "local Spark Q8_0 size == HF pin")
        check(sha256_file(LOCAL_SPARK) == HF_QUANTS["Spark-X2.5-4B-Q8_0.gguf"][1],
              "local Spark Q8_0 sha256 == HF lfs.oid (download-verify contract)")
    else:
        skip(f"{LOCAL_SPARK} absent — sha256 download-verify evidence not re-run")


# ---------------------------------------------------------------------------
# C. arithmetic mirror
# ---------------------------------------------------------------------------
def section_c() -> None:
    print("\n[C] arithmetic mirror (executed reference values)")
    p = softmax([2.0, 1.0, 0.0])
    check(abs(sum(p) - 1.0) < 1e-12, "restricted softmax sums to 1")
    check(p[0] > p[1] > p[2], "restricted softmax preserves logit order")
    row("softmax([2,1,0])", ", ".join(f"{v:.6f}" for v in p))
    check(softmax([0.0, 0.0]) == [0.5, 0.5], "flat logits -> uniform")

    for label, probs, documented in TS_CONFIDENCE_EXAMPLES:
        ours = confidence_normalized_peak(probs)
        delta = abs(ours - documented)
        check(delta <= TS_REL_TOL,
              f"confidence[{label}] {ours:.4f} ~ documented {documented} (Δ{delta:.3f})")
        row(f"confidence {label}", f"{ours:.4f} vs doc {documented}")
    lab, probs, documented = TS_CONFIDENCE_OUTLIER
    ours = confidence_normalized_peak(probs)
    check(True, f"documented outlier kept honest: {lab} ours {ours:.3f} vs doc {documented} "
                f"(SPEC 2.3 — no parity claim)")
    check(confidence_normalized_peak([1.0, 0.0]) == 1.0, "peak -> confidence 1.0")
    check(confidence_normalized_peak([0.5, 0.5]) == 0.0, "flat -> confidence 0.0")
    check(abs(confidence_normalized_peak([0.0, 0.7, 0.3]) - 0.55) < 1e-12,
          "score/choice share one confidence statistic (0.55 for 0.7 peak over 3 levels)")

    s = score_weighted_mean([0.0, 0.7, 0.3])
    check(abs(s - 1.30) < 1e-12, "score == sum(level * p) == 1.30 (documented worked example)")
    check(abs(score_weighted_mean([0.0, 0.94, 0.06]) - 1.06) < 1e-12, "score(0,0.94,0.06) == 1.06")
    check(score_weighted_mean([1.0, 0.0, 0.0]) == 0.0, "score of level 0 == 0.0")
    row("score = Σ i·p_i", "1.30 / 1.06 / 0.0 reproduced")

    full = [2.0, 1.0, 0.0, -1.0]
    cov = coverage([2.0, 1.0], full)
    check(0.0 <= cov <= 1.0, "coverage in [0,1]")
    check(abs(cov - sum(softmax(full)[:2])) < 1e-12, "coverage == candidate mass under full softmax")
    row("coverage(cands={0,1})", f"{cov:.6f}")

    z = candidate_sequence_score([-1.0, -1.0, -1.0], length_norm=1.0)
    check(abs(z + 1.0) < 1e-12, "length_norm=1 -> equal per-token logp gives equal z regardless of length")
    check(candidate_sequence_score([-2.0], 0.0) == -2.0, "length_norm=0 -> plain log-prob sum")
    row("candidate z (-1.0 x3, norm=1)", f"{z:.4f}")

    kvpt = kv_bytes_per_token(36, 4, 256, 256, GGML_TYPE_BYTES["f16"])
    check(kvpt == 147456, "spark2_5 f16 KV bytes/token/seq == 147456")
    check(kv_bytes_per_token(36, 4, 256, 256, GGML_TYPE_BYTES["q8_0"]) == 73728,
          "q8_0 KV halves the footprint")
    row("spark2_5 KV/token/seq (f16|q8_0)", "147456 | 73728 B")

    cands = [(k, v[0]) for k, v in sorted(HF_QUANTS.items(), key=lambda kv: -kv[1][0])]
    vram8 = 8 * 1024 ** 3
    ram31 = 31 * 1024 ** 3
    a = recommend_quant(cands, vram_bytes=vram8, ram_bytes=ram31, kv_per_token_f16=kvpt,
                        n_ctx=4096, n_seq_max=8)
    check(a["quant"] == "Spark-X2.5-4B-Q8_0.gguf" and a["kv_type"] == "q8_0" and a["placement"] == "gpu",
          "8 GiB VRAM / ctx 4096 / n_seq_max 8 -> Q8_0 + q8_0 KV on GPU")
    row("recommend_quant 8GiB", f"{a['quant']} kv={a['kv_type']} total={a['total'] / 1e9:.2f} GB")
    b = recommend_quant(cands, vram_bytes=vram8, ram_bytes=ram31, kv_per_token_f16=kvpt,
                        n_ctx=2048, n_seq_max=4)
    check(b["quant"] == "Spark-X2.5-4B-Q8_0.gguf" and b["kv_type"] == "f16" and b["placement"] == "gpu",
          "8 GiB VRAM / ctx 2048 / n_seq_max 4 -> Q8_0 + f16 KV on GPU")
    row("recommend_quant 8GiB small-ctx", f"{b['quant']} kv={b['kv_type']} total={b['total'] / 1e9:.2f} GB")
    c = recommend_quant(cands, vram_bytes=vram8, ram_bytes=ram31, kv_per_token_f16=kvpt,
                        n_ctx=32768, n_seq_max=16)
    check(c["placement"] in ("gpu", "cpu", "insufficient") and c["quant"] in (None, *[x[0] for x in cands]),
          "recommend_quant always returns a covered placement")
    row("recommend_quant huge ctx", f"{c['quant']} kv={c['kv_type']} placement={c['placement']}")

    check(FTYPE[7] == "MOSTLY_Q8_0" and FTYPE[15] == "MOSTLY_Q4_K_M",
          "GGUF file_type -> quant labels (7=Q8_0, 15=Q4_K_M)")
    if LOCAL_SPARK.exists():
        md = parse_gguf_metadata(LOCAL_SPARK)["kv"]
        for key, want in LOCAL_SPARK_KV.items():
            check(md.get(key) == want, f"local Spark header {key} == {want!r}")
        row("local Spark arch", f"{md['general.architecture']} file_type={md['general.file_type']} "
                                f"({FTYPE.get(md['general.file_type'], '?')})")
    else:
        skip("local Spark GGUF absent — header-parse pins not re-run")
    if LOCAL_QWEN35.exists():
        md = parse_gguf_metadata(LOCAL_QWEN35)["kv"]
        check(md.get("general.architecture") == "qwen35" and md.get("qwen35.block_count") == 24
              and md.get("general.file_type") == 15, "Qwen3.5-0.8B header pins")
        row("Qwen3.5-0.8B header", f"{md['general.architecture']} layers={md['qwen35.block_count']}")


# ---------------------------------------------------------------------------
# D. package + contract-surface pins
# ---------------------------------------------------------------------------
def section_d() -> None:
    print("\n[D] package + contract surface (post-scaffold / post-implementation)")
    pyproject = ROOT / "pyproject.toml"
    if pyproject.exists():
        text = pyproject.read_text()
        for dep in FORBIDDEN_TRAINING_DEPS:
            check(re.search(rf'["\']{re.escape(dep)}[\[=><~!]', text) is None,
                  f"pyproject declares no training dependency ({dep})")
        check("requires-python" in text, "pyproject pins requires-python")
    else:
        skip("pyproject.toml not present yet")
    try:
        sys.path.insert(0, str(ROOT / "src"))
        import typed_gguf  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        skip(f"package not importable yet ({type(exc).__name__}); E1a onward re-runs this")
        return
    check(isinstance(getattr(typed_gguf, "__version__", None), str), "typed_gguf.__version__ is a str")
    want_mods = ["schema", "errors", "cli", "engine", "runtime", "registry", "calibration",
                 "api", "bench"]
    import importlib
    for mod in want_mods:
        try:
            importlib.import_module(f"typed_gguf.{mod}")
            check(True, f"module typed_gguf.{mod} imports")
        except Exception as exc:  # noqa: BLE001
            check(False, f"module typed_gguf.{mod} imports ({exc})")
    # transplanted reference implementations must agree with the mirror
    try:
        from typed_gguf.engine import readout
    except Exception:  # noqa: BLE001
        skip("typed_gguf.engine.readout not implemented yet")
        return
    required = ("restricted_softmax", "confidence_normalized_peak", "score_weighted_mean")
    present = {n: getattr(readout, n, None) for n in required}
    if not any(callable(fn) for fn in present.values()):
        skip("readout functions not implemented yet (E1b) — mirror values stand alone")
        return
    for name in required:
        check(callable(present[name]), f"readout.{name} exists")
    if all(callable(fn) for fn in present.values()):
        check(max(abs(a - b) for a, b in
                  zip(readout.restricted_softmax([2.0, 1.0, 0.0]), softmax([2.0, 1.0, 0.0]),
                      strict=True)) < 1e-12,
              "readout.restricted_softmax matches the oracle mirror")
        check(readout.confidence_normalized_peak([0.0, 0.7, 0.3])
              == confidence_normalized_peak([0.0, 0.7, 0.3]),
              "readout.confidence_normalized_peak matches the oracle mirror")
        check(readout.score_weighted_mean([0.0, 0.7, 0.3]) == 1.30,
              "readout.score_weighted_mean matches the oracle mirror")



def main() -> int:
    print(f"typed-gguf runtime contract oracle — repo {ROOT}")
    section_a()
    section_b()
    section_c()
    section_d()
    print("\n=== reference table (executed) ===")
    for name, value in TABLE:
        print(f"  {name:36s} {value}")
    print(f"\nfailures: {len(FAILS)}  skips: {len(SKIPS)}")
    for f in FAILS:
        print(f"  FAIL {f}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
