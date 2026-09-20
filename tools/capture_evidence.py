#!/usr/bin/env python3
"""Capture spec-time evidence for the typed-gguf runtime contract.

Writes docs/evidence/*.json into the repo. Re-runnable; network required.
Run once at spec time; the oracle (docs/verify_runtime_contract.py) then reads
these files OFFLINE so its pins stay reproducible without a network.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import urllib.request

REPO = pathlib.Path(__file__).resolve().parents[1]
EVID = REPO / "docs" / "evidence"
EVID.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "typed-gguf-evidence/0.1"}


def get_json(url: str, accept: str = "application/json"):
    req = urllib.request.Request(url, headers={**UA, "Accept": accept})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read().decode())


def get_text(url: str) -> str:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=90) as r:
        return r.read().decode()


def capture_llama_release() -> None:
    tag = "b11026"
    d = get_json(f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{tag}",
                 "application/vnd.github+json")
    assets = sorted(
        ({"name": a["name"], "size": a["size"], "browser_download_url": a["browser_download_url"]}
         for a in d.get("assets", [])),
        key=lambda a: a["name"],
    )
    out = {
        "captured_at": "2026-09-17",
        "source": f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{tag}",
        "tag": d["tag_name"],
        "published_at": d["published_at"],
        "asset_count": len(assets),
        "assets": assets,
    }
    (EVID / "llama_cpp_release_b11026.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"llama_cpp_release_b11026.json: {len(assets)} assets, published {d['published_at']}")


def capture_llama_header() -> None:
    tag = "b11026"
    text = get_text(f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{tag}/include/llama.h")
    lines = text.splitlines()
    syms = [
        "llama_get_memory", "llama_n_seq_max", "llama_memory_seq_cp", "llama_memory_seq_rm",
        "llama_memory_seq_keep", "llama_state_seq_save_file", "llama_state_seq_load_file",
        "llama_state_seq_get_size", "llama_batch_init", "llama_decode", "llama_get_logits_ith",
        "llama_tokenize", "llama_detokenize", "llama_vocab_n_tokens", "llama_chat_apply_template",
        "llama_model_load_from_file", "llama_init_from_model", "llama_model_meta_val_str",
        "llama_model_chat_template", "llama_model_desc", "llama_model_n_params",
        "llama_print_system_info",
    ]
    pins = {}
    for s in syms:
        for i, line in enumerate(lines, 1):
            if s in line and "LLAMA_API" in line:
                pins[s] = i
                break
    out = {
        "captured_at": "2026-09-17",
        "source": f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{tag}/include/llama.h",
        "tag": tag,
        "total_lines": len(lines),
        "symbol_lines": pins,
    }
    (EVID / "llama_cpp_b11026_header.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"llama_cpp_b11026_header.json: {len(lines)} lines, {len(pins)} symbol pins")


def capture_hf_model() -> None:
    repo = "XHToken/Spark-X2.5-4B-GGUF"
    meta = get_json(f"https://huggingface.co/api/models/{repo}")
    tree = get_json(f"https://huggingface.co/api/models/{repo}/tree/main?recursive=1")
    files = []
    for e in tree:
        if e.get("type") != "file":
            continue
        rec = {"path": e["path"], "size": e.get("size")}
        if "lfs" in e:
            rec["lfs_oid_sha256"] = e["lfs"].get("oid")
            rec["lfs_size"] = e["lfs"].get("size")
            rec["xet_hash"] = e.get("xetHash")
        files.append(rec)
    files.sort(key=lambda r: r["path"])
    out = {
        "captured_at": "2026-09-17",
        "source": f"https://huggingface.co/api/models/{repo}",
        "repo": repo,
        "repo_sha": meta["sha"],
        "last_modified": meta["lastModified"],
        "gated": meta["gated"],
        "license": meta.get("cardData", {}).get("license"),
        "tags": meta.get("tags", []),
        "files": files,
    }
    (EVID / "hf_spark_x2_5.json").write_text(json.dumps(out, indent=1) + "\n")
    ggufs = [f for f in files if f["path"].endswith(".gguf")]
    print("hf_spark_x2_5.json:", json.dumps(ggufs, indent=1))


def capture_tarball_listing() -> None:
    """File listing + tarball sha256 for the release asset we distribute on this host."""
    local = pathlib.Path("/tmp/typed_gguf_probe/rel/llama-b11026-bin-ubuntu-x64.tar.gz")
    if not local.exists():
        print("tarball not present locally; skipping listing capture")
        return
    sha = subprocess.run(["sha256sum", str(local)], capture_output=True, text=True, check=True)
    listing = subprocess.run(["tar", "tzf", str(local)], capture_output=True, text=True, check=True)
    names = sorted(x.strip() for x in listing.stdout.splitlines() if x.strip())
    out = {
        "captured_at": "2026-09-17",
        "asset": "llama-b11026-bin-ubuntu-x64.tar.gz",
        "sha256": sha.stdout.split()[0],
        "size_bytes": local.stat().st_size,
        "file_count": len(names),
        "shared_libs": sorted(n for n in names if n.endswith(".so") or ".so." in n),
        "executables": sorted(n for n in names if "/" in n and not n.endswith(".so")
                              and not any(p in n for p in (".so.", "/include/", "/lib/", ".h"))),
    }
    (EVID / "llama_b11026_ubuntu_x64_listing.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"tarball listing: {len(names)} entries, sha256 {out['sha256'][:16]}…, "
          f"{len(out['shared_libs'])} shared libs")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "llama"):
        capture_llama_release()
        capture_llama_header()
    if which in ("all", "hf"):
        capture_hf_model()
    if which in ("all", "tarball"):
        capture_tarball_listing()
