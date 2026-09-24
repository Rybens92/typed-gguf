"""Build the offline release fixtures for the `runtime update` story.

Nothing here touches the network: the bundle is re-tarred from the real llama.cpp b11026 tree
that a previous card downloaded (docs/evidence/t07b5/... in this repo), and the release list is
the API's own JSON shape, with the asset names computed by the *app's* own retag rule
(`typed_gguf.runtime.update.retag_asset_name`) so "the newest release carrying this host's
retagged bundle name" has something real to match.

    <venv>/bin/python make_fixtures.py
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import tarfile

from typed_gguf.runtime import pins, update

ROOT = pathlib.Path("/workspace/e2e-t_559ed8c8/fixtures")
TREE = pathlib.Path("/workspace/ggufone/docs/evidence/t07b5/home/runtime/"
                    "b11026-linux-x64-vulkan")
TARGET_TAG = "b99999"
SKIPPED_TAG = "b100001"
#: the one big library we leave out: this fixture plays the `linux-x64-cpu` bundle, and the
#: 42 MB vulkan backend is what the vulkan variant carries, not the cpu one
EXCLUDE = {"libggml-vulkan.so"}


def sha256_file(path: pathlib.Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def build_tarball(name: str) -> pathlib.Path:
    assets = ROOT / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    target = assets / name
    with tarfile.open(target, "w:gz", compresslevel=1) as tar:
        for item in sorted(TREE.iterdir()):
            if item.name in EXCLUDE:
                continue
            tar.add(item, arcname=f"llama-{TARGET_TAG}/{item.name}", recursive=True)
    return target


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = pins.load_lock()
    top = pins.asset_for(lock, "linux-x64-cpu")
    cuda = pins.asset_for(lock, "linux-x64-cuda-12.8")
    vulkan = pins.asset_for(lock, "linux-x64-vulkan")
    cpu_name = update.retag_asset_name(top.asset, TARGET_TAG, pinned_tag=lock.tag)
    cuda_name = update.retag_asset_name(cuda.asset, TARGET_TAG, pinned_tag=lock.tag)
    vulkan_name = update.retag_asset_name(vulkan.asset, TARGET_TAG, pinned_tag=lock.tag)
    print(f"pinned tag {lock.tag}; cpu asset {top.asset} -> {cpu_name}")
    print(f"cuda asset -> {cuda_name}; vulkan asset -> {vulkan_name}")

    tarball = build_tarball(cpu_name)
    size = tarball.stat().st_size
    digest = sha256_file(tarball)
    print(f"bundle: {tarball.name} {size} bytes sha256 {digest}")

    # two extra targets with the same bytes: the `kill -9` leg and the negative download legs need
    # a *fresh* target so the update has to download instead of adopting a bundle already on disk
    kill_tag = "b99998"
    neg_tag = "b99997"
    tags = {}
    for tag in (kill_tag, neg_tag):
        name = update.retag_asset_name(top.asset, tag, pinned_tag=lock.tag)
        target = ROOT / "assets" / name
        if target.exists():
            target.unlink()
        try:
            os.link(tarball, target)
        except OSError:
            shutil.copy2(tarball, target)
        tags[tag] = name
        print(f"extra bundle for {tag}: {target.name} (same bytes)")
    kill_name = tags[kill_tag]
    neg_name = tags[neg_tag]

    skipped_name = update.retag_asset_name(top.asset, SKIPPED_TAG, pinned_tag=lock.tag)
    releases = [
        # the real API's shape: newest first, and a milestone tag that carries no bundle at all
        {"tag_name": "v0.5.0", "published_at": "2026-09-24T09:00:00Z", "prerelease": False,
         "assets": [{"name": "nightly-tag.txt", "size": 12,
                     "digest": "sha256:" + "1" * 64}]},
        # newer than the target and genuinely asset-bearing — but not for this host
        {"tag_name": SKIPPED_TAG, "published_at": "2026-09-24T08:00:00Z", "prerelease": True,
         "assets": [{"name": skipped_name.replace("ubuntu-x64", "win-cpu-x64"), "size": 18439911,
                     "digest": "sha256:" + "2" * 64}]},
        {"tag_name": TARGET_TAG, "published_at": "2026-09-24T07:00:00Z", "prerelease": True,
         "assets": [
             {"name": cpu_name, "size": size, "digest": "sha256:" + digest},
             # present in the listing so the cuda leg gets *past* target resolution — the
             # pre-flight is what has to refuse it, before a byte is fetched
             {"name": cuda_name, "size": cuda.size,
              "digest": "sha256:" + "3" * 64},
             {"name": vulkan_name, "size": vulkan.size,
              "digest": "sha256:" + "4" * 64},
         ]},
        # the `kill -9` leg's target: only reachable with `--tag`, same bundle bytes
        {"tag_name": kill_tag, "published_at": "2026-09-24T06:00:00Z", "prerelease": True,
         "assets": [{"name": kill_name, "size": size, "digest": "sha256:" + digest}]},
        # the negative download legs' target: also only reachable with `--tag`
        {"tag_name": neg_tag, "published_at": "2026-09-24T05:00:00Z", "prerelease": True,
         "assets": [{"name": neg_name, "size": size, "digest": "sha256:" + digest}]},
    ]
    (ROOT / "releases.json").write_text(json.dumps(releases, indent=1) + "\n")
    (ROOT / "scenario.json").write_text(json.dumps(
        {"releases": "ok", "asset": "ok", "throttle_ms": 0}, indent=1) + "\n")
    print(f"releases.json: {[r['tag_name'] for r in releases]}")
    print(f"scenario.json: {(ROOT / 'scenario.json').read_text().strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
