"""Live download tests against the pinned HF repo (A-E1a-4).

Marked `network`: they are skipped unless `--run-network` is passed. They use the *smallest*
LFS object of the pinned repo so they stay cheap, while exercising exactly the production
code path (`hf.model_info` -> tree/lfs.oid -> Range-resume download -> SHA-256 verify).

The 4.38 GB default-model run (kill at ~10%, resume, `sha256 == lfs.oid`) is recorded in
docs/evidence/e1a_baseline.json (`pull_resume`) because re-downloading 4.4 GB per test run is
not something a unit gate should do.
"""
from __future__ import annotations

import hashlib
import pathlib
import urllib.request

import pytest

from typed_gguf.registry import hf

REPO = "XHToken/Spark-X2.5-4B-GGUF"
# pinned in docs/evidence/hf_lfs_oid_semantics.json (verified there by execution)
IMAGE = "images/spark3-hybrid-architecture-light.png"
IMAGE_SHA = "5eddac6baf8e8460cc2a75b96070eff4074428674f3c33665bc2ff9479756bbd"
IMAGE_SIZE = 429_073


def _image_entry() -> hf.FileInfo:
    info = hf.model_info(REPO)
    return next(f for f in info.files if f.path == IMAGE)


@pytest.mark.network
def test_tree_api_reports_the_pinned_oid_for_an_lfs_file() -> None:
    entry = _image_entry()
    assert entry.size == IMAGE_SIZE
    assert entry.oid == IMAGE_SHA  # lfs.oid == sha256(file), asserted live


@pytest.mark.network
def test_download_a_real_lfs_object_and_verify_it(tmp_path: pathlib.Path) -> None:
    entry = _image_entry()
    result = hf.download_file(REPO, entry.path, tmp_path / "image.png",
                              revision=info_sha(), size=entry.size, sha256=entry.oid)
    assert result.verified is True
    assert result.sha256 == IMAGE_SHA
    assert result.bytes_fetched == IMAGE_SIZE
    assert hashlib.sha256((tmp_path / "image.png").read_bytes()).hexdigest() == IMAGE_SHA


@pytest.mark.network
def test_range_resume_against_the_real_server(tmp_path: pathlib.Path) -> None:
    entry = _image_entry()
    dest = tmp_path / "image.png"
    part = dest.with_name(dest.name + ".part")
    half = IMAGE_SIZE // 2
    url = hf.resolve_url(REPO, entry.path, info_sha())
    request = urllib.request.Request(url, headers={"Range": f"bytes=0-{half - 1}"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        part.write_bytes(response.read())
    assert part.stat().st_size == half  # a killed download leaves exactly this

    result = hf.download_file(REPO, entry.path, dest, revision=info_sha(), size=entry.size,
                              sha256=entry.oid)
    assert result.resumed_from == half
    assert result.bytes_fetched == IMAGE_SIZE - half
    assert result.sha256 == IMAGE_SHA
    assert dest.stat().st_size == IMAGE_SIZE


def info_sha() -> str:
    """The pinned commit of the default repo (so the test is reproducible, not `main`)."""
    import json

    root = pathlib.Path(__file__).resolve().parents[1]
    evidence = json.loads((root / "docs" / "evidence" / "hf_spark_x2_5.json").read_text())
    return str(evidence["repo_sha"])


@pytest.mark.network
def test_repo_metadata_reports_the_pinned_sha_and_license() -> None:
    info = hf.model_info(REPO)
    assert info.sha == info_sha()
    assert info.license == "apache-2.0"
    assert info.gated is False
    assert any(f.path == "Spark-X2.5-4B-Q8_0.gguf" and f.size == 4_375_021_152
               for f in info.files)
