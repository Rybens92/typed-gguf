"""HuggingFace resolve/search/download: resume, SHA verify, auth, offline snapshot.

Offline by construction: the HTTP seam (`hf._open`) is faked, so every byte in these tests is
served from memory and the Range/verification logic is still the production one.
"""
from __future__ import annotations

import hashlib
import io
import json
import pathlib
import urllib.error

import pytest

from ggufone.errors import GgufoneError
from ggufone.registry import hf

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_REPO = "XHToken/Spark-X2.5-4B-GGUF"


# ------------------------------------------------------------------ fake server
class FakeServer:
    """Minimal HTTP response stand-in with Range support (same surface as HTTPResponse)."""

    def __init__(self, payload: bytes, *, honor_range: bool = True, status: int = 200,
                 body_limit: int | None = None) -> None:
        self.payload = payload
        self.honor_range = honor_range
        self.status_override = status
        self.body_limit = body_limit
        self.requests: list[dict[str, str]] = []

    def __call__(self, url: str, headers: dict[str, str], timeout: float = 60.0):
        self.requests.append({"url": url, "headers": dict(headers)})
        start = 0
        rng = headers.get("Range")
        if rng and rng.startswith("bytes=") and self.honor_range:
            start = int(rng.split("=", 1)[1].split("-", 1)[0])
        body = self.payload[start:]
        if self.body_limit is not None:
            body = body[: self.body_limit]
        status = self.status_override if start == 0 else 206
        server = self

        class _Response(io.BytesIO):
            def __init__(self) -> None:
                super().__init__(body)
                self.status = status
                self.headers = {"Content-Length": str(len(body))}

            def __enter__(self):
                return self

            def __exit__(self, *exc) -> None:
                server.close()

        return _Response()

    def close(self) -> None:
        pass


def file_entry(path: str, payload: bytes) -> dict[str, object]:
    """Tree-API shape: non-LFS files carry `size`, LFS files carry `lfs.oid` (== sha256)."""
    return {"path": path, "size": len(payload),
            "lfs": {"oid": hashlib.sha256(payload).hexdigest(), "size": len(payload)}}


# ------------------------------------------------------------------ tokens
def test_token_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf_primary")
    assert hf.token() == "hf_primary"


def test_token_prefers_hf_token_then_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_secondary")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    assert hf.token() == "hf_secondary"


def test_token_from_the_hub_cache_file(monkeypatch: pytest.MonkeyPatch,
                                       tmp_path: pathlib.Path) -> None:
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("HF_HOME", raising=False)
    cache = tmp_path / ".cache" / "huggingface"
    cache.mkdir(parents=True)
    (cache / "token").write_text("hf_from_cache\n")
    assert hf.token() == "hf_from_cache"


def test_token_absent_is_none(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_HOME"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert hf.token() is None


# ------------------------------------------------------------------ model info
def test_model_info_maps_the_metadata_and_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"x" * 1024

    def fake(url: str, *, token=None, timeout: float = 60.0):
        if url.endswith("/tree/main?recursive=1"):
            return [{"type": "file", **file_entry("m-Q8_0.gguf", payload)},
                    {"type": "file", "path": "README.md", "size": 3},
                    {"type": "directory", "path": "images"}]
        return {"repo": DEFAULT_REPO, "sha": "deadbeef", "gated": False,
                "cardData": {"license": "apache-2.0"}, "tags": ["gguf"]}

    monkeypatch.setattr(hf, "fetch_json", fake)
    info = hf.model_info(DEFAULT_REPO)
    assert info.repo == DEFAULT_REPO
    assert info.sha == "deadbeef"
    assert info.license == "apache-2.0"
    assert info.gated is False
    assert [f.path for f in info.files] == ["m-Q8_0.gguf", "README.md"]
    assert info.files[0].oid == hashlib.sha256(payload).hexdigest()
    assert info.files[1].oid is None


def test_model_info_uses_the_committed_snapshot_offline() -> None:
    info = hf.model_info(DEFAULT_REPO, offline=True)
    evidence = json.loads((ROOT / "docs" / "evidence" / "hf_spark_x2_5.json").read_text())
    assert info.sha == evidence["repo_sha"]
    assert info.license == "apache-2.0"
    assert {f.path for f in info.files} >= {"Spark-X2.5-4B-Q8_0.gguf"}
    q8_evidence = next(f for f in evidence["files"]
                       if f["path"] == "Spark-X2.5-4B-Q8_0.gguf")
    q8 = next(f for f in info.files if f.path == "Spark-X2.5-4B-Q8_0.gguf")
    assert q8.size == 4_375_021_152
    assert q8.oid == q8_evidence["lfs_oid_sha256"]
    assert info.source == "snapshot"


def test_model_info_offline_for_an_unknown_repo_is_an_error() -> None:
    with pytest.raises(GgufoneError) as exc:
        hf.model_info("someone/other-model", offline=True)
    assert exc.value.code == "E_DOWNLOAD_FAILED"
    assert "\n" not in str(exc.value)


def test_model_info_maps_401_to_hf_auth_required(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(url: str, *, token=None, timeout: float = 60.0):
        raise urllib.error.HTTPError(url, 401, "Unauthorized", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr(hf, "fetch_json", boom)
    with pytest.raises(GgufoneError) as exc:
        hf.model_info("acme/private-gguf")
    assert exc.value.code == "E_HF_AUTH_REQUIRED"
    msg = str(exc.value)
    assert "HF_TOKEN" in msg and "acme/private-gguf" in msg


def test_model_info_maps_403_to_hf_auth_required(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(url: str, *, token=None, timeout: float = 60.0):
        raise urllib.error.HTTPError(url, 403, "Forbidden", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr(hf, "fetch_json", boom)
    with pytest.raises(GgufoneError) as exc:
        hf.model_info("acme/gated-gguf")
    assert exc.value.code == "E_HF_AUTH_REQUIRED"
    assert "gated" in str(exc.value)


def test_model_info_maps_a_dead_network_to_download_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(url: str, *, token=None, timeout: float = 60.0):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(hf, "fetch_json", boom)
    with pytest.raises(GgufoneError) as exc:
        hf.model_info("acme/model")
    assert exc.value.code == "E_DOWNLOAD_FAILED"
    assert "offline" in str(exc.value).lower()


def test_search_filters_to_gguf_repos(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(url: str, *, token=None, timeout: float = 60.0):
        assert "search=" in url
        return [{"id": "acme/gguf-a", "downloads": 5, "tags": ["gguf"]},
                {"id": "acme/text-only", "downloads": 7, "tags": ["text-generation"]}]

    monkeypatch.setattr(hf, "fetch_json", fake)
    got = hf.search("gguf-a")
    assert [r["id"] for r in got] == ["acme/gguf-a"]


# ------------------------------------------------------------------ download / resume
def test_download_writes_the_file_and_verifies_the_hash(monkeypatch: pytest.MonkeyPatch,
                                                        tmp_path: pathlib.Path) -> None:
    payload = b"gguf" * 5000
    server = FakeServer(payload)
    monkeypatch.setattr(hf, "_open", server)
    dest = tmp_path / "m.gguf"
    result = hf.download_file(DEFAULT_REPO, "m.gguf", dest, revision="main",
                              size=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    assert dest.read_bytes() == payload
    assert result.bytes_fetched == len(payload)
    assert result.resumed_from == 0
    assert result.verified is True
    assert not dest.with_name(dest.name + ".part").exists()


def test_download_resumes_from_the_part_file(monkeypatch: pytest.MonkeyPatch,
                                             tmp_path: pathlib.Path) -> None:
    payload = b"abcdefghij" * 1000
    server = FakeServer(payload)
    monkeypatch.setattr(hf, "_open", server)
    dest = tmp_path / "m.gguf"
    part = dest.with_name(dest.name + ".part")
    part.write_bytes(payload[:4000])
    result = hf.download_file(DEFAULT_REPO, "m.gguf", dest, revision="main",
                              size=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    assert dest.read_bytes() == payload
    assert result.resumed_from == 4000
    assert result.bytes_fetched == len(payload) - 4000
    assert server.requests[-1]["headers"].get("Range") == "bytes=4000-"


def test_download_restarts_when_the_server_ignores_the_range(monkeypatch: pytest.MonkeyPatch,
                                                             tmp_path: pathlib.Path) -> None:
    payload = b"0123456789" * 500
    server = FakeServer(payload, honor_range=False)
    monkeypatch.setattr(hf, "_open", server)
    dest = tmp_path / "m.gguf"
    part = dest.with_name(dest.name + ".part")
    part.write_bytes(b"stale-partial")
    result = hf.download_file(DEFAULT_REPO, "m.gguf", dest, revision="main", size=len(payload),
                              sha256=hashlib.sha256(payload).hexdigest())
    assert dest.read_bytes() == payload
    assert result.resumed_from == 0
    assert result.bytes_fetched == len(payload)


def test_download_detects_a_truncated_body(monkeypatch: pytest.MonkeyPatch,
                                           tmp_path: pathlib.Path) -> None:
    payload = b"z" * 9000
    server = FakeServer(payload, body_limit=4000)
    monkeypatch.setattr(hf, "_open", server)
    with pytest.raises(GgufoneError) as exc:
        hf.download_file(DEFAULT_REPO, "m.gguf", tmp_path / "m.gguf", revision="main",
                         size=len(payload))
    assert exc.value.code == "E_DOWNLOAD_FAILED"
    assert "9000" in str(exc.value)
    assert not (tmp_path / "m.gguf").exists()


def test_download_rejects_a_sha_mismatch_and_keeps_the_part_file(monkeypatch: pytest.MonkeyPatch,
                                                                 tmp_path: pathlib.Path) -> None:
    payload = b"q" * 2048
    monkeypatch.setattr(hf, "_open", FakeServer(payload))
    dest = tmp_path / "m.gguf"
    with pytest.raises(GgufoneError) as exc:
        hf.download_file(DEFAULT_REPO, "m.gguf", dest, revision="main", size=len(payload),
                         sha256="0" * 64)
    assert exc.value.code == "E_SHA256_MISMATCH"
    assert not dest.exists()
    assert dest.with_name(dest.name + ".part").exists()  # resumable, not deleted


def test_download_no_verify_records_the_unverified_hash(monkeypatch: pytest.MonkeyPatch,
                                                        tmp_path: pathlib.Path) -> None:
    payload = b"u" * 1024
    monkeypatch.setattr(hf, "_open", FakeServer(payload))
    result = hf.download_file(DEFAULT_REPO, "m.gguf", tmp_path / "m.gguf", revision="main",
                              size=len(payload), sha256="0" * 64, no_verify=True)
    assert result.verified is False
    assert result.sha256 == hashlib.sha256(payload).hexdigest()


def test_download_sends_the_token_and_resolves_the_url(monkeypatch: pytest.MonkeyPatch,
                                                       tmp_path: pathlib.Path) -> None:
    payload = b"t" * 512
    server = FakeServer(payload)
    monkeypatch.setattr(hf, "_open", server)
    hf.download_file(DEFAULT_REPO, "sub/m.gguf", tmp_path / "m.gguf", revision="abc123",
                     size=len(payload), token="hf_secret")
    assert server.requests[0]["url"] == (
        f"{hf.HF_HOST}/{DEFAULT_REPO}/resolve/abc123/sub/m.gguf")
    assert server.requests[0]["headers"]["Authorization"] == "Bearer hf_secret"


def test_download_progress_callback_sees_the_total(monkeypatch: pytest.MonkeyPatch,
                                                   tmp_path: pathlib.Path) -> None:
    payload = b"p" * 3000
    monkeypatch.setattr(hf, "_open", FakeServer(payload))
    seen: list[tuple[int, int | None]] = []

    def note(done: int, total: int | None) -> None:
        seen.append((done, total))

    hf.download_file(DEFAULT_REPO, "m.gguf", tmp_path / "m.gguf", revision="main",
                     size=len(payload), chunk=1000, progress=note)
    assert seen and seen[-1] == (len(payload), len(payload))


# ------------------------------------------------------------------ disk precheck
def test_check_disk_space_names_both_numbers() -> None:
    with pytest.raises(GgufoneError) as exc:
        hf.check_disk_space("/tmp", 4_375_021_152, free_bytes=int(1.2e9))
    assert exc.value.code == "E_INSUFFICIENT_DISK"
    msg = str(exc.value)
    assert "4.38 GB" in msg and "1.20 GB" in msg


def test_check_disk_space_passes_with_room(tmp_path: pathlib.Path) -> None:
    hf.check_disk_space(tmp_path, 1024, free_bytes=1 << 30)  # must not raise


def test_check_disk_space_reads_the_real_filesystem(tmp_path: pathlib.Path) -> None:
    hf.check_disk_space(tmp_path, 1024)  # must not raise on a real directory


def test_required_bytes_helper_is_shared_with_init() -> None:
    assert hf.human_bytes(4_375_021_152) == "4.38 GB"
    assert hf.human_bytes(16_855_810) == "16.86 MB"
    assert hf.human_bytes(512) == "512 B"
