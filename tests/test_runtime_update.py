"""A-E5-6..9: `runtime update|rollback` (SPEC 2.8) — check, switch, rollback, failure paths.

Everything here is offline and deterministic. The *target* is a fixture release list (the GitHub
API leg lives behind `update.DEFAULT_FETCH`, which these gates replace), the *bundle* is a
synthetic tar with fake libs — the convention of `tests/test_runtime_install.py` — and the
*download* is a `file://` URL, so the real resume/size/SHA-256 leg in `install._obtain_archive`
runs without a socket. The one test that touches api.github.com carries `@pytest.mark.network`
and is skipped by name offline (A-E5-10).

The failure legs (A-E5-8) are the card's real subject: for each one, `runtime.json` must come out
byte-identical, the working runtime must be untouched, and only the staging debris may be gone.
`snapshot()` is what makes that a measurement instead of a promise.

The probe is driven with `deep=False` wherever the bundle is a fake one (a real dlopen of a fake
`libllama.so` cannot resolve the 34 symbols, exactly as with `TYPED_GGUF_DEEP_PROBE=0`); where the
verdict under test *is* the probe's own, the gates inject the same seams the probe's own tests use
(`capability.DEFAULT_SCAN`, `capability.probe_symbols`, `capability.load_backend_library`).
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import tarfile
import urllib.error

import pytest

from tests import conftest as suite_environment
from typed_gguf import __version__, cli
from typed_gguf.errors import TypedGgufError
from typed_gguf.registry import hf
from typed_gguf.runtime import capability, install, pins, update

ROOT = pathlib.Path(__file__).resolve().parents[1]
VARIANT = "linux-x64-cpu"
VULKAN_VARIANT = "linux-x64-vulkan"
CUDA_VARIANT = "linux-x64-cuda-12.8"
PINNED_TAG = "b11026"
PINNED_ASSET = "llama-b11026-bin-ubuntu-x64.tar.gz"
CUDA_ASSET = "llama-b11026-bin-ubuntu-cuda-12.8-x64.tar.gz"
NEXT_TAG = "b11160"
NEXT_ASSET = "llama-b11160-bin-ubuntu-x64.tar.gz"
PINNED_BUILD = 11026
NEXT_BUILD = 11160
HOST = pins.fake_host(system="linux", machine="x86_64")
#: A box that *detects* cuda: what the coordinator's host and the reviewer's look like (`nvidia-smi`
#: present, `libcudart.so.12` absent) — and what makes the target-variant rule observable.
GPU_HOST = pins.fake_host(system="linux", machine="x86_64", has_nvidia_smi=True)


# ------------------------------------------------------------------------------- fixtures
def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_bundle(directory: pathlib.Path, *, build: int = PINNED_BUILD,
                 libs: tuple[str, ...] = ("libllama.so", "libggml.so", "libggml-base.so",
                                          "libggml-cpu.so"),
                 arch_symbol: bool = True, tools: tuple[str, ...] = ("llama-fit-params",)) -> None:
    """A tiny stand-in for an extracted llama.cpp bundle (same shape, fake libs)."""
    directory.mkdir(parents=True, exist_ok=True)
    for index, lib in enumerate(libs):
        payload = b"\x7fELF fake\n" + (b"llama_model_spark2_5\x00" if arch_symbol else b"")
        (directory / lib).write_bytes(payload + bytes([index]))
    cli_script = directory / "llama-cli"
    cli_script.write_text(
        f"#!/bin/sh\necho 'version: 0.4.1-dev (build {build}, commit b49650adb)' >&2\n")
    for name in tools:
        (directory / name).write_text("#!/bin/sh\nexit 0\n")
    for exe in (cli_script, *(directory / name for name in tools)):
        exe.chmod(0o755)


def make_archive(tmp_path: pathlib.Path, name: str, *, tag: str = "b11026",
                 **kwargs: object) -> pathlib.Path:
    inner = tmp_path / f"staging-{name}" / f"llama-{tag}"
    write_bundle(inner, **kwargs)                                   # type: ignore[arg-type]
    archive = tmp_path / name
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(inner, arcname=f"llama-{tag}")
    return archive


def fake_lock(tmp_path: pathlib.Path, *, variants: tuple[str, ...] = (VARIANT,),
              tag: str = PINNED_TAG, asset: str = PINNED_ASSET, size: int = 1,
              digest: str | None = None, repo: str | None = "ggml-org/llama.cpp",
              min_build: int = 10828,
              required_files: tuple[str, ...] = ("libllama.so", "libggml.so", "libggml-base.so"),
              extra_assets: dict[str, dict] | None = None,
              system_libs: dict[str, tuple[str, ...]] | None = None,
              name: str = "fake-runtime.lock") -> pins.RuntimeLock:
    assets: dict[str, dict] = {variant: {"asset": asset, "size": size, "sha256": digest}
                               for variant in variants}
    assets.update(extra_assets or {})
    llama: dict[str, object] = {
        "tag": tag, "published_at": "2026-09-17T13:31:47Z", "commit": "b49650adb",
        "min_build_for_spark2_5": min_build,
        "assets": assets,
        "url_template": "https://example.invalid/releases/download/{tag}/{asset}",
        "required_files": list(required_files),
        "required_tools": ["llama-fit-params"],
        "required_symbols_llama": ["llama_backend_init"],
        "required_symbols_ggml": ["ggml_backend_load_all_from_path"],
        "mandatory_call_order": ["CDLL(libggml.so, RTLD_GLOBAL)"],
    }
    if repo is not None:
        llama["repo"] = repo
    if system_libs:
        # `init`'s pre-flight reads these (install.py:331): what the pinned bundle links and does
        # not ship. Only the locks that are *about* the pre-flight carry them.
        llama["system_libs"] = {variant: list(names) for variant, names in system_libs.items()}
    payload = {
        "schema": "typed_gguf.runtime.lock/v1",
        "llama_cpp": llama,
        "default_model": {"repo": "XHToken/Spark-X2.5-4B-GGUF", "repo_sha": "0" * 40,
                          "quant": "Q8_0", "file": "Spark-X2.5-4B-Q8_0.gguf", "size": 1,
                          "sha256": "a" * 64, "arch": "spark2_5", "license": "apache-2.0",
                          "alternates": {}},
    }
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return pins.load_lock(path)


def release(tag: str = NEXT_TAG, *, asset: str = NEXT_ASSET, size: int = 1234,
            digest: str | None = None,
            published_at: str = "2026-09-24T09:00:00Z") -> update.Release:
    return update.Release(tag=tag, published_at=published_at,
                          assets=(update.ReleaseAsset(name=asset, size=size, sha256=digest),))


class InstalledHome:
    """A data home `init` installed, plus the facts its gates compare against."""

    def __init__(self, home: pathlib.Path, lock: pins.RuntimeLock, archive: pathlib.Path,
                 result: dict) -> None:
        self.home = home
        self.lock = lock
        self.archive = archive
        self.result = result
        self.dir = pathlib.Path(result["dir"])
        self.record = result["record"]
        self.record_bytes = (home / "runtime.json").read_bytes()

    def snapshot(self) -> dict[str, tuple]:
        return snapshot(self.home)

    def runtime_snapshot(self) -> dict[str, tuple]:
        """The working runtime itself: what a failed leg must not be able to change."""
        return snapshot(self.home / "runtime")

    def pending(self) -> list[str]:
        return sorted(path.name for path in (self.home / "runtime").glob(".pending-*"))

    def runtime_names(self) -> list[str]:
        return sorted(path.name for path in (self.home / "runtime").iterdir())


def installed_home(tmp_path: pathlib.Path, *, variants: tuple[str, ...] = (VARIANT,),
                   extra_assets: dict[str, dict] | None = None,
                   home_name: str = "home") -> InstalledHome:
    """`init`'s own rung 1 against a fake bundle: the real fetch, extract, probe and record.

    Driven with `deep=False` (the `TYPED_GGUF_DEEP_PROBE=0` shape) so this fixture needs no child
    process, and with a `file://` URL so the whole path runs offline.
    """
    archive = make_archive(tmp_path, PINNED_ASSET, tag=PINNED_TAG, build=PINNED_BUILD)
    lock = fake_lock(tmp_path, variants=variants, size=archive.stat().st_size,
                     digest=sha256(archive), extra_assets=extra_assets)
    home = tmp_path / home_name
    result = install.install("cpu", home=home, lock=lock, url=archive.as_uri(), deep=False,
                             free_bytes=1 << 40)
    return InstalledHome(home, lock, archive, result)


def snapshot(root: pathlib.Path) -> dict[str, tuple]:
    """Every path under `root` with its size and content hash — what a command left behind."""
    if not root.exists():
        return {}
    found: dict[str, tuple] = {}
    for path in sorted(root.rglob("*")):
        relative = str(path.relative_to(root))
        found[relative] = (("dir",) if path.is_dir()
                           else ("file", path.stat().st_size, sha256(path)))
    return found


class FakeClient:
    """The keep client's `stop(drain=True)` surface, with every call and its moment recorded."""

    def __init__(self, report: dict | None = None, *, observe=None) -> None:
        self.report = report or {"stopped": False, "pid": None, "reason": "no host",
                                 "cleaned": False}
        self.calls: list[dict] = []
        self.seen_at_stop: list[object] = []
        self.observe = observe

    def stop(self, *, drain: bool = False) -> dict:
        self.calls.append({"drain": drain})
        if self.observe is not None:
            self.seen_at_stop.append(self.observe())
        return self.report


class Answer:
    """The `urlopen` context manager, answering with bytes."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> Answer:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def read(self) -> bytes:
        return self.payload


def must_not_fetch(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("upstream must not be queried for a runtime we will not switch")


# ------------------------------------------------------- the target rules (the lock's policy)
def test_the_release_api_repo_is_the_lock_s_own(tmp_path: pathlib.Path) -> None:
    """Universal: official bundles per `runtime.lock` — the API repo is the lock's own fact."""
    lock = pins.load_lock(ROOT / "runtime.lock")
    assert lock.repo == "ggml-org/llama.cpp"
    assert update.releases_url(lock) == (
        "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=20")
    with pytest.raises(TypedGgufError) as exc:
        update.releases_url(fake_lock(tmp_path, repo=None))
    assert exc.value.code == "E_UPDATE_UNAVAILABLE"
    assert "upstream" in str(exc.value)


def test_releases_url_asks_for_the_named_tag_when_one_is_given(tmp_path: pathlib.Path) -> None:
    lock = fake_lock(tmp_path)
    assert update.releases_url(lock, tag=NEXT_TAG) == (
        "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/b11160")
    assert update.releases_url(lock, limit=5).endswith("per_page=5")


def test_the_asset_name_is_the_pinned_one_retagged(tmp_path: pathlib.Path) -> None:
    """SPEC 2.8 item 2: `llama-b11026-…-x64.tar.gz` -> `llama-b11160-…-x64.tar.gz`."""
    lock = fake_lock(tmp_path)
    assert update.retag_asset_name(PINNED_ASSET, NEXT_TAG, pinned_tag=lock.tag) == NEXT_ASSET
    # ...and a pinned name that does not carry the pinned tag is never guessed at
    assert update.retag_asset_name("llama-x64.tar.gz", NEXT_TAG, pinned_tag=lock.tag) is None
    assert update.build_of_tag(NEXT_TAG) == NEXT_BUILD
    assert update.build_of_tag("v0.5.0") is None


def test_the_newest_asset_bearing_release_wins_over_releases_latest() -> None:
    """The recon of 2026-09-24: `releases/latest` is a milestone release with no bundle."""
    latest = update.Release(tag="v0.5.0", published_at="2026-09-24T08:00:00Z",
                            assets=(update.ReleaseAsset("nightly-tag.txt", 12),))
    older = release("b11050", asset="llama-b11050-bin-ubuntu-x64.tar.gz", size=99)
    chosen, asset = update.pick_target([latest, release(), older], pinned_name=PINNED_ASSET,
                                       pinned_tag=PINNED_TAG)
    assert (chosen.tag, asset.name, asset.size) == (NEXT_TAG, NEXT_ASSET, 1234)
    with pytest.raises(TypedGgufError) as exc:
        update.pick_target([latest], pinned_name=PINNED_ASSET, pinned_tag=PINNED_TAG)
    assert exc.value.code == "E_UPDATE_UNAVAILABLE"
    assert "never guesses" in str(exc.value)
    # a release that does not carry the bundle is skipped, not a wall: the match behind it wins
    chosen, _ = update.pick_target([older, latest, release()], pinned_name=PINNED_ASSET,
                                   pinned_tag=PINNED_TAG, tag=NEXT_TAG)
    assert chosen.tag == NEXT_TAG
    # ...and a pinned name with no pinned tag in it has no rule to re-tag: nothing is chosen
    with pytest.raises(TypedGgufError) as exc:
        update.pick_target([release()], pinned_name="llama-ubuntu-x64.tar.gz",
                           pinned_tag=PINNED_TAG)
    assert exc.value.code == "E_UPDATE_UNAVAILABLE"


def test_a_named_tag_must_carry_this_host_s_bundle() -> None:
    other = release("b11050", asset="llama-b11050-bin-macos-arm64.tar.gz")
    with pytest.raises(TypedGgufError) as exc:
        update.pick_target([release(), other], pinned_name=PINNED_ASSET, pinned_tag=PINNED_TAG,
                           tag="b11050")
    assert exc.value.code == "E_UPDATE_UNAVAILABLE"
    assert "b11050" in str(exc.value)
    chosen, _ = update.pick_target([other, release()], pinned_name=PINNED_ASSET,
                                   pinned_tag=PINNED_TAG, tag=NEXT_TAG)
    assert chosen.tag == NEXT_TAG


def test_a_pinned_name_without_the_pinned_tag_refuses_before_any_fetch(
        tmp_path: pathlib.Path) -> None:
    home = installed_home(tmp_path)
    lock = fake_lock(tmp_path, asset="llama-ubuntu-x64.tar.gz",
                     size=home.archive.stat().st_size, name="other.lock")
    with pytest.raises(TypedGgufError) as exc:
        update.update(home=home.home, lock=lock, fetch=must_not_fetch, deep=False,
                      probes=HOST)
    assert exc.value.code == "E_UPDATE_UNAVAILABLE"
    assert "b11026" in str(exc.value)


def test_a_release_entry_without_a_tag_is_skipped_not_a_wall() -> None:
    """The API's answer is data: a tagless entry, a non-object entry and a single release object
    must not hide the releases behind them (the live list carries both shapes)."""
    full = {"tag_name": NEXT_TAG, "published_at": "2026-09-24T13:50:41Z",
            "assets": [{"name": NEXT_ASSET, "size": 17002550, "digest": "sha256:" + "a" * 64}]}
    assert [entry.tag for entry in update.parse_releases([{"tag_name": ""}, "junk", full])] == \
        [NEXT_TAG]
    assert [entry.tag for entry in update.parse_releases(full)] == [NEXT_TAG]
    assert update.parse_releases([{"assets": []}, None]) == []


def test_an_asset_without_a_usable_size_records_none_not_a_guess() -> None:
    """A missing or non-numeric `size` is None: the payload's `size_human` says so instead of a
    number nobody sent."""
    missing = update.parse_releases([{"tag_name": NEXT_TAG, "assets": [{"name": NEXT_ASSET}]}])
    assert missing[0].assets[0].size is None
    broken = update.parse_releases([{"tag_name": NEXT_TAG,
                                     "assets": [{"name": NEXT_ASSET, "size": "big"}]}])
    assert broken[0].assets[0].size is None


def test_a_record_naming_another_directory_is_not_this_runtime(tmp_path: pathlib.Path) -> None:
    """`runtime.json` is data, not an answer: when it names a directory that is not the one
    `find_runtime` found, its tag/build/installed_at describe *that* bundle — the build of this one
    is read from the binary (and a record that names this directory without a build still
    answers it)."""
    home = installed_home(tmp_path)
    record_path = home.home / "runtime.json"
    record = json.loads(record_path.read_bytes())
    record.update({"dir": str(tmp_path / "elsewhere"), "tag": "b1", "build": 1,
                   "installed_at": "2020-01-01T00:00:00Z"})
    record_path.write_text(json.dumps(record))
    current = update.current_runtime(home.home)
    assert current["dir"] == str(home.dir) and current["build"] == PINNED_BUILD
    assert current["tag"] is None and current["variant"] is None
    assert current["installed_at"] is None
    record.update({"dir": str(home.dir), "build": None})
    record_path.write_text(json.dumps(record))
    current = update.current_runtime(home.home)
    assert current["tag"] == "b1" and current["build"] == PINNED_BUILD


def test_an_update_with_a_named_tag_walks_that_release_only(tmp_path: pathlib.Path) -> None:
    """`--tag` narrows the walk: a named tag that carries no bundle for this host refuses instead
    of quietly moving to a newer release."""
    home = installed_home(tmp_path)
    named = release("b11050", asset="llama-b11050-bin-ubuntu-x64.tar.gz", size=99)
    other = release("b11050", asset="llama-b11050-bin-macos-arm64.tar.gz")
    with pytest.raises(TypedGgufError) as exc:
        update.update(home=home.home, lock=home.lock, tag="b11050", releases=[release(), other],
                      deep=False, probes=HOST, client=FakeClient())
    assert exc.value.code == "E_UPDATE_UNAVAILABLE" and "b11050" in str(exc.value)
    payload = update.update(check=True, home=home.home, lock=home.lock, tag="b11050",
                            releases=[release(), named], deep=False, probes=HOST)
    assert payload["to"]["tag"] == "b11050" and payload["to"]["build"] == 11050


def test_the_fetch_is_asked_for_the_list_the_way_the_lock_says(tmp_path: pathlib.Path) -> None:
    """The upstream leg is called with this run's tag, the pinned page size and the HTTP timeout —
    a fetch that was asked for something else is a different query."""
    home = installed_home(tmp_path)
    seen: list[dict] = []

    def fake_fetch(lock: pins.RuntimeLock, **kwargs: object) -> list[update.Release]:
        assert lock is home.lock
        seen.append(kwargs)
        return [release()]

    update.update(check=True, home=home.home, lock=home.lock, fetch=fake_fetch, deep=False,
                  probes=HOST)
    assert seen == [{"tag": None, "limit": update.RELEASES_LIMIT,
                     "timeout": update.REQUEST_TIMEOUT}]
    update.update(check=True, home=home.home, lock=home.lock, tag=NEXT_TAG, fetch=fake_fetch,
                  deep=False, probes=HOST)
    assert seen[1]["tag"] == NEXT_TAG


# --------------------------------------------------------------------------- A-E5-6: check mode
def test_check_prints_current_vs_target_and_touches_nothing(tmp_path: pathlib.Path) -> None:
    home = installed_home(tmp_path)
    archive = make_archive(tmp_path, NEXT_ASSET, tag=NEXT_TAG, build=NEXT_BUILD)
    before = home.snapshot()
    payload = update.update(check=True, home=home.home, lock=home.lock,
                            releases=[release(size=archive.stat().st_size)],
                            url=archive.as_uri(), deep=False, probes=HOST)
    assert payload["check"] is True and payload["updated"] is False
    assert payload["from"] == {"tag": PINNED_TAG, "build": PINNED_BUILD,
                              "dir": str(home.dir), "variant": VARIANT}
    assert payload["to"] == {"tag": NEXT_TAG, "build": NEXT_BUILD,
                             "dir": str(home.home / "runtime" / f"{NEXT_TAG}-{VARIANT}"),
                             "variant": VARIANT}
    assert payload["asset"]["name"] == NEXT_ASSET
    assert payload["asset"]["size"] == archive.stat().st_size
    assert payload["asset"]["size_human"].endswith("B")
    assert payload["asset"]["url"] == (
        f"https://example.invalid/releases/download/{NEXT_TAG}/{NEXT_ASSET}")
    assert payload["probe"] is None and payload["previous"] is None
    assert payload["reason"] is None
    assert home.snapshot() == before, "--check must touch nothing"


def test_the_wire_shapes_are_the_pinned_key_sets(tmp_path: pathlib.Path) -> None:
    home = installed_home(tmp_path)
    checked = update.update(check=True, home=home.home, lock=home.lock, releases=[release()],
                            deep=False, probes=HOST)
    assert tuple(checked) == update.UPDATE_KEYS
    assert tuple(checked["from"]) == update.FROM_KEYS
    assert tuple(checked["to"]) == update.FROM_KEYS
    assert tuple(checked["asset"]) == update.ASSET_KEYS
    assert update.PREVIOUS_KEYS == ("dir", "tag", "build", "installed_at")
    assert update.PROBE_KEYS == ("build", "tools", "symbols_ok", "symbols_probed",
                                 "missing_symbols", "backends", "expect_backend")


def test_a_target_equal_to_the_current_tag_is_not_an_update(tmp_path: pathlib.Path,
                                                            monkeypatch: pytest.MonkeyPatch,
                                                            capsys: pytest.CaptureFixture[str]
                                                            ) -> None:
    home = installed_home(tmp_path)
    monkeypatch.setenv("TYPED_GGUF_HOME", str(home.home))
    before = home.snapshot()
    target = release(PINNED_TAG, asset=PINNED_ASSET, size=home.archive.stat().st_size)
    monkeypatch.setattr(update, "DEFAULT_FETCH", lambda lock, **kw: [target])
    payload = update.update(home=home.home, lock=home.lock, releases=[target], deep=False,
                            url="file:///no/such/bundle.tar.gz", client=FakeClient(),
                            probes=HOST)
    assert payload["updated"] is False
    assert payload["reason"] == f"already at {PINNED_TAG}"
    assert payload["from"]["tag"] == PINNED_TAG
    assert payload["previous"] is None and payload["host_stopped"] is None
    assert home.snapshot() == before
    assert cli.main(["runtime", "update", "--check", "--backend", "cpu"]) == 0
    assert f"already at {PINNED_TAG}" in capsys.readouterr().out


def test_check_offline_is_a_typed_message_and_leaves_no_partial_state(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    """SPEC 2.8 item 1: no network -> `E_DOWNLOAD_FAILED` (exit 3) naming the URL it could not
    reach, and no partial state. `TYPED_GGUF_TEST_BLOCK_NET` is the real off switch."""
    home = installed_home(tmp_path)
    monkeypatch.setenv("TYPED_GGUF_HOME", str(home.home))
    suite_environment.block_network(monkeypatch)
    before = home.snapshot()
    with pytest.raises(TypedGgufError) as exc:
        update.update(check=True, home=home.home, lock=home.lock, deep=False, probes=HOST)
    assert exc.value.code == "E_DOWNLOAD_FAILED"
    assert "api.github.com/repos/ggml-org/llama.cpp/releases" in str(exc.value)
    assert home.snapshot() == before
    assert cli.main(["runtime", "update", "--check"]) == 3
    assert "error: E_DOWNLOAD_FAILED" in capsys.readouterr().err


def test_a_bad_api_answer_is_a_download_failure(tmp_path: pathlib.Path) -> None:
    lock = fake_lock(tmp_path)
    url = update.releases_url(lock)

    def http_404(*_args: object, **_kwargs: object) -> None:
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)  # type: ignore[arg-type]

    with pytest.raises(TypedGgufError) as exc:
        update.fetch_releases(lock, urlopen=http_404)
    assert exc.value.code == "E_DOWNLOAD_FAILED" and url in str(exc.value)
    with pytest.raises(TypedGgufError) as exc:
        update.fetch_releases(lock, urlopen=lambda *a, **k: Answer(b"<html>not json</html>"))
    assert exc.value.code == "E_DOWNLOAD_FAILED"


def test_the_release_list_is_parsed_into_tags_assets_sizes_and_digests(
        tmp_path: pathlib.Path) -> None:
    lock = fake_lock(tmp_path)
    body = json.dumps([
        {"tag_name": NEXT_TAG, "published_at": "2026-09-24T09:00:00Z",
         "assets": [{"name": NEXT_ASSET, "size": 30700000, "digest": "sha256:" + "d" * 64},
                    {"name": "cudart-llama-bin-win-cuda-12.4-x64.zip", "size": 3}]},
        {"tag_name": "b11050",
         "assets": [{"name": "llama-b11050-bin-ubuntu-x64.tar.gz", "size": 5, "digest": None}]},
    ]).encode()
    releases = update.fetch_releases(lock, urlopen=lambda *a, **k: Answer(body))
    assert [item.tag for item in releases] == [NEXT_TAG, "b11050"]
    assert releases[0].published_at == "2026-09-24T09:00:00Z"
    first = releases[0].assets[0]
    assert (first.name, first.size, first.sha256) == (NEXT_ASSET, 30700000, "d" * 64)
    assert releases[1].assets[0].size == 5 and releases[1].assets[0].sha256 is None


# ------------------------------------------------------------------ A-E5-7: the real switch
def test_a_successful_update_switches_the_record_and_keeps_the_previous_bundle(
        tmp_path: pathlib.Path) -> None:
    home = installed_home(tmp_path)
    archive = make_archive(tmp_path, NEXT_ASSET, tag=NEXT_TAG, build=NEXT_BUILD)
    lock_path = home.lock.source_path
    lock_bytes = lock_path.read_bytes()
    registry = home.home / "registry.json"
    registry.write_text(json.dumps({"schema": "typed_gguf.registry/v1", "models": {},
                                    "current": None}))
    registry_bytes = registry.read_bytes()
    client = FakeClient()
    payload = update.update(home=home.home, lock=home.lock,
                            releases=[release(size=archive.stat().st_size,
                                              digest=sha256(archive))],
                            url=archive.as_uri(), deep=False, client=client, probes=HOST)
    final = home.home / "runtime" / f"{NEXT_TAG}-{VARIANT}"
    record = json.loads((home.home / "runtime.json").read_bytes())
    assert payload["updated"] is True and payload["check"] is False
    assert payload["from"]["tag"] == PINNED_TAG and payload["from"]["build"] == PINNED_BUILD
    assert payload["to"] == {"tag": NEXT_TAG, "build": NEXT_BUILD, "dir": str(final),
                             "variant": VARIANT}
    assert payload["probe"] == {"build": NEXT_BUILD,
                                "tools": {"llama-cli": str(final / "llama-cli"),
                                          "llama-fit-params": str(final / "llama-fit-params")},
                                "symbols_ok": False, "symbols_probed": False,
                                "missing_symbols": [], "backends": ["cpu"],
                                "expect_backend": "cpu"}
    assert payload["asset"]["sha256"] == sha256(archive)
    assert payload["host_stopped"] == client.report
    assert tuple(payload) == update.UPDATE_KEYS
    assert tuple(payload["probe"]) == update.PROBE_KEYS
    assert tuple(payload["asset"]) == update.ASSET_KEYS
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", record["updated_at"]), (
        record["updated_at"])
    # the bundle: staged, probed, moved — and the previous one still on disk
    assert (final / "libllama.so").exists() and (final / "llama-cli").exists()
    assert (home.dir / "libllama.so").exists()
    assert home.runtime_names() == sorted([home.dir.name, final.name])
    assert (home.home / "downloads" / NEXT_ASSET).exists()
    assert home.pending() == []
    # the record: the new bundle, the old one retained for rollback
    assert record["dir"] == str(final) and record["tag"] == NEXT_TAG
    assert record["build"] == NEXT_BUILD and record["asset"] == NEXT_ASSET
    assert record["asset_sha256"] == sha256(archive)
    assert record["asset_verified"] is True
    assert record["libllama_sha256"] == sha256(final / "libllama.so")
    # the tools the record names live in the bundle the update moved into place, not in the
    # staging directory the probe saw (`doctor`/`version` read this map)
    assert record["tools"] == payload["probe"]["tools"]
    assert record["tools"]["llama-cli"] == str(final / "llama-cli")
    assert set(record["previous"]) == set(update.PREVIOUS_KEYS)
    assert record["previous"] == {"dir": str(home.dir), "tag": PINNED_TAG,
                                  "build": PINNED_BUILD,
                                  "installed_at": home.record["installed_at"]}
    assert payload["previous"] == record["previous"]
    # ...and nothing else the tool owns was touched
    assert lock_path.read_bytes() == lock_bytes, "an update never rewrites runtime.lock"
    assert registry.read_bytes() == registry_bytes


def test_the_update_record_carries_every_key_init_writes(tmp_path: pathlib.Path) -> None:
    """`doctor`/`version` read the same keys after an update as after an install."""
    home = installed_home(tmp_path)
    archive = make_archive(tmp_path, NEXT_ASSET, tag=NEXT_TAG, build=NEXT_BUILD)
    update.update(home=home.home, lock=home.lock,
                  releases=[release(size=archive.stat().st_size)], url=archive.as_uri(),
                  deep=False, client=FakeClient(), probes=HOST)
    record = json.loads((home.home / "runtime.json").read_bytes())
    assert set(record) - set(home.record) == {"previous", "updated_at", "update_from"}
    assert set(home.record) - set(record) == set()
    assert record["update_from"] == {"dir": str(home.dir), "tag": PINNED_TAG,
                                     "build": PINNED_BUILD}


def test_version_and_doctor_report_the_new_build(tmp_path: pathlib.Path,
                                                 monkeypatch: pytest.MonkeyPatch,
                                                 capsys: pytest.CaptureFixture[str]) -> None:
    home = installed_home(tmp_path)
    archive = make_archive(tmp_path, NEXT_ASSET, tag=NEXT_TAG, build=NEXT_BUILD)
    update.update(home=home.home, lock=home.lock,
                  releases=[release(size=archive.stat().st_size)], url=archive.as_uri(),
                  deep=False, client=FakeClient(), probes=HOST)
    monkeypatch.setenv("TYPED_GGUF_HOME", str(home.home))
    assert cli.main(["version", "--json"]) == 0
    version = json.loads(capsys.readouterr().out)
    assert version["runtime"]["build"] == NEXT_BUILD
    assert version["runtime"]["dir"] == str(home.home / "runtime" / f"{NEXT_TAG}-{VARIANT}")
    assert version["lock"]["tag"] == PINNED_TAG          # the pin itself never moves
    report = cli.doctor_checks(home=home.home, lock=home.lock)
    assert report["runtime"]["build"] == NEXT_BUILD
    assert report["runtime"]["tag"] == NEXT_TAG
    assert report["runtime"]["dir"] == str(home.home / "runtime" / f"{NEXT_TAG}-{VARIANT}")


def test_the_host_is_stopped_before_the_bundle_is_switched(tmp_path: pathlib.Path) -> None:
    """SPEC 2.12: a process that dlopen'd the old libraries must not survive the switch."""
    home = installed_home(tmp_path)
    archive = make_archive(tmp_path, NEXT_ASSET, tag=NEXT_TAG, build=NEXT_BUILD)
    final = home.home / "runtime" / f"{NEXT_TAG}-{VARIANT}"
    record_before = home.record_bytes
    client = FakeClient(observe=lambda: (
        (home.home / "runtime.json").read_bytes() == record_before,
        final.exists(), home.pending()))
    update.update(home=home.home, lock=home.lock,
                  releases=[release(size=archive.stat().st_size)], url=archive.as_uri(),
                  deep=False, client=client, probes=HOST)
    assert client.calls == [{"drain": True}], "the host is stopped, with drain (the swap rule)"
    assert len(client.seen_at_stop) == 1
    record_at_stop, switched_at_stop, pending_at_stop = client.seen_at_stop[0]
    assert record_at_stop is True, "the record must not move before the host is stopped"
    assert switched_at_stop is False, "the switch happens after the stop"
    assert pending_at_stop == [f".pending-{NEXT_ASSET}"], pending_at_stop


def test_a_box_with_no_host_is_not_a_failure(tmp_path: pathlib.Path) -> None:
    home = installed_home(tmp_path)
    archive = make_archive(tmp_path, NEXT_ASSET, tag=NEXT_TAG, build=NEXT_BUILD)
    payload = update.update(home=home.home, lock=home.lock,
                            releases=[release(size=archive.stat().st_size)],
                            url=archive.as_uri(), deep=False, client=FakeClient(), probes=HOST)
    assert payload["updated"] is True
    assert payload["host_stopped"] == {"stopped": False, "pid": None, "reason": "no host",
                                       "cleaned": False}


def test_a_re_update_after_a_rollback_adopts_the_bundle_it_already_has(
        tmp_path: pathlib.Path) -> None:
    """Nothing is ever deleted, so the bundle a rollback left behind is reused, not re-fetched."""
    home = installed_home(tmp_path)
    archive = make_archive(tmp_path, NEXT_ASSET, tag=NEXT_TAG, build=NEXT_BUILD)
    update.update(home=home.home, lock=home.lock,
                  releases=[release(size=archive.stat().st_size)], url=archive.as_uri(),
                  deep=False, client=FakeClient(), probes=HOST)
    update.rollback(home=home.home, client=FakeClient())
    assert json.loads((home.home / "runtime.json").read_bytes())["dir"] == str(home.dir)
    again = update.update(home=home.home, lock=home.lock, releases=[release()],
                          url="file:///definitely-not-here", deep=False, client=FakeClient(),
                          probes=HOST)
    record = json.loads((home.home / "runtime.json").read_bytes())
    assert again["updated"] is True and record["tag"] == NEXT_TAG
    assert record["source"] == "already-downloaded"
    assert record["previous"]["dir"] == str(home.dir)
    assert record["previous"]["build"] == PINNED_BUILD


def test_the_deep_probe_follows_the_same_knob_init_uses(tmp_path: pathlib.Path,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[bool] = []
    asked: list[dict] = []

    def spy(runtime_dir: object, **kwargs: object) -> capability.ProbeResult:
        seen.append(bool(kwargs["deep"]))
        asked.append({key: kwargs.get(key) for key in
                      ("lock", "expect_backend", "run_tools", "system")})
        return capability.ProbeResult(runtime_dir=pathlib.Path(str(runtime_dir)),
                                      build=NEXT_BUILD, backends=("cpu",),
                                      expect_backend="cpu", symbols_checked=True)

    for index, (env, expected) in enumerate(((None, True), ("0", False))):
        home = installed_home(tmp_path, home_name=f"home-{index}")
        archive = make_archive(tmp_path, NEXT_ASSET, tag=NEXT_TAG, build=NEXT_BUILD)
        if env is None:
            monkeypatch.delenv("TYPED_GGUF_DEEP_PROBE", raising=False)
        else:
            monkeypatch.setenv("TYPED_GGUF_DEEP_PROBE", env)
        payload = update.update(home=home.home, lock=home.lock,
                                releases=[release(size=archive.stat().st_size)],
                                url=archive.as_uri(), probe=spy, client=FakeClient(),
                                probes=HOST)
        assert seen[-1] is expected, env
        assert payload["probe"]["build"] == NEXT_BUILD
        assert asked[-1] == {"lock": home.lock, "expect_backend": "cpu", "run_tools": True,
                             "system": None}, env


# --------------------------------------------------------- A-E5-8: a failure leaves the old one
def test_a_failed_download_leaves_the_runtime_untouched(tmp_path: pathlib.Path) -> None:
    home = installed_home(tmp_path)
    before = home.runtime_snapshot()
    with pytest.raises(TypedGgufError) as exc:
        update.update(home=home.home, lock=home.lock, releases=[release(size=64)],
                      url="file:///no/such/bundle.tar.gz", deep=False, client=FakeClient(),
                      probes=HOST)
    assert exc.value.code == "E_DOWNLOAD_FAILED"
    assert "no/such/bundle.tar.gz" in str(exc.value)
    assert (home.home / "runtime.json").read_bytes() == home.record_bytes
    assert home.runtime_snapshot() == before
    assert home.pending() == []
    assert not (home.home / "downloads" / NEXT_ASSET).exists()


def test_the_github_asset_leg_names_github_and_sends_the_real_user_agent(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """P2 (card t_16067777, E2E proposal): the GitHub download speaks for GitHub.

    The E2E's U5 leg read `E_DOWNLOAD_FAILED: …: HuggingFace returned HTTP 404` and the fixture's
    request log showed `User-Agent: typed-gguf/0.1` on the asset GET while the release API sent
    `typed-gguf/0.2.3`: `registry/hf.py` — the module the GitHub asset download borrows its
    transport from — hardcoded the version *and* worded every failure with HuggingFace's name.
    The asset leg is GitHub's, and the UA carries the packaged version (the same source as the
    release API's), so a reader of the failure never has to know which module fetched it.
    """
    home = installed_home(tmp_path)
    archive = make_archive(tmp_path, NEXT_ASSET, tag=NEXT_TAG, build=NEXT_BUILD)
    seen: list[dict[str, str]] = []

    def not_found(url: str, headers: dict[str, str], timeout: float = 60.0):
        seen.append({"url": url, "headers": dict(headers)})
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr(hf, "_open", not_found)
    before = (home.home / "runtime.json").read_bytes()
    with pytest.raises(TypedGgufError) as exc:
        update.update(home=home.home, lock=home.lock,
                      releases=[release(size=archive.stat().st_size)], deep=False,
                      client=FakeClient(), probes=HOST)
    message = str(exc.value)
    assert exc.value.code == "E_DOWNLOAD_FAILED"
    assert "GitHub returned HTTP 404" in message, message
    assert "HuggingFace" not in message, message
    assert "huggingface.co" not in message, message
    assert seen, "the failing request is the asset GET"
    agent = seen[0]["headers"].get("User-Agent", "")
    assert agent.startswith(update.USER_AGENT), agent
    assert agent.startswith(f"typed-gguf/{__version__}"), agent
    assert (home.home / "runtime.json").read_bytes() == before


def test_a_bundle_missing_a_required_file_aborts_before_the_probe(
        tmp_path: pathlib.Path) -> None:
    home = installed_home(tmp_path)
    archive = make_archive(tmp_path, NEXT_ASSET, tag=NEXT_TAG,
                           libs=("libllama.so", "libggml.so"))
    before = home.runtime_snapshot()

    def must_not_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the probe must not run on an incomplete bundle")

    with pytest.raises(TypedGgufError) as exc:
        update.update(home=home.home, lock=home.lock,
                      releases=[release(size=archive.stat().st_size)], url=archive.as_uri(),
                      deep=False, client=FakeClient(), probe=must_not_run, probes=HOST)
    assert exc.value.code == "E_RUNTIME_SYMBOLS"
    assert "libggml-base.so" in str(exc.value)
    assert (home.home / "runtime.json").read_bytes() == home.record_bytes
    assert home.runtime_snapshot() == before
    assert home.runtime_names() == [home.dir.name]
    assert home.pending() == []
    assert not (home.home / "runtime" / f"{NEXT_TAG}-{VARIANT}").exists()


def test_a_bundle_whose_symbols_do_not_resolve_aborts(tmp_path: pathlib.Path,
                                                      monkeypatch: pytest.MonkeyPatch,
                                                      in_process_scan: None) -> None:
    home = installed_home(tmp_path)
    archive = make_archive(tmp_path, NEXT_ASSET, tag=NEXT_TAG, build=NEXT_BUILD)
    monkeypatch.setattr(capability, "probe_symbols",
                        lambda *a, **k: (["llama_decode"], [], None))
    with pytest.raises(TypedGgufError) as exc:
        update.update(home=home.home, lock=home.lock,
                      releases=[release(size=archive.stat().st_size)], url=archive.as_uri(),
                      deep=True, client=FakeClient(), probes=HOST)
    assert exc.value.code == "E_RUNTIME_SYMBOLS"
    assert "llama_decode" in str(exc.value)
    assert (home.home / "runtime.json").read_bytes() == home.record_bytes
    assert home.pending() == []


def test_a_bundle_below_the_arch_build_aborts(tmp_path: pathlib.Path) -> None:
    home = installed_home(tmp_path)
    archive = make_archive(tmp_path, NEXT_ASSET, tag=NEXT_TAG, build=10000)
    with pytest.raises(TypedGgufError) as exc:
        update.update(home=home.home, lock=home.lock,
                      releases=[release(size=archive.stat().st_size)], url=archive.as_uri(),
                      deep=False, client=FakeClient(), probes=HOST)
    assert exc.value.code == "E_RUNTIME_BUILD_OLD"
    assert "b10828" in str(exc.value)
    assert (home.home / "runtime.json").read_bytes() == home.record_bytes
    assert home.pending() == []


def test_a_bundle_without_the_model_architecture_aborts(tmp_path: pathlib.Path) -> None:
    home = installed_home(tmp_path)
    archive = make_archive(tmp_path, NEXT_ASSET, tag=NEXT_TAG, build=NEXT_BUILD,
                           arch_symbol=False)
    with pytest.raises(TypedGgufError) as exc:
        update.update(home=home.home, lock=home.lock,
                      releases=[release(size=archive.stat().st_size)], url=archive.as_uri(),
                      deep=False, client=FakeClient(), probes=HOST)
    assert exc.value.code == "E_MODEL_ARCH_UNSUPPORTED"
    assert "spark2_5" in str(exc.value)
    assert (home.home / "runtime.json").read_bytes() == home.record_bytes
    assert home.pending() == []


def test_a_gpu_bundle_that_does_not_load_here_aborts_instead_of_falling_back(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, in_process_scan: None) -> None:
    """No fallback ladder: an update never silently changes the backend it was asked for."""
    pinned_vulkan = "llama-b11026-bin-ubuntu-vulkan-x64.tar.gz"
    home = installed_home(tmp_path, variants=(VARIANT,),
                          extra_assets={VULKAN_VARIANT: {"asset": pinned_vulkan, "size": 0,
                                                         "sha256": None}})
    vulkan_asset = pinned_vulkan.replace(PINNED_TAG, NEXT_TAG)
    archive = make_archive(tmp_path, vulkan_asset, tag=NEXT_TAG, build=NEXT_BUILD,
                           libs=("libllama.so", "libggml.so", "libggml-base.so",
                                 "libggml-vulkan.so"))
    monkeypatch.setattr(capability, "probe_symbols", lambda *a, **k: ([], [], None))
    monkeypatch.setattr(
        capability, "load_backend_library",
        lambda path: ("libvulkan.so.1: cannot open shared object file" if "vulkan" in str(path)
                      else None))
    with pytest.raises(TypedGgufError) as exc:
        update.update(backend="vulkan", home=home.home, lock=home.lock,
                      releases=[release(asset=vulkan_asset, size=archive.stat().st_size)],
                      url=archive.as_uri(), deep=True, client=FakeClient(), probes=HOST)
    assert exc.value.code == "E_RUNTIME_SYMBOLS"
    assert "vulkan" in str(exc.value)
    assert (home.home / "runtime.json").read_bytes() == home.record_bytes
    assert home.pending() == []
    assert not (home.home / "runtime" / f"{NEXT_TAG}-{VULKAN_VARIANT}").exists()
    assert home.runtime_names() == [home.dir.name]


def test_a_switch_refused_by_a_host_that_would_not_stop_keeps_the_old_runtime(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = installed_home(tmp_path)
    archive = make_archive(tmp_path, NEXT_ASSET, tag=NEXT_TAG, build=NEXT_BUILD)
    occupied = FakeClient({"stopped": False, "pid": 4321,
                           "reason": "pid 4321 survived SIGKILL", "cleaned": True})
    monkeypatch.setattr(update, "pid_alive", lambda pid: pid == 4321)
    with pytest.raises(TypedGgufError) as exc:
        update.update(home=home.home, lock=home.lock,
                      releases=[release(size=archive.stat().st_size)], url=archive.as_uri(),
                      deep=False, client=occupied, probes=HOST)
    assert exc.value.code == "E_UPDATE_UNAVAILABLE"
    assert "would not stop" in str(exc.value)
    assert (home.home / "runtime.json").read_bytes() == home.record_bytes
    assert home.pending() == []
    assert not (home.home / "runtime" / f"{NEXT_TAG}-{VARIANT}").exists()
    # ...and the same refusal reaches the CLI as a user error (exit 2), not a crash
    assert cli._fail(exc.value) == 2


def test_an_update_without_disk_space_stages_nothing(tmp_path: pathlib.Path) -> None:
    home = installed_home(tmp_path)
    with pytest.raises(TypedGgufError) as exc:
        update.update(home=home.home, lock=home.lock, releases=[release(size=4096)],
                      url="file:///no/such/bundle.tar.gz", free_bytes=1024, deep=False,
                      client=FakeClient(), probes=HOST)
    assert exc.value.code == "E_INSUFFICIENT_DISK"
    assert (home.home / "runtime.json").read_bytes() == home.record_bytes
    assert home.pending() == []
    assert home.runtime_names() == sorted([home.dir.name])


# ------------------------------------------------------------------- rollback (SPEC 2.8 tail)
def test_rollback_flips_the_record_back_and_keeps_both_bundles(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    home = installed_home(tmp_path)
    archive = make_archive(tmp_path, NEXT_ASSET, tag=NEXT_TAG, build=NEXT_BUILD)
    update.update(home=home.home, lock=home.lock,
                  releases=[release(size=archive.stat().st_size)], url=archive.as_uri(),
                  deep=False, client=FakeClient(), probes=HOST)
    new_dir = home.home / "runtime" / f"{NEXT_TAG}-{VARIANT}"
    monkeypatch.setenv("TYPED_GGUF_HOME", str(home.home))
    assert cli.main(["runtime", "rollback", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert tuple(payload) == update.ROLLBACK_KEYS
    assert payload["rolled_back"] is True
    assert payload["to"] == {"dir": str(home.dir), "tag": PINNED_TAG, "build": PINNED_BUILD,
                             "variant": VARIANT}
    assert payload["from"] == {"dir": str(new_dir), "tag": NEXT_TAG, "build": NEXT_BUILD,
                               "variant": VARIANT}
    assert payload["previous"] is None
    record = json.loads((home.home / "runtime.json").read_bytes())
    assert record["dir"] == str(home.dir) and record["tag"] == PINNED_TAG
    assert record["build"] == PINNED_BUILD and record["variant"] == VARIANT
    assert record["previous"] is None
    assert record["libllama_sha256"] == sha256(home.dir / "libllama.so")
    assert new_dir.exists(), "rollback never deletes a bundle"
    assert record["rolled_back_from"] == {"dir": str(new_dir), "tag": NEXT_TAG,
                                          "build": NEXT_BUILD}
    assert cli.main(["version", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["runtime"]["build"] == PINNED_BUILD
    report = cli.doctor_checks(home=home.home, lock=home.lock)
    assert report["runtime"]["tag"] == PINNED_TAG
    # ...and a second rollback has nothing to go back to: it refuses and changes nothing
    frozen = (home.home / "runtime.json").read_bytes()
    assert cli.main(["runtime", "rollback"]) == 2
    assert "E_UPDATE_UNAVAILABLE" in capsys.readouterr().err
    assert (home.home / "runtime.json").read_bytes() == frozen


def test_rollback_keeps_the_probe_facts_the_update_recorded(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    """P1 (card t_16067777, E2E proposal): the record `update` wrote is the one that comes back.

    `runtime rollback` used to rebuild a ten-key record, so the ~30 keys the update wrote — the
    probe's `backends`, `symbols_*`, the asset facts — were gone and `typed-gguf version` printed
    `backends unknown`. SPEC 2.8 promises "all existing record keys stay" for *update*; rolling
    back is that same record moving to the other bundle, so the probe facts survive it too — and
    the keys that describe the *active* bundle (its dir, its `tools`, its `libllama` hash) are
    rewritten to the bundle the record now names.
    """
    home = installed_home(tmp_path)
    archive = make_archive(tmp_path, NEXT_ASSET, tag=NEXT_TAG, build=NEXT_BUILD)
    payload = update.update(home=home.home, lock=home.lock,
                            releases=[release(size=archive.stat().st_size)],
                            url=archive.as_uri(), deep=False, client=FakeClient(), probes=HOST)
    probe = payload["probe"]
    updated = json.loads((home.home / "runtime.json").read_bytes())
    assert probe["backends"], "the fixture's probe reports at least one backend"
    monkeypatch.setenv("TYPED_GGUF_HOME", str(home.home))
    report = update.rollback(home=home.home, client=FakeClient())
    record = json.loads((home.home / "runtime.json").read_bytes())

    # every key the update wrote is still there — the merged record, not a fresh ten-key one
    assert set(updated) - set(record) == set()
    assert set(home.record) - set(record) == set(), "init's keys survive the round trip too"
    # ...the probe facts included, which is what `version` reads
    assert record["backends"] == probe["backends"]
    for key in ("symbols_ok", "symbols_probed", "missing_symbols", "probe_warnings",
                "backend_requested", "asset", "asset_sha256", "rung", "url"):
        assert key in record, key
    # the keys that describe the *active* bundle are the rollback target's, not the old record's
    assert record["dir"] == str(home.dir) and record["previous"] is None
    assert record["libllama_sha256"] == sha256(home.dir / "libllama.so")
    # Tier-M (the update.py sweep left `previous.get("installed_at")` alive): the stamp is the
    # bundle-that-is-installed-now's own, carried over from the retained block — not the update's
    # (the `**record` spread would hand that one over for free) and never `None`.
    assert record["installed_at"] == home.record["installed_at"]
    assert record["installed_at"] is not None
    # ...and the report names the home it acted on and the direction it moved in (the sweep left
    # `str(None)` and a `from`/`to` swap alive here)
    assert report["home"] == str(home.home)
    assert report["rolled_back"] is True
    assert report["from"]["build"] == NEXT_BUILD and report["to"]["build"] == PINNED_BUILD
    assert report["to"]["dir"] == str(home.dir)
    assert record["tools"]["llama-cli"] == str(home.dir / "llama-cli")
    assert record["tools"]["llama-fit-params"] == str(home.dir / "llama-fit-params")
    # and the CLI reports the probe facts instead of "backends unknown"
    assert cli.main(["version", "--json"]) == 0
    version = json.loads(capsys.readouterr().out)
    assert version["runtime"]["backends"] == probe["backends"]
    assert cli.main(["version"]) == 0
    assert f"backends {', '.join(probe['backends'])}" in capsys.readouterr().out


def test_rollback_refuses_when_nothing_was_ever_updated(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    home = installed_home(tmp_path)
    monkeypatch.setenv("TYPED_GGUF_HOME", str(home.home))
    before = home.snapshot()
    with pytest.raises(TypedGgufError) as exc:
        update.rollback(home=home.home, client=FakeClient())
    assert exc.value.code == "E_UPDATE_UNAVAILABLE"
    assert "previous" in str(exc.value)
    assert home.snapshot() == before
    assert cli.main(["runtime", "rollback"]) == 2
    assert "E_UPDATE_UNAVAILABLE" in capsys.readouterr().err
    assert cli._fail(exc.value) == 2                       # a user error, not a crash


def test_rollback_refuses_when_the_retained_bundle_is_gone(tmp_path: pathlib.Path) -> None:
    home = installed_home(tmp_path)
    archive = make_archive(tmp_path, NEXT_ASSET, tag=NEXT_TAG, build=NEXT_BUILD)
    update.update(home=home.home, lock=home.lock,
                  releases=[release(size=archive.stat().st_size)], url=archive.as_uri(),
                  deep=False, client=FakeClient(), probes=HOST)
    (home.dir / "libllama.so").unlink()
    before = (home.home / "runtime.json").read_bytes()
    with pytest.raises(TypedGgufError) as exc:
        update.rollback(home=home.home, client=FakeClient())
    assert exc.value.code == "E_UPDATE_UNAVAILABLE"
    assert str(home.dir) in str(exc.value)
    assert (home.home / "runtime.json").read_bytes() == before


# --------------------------------------------------------------------------- A-E5-9: refusals
def test_rollback_stops_the_warm_host_and_timestamps_the_record(tmp_path: pathlib.Path) -> None:
    """The rollback leg stops the host *before* it flips the record (the same rule the update
    follows: no process keeps answering on the old libraries) and stamps the switch."""
    home = installed_home(tmp_path)
    archive = make_archive(tmp_path, NEXT_ASSET, tag=NEXT_TAG, build=NEXT_BUILD)
    update.update(home=home.home, lock=home.lock,
                  releases=[release(size=archive.stat().st_size)], url=archive.as_uri(),
                  deep=False, client=FakeClient(), probes=HOST)
    client = FakeClient()
    payload = update.rollback(home=home.home, client=client)
    assert payload["host_stopped"] == client.report
    assert client.calls == [{"drain": True}]
    record = json.loads((home.home / "runtime.json").read_bytes())
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", record["rolled_back_at"]), (
        record["rolled_back_at"])
    assert record["rolled_back_from"]["dir"] == str(home.home / "runtime" / f"{NEXT_TAG}-{VARIANT}")


def test_update_refuses_on_a_runtime_managed_outside_typed_gguf(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    home = installed_home(tmp_path)
    monkeypatch.setenv("TYPED_GGUF_RUNTIME_DIR", str(home.dir))
    monkeypatch.setenv("TYPED_GGUF_HOME", str(home.home))
    before = home.snapshot()
    for check in (True, False):
        with pytest.raises(TypedGgufError) as exc:
            update.update(check=check, home=home.home, lock=home.lock, fetch=must_not_fetch,
                          deep=False, probes=HOST)
        assert exc.value.code == "E_UPDATE_UNAVAILABLE"
        assert "TYPED_GGUF_RUNTIME_DIR" in str(exc.value)
        assert "unset" in str(exc.value)
    assert home.snapshot() == before
    assert cli.main(["runtime", "update", "--check"]) == 2
    assert "E_UPDATE_UNAVAILABLE" in capsys.readouterr().err


def test_update_refuses_on_an_empty_data_home(tmp_path: pathlib.Path,
                                              monkeypatch: pytest.MonkeyPatch,
                                              capsys: pytest.CaptureFixture[str]) -> None:
    empty = tmp_path / "empty"
    monkeypatch.setenv("TYPED_GGUF_HOME", str(empty))
    monkeypatch.delenv("TYPED_GGUF_RUNTIME_DIR", raising=False)
    for check in (True, False):
        with pytest.raises(TypedGgufError) as exc:
            update.update(check=check, home=empty, deep=False, client=FakeClient())
        assert exc.value.code == "E_RUNTIME_MISSING"
        assert "init" in str(exc.value)
    assert not empty.exists()
    assert cli.main(["runtime", "update", "--check"]) == 3
    assert "E_RUNTIME_MISSING" in capsys.readouterr().err


def test_the_runtime_command_says_what_it_accepts(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["runtime"]) == 2
    assert "update|rollback" in capsys.readouterr().err
    assert cli.main(["runtime", "update", "--nope"]) == 2
    err = capsys.readouterr().err
    assert "E_UNKNOWN_KEY" in err and "typed-gguf runtime --help" in err
    assert cli.main(["runtime", "update", "--help"]) == 0
    update_page = capsys.readouterr().out
    assert update_page.splitlines()[0] == ("usage: typed-gguf runtime update "
                                          "--check|--dry-run --tag TAG --backend "
                                          "auto|cpu|vulkan|cuda|metal --json")
    assert update_page.splitlines()[2] == "refresh the installed llama.cpp runtime"
    assert cli.main(["runtime", "rollback", "--help"]) == 0
    rollback_page = capsys.readouterr().out
    assert rollback_page.splitlines()[0] == "usage: typed-gguf runtime rollback --json"
    assert rollback_page.splitlines()[2] == "roll back to the runtime the last update replaced"
    for page in (update_page, rollback_page):
        assert "run `typed-gguf --help` for the command list" in page


def test_the_update_record_does_not_tell_the_user_to_re_run_init_force(
        tmp_path: pathlib.Path) -> None:
    """A successful update must not leave `init --force` advice in the record.

    `probe.warnings()` compares the live build against the *pinned* tag. That is true, but on a
    runtime the user explicitly asked to move it is the one warning the update itself answers —
    and `doctor`/`version` print the record's warnings, so "re-run `typed-gguf init --force`"
    would point the user at undoing the update. The update path names the move instead.
    """
    home = installed_home(tmp_path)
    archive = make_archive(tmp_path, NEXT_ASSET, tag=NEXT_TAG, build=NEXT_BUILD)

    def moved_probe(runtime_dir: object, **kwargs: object) -> capability.ProbeResult:
        staged = pathlib.Path(str(runtime_dir))
        return capability.ProbeResult(
            runtime_dir=staged, deep=False, symbols_checked=True, build=NEXT_BUILD,
            expected_tag=PINNED_TAG, min_build=10828, backends=("cpu",),
            tools={"llama-fit-params": str(staged / "llama-fit-params")},
            fit_params_help_exit=0, expect_backend="cpu")

    assert any("init --force" in line for line in moved_probe(home.dir).warnings()), (
        "the fixture no longer reproduces the pin-difference warning")
    payload = update.update(home=home.home, lock=home.lock,
                            releases=[release(size=archive.stat().st_size,
                                              digest=sha256(archive))],
                            url=archive.as_uri(), deep=False, client=FakeClient(), probes=HOST,
                            probe=moved_probe)
    # a probe that resolved every symbol reports exactly that (`symbols_ok` is not `probed`)
    assert payload["probe"]["symbols_ok"] is True
    assert payload["probe"]["symbols_probed"] is True
    record = json.loads((home.home / "runtime.json").read_bytes())
    assert record["symbols_ok"] is True and record["symbols_probed"] is True
    assert not any("init --force" in line for line in record["probe_warnings"]), (
        record["probe_warnings"])
    assert any(NEXT_TAG in line and PINNED_TAG in line
               for line in record["probe_warnings"]), record["probe_warnings"]


# ------------------------------------------------------------------ the docs story (A-E5-10)
def test_the_readme_documents_the_one_command_update_and_rollback_story() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    interfaces = readme.split("## Interfaces", 1)[1].split("\n## ", 1)[0]
    rows = [line for line in interfaces.splitlines() if "`typed-gguf runtime" in line]
    assert len(rows) == 1, rows
    for needle in ("runtime update", "--check", "--tag", "runtime rollback",
                   "previous bundle"):
        assert needle in rows[0], f"the update row no longer names {needle!r}"
    assert "## Updating the runtime" in readme
    story = readme.split("## Updating the runtime", 1)[1].split("\n## ", 1)[0]
    for needle in ("typed-gguf runtime update --check", "typed-gguf runtime update",
                   "typed-gguf runtime rollback", "runtime.lock", "init"):
        assert needle in story, f"the update story no longer names {needle!r}"
    assert "runtime" in cli.COMMANDS and cli.COMMAND_HELP["runtime"]


@pytest.mark.network
def test_the_live_release_list_carries_this_host_s_pinned_asset_name() -> None:
    """The network leg on its own (A-E5-10): the real GitHub API, skipped by name offline."""
    from typed_gguf.registry import store

    lock = pins.load_lock(ROOT / "runtime.lock")
    assert store.data_home()                                # the CLI's own home resolution runs
    releases = update.fetch_releases(lock)
    assert releases, "the live release list answered nothing"
    pinned = pins.asset_for(lock, VARIANT).asset
    chosen, asset = update.pick_target(releases, pinned_name=pinned, pinned_tag=lock.tag)
    assert asset.name == update.retag_asset_name(pinned, chosen.tag, pinned_tag=lock.tag)
    assert asset.size is not None and asset.size > 0
    assert chosen.published_at, "the picked release carries no published_at"


# ------------------------------------------- the fix card (t_ba767a2b): M2's pre-flight + M2b
def release_pair(size: int = 1234) -> update.Release:
    """One release carrying *both* hosts' bundles — what makes the target-variant rule readable."""
    return update.Release(tag=NEXT_TAG, published_at="2026-09-24T09:00:00Z", assets=(
        update.ReleaseAsset(name=NEXT_ASSET, size=size),
        update.ReleaseAsset(name=CUDA_ASSET.replace(PINNED_TAG, NEXT_TAG), size=size + 1)))


def gpu_box_home(tmp_path: pathlib.Path, **kwargs: object) -> InstalledHome:
    """The reviewer's box: detection says cuda, `init` installed cpu, and the lock pins both."""
    return installed_home(tmp_path,
                          extra_assets={CUDA_VARIANT: {"asset": CUDA_ASSET, "size": 0,
                                                       "sha256": None}},
                          **kwargs)  # type: ignore[arg-type]


def test_a_bundle_this_host_cannot_load_refuses_before_any_download(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """M2: `update` runs `init`'s own pre-flight, so a box that cannot load the target bundle's
    system libraries refuses *before* spending the download (install.py:331; SPEC 2.8 step 3).

    The refusal keeps `init`'s vocabulary (`E_RUNTIME_SYMBOLS`, the reason, "nothing was changed")
    and stays a refusal: there is no ladder here, so a missing `libcudart.so.12` is a dead end that
    must not cost 168.8 MB per attempt. The URL is a real `file://` archive, so a download that
    happened would have *succeeded* and the run would have updated — the typed error is the proof
    that nothing was fetched.
    """
    home = installed_home(tmp_path)
    archive = make_archive(tmp_path, NEXT_ASSET, tag=NEXT_TAG, build=NEXT_BUILD)
    lock = fake_lock(tmp_path, size=home.archive.stat().st_size, digest=sha256(home.archive),
                     name="preflight.lock",
                     system_libs={VARIANT: ("libcudart.so.12", "libcuda.so.1")})
    monkeypatch.setattr(install, "PREFLIGHT_SYSTEM_LIBS", lambda names: {
        "libcudart.so.12": "libcudart.so.12: cannot open shared object file: "
                           "No such file or directory"})
    before = home.snapshot()
    with pytest.raises(TypedGgufError) as exc:
        update.update(home=home.home, lock=lock,
                      releases=[release(size=archive.stat().st_size)], url=archive.as_uri(),
                      deep=False, client=FakeClient(), probes=HOST)
    assert exc.value.code == "E_RUNTIME_SYMBOLS", str(exc.value)
    assert "libcudart.so.12" in str(exc.value) and "nothing was changed" in str(exc.value)
    assert not (home.home / "downloads" / NEXT_ASSET).exists(), "the archive was never fetched"
    assert home.pending() == [], "nothing was staged"
    assert home.snapshot() == before, "the runtime and its record are untouched"
    # the same refusal reaches the CLI as a runtime error (exit 3), not a crash
    assert cli._fail(exc.value) == 3


def test_check_keeps_its_contract_and_reports_the_target_the_pre_flight_would_refuse(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`--check` is the read-only plan report (SPEC A-E5-6) and stays one: it prints current vs
    target without a download, so the pre-flight has nothing to say about it (card t_ba767a2b)."""
    home = installed_home(tmp_path)
    archive = make_archive(tmp_path, NEXT_ASSET, tag=NEXT_TAG, build=NEXT_BUILD)
    lock = fake_lock(tmp_path, size=home.archive.stat().st_size, digest=sha256(home.archive),
                     name="preflight.lock", system_libs={VARIANT: ("libcudart.so.12",)})
    monkeypatch.setattr(install, "PREFLIGHT_SYSTEM_LIBS", lambda names: {
        "libcudart.so.12": "libcudart.so.12: cannot open shared object file"})
    before = home.snapshot()
    payload = update.update(check=True, home=home.home, lock=lock,
                            releases=[release(size=archive.stat().st_size)], deep=False,
                            probes=HOST)
    assert payload["check"] is True and payload["updated"] is False
    assert payload["to"]["variant"] == VARIANT and payload["reason"] is None
    assert home.snapshot() == before, "--check must touch nothing"


def test_the_default_target_is_the_variant_init_installed_not_a_fresh_detection(
        tmp_path: pathlib.Path) -> None:
    """M2b (the coordinator's option (b)): `update` MAINTAINS the bundle `init` installed.

    On a CUDA-driver box whose ladder fell back to cpu, the *default* target is the installed
    record's variant — `--backend` is how a backend switch is asked for (SPEC 2.8 step 2). The
    release carries both bundles, so a detection-driven target is a visible wrong answer, not an
    `E_UPDATE_UNAVAILABLE` that would hide the rule.
    """
    home = gpu_box_home(tmp_path)
    assert home.record["variant"] == VARIANT and home.record["dir"] == str(home.dir)
    payload = update.update(check=True, home=home.home, lock=home.lock,
                            releases=[release_pair(home.archive.stat().st_size)], deep=False,
                            probes=GPU_HOST)
    assert payload["from"]["variant"] == VARIANT
    assert payload["to"]["variant"] == VARIANT, "detection must not re-decide what is installed"
    assert payload["to"]["dir"] == str(home.home / "runtime" / f"{NEXT_TAG}-{VARIANT}")
    assert payload["asset"]["name"] == NEXT_ASSET


def test_a_real_update_on_a_gpu_box_switches_the_installed_variant_it_already_had(
        tmp_path: pathlib.Path) -> None:
    """The same rule end to end: the download, the probe, the put and the record are the cpu ones
    the box already runs — a GPU detection changes nothing about what is being maintained."""
    home = gpu_box_home(tmp_path)
    archive = make_archive(tmp_path, NEXT_ASSET, tag=NEXT_TAG, build=NEXT_BUILD)
    payload = update.update(home=home.home, lock=home.lock,
                            releases=[release_pair(archive.stat().st_size)],
                            url=archive.as_uri(), deep=False, client=FakeClient(),
                            probes=GPU_HOST)
    final = home.home / "runtime" / f"{NEXT_TAG}-{VARIANT}"
    record = json.loads((home.home / "runtime.json").read_bytes())
    assert payload["updated"] is True and payload["to"]["variant"] == VARIANT
    assert payload["probe"]["expect_backend"] == "cpu"
    assert record["variant"] == VARIANT and record["dir"] == str(final)
    assert record["tag"] == NEXT_TAG and (final / "libllama.so").exists()
    assert not (home.home / "runtime" / f"{NEXT_TAG}-{CUDA_VARIANT}").exists()


def test_an_explicit_backend_still_overrides_the_installed_record(tmp_path: pathlib.Path) -> None:
    """`--backend` is the switch: the record answers the *default*, never an explicit request."""
    home = gpu_box_home(tmp_path)
    payload = update.update(check=True, backend="cuda", home=home.home, lock=home.lock,
                            releases=[release_pair()], deep=False, probes=HOST)
    assert payload["to"]["variant"] == CUDA_VARIANT
    assert payload["asset"]["name"] == CUDA_ASSET.replace(PINNED_TAG, NEXT_TAG)


def test_a_record_that_does_not_name_the_active_runtime_leaves_detection_alone(
        tmp_path: pathlib.Path) -> None:
    """The third clause: a record naming another directory is a fact about *that* bundle, so the
    target variant comes from detection exactly as it did before the rule (SPEC 2.8 step 2)."""
    home = gpu_box_home(tmp_path)
    record_path = home.home / "runtime.json"
    record = json.loads(record_path.read_bytes())
    record["dir"] = str(tmp_path / "elsewhere")
    record_path.write_text(json.dumps(record))
    assert update.current_runtime(home.home)["variant"] is None, "the record is not this runtime"
    payload = update.update(check=True, home=home.home, lock=home.lock,
                            releases=[release_pair()], deep=False, probes=GPU_HOST)
    assert payload["to"]["variant"] == CUDA_VARIANT


def test_a_record_with_no_variant_leaves_detection_alone(tmp_path: pathlib.Path) -> None:
    """A record that names this runtime but carries no `variant` has nothing to read, so the update
    falls back to detection — the same answer the record about another directory gets (SPEC 2.8
    step 2, card t_ba767a2b).

    This is the pin for `installed_variant`'s *falsy* answer: a truthy-but-empty one (`str(None)`,
    the shape a mutation takes) would ask the lock for a variant named `None` and fail the update
    with `E_RUNTIME_MISSING` instead of updating this box.
    """
    home = gpu_box_home(tmp_path)
    record_path = home.home / "runtime.json"
    record = json.loads(record_path.read_bytes())
    record.pop("variant", None)
    record_path.write_text(json.dumps(record))
    assert update.current_runtime(home.home)["dir"] == str(home.dir), "this runtime is the record's"
    payload = update.update(check=True, home=home.home, lock=home.lock,
                            releases=[release_pair()], deep=False, probes=GPU_HOST)
    assert payload["to"]["variant"] == CUDA_VARIANT, "detection answers when the record is silent"


def test_an_installed_variant_the_lock_no_longer_pins_refuses_instead_of_switching(
        tmp_path: pathlib.Path) -> None:
    """A lock that dropped the installed variant has no bundle to maintain: refuse, and let
    `--backend` (or a fresh `init`) be the deliberate way to a different backend."""
    home = installed_home(tmp_path)
    lock = fake_lock(tmp_path, variants=(CUDA_VARIANT,), asset=CUDA_ASSET, size=0,
                     name="no-cpu.lock")
    before = home.snapshot()
    with pytest.raises(TypedGgufError) as exc:
        update.update(check=True, home=home.home, lock=lock, releases=[release_pair()],
                      deep=False, probes=GPU_HOST)
    assert exc.value.code == "E_RUNTIME_MISSING"
    assert VARIANT in str(exc.value)
    assert home.snapshot() == before
