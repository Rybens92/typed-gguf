"""The default model: `current`, and the sole-alias fallback when nothing set one.

Card t_a0fa2dc0 (UX, 0.2.3): after `typed-gguf models pull` a plain `ask` must work without
hunting a path.

`pull` already leaves `current` on the entry it wrote, so a home that pulled is fine (receipt
`docs/evidence/t_a0fa2dc0_default_model.md`, [B4]/[A4]). The two gaps this file pins are the ones
the coordinator hit live:

* a registry with exactly **one** alias and no `current` — the key was added after some homes were
  written, and a lost/stale default used to turn bare `ask` into `E_MODEL_NOT_FOUND: None is not a
  registry alias …` even though the choice was unambiguous. SPEC 2.7 is silent on a missing
  default; the card's decision slack picks the sole-alias fallback, and that is what
  `store.find_default` implements (`source="current"` / `source="sole"`);
* the no-model error, which named `None` as the thing that is not an alias and left the reader
  without the two remedies that exist (`models pull` / `models use`).

Two aliases with no `current` stay unresolved on purpose: *which* model to run is the user's call,
never a guess between models. Every explicit reference (`--model alias|path`) is untouched — the
third gate below pins that the CLI's own `requested -> resolved` note only appears for a call that
named no model at all.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import struct

import pytest

from typed_gguf import cli, schema
from typed_gguf.errors import TypedGgufError
from typed_gguf.registry import store


# ------------------------------------------------------------------ fixtures
def make_gguf(arch: str = "spark2_5", file_type: int = 7) -> bytes:
    """The tiny but valid GGUF `tests/test_cli_e1a.py` writes (header only, no tensors)."""
    def gstr(value: str) -> bytes:
        raw = value.encode()
        return struct.pack("<Q", len(raw)) + raw

    kvs = [gstr("general.architecture") + struct.pack("<I", 8) + gstr(arch),
           gstr("general.file_type") + struct.pack("<I", 4) + struct.pack("<I", file_type)]
    return (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 1) + struct.pack("<Q", len(kvs))
            + b"".join(kvs))


def home_with(tmp_path: pathlib.Path, *aliases: str,
              current: str | None = None) -> tuple[pathlib.Path, dict[str, store.Entry]]:
    """A data home whose registry lists `aliases`, each pointing at a real (tiny) file."""
    home = tmp_path / "home"
    entries: dict[str, store.Entry] = {}
    for index, alias in enumerate(aliases):
        model = home / "models" / f"{alias}.gguf"
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_bytes(make_gguf())
        entries[alias] = store.Entry(
            alias=alias, path=str(model), sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
            arch="spark2_5", quant="Q8_0", size=model.stat().st_size, license="apache-2.0",
            source="acme/fake", added_at="2026-09-24T00:00:00Z", file_type=7, repo="acme/fake")
    store.save_registry(store.Registry(aliases=entries, current=current),
                        store.registry_path(home))
    return home, entries


def registry_of(*aliases: str, current: str | None = None) -> store.Registry:
    """The same table in memory (no disk): the unit gates below do not need a home."""
    entries = {alias: store.Entry(alias=alias, path=f"/models/{alias}.gguf") for alias in aliases}
    return store.Registry(aliases=entries, current=current)


def bare_request() -> schema.Request:
    """The request a plain `ask --state … --noul …` builds: no `model` at all."""
    return schema.parse_request({"state": "hello", "questions": {"q1": {"type": "noul"}}})


# ------------------------------------------------------- store.find_default
def test_current_is_the_default_when_it_is_set() -> None:
    found = store.find_default(registry_of("aaa-small", "spark", current="spark"))
    assert found is not None
    assert (found.entry.alias, found.source) == ("spark", "current")


def test_a_sole_alias_is_the_default_when_current_is_unset() -> None:
    found = store.find_default(registry_of("stories260k"))
    assert found is not None
    assert (found.entry.alias, found.source) == ("stories260k", "sole")


def test_two_aliases_without_current_have_no_default() -> None:
    """Which of two models to run is the user's call (`models use`), never a guess."""
    assert store.find_default(registry_of("aaa-small", "spark")) is None


def test_a_stale_current_falls_back_to_the_sole_alias() -> None:
    """A `current` naming an alias that is gone must not hide the one that is left."""
    found = store.find_default(registry_of("stories260k", current="uninstalled"))
    assert found is not None
    assert (found.entry.alias, found.source) == ("stories260k", "sole")


def test_an_empty_registry_has_no_default() -> None:
    assert store.find_default(registry_of()) is None
    assert store.find_default(store.Registry()) is None


def test_resolve_hands_out_a_default_only_when_it_is_asked_for() -> None:
    """`use_current=False` (the default) keeps `None` meaning "the caller named nothing"."""
    registry = registry_of("stories260k")
    assert store.resolve(registry, None) is None
    assert store.resolve(registry, None, use_current=True).alias == "stories260k"


# ------------------------------------------------------- the CLI's own resolution
def test_a_bare_request_resolves_the_sole_alias(tmp_path: pathlib.Path,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    home, entries = home_with(tmp_path, "stories260k")
    monkeypatch.setenv("TYPED_GGUF_HOME", str(home))
    alias, path = cli._resolve_model(bare_request(), home=home)
    assert (alias, path) == ("stories260k", entries["stories260k"].path)


def test_a_bare_fit_resolves_the_sole_alias(tmp_path: pathlib.Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    """`fit` / `calibrate` reach the registry through `_resolve_model_ref(None)`."""
    home, entries = home_with(tmp_path, "stories260k")
    monkeypatch.setenv("TYPED_GGUF_HOME", str(home))
    assert cli._resolve_model_ref(None, home=home) == ("stories260k",
                                                      entries["stories260k"].path)


def test_two_aliases_without_current_still_refuse_a_bare_request(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home, _ = home_with(tmp_path, "aaa-small", "spark")
    monkeypatch.setenv("TYPED_GGUF_HOME", str(home))
    with pytest.raises(TypedGgufError) as exc:
        cli._resolve_model(bare_request(), home=home)
    assert exc.value.code == "E_MODEL_NOT_FOUND"
    assert "aaa-small" in str(exc.value) and "spark" in str(exc.value)


def test_an_explicit_reference_is_never_rerouted_by_the_default(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller who names a model keeps exactly what it named (the card's constraint)."""
    home, entries = home_with(tmp_path, "stories260k", current="stories260k")
    monkeypatch.setenv("TYPED_GGUF_HOME", str(home))
    elsewhere = tmp_path / "somewhere-else.gguf"
    elsewhere.write_bytes(make_gguf())
    assert cli._resolve_model_ref(str(elsewhere), home=home) == ("somewhere-else", str(elsewhere))
    assert cli._resolve_model_ref("stories260k", home=home) == ("stories260k",
                                                                entries["stories260k"].path)


# ------------------------------------------------------- the no-model error
def test_the_no_model_error_names_both_remedies(tmp_path: pathlib.Path,
                                                monkeypatch: pytest.MonkeyPatch,
                                                capsys: pytest.CaptureFixture[str]) -> None:
    """An empty registry is a dead end unless the message says what to do about it."""
    monkeypatch.setenv("TYPED_GGUF_HOME", str(tmp_path / "empty-home"))
    assert cli.main(["ask", "--state", "hello", "--noul", "q1=answer this"]) == 2
    err = capsys.readouterr().err
    assert "E_MODEL_NOT_FOUND" in err
    assert "models pull" in err and "models use" in err
    assert "--model" in err, "the escape hatch a caller already has must be named too"
    assert "None is not a registry alias" not in err, "`None` is not what the user asked for"
    assert "known aliases: <none>" in err, "the empty registry itself is worth naming"


def test_the_no_model_error_lists_the_aliases_it_could_not_choose_between(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    home, _ = home_with(tmp_path, "aaa-small", "spark")
    monkeypatch.setenv("TYPED_GGUF_HOME", str(home))
    assert cli.main(["ask", "--state", "hello", "--noul", "q1=answer this"]) == 2
    err = capsys.readouterr().err
    assert "aaa-small" in err and "spark" in err
    assert "models use" in err and "models pull" in err


# ------------------------------------------------------- the resolution is visible
def stub_decide(monkeypatch: pytest.MonkeyPatch, model: str = "stories260k") -> None:
    """Answer without a model: the CLI's own output is what these gates are about."""
    def fake(payload: dict, **_kwargs: object) -> dict:
        return {"model": model, "engine": {}, "answers": {}, "usage": {}, "timings": {}}
    monkeypatch.setattr(cli, "decide_payload_warm", fake)


def test_a_bare_ask_says_which_alias_it_resolved(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    home, _ = home_with(tmp_path, "stories260k")
    monkeypatch.setenv("TYPED_GGUF_HOME", str(home))
    stub_decide(monkeypatch)
    assert cli.main(["ask", "--state", "hello", "--noul", "q1=answer this"]) == 0
    captured = capsys.readouterr()
    assert "requested <default>" in captured.err, "the request named no model: say so"
    assert "resolved stories260k" in captured.err
    assert "only alias" in captured.err, "and why that alias is the default"
    assert "resolved" not in captured.out, "stdout stays the response, byte for byte"


def test_the_note_says_current_when_current_chose(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    home, _ = home_with(tmp_path, "aaa-small", "stories260k", current="stories260k")
    monkeypatch.setenv("TYPED_GGUF_HOME", str(home))
    stub_decide(monkeypatch)
    assert cli.main(["ask", "--state", "hello", "--noul", "q1=answer this"]) == 0
    err = capsys.readouterr().err
    assert "requested <default>" in err and "resolved stories260k" in err
    assert "current" in err and "only alias" not in err


def test_an_explicit_model_prints_no_note(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    """No behaviour change for a caller who already passes `--model`."""
    home, _ = home_with(tmp_path, "stories260k", current="stories260k")
    monkeypatch.setenv("TYPED_GGUF_HOME", str(home))
    stub_decide(monkeypatch)
    assert cli.main(["ask", "--state", "hello", "--noul", "q1=answer this",
                     "--model", "stories260k"]) == 0
    assert "requested" not in capsys.readouterr().err


def test_the_run_command_says_the_same_thing(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    home, _ = home_with(tmp_path, "stories260k")
    monkeypatch.setenv("TYPED_GGUF_HOME", str(home))
    questions = tmp_path / "q.json"
    questions.write_text(json.dumps({"state": "hello",
                                     "questions": {"q1": {"type": "noul"}}}), encoding="utf-8")
    stub_decide(monkeypatch)
    assert cli.main(["run", "--questions", str(questions)]) == 0
    captured = capsys.readouterr()
    assert "requested <default>" in captured.err and "resolved stories260k" in captured.err
