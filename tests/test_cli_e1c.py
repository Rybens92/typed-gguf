"""E1c CLI surface: per-command help (carried finding #1), the fit knobs, `fit` itself.

The card carried a UX gap from E1b verification: `typed-gguf ask --help` used to answer
`error: E_UNKNOWN_KEY: unknown option --help`. These tests pin the fix — every command explains
itself, and any unknown flag now points at that help.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from tests.test_fit import write_gguf
from typed_gguf import cli, schema
from typed_gguf.registry import store


# ------------------------------------------------------------------ per-command help
@pytest.mark.parametrize("command", [cmd for cmd in cli.COMMANDS if cmd != "models"])
def test_every_command_has_a_help_page(command: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([command, "--help"]) == 0
    out = capsys.readouterr().out
    assert f"typed-gguf {command}" in out
    assert f"milestone: {cli.MILESTONES[command]}" in out


@pytest.mark.parametrize("command", ["run", "ask", "fit"])
def test_the_help_page_lists_the_flags_that_command_takes(
        command: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([command, "--help"]) == 0
    out = capsys.readouterr().out
    for flag in cli.COMMAND_HELP[command]:
        assert flag in out


def test_the_help_page_covers_the_e1c_flags(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["run", "--help"]) == 0
    out = capsys.readouterr().out
    for flag in ("--template", "--thinking", "--no-fit", "--fit-target"):
        assert flag in out, flag
    assert cli.main(["fit", "--help"]) == 0
    assert "--no-cache" in capsys.readouterr().out


def test_models_subcommands_have_help_too(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["models", "--help"]) == 0
    out = capsys.readouterr().out
    for sub in cli.MODELS_SUBCOMMANDS:
        assert sub in out


@pytest.mark.parametrize("sub", cli.MODELS_SUBCOMMANDS)
def test_a_models_subcommand_help_page_names_the_subcommand_once(
        sub: str, capsys: pytest.CaptureFixture[str]) -> None:
    """Release review F1/F2 pass (card `t_a25bd190`): the usage line used to read
    `usage: typed-gguf models search search <query>` — the matched `COMMAND_HELP["models"]` entry
    already begins with the subcommand name, so the line must add it exactly once."""
    assert cli.main(["models", sub, "--help"]) == 0
    usage = capsys.readouterr().out.splitlines()[0]
    assert usage.startswith(f"usage: typed-gguf models {sub} "), usage
    assert not usage.startswith(f"usage: typed-gguf models {sub} {sub}"), usage


def test_an_unknown_flag_points_at_the_command_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["run", "--nope"]) == 2
    err = capsys.readouterr().err
    assert "E_UNKNOWN_KEY" in err
    assert "typed-gguf run --help" in err
    assert "Traceback" not in err


def test_help_does_not_swallow_a_real_flag(capsys: pytest.CaptureFixture[str]) -> None:
    """`--help` intercepts only itself: ordinary flags keep working next to it."""
    assert cli.main(["version", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "typed-gguf"
    assert cli.main(["doctor", "--help"]) == 0
    assert "usage: typed-gguf doctor" in capsys.readouterr().out


# ------------------------------------------------------------- the fit/template knobs
@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> dict:
    seen: dict = {}

    def fake_decide(payload: dict, *, home=None, **kwargs) -> dict:
        request = schema.parse_request(payload)
        seen["payload"] = payload
        seen["options"] = request.options
        seen["kwargs"] = kwargs
        return {"model": None, "engine": {}, "answers": {}, "usage": {}, "timings": {},
                "warnings": []}

    monkeypatch.setattr(cli, "decide_payload_warm", fake_decide)   # the E4 seam (see test_cli.py)
    return seen


def test_run_passes_the_fit_knobs_through(recorder: dict, tmp_path: pathlib.Path) -> None:
    questions = tmp_path / "q.json"
    questions.write_text(json.dumps({"state": "S", "questions": {
        "q": {"type": "choice", "criteria": {"a": None, "b": None}}}}), encoding="utf-8")
    assert cli.main(["run", "--questions", str(questions), "--no-fit", "--no-fit-cache"]) == 0
    assert recorder["kwargs"]["fit_enabled"] is False
    assert recorder["kwargs"]["fit_cache"] is False
    assert cli.main(["run", "--questions", str(questions), "--fit-target", "512",
                     "--fit-ctx", "2048"]) == 0
    assert recorder["kwargs"]["fit_target_mb"] == 512
    assert recorder["kwargs"]["fit_ctx"] == 2048


def test_the_template_and_thinking_flags_reach_the_request(
        recorder: dict, tmp_path: pathlib.Path) -> None:
    questions = tmp_path / "q.json"
    questions.write_text(json.dumps({"state": "S", "questions": {
        "q": {"type": "choice", "criteria": {"a": None, "b": None}}}}), encoding="utf-8")
    assert cli.main(["run", "--questions", str(questions), "--template", "plain",
                     "--thinking"]) == 0
    assert recorder["options"].template == "plain"
    assert recorder["options"].thinking is True
    assert cli.main(["run", "--questions", str(questions)]) == 0
    assert recorder["options"].template is None
    assert recorder["options"].thinking is False


def test_ask_accepts_the_fit_flags_too(recorder: dict) -> None:
    assert cli.main(["ask", "--state", "S", "--choice", "q=Which?:a|b", "--no-fit"]) == 0
    assert recorder["kwargs"]["fit_enabled"] is False
    assert recorder["options"].template is None


# --------------------------------------------------------------------------- fit
def test_fit_resolves_an_alias_from_the_registry(tmp_path: pathlib.Path,
                                                 capsys: pytest.CaptureFixture[str],
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("TYPED_GGUF_HOME", str(home))
    monkeypatch.delenv("TYPED_GGUF_RUNTIME_DIR", raising=False)
    model = write_gguf(tmp_path / "synthetic.gguf")
    registry, _warnings = store.load_registry(store.registry_path(home))
    store.add_entry(registry, store.Entry(alias="tiny", path=str(model), size=1), alias="tiny")
    store.save_registry(registry, store.registry_path(home))
    assert cli.main(["fit", "tiny", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["model"] == "tiny"
    assert payload["path"] == str(model)
    assert payload["source"] == "estimate"


def test_fit_writes_and_reuses_the_cache(tmp_path: pathlib.Path,
                                         capsys: pytest.CaptureFixture[str],
                                         monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("TYPED_GGUF_HOME", str(home))
    monkeypatch.delenv("TYPED_GGUF_RUNTIME_DIR", raising=False)
    model = write_gguf(tmp_path / "synthetic.gguf")
    assert cli.main(["fit", str(model), "--json"]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["cache"] is None or pathlib.Path(first["cache"]).exists()
    cached = list((home / "fit").glob("*.json"))
    assert len(cached) == 1
    assert cli.main(["fit", str(model), "--json"]) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["cache"] == first["cache"]
    assert cli.main(["fit", str(model), "--no-cache", "--json"]) == 0
    third = json.loads(capsys.readouterr().out)
    assert third["est_total_bytes"] == first["est_total_bytes"]


def test_fit_text_output_prints_the_fields_and_the_notes(
        tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYPED_GGUF_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("TYPED_GGUF_RUNTIME_DIR", raising=False)
    model = write_gguf(tmp_path / "synthetic.gguf")
    assert cli.main(["fit", str(model)]) == 0
    out = capsys.readouterr().out
    for field in ("n_ctx", "kv_type", "n_seq_max", "source", "backend"):
        assert f"{field}:" in out
    assert "source: estimate" in out
