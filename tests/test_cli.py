"""A-E1b-10: the CLI surface — `run` and `ask`, JSON on stdout, pinned exit codes 0/2/3/4.

Offline tests drive `cli.main()` with the decision layer replaced by the deterministic fake
session (so flag plumbing, request assembly and exit codes are pinned without a model); the
`model`-marked tests at the bottom run the real binary against a real GGUF.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

from tests.fake_engine import FakeSession, biased_row
from typed_gguf import cli, schema
from typed_gguf.engine import decide
from typed_gguf.registry import store


def _fake_decide(payload: dict, *, home: pathlib.Path | None = None, **kwargs) -> dict:
    """`cli.decide_payload` with the fake session: same validation, same rendering.

    `**kwargs` absorbs the E1c fit knobs (`fit_enabled`, `fit_target_mb`, `fit_ctx`,
    `fit_cache`) — the fake never loads a model, so a plan cannot be involved.
    """
    request = schema.parse_request(payload)
    session = FakeSession(n_vocab=512)
    word = "billing"
    billing = session.tokenize(word)[0]
    session.row_fn = (lambda ctx, session=session, billing=billing:
                      biased_row(session.n_vocab, {billing: 6.0}))
    result = decide.decide_request(request, session)
    return schema.render_response(result.payload(), format=request.format)


@pytest.fixture
def fake_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """The seam `run`/`ask` call: E4 made it `decide_payload_warm` (which owns keep-alive and
    falls back to the cold `decide_payload` when there is no host to be had)."""
    monkeypatch.setattr(cli, "decide_payload_warm", _fake_decide)


@pytest.fixture
def home(tmp_path: pathlib.Path) -> pathlib.Path:
    return tmp_path / "typed-gguf-home"


def test_run_reads_a_full_request_file(fake_engine, home, tmp_path, capsys) -> None:
    questions = tmp_path / "q.json"
    questions.write_text(json.dumps({
        "state": "Blank dashboard for everyone.",
        "questions": {"area": {"type": "choice", "criteria": {"billing": None, "api": None}}},
    }), encoding="utf-8")
    code = cli.main(["run", "--questions", str(questions)])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["answers"]["area"]["type"] == "choice"
    assert set(out) == {"model", "engine", "answers", "usage", "timings", "warnings",
                        "calibrated", "calibration"}


def test_run_accepts_a_bare_questions_map_plus_state_flags(fake_engine, home, tmp_path,
                                                           capsys) -> None:
    questions = tmp_path / "q.json"
    questions.write_text(json.dumps(
        {"area": {"type": "choice", "criteria": {"billing": None, "api": None}}}),
        encoding="utf-8")
    state = tmp_path / "state.txt"
    state.write_text("Blank dashboard.", encoding="utf-8")
    code = cli.main(["run", "--questions", str(questions), "--state", f"@{state}"])
    assert code == 0
    assert "area" in json.loads(capsys.readouterr().out)["answers"]


def test_run_writes_the_response_to_out_and_typesafe_drops_native_keys(fake_engine, home,
                                                                       tmp_path) -> None:
    questions = tmp_path / "q.json"
    questions.write_text(json.dumps(
        {"area": {"type": "choice", "criteria": {"billing": None, "api": None}}}),
        encoding="utf-8")
    out = tmp_path / "r.json"
    code = cli.main(["run", "--questions", str(questions), "--state", "S", "--format",
                     "typesafe", "--out", str(out)])
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert set(payload) == {"model", "answers", "usage"}
    assert set(payload["answers"]["area"]) == {"type", "choice", "probabilities", "confidence"}


def test_ask_builds_questions_from_flags(fake_engine, home, tmp_path, capsys) -> None:
    code = cli.main(["ask", "--state", "Blank dashboard for every user.",
                     "--choice", "area=Which team owns this?:billing|billing api",
                     "--score", "sev=How bad?:cosmetic|annoying|blocking",
                     "--noul", "page=Should we page?"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert set(out["answers"]) == {"area", "sev", "page"}
    assert out["answers"]["area"]["type"] == "choice"
    assert set(out["answers"]["sev"]["probabilities"]) == {"0", "1", "2"}
    assert set(out["answers"]["page"]) == {"type", "noul", "probabilities", "coverage",
                                          "reliability", "decode_steps", "cue"}
    # card t_6c119626: every answer carries the cue verdict it was read from (the closer, its
    # mass and the doc pointer when the row is a refusal) — a flat warnings list cannot
    assert {"refused", "token", "closer", "mass"} <= set(out["answers"]["page"]["cue"])
    assert out["answers"]["page"]["cue"]["refused"] is False


def test_ask_engine_flags_reach_the_request(fake_engine, home, tmp_path, capsys) -> None:
    code = cli.main(["ask", "--state", "S", "--readout", "single_token",
                     "--temperature", "0.5", "--n-seq-max", "5",
                     "--choice", "area=Which?:billing|api"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["engine"]["readout"] == "single_token"


def test_run_needs_a_questions_file(fake_engine, home, capsys) -> None:
    assert cli.main(["run"]) == 2
    assert "E_UNKNOWN_KEY" in capsys.readouterr().err


def test_ask_needs_a_state(fake_engine, home, capsys) -> None:
    assert cli.main(["ask", "--choice", "area=Which?:a|b"]) == 2
    assert "E_STATE_EMPTY" in capsys.readouterr().err


def test_ask_rejects_a_malformed_question_spec(fake_engine, home, capsys) -> None:
    assert cli.main(["ask", "--state", "S", "--choice", "area-without-labels"]) == 2
    assert "E_QID_INVALID" in capsys.readouterr().err
    assert cli.main(["ask", "--state", "S", "--choice", "area=Which?"]) == 2
    assert "E_CHOICE_CRITERIA" in capsys.readouterr().err


def test_a_missing_questions_file_is_a_user_error(fake_engine, home, capsys) -> None:
    assert cli.main(["run", "--questions", "/nonexistent/q.json"]) == 2
    assert "E_UNKNOWN_KEY" in capsys.readouterr().err


def test_schema_errors_exit_two_without_a_traceback(fake_engine, home, tmp_path, capsys) -> None:
    questions = tmp_path / "q.json"
    questions.write_text(json.dumps({"questions": {
        "q": {"type": "choice", "criteria": {}}}}), encoding="utf-8")
    assert cli.main(["run", "--questions", str(questions), "--state", "S"]) == 2
    err = capsys.readouterr().err
    assert "E_CHOICE_CRITERIA" in err and "Traceback" not in err


def test_runtime_errors_exit_three(fake_engine, home, tmp_path, capsys, monkeypatch) -> None:
    def boom(payload, *, home=None, **kwargs):
        from typed_gguf.errors import RuntimeMissingError
        raise RuntimeMissingError("E_RUNTIME_MISSING: no runtime installed")

    monkeypatch.setattr(cli, "decide_payload_warm", boom)     # the seam `run`/`ask` call (E4)
    assert cli.main(["ask", "--state", "S", "--choice", "a=Which?:x|y"]) == 3
    assert "E_RUNTIME_MISSING" in capsys.readouterr().err


def test_internal_errors_exit_four_and_never_leak_a_traceback(fake_engine, home, capsys,
                                                              monkeypatch) -> None:
    def boom(payload, *, home=None, **kwargs):
        raise ZeroDivisionError("kaboom")

    monkeypatch.setattr(cli, "decide_payload_warm", boom)     # the seam `run`/`ask` call (E4)
    assert cli.main(["ask", "--state", "S", "--choice", "a=Which?:x|y"]) == 4
    err = capsys.readouterr().err
    assert "E_INTERNAL" in err and "Traceback" not in err and "kaboom" in err


def test_unknown_model_alias_is_a_pinned_model_not_found(home) -> None:
    home.mkdir(parents=True, exist_ok=True)
    request = schema.parse_request({"state": "S", "model": "nope",
                                    "questions": {"a": {"type": "noul"}}})
    with pytest.raises(Exception) as exc:
        cli._resolve_model(request, home=home)
    assert getattr(exc.value, "code", None) == "E_MODEL_NOT_FOUND"


def test_a_registry_alias_resolves_to_its_path(home) -> None:
    home.mkdir(parents=True, exist_ok=True)
    entry = store.Entry(alias="spark", path="/models/spark.gguf", size=1, sha256="0" * 64,
                        arch="spark2_5", quant="Q8_0", added_at="now")
    registry = store.Registry()
    registry.aliases["spark"] = entry
    registry.current = "spark"
    store.save_registry(registry, store.registry_path(home))
    request = schema.parse_request({"state": "S", "model": "spark",
                                    "questions": {"a": {"type": "noul"}}})
    assert cli._resolve_model(request, home=home) == ("spark", "/models/spark.gguf")
    # a bare, unknown alias in typesafe mode falls back to the default alias (SPEC 2.6)
    typesafe = schema.parse_request({"state": "S", "model": "jev-latest", "format": "typesafe",
                                     "questions": {"a": {"type": "noul"}}})
    assert cli._resolve_model(typesafe, home=home) == ("spark", "/models/spark.gguf")


def test_a_model_path_is_used_directly(home, tmp_path) -> None:
    model = tmp_path / "some-model.gguf"
    model.write_bytes(b"GGUF")
    request = schema.parse_request({"state": "S", "model": str(model),
                                    "questions": {"a": {"type": "noul"}}})
    assert cli._resolve_model(request, home=home) == ("some-model", str(model))


def test_engine_request_payload_overlays_flags() -> None:
    body = cli.engine_request_payload(
        {"questions": {"a": {"type": "noul"}}}, state="S", model="m", fmt="typesafe",
        engine_options={"readout": "single_token"})
    assert body["state"] == "S" and body["model"] == "m" and body["format"] == "typesafe"
    assert body["options"] == {"readout": "single_token"}
    # a bare questions map is wrapped, and file-level options survive the overlay
    body = cli.engine_request_payload({"q": {"type": "noul"}}, engine_options={"threads": 2})
    assert body["questions"] == {"q": {"type": "noul"}}
    assert body["options"] == {"threads": 2}


# --------------------------------------------------------------- real end-to-end (A-E1b-10)
MODELS = {
    "qwen35": pathlib.Path.home() / ".cache" / "llama.cpp" / "Qwen3.5-0.8B-UD-Q4_K_XL.gguf",
    "spark2_5": pathlib.Path.home() / ".hermes" / "models" / "Spark-X2.5-4B-Q8_0.gguf",
}


def _runtime_dir() -> pathlib.Path:
    from typed_gguf.runtime import finder
    env = os.environ.get("TYPED_GGUF_RUNTIME_DIR")
    if env and (pathlib.Path(env) / "libllama.so").exists():
        return pathlib.Path(env)
    found = finder.find_runtime()
    if found:
        return found
    for base in (pathlib.Path.home() / ".hermes" / "runtime",
                 pathlib.Path.home() / ".local" / "share" / "typed-gguf" / "runtime"):
        for candidate in sorted(base.glob("*/")):
            if (candidate / "libllama.so").exists():
                return candidate
    pytest.skip("no llama.cpp runtime on this box (set TYPED_GGUF_RUNTIME_DIR or run init)")


@pytest.mark.model
def test_cli_run_and_ask_end_to_end_on_a_real_gguf(tmp_path, capsys) -> None:
    model = MODELS["qwen35"]
    if not model.exists():
        pytest.skip(f"{model} is not on this box")
    home = tmp_path / "home"
    environment = {**os.environ, "TYPED_GGUF_HOME": str(home),
                   "TYPED_GGUF_RUNTIME_DIR": str(_runtime_dir()),
                   "PYTHONPATH": str(pathlib.Path(__file__).resolve().parents[1] / "src")}
    questions = tmp_path / "q.json"
    questions.write_text(json.dumps(
        {"area": {"type": "choice", "criteria": {"billing": None, "technical": None}}}),
        encoding="utf-8")
    run = subprocess.run([sys.executable, "-m", "typed_gguf", "run", "--questions", str(questions),
                          "--state", "The billing page is blank for every user.",
                          "--model", str(model), "--threads", "4", "--state-id", "e1b-cli"],
                         capture_output=True, text=True, env=environment, timeout=600,
                         check=False)
    assert run.returncode == 0, run.stderr
    payload = json.loads(run.stdout)
    assert payload["answers"]["area"]["choice"] in ("billing", "technical")
    assert payload["engine"]["prefix_tokens"] > 0
    assert payload["usage"]["prefill_tokens"] == payload["engine"]["prefix_tokens"]

    ask = subprocess.run([sys.executable, "-m", "typed_gguf", "ask", "--state", "Billing is down.",
                          "--model", str(model), "--threads", "4",
                          "--noul", "page=Should we page the on-call engineer?"],
                         capture_output=True, text=True, env=environment, timeout=600,
                         check=False)
    assert ask.returncode == 0, ask.stderr
    assert 0.0 <= json.loads(ask.stdout)["answers"]["page"]["noul"] <= 1.0
    capsys.readouterr()

    bad = subprocess.run([sys.executable, "-m", "typed_gguf", "ask", "--state", "S",
                          "--choice", "broken"], capture_output=True, text=True,
                         env=environment, timeout=120, check=False)
    assert bad.returncode == 2 and "E_QID_INVALID" in bad.stderr
    # E_MODEL_NOT_FOUND is a user error (exit 2, E1a's registry classification)
    missing = subprocess.run([sys.executable, "-m", "typed_gguf", "run", "--questions",
                              str(questions), "--state", "S", "--model", "not-a-model"],
                             capture_output=True, text=True, env=environment, timeout=120,
                             check=False)
    assert missing.returncode == 2 and "E_MODEL_NOT_FOUND" in missing.stderr
    # a runtime problem is exit 3
    broken = {**environment, "TYPED_GGUF_RUNTIME_DIR": str(tmp_path / "no-runtime")}
    no_runtime = subprocess.run([sys.executable, "-m", "typed_gguf", "run", "--questions",
                                 str(questions), "--state", "S", "--model", str(model)],
                                capture_output=True, text=True, env=broken, timeout=120,
                                check=False)
    assert no_runtime.returncode == 3 and "E_RUNTIME_MISSING" in no_runtime.stderr
