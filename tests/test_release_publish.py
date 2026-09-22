"""Release mechanics gate (card `t_6027fb77`): the publish workflow and the version that ships.

`publish.yml` is the one file that can put an artifact on PyPI, and PyPI's trusted publisher for
this project names it *literally* — owner `Rybens92`, repository `typed-gguf`, workflow name
`publish.yml`, environment `pypi`. Those four values are a contract, not a style choice: a renamed
file or a renamed environment turns the upload into a 403 that no log explains, and a job that
reaches for a token secret is a release that cannot work at all (the OIDC permission —
`id-token: write` — is the only credential this repository is allowed to have).

The other half is the version the release carries. `pyproject.toml` and `typed_gguf.__version__`
are two spellings of one number, the release notes are *named* for it (`test_public_docs.py` says
so in its own failure message), and the workflow's gate compares the **built artifact** against
the tag. All four are pinned together here.

The workflow's gate script is not read but *executed* (bash, in a tmp dir with a fake `dist/`):
"a mismatch refuses to publish" is the one behaviour a release cannot get wrong, and a regex over
a `run:` block cannot prove it. The YAML itself is parsed with PyYAML (a `dev`-extra dependency):
a workflow that does not parse is a workflow that never runs.
"""
from __future__ import annotations

import os
import pathlib
import re
import subprocess
import tomllib

import yaml

from typed_gguf import __version__

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
WORKFLOW = WORKFLOWS / "publish.yml"
#: the workflow *file name* PyPI's publisher is configured with — never rename it alone
WORKFLOW_NAME = "publish.yml"
#: the environment the trusted publisher declares, and the page the release points at
ENVIRONMENT = "pypi"
ENVIRONMENT_URL = "https://pypi.org/project/typed-gguf/"
#: the upload itself: Trusted Publishing forced on (the flag is what makes OIDC the credential)
PUBLISH_RUN = "uv publish --trusted-publishing always"
#: the runner/actions the other workflows are pinned to (mirror them, do not float)
CHECKOUT = "actions/checkout@v4"
SETUP_UV = "astral-sh/setup-uv@v5"
RUNNER = "ubuntu-24.04"

WORKFLOW_TEXT = WORKFLOW.read_text(encoding="utf-8") if WORKFLOW.exists() else ""
WORKFLOW_YAML = yaml.safe_load(WORKFLOW_TEXT) if WORKFLOW_TEXT else {}
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
VERSION = PYPROJECT["project"]["version"]
NOTES = ROOT / "docs" / f"RELEASE_NOTES_v{VERSION}.md"
#: the docs gate that pins the notes to the packaged version, and the sentence it asks the next
#: bump to obey ("this file pins the v0.1.0 notes: rename it with the version")
DOCS_GATE = ROOT / "tests" / "test_public_docs.py"
#: credential spellings a Trusted-Publishing workflow must not carry (the file may *talk* about
#: tokens — these are the ways one becomes a credential)
TOKEN_SURFACE = ("secrets.", "--token", "--password", "api-token", "api_token", "UV_PUBLISH_TOKEN")


def _publish() -> dict:
    """The workflow, or a dict with a message instead — a missing file fails with the reason."""
    if not WORKFLOW.exists():
        raise AssertionError(
            f"{WORKFLOW.relative_to(ROOT)} is missing: PyPI's trusted publisher names it, so "
            "without it there is no release path at all")
    return WORKFLOW_YAML


def _triggers(workflow: dict) -> dict:
    """`on:` — PyYAML's YAML 1.1 resolution reads the bare key as the boolean `True`.

    GitHub Actions' own parser does not (the file stays spelled `on:`), so the gate has to accept
    whichever the parser hands back instead of pinning the quirk.
    """
    return workflow.get("on") or workflow.get(True) or {}


def _job() -> dict:
    return _publish().get("jobs", {}).get("pypi", {})


def _steps() -> list[dict]:
    return _job().get("steps", [])


def _run_steps() -> list[str]:
    return [step["run"] for step in _steps() if "run" in step]


def _gate_script() -> str:
    hits = [run for run in _run_steps() if "refs/tags/" in run]
    assert len(hits) == 1, (
        f"publish.yml must carry exactly one tag-vs-artifact version gate, found {len(hits)}")
    return hits[0]


def _run_gate(workdir: pathlib.Path, ref: str, ref_name: str,
              wheels: tuple[str, ...] = (VERSION,)) -> subprocess.CompletedProcess[str]:
    """Run the workflow's own gate step against a fake `dist/`, the way the job would."""
    dist = workdir / "dist"
    dist.mkdir(parents=True, exist_ok=True)
    for version in wheels:
        (dist / f"typed_gguf-{version}-py3-none-any.whl").write_bytes(b"")
    return subprocess.run(
        ["bash", "-c", _gate_script()], cwd=workdir, capture_output=True, text=True, timeout=60,
        env={**os.environ, "GITHUB_REF": ref, "GITHUB_REF_NAME": ref_name})


def _output(result: subprocess.CompletedProcess[str]) -> str:
    return result.stdout + result.stderr


# ------------------------------------------------------------------- the publisher contract


def test_the_workflow_is_the_file_the_pypi_publisher_names() -> None:
    """Owner/repository/workflow-name/environment: the four values PyPI matches on."""
    assert WORKFLOW.parent == WORKFLOWS and WORKFLOW.name == WORKFLOW_NAME, (
        f"PyPI's trusted publisher is configured for {WORKFLOW_NAME!r}; this file is "
        f"{WORKFLOW.relative_to(ROOT)}")
    assert _publish().get("name") == "publish", "the workflow's display name should say so"


def test_the_triggers_are_the_published_release_and_the_manual_dispatch() -> None:
    triggers = _triggers(_publish())
    assert triggers.get("release", {}).get("types") == ["published"], (
        "the release is the human decision to ship: `release: types: [published]`")
    assert "workflow_dispatch" in triggers, "the re-run path must stay manual-dispatched"
    # …and nothing else: a push or a pull request must never reach an upload
    assert set(triggers) == {"release", "workflow_dispatch"}, (
        f"unexpected trigger surface: {sorted(triggers)}")


def test_the_upload_credential_is_oidc_only() -> None:
    """`id-token: write` is the credential; every token spelling is a second one to leak."""
    assert _publish().get("permissions") == {"contents": "read"}, (
        "the workflow's own scope is read-only; the job asks for the one extra it needs")
    assert _job().get("permissions") == {"id-token": "write"}, (
        "Trusted Publishing is the job-level `id-token: write` exchange")
    offenders = [needle for needle in TOKEN_SURFACE if needle in WORKFLOW_TEXT]
    assert offenders == [], (
        f"publish.yml names a credential ({offenders}): Trusted Publishing needs no token, and "
        "a token in the workflow is a token in the release path")


def test_the_job_runs_in_the_environment_pypi_declares() -> None:
    job = _job()
    assert job.get("runs-on") == RUNNER, f"the other workflows run on {RUNNER}"
    assert job.get("environment") == {"name": ENVIRONMENT, "url": ENVIRONMENT_URL}, (
        f"PyPI's publisher declares the environment {ENVIRONMENT!r}, and the run should link "
        "the project page")


def test_the_steps_are_checkout_uv_build_gate_and_publish() -> None:
    """The order is the contract: nothing may run before the build except the checkout + uv."""
    steps = _steps()
    uses = [step["uses"] for step in steps if "uses" in step]
    assert uses == [CHECKOUT, SETUP_UV], f"unexpected action surface: {uses}"
    runs = [run.strip() for run in _run_steps()]
    assert len(runs) == 3, f"expected build + version gate + publish, found {len(runs)} run steps"
    build, gate, publish = runs
    assert build == "uv build", build
    assert "refs/tags/" in gate, "the second run step is the version gate"
    assert publish == PUBLISH_RUN, publish


def test_the_publish_step_keeps_the_attestations_and_takes_no_credential() -> None:
    publish = [step for step in _steps() if step.get("run", "").strip() == PUBLISH_RUN]
    assert len(publish) == 1, f"expected one publish step, found {len(publish)}"
    assert set(publish[0]) == {"name", "run"}, (
        f"the publish step takes no inputs beyond its command: {sorted(publish[0])}")
    assert "--no-attestations" not in WORKFLOW_TEXT, (
        "the PEP 740 attestations are the provenance the OIDC upload exists for — keep them on")


def test_this_repository_has_exactly_one_publisher() -> None:
    """A second workflow that could upload is a second release path to keep trusted."""
    publishers = sorted(path.name for path in WORKFLOWS.glob("*.yml")
                        if "uv publish" in path.read_text(encoding="utf-8"))
    assert publishers == [WORKFLOW_NAME], f"workflows that publish: {publishers}"


# ------------------------------------------------------------------- the gate's own behaviour


def test_the_version_gate_passes_when_the_artifact_matches_the_tag(tmp_path) -> None:
    result = _run_gate(tmp_path, "refs/tags/v0.1.1", "v0.1.1")
    assert result.returncode == 0, _output(result)
    assert f"built version: {VERSION}" in result.stdout, _output(result)


def test_the_version_gate_refuses_an_artifact_that_is_not_the_tag(tmp_path) -> None:
    result = _run_gate(tmp_path, "refs/tags/v9.9.9", "v9.9.9")
    assert result.returncode != 0, (
        "a wheel that is not the release must never reach PyPI:\n" + _output(result))
    assert "::error::" in _output(result), _output(result)


def test_the_version_gate_prints_and_continues_on_a_manual_dispatch(tmp_path) -> None:
    result = _run_gate(tmp_path, "refs/heads/main", "main")
    assert result.returncode == 0, _output(result)
    assert f"built version: {VERSION}" in result.stdout, _output(result)


def test_the_version_gate_fails_closed_when_nothing_was_built(tmp_path) -> None:
    """No artifact in `dist/` is not "no mismatch" — it is a run that must stop."""
    result = _run_gate(tmp_path, "refs/tags/v0.1.1", "v0.1.1", wheels=())
    assert result.returncode != 0, _output(result)


# ------------------------------------------------------------------- the version that ships


def test_the_packaged_version_is_the_two_spellings_of_one_number() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", VERSION), (
        f"pyproject's version is not a release number: {VERSION!r}")
    assert __version__ == VERSION, (
        f"typed_gguf.__version__ ({__version__}) is not pyproject's version ({VERSION}): the "
        "wheel would report one number and the tag would carry another")


def test_the_release_notes_are_named_and_titled_for_the_packaged_version() -> None:
    assert NOTES.exists(), (
        f"{NOTES.relative_to(ROOT)} is missing: the release notes for the packaged version are "
        "part of the release")
    head = NOTES.read_text(encoding="utf-8").splitlines()[0]
    assert head.startswith("# typed-gguf ") and f"v{VERSION}" in head, head


def test_the_docs_gate_is_renamed_with_the_version_it_pins() -> None:
    """`test_public_docs` fails with "this file pins the v0.1.0 notes: rename it with the version"
    when the packaged version moves — this is that instruction, made executable: the two version
    pins (this file's derivation and the docs gate's literal) cannot drift apart."""
    pins = DOCS_GATE.read_text(encoding="utf-8")
    assert f'"RELEASE_NOTES_v{VERSION}.md"' in pins, (
        f"the docs gate still reads another release-notes file than v{VERSION}")
    assert f'assert version == "{VERSION}"' in pins, (
        f"the docs gate still pins another packaged version than {VERSION}")
