"""The runtime-matrix oracle step's known-skip filter (card t_f96fed7f, AC1).

`.github/workflows/runtime-matrix.yml` is the live acceptance harness: it unpacks the pinned
llama.cpp bundle and runs `docs/verify_runtime_contract.py` against it, and a skip in that run is
treated as a failure — a distribution path that silently regressed must not read green. The first
run the harness ever had (36151935399, 2026-09-25) failed on that rule in both Linux jobs, with
exactly two skips:

    SKIP /home/runner/.hermes/models/Spark-X2.5-4B-Q8_0.gguf absent — sha256 download-verify
    evidence not re-run
    SKIP local Spark GGUF absent — header-parse pins not re-run

Both are *inherent* on a runner: they are about a 4.4 GB model the card forbids downloading. The
assertion was written before the harness ever ran, so the policy was never exercised.

`tools/matrix_oracle_skips.py` is the policy, named: the two model-absent skips are allowed
(path-independently), every other skip refuses the step, and the filter refuses a run whose own
`failures: … skips: …` summary disagrees with the lines it printed.

The `MATRIX_RUN` fixture below is that run's output (the two SKIP lines verbatim, from the job log
in the card); `_run` is the tool as the workflow calls it — a pipe, exit code and all.
"""
from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "matrix_oracle_skips.py"

#: The two SKIP lines run 36151935399 printed, copied from the job log — path and em dashes intact.
MATRIX_SKIPS = (
    "  SKIP /home/runner/.hermes/models/Spark-X2.5-4B-Q8_0.gguf absent — sha256 download-verify "
    "evidence not re-run",
    "  SKIP local Spark GGUF absent — header-parse pins not re-run",
)
#: The oracle's own tail line for that run (docs/verify_runtime_contract.py `main`).
MATRIX_SUMMARY = "failures: 0  skips: 2"
#: What the failing job printed around the skips: the oracle's live section, on a runner with the
#: bundle but without the model.
MATRIX_RUN = "\n".join([
    "typed-gguf runtime contract oracle — repo /home/runner/work/typed-gguf/typed-gguf",
    "",
    "[B] live runtime probes (ctypes vs pinned release)",
    "  runtime dir: /tmp/typed-gguf-rt",
    "  ok   ctypes resolves all 34 required symbols (0 missing)",
    "  ok   libggml.so resolves the backend loader (2/2) — PoC pitfall 1",
    *MATRIX_SKIPS,
    "",
    "=== reference table (executed) ===",
    "",
    MATRIX_SUMMARY,
    "",
])
#: The same run on a box that *does* hold the file: the two skips become real checks.
CLEAN_RUN = "\n".join([
    "[B] live runtime probes (ctypes vs pinned release)",
    "  runtime dir: /tmp/typed-gguf-rt",
    "  ok   local Spark Q8_0 sha256 == HF lfs.oid (download-verify contract)",
    "",
    "failures: 0  skips: 0",
    "",
])


def load_tool():
    """The tool as a module (it is a script under `tools/`, not an installed package)."""
    if not TOOL.exists():
        raise AssertionError(
            f"{TOOL.relative_to(ROOT)} is missing: the matrix's oracle steps pipe the oracle's "
            "output through it, so without it the harness has the old `grep -q SKIP` policy — "
            "which failed the first run it ever had (card t_f96fed7f)")
    spec = importlib.util.spec_from_file_location("matrix_oracle_skips", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module          # a dataclass needs its module in `sys.modules`
    spec.loader.exec_module(module)
    return module


def _run(text: str, *args: str) -> subprocess.CompletedProcess[str]:
    """The tool exactly as the workflow runs it: the oracle's stdout on stdin, exit code out."""
    return subprocess.run([sys.executable, str(TOOL), *args], input=text, capture_output=True,
                          text=True, timeout=60)


# ------------------------------------------------------- the policy: named, not substring-wide
def test_the_two_inherent_skips_of_the_first_matrix_run_are_allowed() -> None:
    result = _run(MATRIX_RUN)
    assert result.returncode == 0, (
        "the two model-absent skips are inherent on a runner (the 4.4 GB Spark GGUF is not there "
        "by design), so this is the run the harness must accept:\n" + result.stdout + result.stderr)
    assert "model-absent skips: 2" in result.stdout, result.stdout
    for line in MATRIX_SKIPS:
        assert line.strip()[len("SKIP "):] in result.stdout, result.stdout


def test_an_injected_third_skip_is_refused() -> None:
    injected = ("  SKIP llama-cli binary absent in runtime dir "
                "(binaries not required for the engine)")
    result = _run(MATRIX_RUN.replace(MATRIX_SUMMARY, f"{injected}\n\nfailures: 0  skips: 3"))
    assert result.returncode != 0, (
        "a skip this card did not name is exactly what the harness exists to catch:\n"
        + result.stdout + result.stderr)
    assert injected.strip() in result.stderr, result.stderr
    assert "model-absent skips: 2" in result.stdout, (
        "the two named skips are still reported as allowed")


def test_a_run_with_no_skips_is_green() -> None:
    result = _run(CLEAN_RUN)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "model-absent skips: 0" in result.stdout, result.stdout


def test_a_near_miss_message_is_still_refused() -> None:
    """"local Spark GGUF absent" is not a licence for every sentence that starts with it."""
    near = "  SKIP local Spark GGUF absent — header-parse pins skipped for now"
    result = _run(CLEAN_RUN.replace("failures: 0  skips: 0", f"{near}\n\nfailures: 0  skips: 1"))
    assert result.returncode != 0, result.stdout + result.stderr


def test_the_word_skip_inside_another_line_is_not_a_skip() -> None:
    """The policy is about skip *lines*: the old `grep -q SKIP` over the whole output could not
    tell a skip from a message that mentions one."""
    text = CLEAN_RUN.replace("failures: 0  skips: 0",
                             "  ok   the SKIP allow-list matches the two model-absent pins\n"
                             "\n  FAIL the SKIP list drifted\n\nfailures: 1  skips: 0")
    assert _run(text).returncode == 0, (
        "no line starts with SKIP, so this run has no skips to explain (its FAIL is the oracle's "
        "business, and `set -e` in the workflow is what catches that)")
    module = load_tool()
    assert module.skip_lines(text) == []


def test_a_summary_that_disagrees_with_the_lines_is_refused() -> None:
    """A capture that lost a skip line must not read as "no such skip"."""
    text = MATRIX_RUN.replace(MATRIX_SUMMARY, "failures: 0  skips: 3")
    result = _run(text)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "skips: 3" in result.stderr, result.stderr


def test_a_run_without_the_oracle_summary_is_refused() -> None:
    """The filter reads a *complete* oracle run; a truncated capture is not one."""
    text = "\n".join([*MATRIX_SKIPS, ""])
    result = _run(text)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "summary" in result.stderr.lower(), result.stderr


def test_the_allow_list_is_path_independent() -> None:
    """The first skip names `$HOME`, which is the runner's path — not a stable substring."""
    text = MATRIX_RUN.replace("/home/runner/.hermes/models", "/root/.hermes/models")
    assert _run(text).returncode == 0, "the same skip under another home is the same skip"


def test_a_skip_about_another_reason_is_refused() -> None:
    """"no runtime installed" is a distribution story, never an allowance."""
    runtime = "  SKIP no runtime installed (set TYPED_GGUF_RUNTIME_DIR or run `typed-gguf init`)"
    text = CLEAN_RUN.replace("failures: 0  skips: 0", f"{runtime}\n\nfailures: 0  skips: 1")
    assert _run(text).returncode != 0, text


def test_the_filter_can_read_a_file_instead_of_stdin(tmp_path: pathlib.Path) -> None:
    captured = tmp_path / "oracle.out"
    captured.write_text(MATRIX_RUN, encoding="utf-8")
    result = subprocess.run([sys.executable, str(TOOL), "--input", str(captured)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_allow_list_names_why_each_skip_is_acceptable() -> None:
    module = load_tool()
    reasons = dict(module.KNOWN_SKIPS)
    assert len(reasons) == 2, f"this card names exactly two inherent skips: {sorted(reasons)}"
    for line in MATRIX_SKIPS:
        hits = [substring for substring in reasons if substring in line]
        assert len(hits) == 1, f"{line!r} matches {hits}"
        assert reasons[hits[0]], "every allowance carries the reason it is one"


def test_the_usage_error_is_not_a_silent_green() -> None:
    """A no-argument call reads stdin; an unreadable `--input` is an error, never an empty run."""
    result = subprocess.run([sys.executable, str(TOOL), "--input", "/nonexistent/oracle.out"],
                            input="", capture_output=True, text=True, timeout=60)
    assert result.returncode != 0, result.stdout + result.stderr
    assert result.returncode != 2 or "usage" in (result.stderr + result.stdout).lower(), (
        result.stderr)


@pytest.mark.parametrize("line", MATRIX_SKIPS)
def test_each_matrix_skip_line_is_classified_by_the_tool(line: str) -> None:
    module = load_tool()
    assert module.classify(line) is not None, f"{line!r} must be an allowed skip"
