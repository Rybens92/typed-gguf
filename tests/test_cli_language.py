"""The CLI's user-visible language: plain product wording, no developer-process jargon.

Card `t_bf6bb78a`. Every string a user can reach through the CLI — the root `--help`, each
command's help page (including `typed-gguf models <sub> --help`) and the message the two
not-yet-shipped commands print — has to read as product language. Milestones, milestone codes
(`E1a`, `E2.5`), "(implemented in …)" notes and bare "SPEC §…" pointers are internal planning
artefacts: they may live in the source's own docstrings and comments (that is where the history
belongs) and must never show up in output.

The gate is a *pattern* gate, not a wording gate: the descriptions stay free to change, what may
never come back is the jargon. `test_the_gate_flags_the_strings_this_card_removed` proves the gate
can fail — it feeds it the exact strings this card took off the surface — and
`test_the_gate_does_not_flag_the_product_s_own_vocabulary` keeps the pattern honest (the frozen
`E_*` error codes and bare flags must never trip it).
"""
from __future__ import annotations

import re

import pytest

from typed_gguf import cli

#: The four forms the card bans, verbatim.
JARGON = ("milestone", "implemented in", "SPEC §")
#: A milestone code as a standalone token (`E1a`, `E4`) — never part of a name. Matches inside
#: "E1a-E4" too (the token boundaries are what make that work).
MILESTONE_CODE = re.compile(r"\bE[0-9][a-z]?\b")
#: What the root help says about the two commands that are specified but not shipped.
PLANNED = "planned; not in this version"


def _jargon_in(text: str) -> list[str]:
    """Every banned form in `text` — an empty list means the gate passes."""
    lowered = text.lower()
    found = [needle for needle in JARGON if needle.lower() in lowered]
    found += [match.group(0) for match in MILESTONE_CODE.finditer(text)]
    return found


def _help_output(capsys: pytest.CaptureFixture[str], argv: list[str]) -> str:
    assert cli.main(argv) == 0, argv
    return capsys.readouterr().out


def _root_help_lines(out: str) -> dict[str, str]:
    """The `commands:` block, keyed by command: one `  <cmd> …` line each."""
    return {line.split()[0]: line for line in out.splitlines()
            if line.startswith("  ") and line.strip()}


# ------------------------------------------------------------------------- the gate
def test_the_root_help_is_free_of_developer_jargon(
        capsys: pytest.CaptureFixture[str]) -> None:
    out = _help_output(capsys, ["--help"])
    assert _jargon_in(out) == [], _jargon_in(out)


@pytest.mark.parametrize("command", cli.COMMANDS)
def test_every_command_help_is_free_of_developer_jargon(
        command: str, capsys: pytest.CaptureFixture[str]) -> None:
    out = _help_output(capsys, [command, "--help"])
    assert _jargon_in(out) == [], (command, _jargon_in(out))


@pytest.mark.parametrize("sub", cli.MODELS_SUBCOMMANDS)
def test_every_models_subcommand_help_is_free_of_developer_jargon(
        sub: str, capsys: pytest.CaptureFixture[str]) -> None:
    out = _help_output(capsys, ["models", sub, "--help"])
    assert _jargon_in(out) == [], (sub, _jargon_in(out))


@pytest.mark.parametrize("command", cli.NOT_IMPLEMENTED)
def test_the_not_shipped_message_is_free_of_developer_jargon(
        command: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([command]) == 3
    err = capsys.readouterr().err
    assert _jargon_in(err) == [], (command, _jargon_in(err))


# ------------------------------------------------- the language the user actually reads
def test_every_command_line_in_the_root_help_carries_a_plain_description(
        capsys: pytest.CaptureFixture[str]) -> None:
    out = _help_output(capsys, ["--help"])
    lines = _root_help_lines(out)
    assert set(lines) == set(cli.COMMANDS), sorted(lines)
    assert set(cli.COMMAND_DESCRIPTIONS) == set(cli.COMMANDS), sorted(
        cli.COMMAND_DESCRIPTIONS)
    for command in cli.COMMANDS:
        description = cli.COMMAND_DESCRIPTIONS[command]
        assert description in lines[command], (command, lines[command])


@pytest.mark.parametrize("command", ["init", "doctor", "models", "ask", "run", "fit", "keep"])
def test_a_help_page_that_used_to_be_a_usage_line_alone_says_what_the_command_does(
        command: str, capsys: pytest.CaptureFixture[str]) -> None:
    out = _help_output(capsys, [command, "--help"])
    assert cli.COMMAND_DESCRIPTIONS[command] in out, out
    assert "run `typed-gguf --help` for the command list" in out, out


def test_the_two_commands_that_are_not_shipped_are_labelled_planned(
        capsys: pytest.CaptureFixture[str]) -> None:
    out = _help_output(capsys, ["--help"])
    lines = _root_help_lines(out)
    for command in cli.NOT_IMPLEMENTED:
        assert cli.COMMAND_DESCRIPTIONS[command] == PLANNED
        assert lines[command].endswith(PLANNED), lines[command]


@pytest.mark.parametrize("command", cli.NOT_IMPLEMENTED)
def test_the_not_shipped_message_is_plain_and_still_exits_3(
        command: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([command]) == 3
    err = capsys.readouterr().err
    assert f"'{command}' is not available in this version (planned)" in err, err
    assert "typed-gguf --help" in err, err


# --------------------------------------------------------------- the gate can fail
#: The strings this card took off the surface, quoted as they were in the tree before it (see the
#: card's evidence: `git show <parent>:src/typed_gguf/cli.py`).
OLD_STRINGS = (
    "  run          (implemented in E1b)",
    "milestone: E1a",
    "'serve' is not implemented yet (milestone E1b); see SPEC.md 5",
    "  serve        (specified in SPEC §2.9, not implemented in v0.1.0; exits 3)",
)


@pytest.mark.parametrize("old", OLD_STRINGS)
def test_the_gate_flags_the_strings_this_card_removed(old: str) -> None:
    assert _jargon_in(old), f"the gate let the old string {old!r} through"


@pytest.mark.parametrize("plain", (
    "error: E_UNKNOWN_KEY: unknown option --nope",
    "usage: typed-gguf fit [<model>] --print --no-cache",
    "answer a batch of questions from a file",
    "E_BACKEND_OOM: the backend could not allocate device memory for the plan",
))
def test_the_gate_does_not_flag_the_product_s_own_vocabulary(plain: str) -> None:
    assert _jargon_in(plain) == [], _jargon_in(plain)
