"""Public-docs gate (card `t_07b5cc01`): the README and the release notes read the way the code is.

Policy v2 (card `t_5b754458`) made `role_split` + `json_instructed` the product defaults. A public
document that says otherwise — or that quotes a **pre-v2** cell as "the default" — is exactly the
drift that passes every unit test and misleads the next reader, so the three things a stranger must
be able to trust are pinned here:

* the quickstart shows the defaults the *code* ships, read from `schema.Options()` rather than
  typed in as a literal (flip the defaults and this file fails with the README still quoting the
  old cell);
* the pre-v2 switch names (`answer_sheet`, `shipped`, `two_step`, `json_field`) appear only in
  lines that also name a switch, the flag, the cell or the policy they belong to — never as
  something a request gets by naming nothing;
* the release notes exist, carry the version from `pyproject.toml`, quote the measured rows they
  claim (the 4B default row and the two `qwen35moe` rows behind policy v2), and keep the two
  honesty statements the release must not lose (MIT + credits, and "serve/mcp are not in this
  release").

The docs are prose, so this gate is a floor, not a proof: it fails a document that *contradicts*
the code, and cannot bless one that misreads it.
"""
from __future__ import annotations

import dataclasses
import pathlib
import re
import tomllib

import pytest

from typed_gguf import cli, schema
from typed_gguf.errors import ERROR_CODES, WARNING_CODES

ROOT = pathlib.Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
NOTES = (ROOT / "docs" / "RELEASE_NOTES_v0.1.0.md").read_text(encoding="utf-8")
SPEC = (ROOT / "SPEC.md").read_text(encoding="utf-8")
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

#: the prompt-policy switches that are *not* the default any more (policy v2, card `t_5b754458`)
PRE_V2_SWITCHES = ("answer_sheet", "two_step", "json_field", "`shipped")
#: what makes a mention a mention-of-a-switch rather than a statement about the defaults
SWITCH_MARKERS = ("--cue", "--chat-format", "pre-v2", "pre-policy-v2", "switch", "flag", "cell",
                  "arm", "not the default", "published as", "stay reachable", "reachable")
#: docs the policy gate covers (the public pair a release ships)
PUBLIC = (("README.md", README), ("docs/RELEASE_NOTES_v0.1.0.md", NOTES))


def test_the_quickstart_quotes_the_defaults_the_code_ships() -> None:
    options = schema.Options()
    assert options.cue == "json_instructed" and options.chat_format == "role_split", (
        "policy v2 moved: update this file together with the public docs")
    # the quoted response is the request a reader gets with no policy flag at all
    assert f'"cue": "{options.cue}"' in README, "README no longer quotes the default cue"
    assert f'"kind": "{options.chat_format}"' in README, (
        "README no longer quotes the default chat format")
    assert f'"contract": "{options.json_contract}"' in README


def test_the_default_cue_and_chat_format_are_named_in_the_quickstart_prose() -> None:
    """The quoted block can be right while the sentence above it still describes the old cell."""
    quickstart = README.split("## Quickstart", 1)[1].split("\n## ", 1)[0]
    assert "`role_split`" in quickstart and "`json_instructed`" in quickstart
    assert "defaults" in quickstart


def test_the_pre_v2_switches_are_never_written_as_the_default() -> None:
    offenders = []
    for name, text in PUBLIC:
        for number, line in enumerate(text.splitlines(), start=1):
            if not any(switch in line for switch in PRE_V2_SWITCHES):
                continue
            if not any(marker in line for marker in SWITCH_MARKERS):
                offenders.append(f"{name}:{number}: {line.strip()[:140]}")
    assert offenders == [], (
        "a pre-v2 switch is named without saying what it is (a switch, a flag, a cell, or the "
        "policy a published row was measured under):\n" + "\n".join(offenders))


def test_the_release_notes_are_pinned_to_the_packaged_version() -> None:
    version = PYPROJECT["project"]["version"]
    assert version == "0.1.0", "this file pins the v0.1.0 notes: rename it with the version"
    head = NOTES.splitlines()[0]
    assert head.startswith("# typed-gguf ") and f"v{version}" in head


def test_the_release_notes_quote_the_measured_rows_they_claim() -> None:
    for needle in ("50/60", "53/60", "54/60", "22/60", "26/60"):
        assert needle in NOTES, f"the release notes no longer quote the measured row {needle!r}"
    assert "docs/BENCHMARKS.md" in NOTES, "the release notes must point at the tables"


def test_the_release_notes_keep_the_license_and_not_shipped_statements() -> None:
    assert "MIT" in NOTES
    assert "no parity claim" in NOTES                        # the adapter's posture stays explicit
    assert "not implemented in this release" in NOTES, (
        "the notes must say serve/mcp are specified but not shipped")
    assert "sequential" in NOTES.lower(), (
        "the notes must carry the per-question sequencing limit")


def test_the_readme_marks_the_serving_surface_as_not_shipped() -> None:
    interfaces = README.split("## Interfaces", 1)[1].split("\n## ", 1)[0]
    # card `t_c0080933` took the current-scope version numbers out of the README (the row says
    # "this release" now), so the pinned literal moves with the wording it quotes.
    assert "not implemented in this release" in interfaces
    # …and the claim is the CLI's own behaviour, not a wish: both commands are stubs by design
    assert cli.main(["serve"]) == 3 and cli.main(["mcp"]) == 3


def test_the_readme_keeps_the_deduplicated_shape() -> None:
    """The rewrite removed the duplicated developer scaffolding — a later paste must not bring it
    back (one `## Verification`, one `## License`, one quickstart)."""
    for title in ("## Quickstart", "## Verification", "## License"):
        assert README.count(title) == 1, f"{title} appears {README.count(title)} times"


# ------------------------------------------- the F1/F3/N2 polish (card t_a25bd190)
def test_the_root_help_marks_the_serving_surface_the_way_the_readme_does(
        capsys: pytest.CaptureFixture[str]) -> None:
    """Release review F1: the root help called `serve`/`mcp` *"implemented in E1b"* while the README
    and SPEC §2.9 say they are specified, not shipped. Card `t_bf6bb78a` then took the milestone
    jargon off the whole surface, so the honesty claim is pinned in the README's own plain words —
    and, because the tool no longer talks about milestones at all, in the product's own description
    of each command (the pattern gate over every page is `tests/test_cli_language.py`)."""
    assert cli.main(["--help"]) == 0
    out = capsys.readouterr().out
    lines = {line.split()[0]: line for line in out.splitlines()
             if line.startswith("  ") and line.strip()}
    for command in ("serve", "mcp"):
        line = lines[command]
        assert line.endswith("planned; not in this version"), line
    # …and a shipped command keeps its own description: the note is built from
    # COMMAND_DESCRIPTIONS, so losing that lookup prints "None"/"XX…XX" here. (Mutation sweep, card
    # t_a25bd190: these exact tails are what kills the padding mutants on the two branches this
    # pass rewrote; the tail is spelled out rather than read from the constant on purpose.)
    assert lines["run"].endswith("- answer a batch of questions from a file"), lines["run"]
    # …and the claim is the commands' own behaviour, not a wish: both stubs exit 3
    assert cli.main(["serve"]) == 3 and cli.main(["mcp"]) == 3


def test_the_limitations_carry_the_two_release_findings_this_pass_adds() -> None:
    """F3 and N2: the fit-plan cache only ever shrinks, and the exotic-platform wheels are future
    work (the stub is gone) — both are limitations a reader must be able to find."""
    limitations = README.split("## Limitations and known issues", 1)[1].split("\n## ", 1)[0]
    assert "never re-expanded" in limitations, (
        "F3: the limitations must say a cached fit plan is never re-expanded")
    assert "--no-fit-cache" in limitations, (
        "F3: the limitations must name the escape hatch (`--no-fit-cache`)")
    assert "wheel" in limitations and "future work" in limitations, (
        "N2: the limitations must say exotic-platform wheels are future work")


def test_the_spec_schema_lists_the_policy_options_the_code_ships() -> None:
    """F4: the README calls SPEC "the contract", and a reader who goes there for the wire schema
    must find the switches the shipped defaults are made of — read from `schema.Options()`, so a
    later default flip fails here instead of drifting."""
    section = SPEC.split("### 2.5 Native schema", 1)[1].split("### 2.6", 1)[0]
    options = schema.Options()
    for name, value in (("cue", options.cue), ("chat_format", options.chat_format),
                        ("json_contract", options.json_contract)):
        line = next((line for line in section.splitlines() if f'"{name}"' in line), None)
        assert line is not None, f"SPEC 2.5 lists no `{name}` option"
        assert f'"{value}"' in line, (
            f"SPEC 2.5 names `{name}` without its shipped default {value!r}: {line}")
    assert '"thinking"' in section, "SPEC 2.5 lists no `thinking` option"


# ------------------------------------------- the E4 + uvx catch-up (card t_67bb0409)
#: E4 (the warm engine host) and the uvx fix landed *after* the draft these notes were written from,
#: so the two things a reader cannot get anywhere else are pinned here: what the warm host bought
#: (in the live gate's own numbers) and how the out-of-tree install spells itself.
WARM_RECEIPT = ROOT / "docs" / "evidence" / "v0_1_0_t_7e24cea4_warm_host.md"
UVX_RECEIPT = ROOT / "docs" / "evidence" / "v0_1_0_t_eff926f9_uvx_install.md"
UVX_ONE_LINER = "uvx --from git+https://github.com/Rybens92/typed-gguf typed-gguf"
#: the facts the README (the `keep` row + the *Warm host* section) and the notes must both carry
WARM_FACTS = ("17.50", "2280", "2.58", "--keep-alive", "--keep-alive 0", "TYPED_GGUF_KEEP_ALIVE",
              "flag > env", "keep status", "keep stop", "one model at a time")


def test_the_notes_make_the_warm_host_the_headline_of_this_build() -> None:
    assert "keep host" in NOTES, "the notes must name the warm host (card t_7e24cea4)"
    assert NOTES.index("keep host") < NOTES.index("## Measured highlights"), (
        "the warm host is this build's headline: it belongs above the benchmark tables")


def test_the_warm_host_numbers_are_the_live_receipts_own() -> None:
    """No number without a receipt (requirement 5): every measured value the notes quote must still
    be the one `tests/test_keep_live.py` produced — re-measure, never re-quote from nowhere."""
    receipt = WARM_RECEIPT.read_text(encoding="utf-8")
    for needle in ("17.50", "2280", "2.58", "0 ms", "5.3", "6384", "2314", "6409"):
        assert needle in receipt, f"the E4 receipt lost {needle!r}: re-measure before re-quoting"
        assert needle in NOTES, f"the notes no longer quote the measured {needle!r}"
    assert "docs/evidence/v0_1_0_t_7e24cea4_warm_host.md" in NOTES, (
        "the warm-host headline needs its receipt (docs/evidence/)")


def test_the_notes_and_the_readme_agree_on_the_warm_host() -> None:
    """Requirement 3: the two public documents must not contradict each other — one fact list,
    asserted against both, so a later edit to either one fails here instead of drifting. Compared
    case-insensitively: a bullet that opens with a capital is the same fact."""
    for source, text in PUBLIC:
        lowered = text.lower()
        for needle in WARM_FACTS:
            assert needle.lower() in lowered, f"{source} no longer carries {needle!r}"


def test_the_notes_and_the_readme_carry_the_same_uvx_one_liner_with_its_honest_limit() -> None:
    assert UVX_ONE_LINER in README, "the README's uvx install line moved: update the notes with it"
    assert UVX_ONE_LINER in NOTES, "the notes must carry the uvx one-liner (card t_eff926f9)"
    for needle in ("out-of-tree", "git fetch", "post-publish"):
        assert needle in NOTES, f"the uvx limit must be stated honestly ({needle!r})"
    assert "docs/evidence/v0_1_0_t_eff926f9_uvx_install.md" in NOTES, (
        "the uvx claim needs its receipt (docs/evidence/)")


def test_no_public_doc_presents_a_default_the_schema_does_not_carry() -> None:
    """Requirement 3, second half: `schema.Options()` is where a native-request default lives. The
    warm host's window is a CLI flag / env knob (`keep/identity.py::DEFAULT_KEEP_ALIVE` = 600 s), so
    a document that spells it `options.keep_alive` sends the reader into `E_UNKNOWN_KEY`."""
    fields = {field.name for field in dataclasses.fields(schema.Options)}
    for source, text in PUBLIC:
        for hit in re.findall(r"options\.([a-z_]+)", text):
            assert hit in fields, f"{source} names `options.{hit}`, which schema.Options() lacks"
        assert '"keep_alive"' not in text, (
            f"{source} presents keep-alive as a request option; it is `--keep-alive` plus "
            "$TYPED_GGUF_KEEP_ALIVE, not part of the native schema")


def test_the_spec_catalog_is_the_frozen_registry() -> None:
    """F4: SPEC §2.5 is the contract's catalog, so every code the code may raise has to be findable
    there — the release review found the prompt-policy and bench-era additions missing."""
    section = SPEC.split("### 2.5 Native schema", 1)[1].split("### 2.6", 1)[0]
    missing = [code for code in (*ERROR_CODES, *WARNING_CODES) if f"`{code}`" not in section]
    assert missing == [], f"SPEC 2.5 does not list: {', '.join(missing)}"
