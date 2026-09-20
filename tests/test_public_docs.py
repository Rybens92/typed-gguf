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

import pathlib
import tomllib

from typed_gguf import cli, schema

ROOT = pathlib.Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
NOTES = (ROOT / "docs" / "RELEASE_NOTES_v0.1.0.md").read_text(encoding="utf-8")
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
    assert "not implemented in v0.1.0" in interfaces
    # …and the claim is the CLI's own behaviour, not a wish: both commands are stubs by design
    assert cli.main(["serve"]) == 3 and cli.main(["mcp"]) == 3


def test_the_readme_keeps_the_deduplicated_shape() -> None:
    """The rewrite removed the duplicated developer scaffolding — a later paste must not bring it
    back (one `## Verification`, one `## License`, one quickstart)."""
    for title in ("## Quickstart", "## Verification", "## License"):
        assert README.count(title) == 1, f"{title} appears {README.count(title)} times"
