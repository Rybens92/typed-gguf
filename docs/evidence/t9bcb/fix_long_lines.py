#!/usr/bin/env python3
"""Split/shorten the E501 lines of this card's own scripts (each replacement asserted once).

Rewrites are (old, new) pairs applied verbatim; the script fails loudly if a pair does not match
exactly once, so it can never half-edit a file. Prose splits keep the rendered text identical; the
few shortenings only drop redundant words (the numbers all come from `stats.json`).
"""
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

EDITS: dict[str, list[tuple[str, str]]] = {
    ".t9bcb/analyse.py": [
        ("* the corrected-instrument baseline of card `t_7c926398` — "
         "`docs/evidence/tiel_corrected_quality.json`\n"
         "  plus its own per-chunk reports and placement sinks "
         "(`docs/evidence/tiel_corrected_chunks/`);",
         "* the corrected-instrument baseline of card `t_7c926398` —\n"
         "  `docs/evidence/tiel_corrected_quality.json`, plus its own per-chunk reports and\n"
         "  placement sinks (`docs/evidence/tiel_corrected_chunks/`);"),
        ('            "mismatch_warnings": sorted({str(w).split(":")[0] '
         'for w in (report.get("warnings") or [])\n',
         '            "mismatch_warnings": sorted({str(w).split(":")[0]\n'
         '                                         for w in (report.get("warnings") or [])\n'),
        ('                smokes["collapse"], '
         'load(pathlib.Path(ROOT / aux["two_step_role_split"]["merged"]))),\n',
         '                smokes["collapse"],\n'
         '                load(pathlib.Path(ROOT / aux["two_step_role_split"]["merged"]))),\n'),
    ],
    ".t9bcb/render_doc.py": [
        ('            f"`measured` {cell[\'measured\']}/{cell[\'items\']}, refusals '
         '{cell[\'refusals\']}/{cell[\'items\']}")\n',
         '            f"`measured` {cell[\'measured\']}/{cell[\'items\']}, refusals "\n'
         '            f"{cell[\'refusals\']}/{cell[\'items\']}")\n'),
        ('        f"Paired by item (exact McNemar + the closed-form interval, '
         '`tools/e3e_roles_decision.py`): "\n',
         '        f"Paired by item (exact McNemar + the closed-form interval, "\n'
         '        f"`tools/e3e_roles_decision.py`): "\n'),
        ('        f"{base[\'refusals\']}/{base[\'items\']} refusals, `low_mass` '
         '{base[\'low_mass\']}/{base[\'items\']}, "\n',
         '        f"{base[\'refusals\']}/{base[\'items\']} refusals, "\n'
         '        f"`low_mass` {base[\'low_mass\']}/{base[\'items\']}, "\n'),
        ('           f"{_num(aux_role[\'coverage_quantiles\'][\'p50\'])}: the placement moves the '
         'question out of "\n',
         '           f"{_num(aux_role[\'coverage_quantiles\'][\'p50\'])}: the placement moves "\n'
         '           f"the question out of "\n'),
        ('         ["the pair is identical", '
         'f"`{log[\'sha_identical\']}` over `{log[\'sha_lines\']}` sha line(s)"],\n',
         '         ["the pair is identical",\n'
         '          f"`{log[\'sha_identical\']}` over `{log[\'sha_lines\']}` sha line(s)"],\n'),
        ('        f"committed at `{stats[\'generated_from\'][\'baseline\']}`; '
         '**not re-measured here** (the card\'s "\n',
         '        f"committed at `{stats[\'generated_from\'][\'baseline\']}`; '
         '**not re-measured here** "\n'
         '        f"(the card\'s "\n'),
        # (`... it is not used anywhere in "` → two shorter lines) was applied by an earlier run of
        # this script and is therefore not repeated here: the skip rule above keeps it idempotent.
        ('        "runtime   /home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan '
         '(pinned b11026)",\n',
         '        "runtime   /home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan",\n'
         '        "          (pinned b11026)",\n'),
        ('        "driver    .t9bcb/run_chunks.sh → .t9bcb/tiel_e3e_arm.py '
         '(the committed observer + the flags)",\n',
         '        "driver    .t9bcb/run_chunks.sh → .t9bcb/tiel_e3e_arm.py "\n'
         '        "(the committed observer + the flags)",\n'),
        ('                       for dev, count in sorted((chunk.get("device_buffers") or '
         '{}).items())) or "—",\n',
         '                       for dev, count in sorted(\n'
         '                           (chunk.get("device_buffers") or {}).items())) or "—",\n'),
        ('        "* the row shape of a `quality` row carries no per-item `effective_backend`, '
         'so \\"per row\\" "\n',
         '        "* the row shape carries no per-item `effective_backend`, so \\"per row\\" "\n'),
        ('                     _num(paired.get("difference")) '
         'if paired.get("difference") is not None else "—",\n',
         '                     _num(paired.get("difference"))\n'
         '                     if paired.get("difference") is not None else "—",\n'),
        ('    lines += _table(["type", "baseline", "challenger", '
         '"discordant (challenger-only / baseline-only)",\n',
         '    lines += _table(["type", "baseline", "challenger",\n'
         '                     "discordant (challenger-only / baseline-only)",\n'),
        ('    lines += ["", "**The cue verdicts (`value_verdict` breakdown of the instructed '
         'contract).**", ""]\n',
         '    lines += ["", "**The cue verdicts (`value_verdict` breakdown).**", ""]\n'),
        ('            f"{arm[\'vs_baseline\'][\'difference\']:+.3f} '
         '(p = {_num(arm[\'vs_baseline\'][\'mcnemar_p\'])})",\n',
         '            f"{arm[\'vs_baseline\'][\'difference\']:+.3f} "\n'
         '            f"(p = {_num(arm[\'vs_baseline\'][\'mcnemar_p\'])})",\n'),
        ('            f"{stats[\'aux\'][\'role_split_only\'][\'cell\'][\'refusals\']}/60) and '
         '`decide._advance_token` "\n',
         '            f"{stats[\'aux\'][\'role_split_only\'][\'cell\'][\'refusals\']}/60) and "\n'
         '            f"`decide._advance_token` "\n'),
        ('            f"never advances past a cue the model closed (the rule `t_7c926398` §5.1 '
         'measured on the "\n',
         '            f"never advances past a cue the model closed (the rule "\n'
         '            f"`t_7c926398` §5.1 measured on the "\n'),
        ('            f"shipped placement). The 4B\'s collapse cell (28/60) is the *other* branch '
         'of that rule: "\n',
         '            f"shipped placement). The 4B\'s collapse cell (28/60) is the "\n'
         '            f"*other* branch of that rule: "\n'),
        ('            f"there the label sits **at** the cue row, the advance succeeds, and the row '
         'it reaches "\n',
         '            f"there the label sits **at** the cue row, the advance succeeds, and "\n'
         '            f"the row it reaches "\n'),
        ('            f"({moved.get(\'decisions_identical\')}/{moved.get(\'n\')} identical '
         'decisions) and is what "\n',
         '            f"({moved.get(\'decisions_identical\')}/{moved.get(\'n\')} identical "\n'
         '            f"decisions) and is what "\n'),
        ('        f"{base[\'correct\']}/{base[\'items\']} is measured on the same committed items, '
         'the same model "\n',
         '        f"{base[\'correct\']}/{base[\'items\']} is measured on the same committed "\n'
         '        f"items, the same model "\n'),
        ('        "| test suite | `uv run --frozen --offline --extra dev pytest -q -rs '
         '-p no:cacheprovider` | "\n',
         '        "| test suite | `uv run --frozen --offline --extra dev pytest -q -rs` | "\n'),
        ('        "* the same numbers in the benchmark document: `docs/BENCHMARKS.md` §7.4.2, '
         'spliced by this "\n',
         '        "* the same numbers in the benchmark document: `docs/BENCHMARKS.md` §7.4.2, "\n'
         '        "spliced by this "\n'),
    ],
}


def main() -> int:
    for relative, pairs in EDITS.items():
        path = ROOT / relative
        text = path.read_text(encoding="utf-8")
        for old, new in pairs:
            count = text.count(old)
            if count == 0 and new in text:
                continue                      # already applied (the script is re-runnable)
            if count != 1:
                raise SystemExit(f"{relative}: {count} match(es) for {old[:60]!r} — refusing")
            text = text.replace(old, new)
        path.write_text(text, encoding="utf-8")
        print(f"{relative}: {len(pairs)} edit(s) applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
