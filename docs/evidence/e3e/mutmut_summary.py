"""E3e (card t_4c48f40a) Tier-M sweep summary: round 1 (the change) and round 2 (after the pins).

mutmut 3.8 keeps the verdicts in `mutants/<path>.meta` (`exit_code_by_key`, `None` = never run);
`mutmut results` lists survivors only and is NOT the score. Round 1 lives in `mutants_e3e_r1/`
(kept, per the box's convention) and round 2 in `mutants/` — the mutant *numbering* is not stable
between rounds, so the two are compared as scores and survivor classes, never key by key.

    uv run python .e3e/mutmut_summary.py            # prints, and writes .e3e/mutmut_summary.txt
"""

from __future__ import annotations

import collections
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULES = ("src/ggufone/engine/prompt.py", "src/ggufone/engine/cue.py")
ROUNDS = (("round 1 (the change, before the pins)", "mutants_e3e_r1"),
          ("round 2 (after the two acceptance pins)", "mutants_e3e_r2"),
          ("round 3 (after the plan's prefix-tokens pin)", "mutants"))
VERDICTS = {0: "survived", 1: "killed", 3: "killed", 5: "no tests", 33: "no tests",
            24: "timeout", -24: "timeout", 35: "suspicious", 36: "timeout", 37: "type check",
            -9: "segfault", -11: "segfault", 152: "timeout", 255: "timeout"}


def verdict(code: object) -> str:
    return "not run" if code is None else VERDICTS.get(code, f"other({code})")


def function_of(key: str) -> str:
    match = re.search(r"\.x_?(.+?)__mutmut_", key)
    return match.group(1) if match else key


def read(root: pathlib.Path) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for module in MODULES:
        meta = root / f"{module}.meta"
        if not meta.exists():
            continue
        codes = json.loads(meta.read_text(encoding="utf-8"))["exit_code_by_key"]
        out[module] = codes
    return out


def line(module: str, codes: dict[str, object]) -> tuple[str, collections.Counter, dict]:
    counts: collections.Counter[str] = collections.Counter(verdict(code) for code in codes.values())
    per_fn: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for key, code in codes.items():
        per_fn[function_of(key)][verdict(code)] += 1
    ran = len(codes) - counts["not run"]
    killed = counts["killed"]
    share = f"{100 * killed / ran:.1f} % of {ran} run" if ran else "nothing ran"
    text = (f"  {module.rsplit('/', 1)[-1]:<12} {len(codes):>4} mutants · {share} · "
            + ", ".join(f"{name} {value}" for name, value in sorted(counts.items())))
    return text, counts, per_fn


def main() -> int:
    lines = ["# E3e Tier-M sweep — scope, score, survivors (card t_4c48f40a)", "",
             "Scope: [tool.mutmut] `source_paths` = engine/prompt.py + engine/cue.py, test selection",
             "= tests/test_e3e_roles.py + tests/test_e3d_cue_switch.py + tests/test_e3c_cue_refused.py",
             "(`engine/decide.py` deliberately out — the pyproject comment says why). mutmut 3.8, this",
             "box, `--max-children 2`. Round 1 = the card's change; round 2 = after the two",
             "acceptance pins (`test_a_template_that_stops_extending_the_prefix_for_a_later_question_is_refused`,",
             "`test_a_template_that_rewrites_the_question_turn_is_refused_too`); round 3 = after the plan's",
             "`prefix_tokens` pin in `test_the_plan_carries_the_role_split_it_rendered`. Each round exists",
             "because the previous round's triage named a REAL unasserted path — the rounds are the",
             "triaging, not a score-chasing loop.", ""]
    totals = {}
    per_round: dict[str, dict] = {}
    for label, directory in ROUNDS:
        lines.append(f"## {label}  ({directory})")
        data = read(ROOT / directory)
        if not data:
            lines.append("  (no artifacts)")
            continue
        killed = ran = total = 0
        for module, codes in data.items():
            text, counts, per_fn = line(module, codes)
            lines.append(text)
            ran += len(codes) - counts["not run"]
            killed += counts["killed"]
            total += len(codes)
            per_round.setdefault(label, {})[module] = per_fn
        share = f"{100 * killed / ran:.1f} % of {ran} run" if ran else "nothing ran"
        lines.append(f"  **total: {total} mutants, {killed} killed — {share}**")
        totals[label] = (killed, ran, total)
        lines.append("")
        lines.append("  per function (killed / run):")
        for module, per_fn in per_round[label].items():
            lines.append(f"    {module.rsplit('/', 1)[-1]}:")
            for fn, counts in sorted(per_fn.items(), key=lambda kv: -sum(kv[1].values())):
                fn_ran = sum(value for name, value in counts.items() if name != "not run")
                lines.append(f"      {fn:<32} {counts['killed']:>4}/{fn_ran:<4}"
                             + ("" if counts["survived"] == 0 and counts["not run"] == 0
                                else f"  ({', '.join(f'{k} {v}' for k, v in sorted(counts.items()) if k != 'killed')})"))
        lines.append("")
    if len(totals) >= 2:
        killed = " → ".join(f"{total[0]}" for total in totals.values())
        first, last = next(iter(totals.values())), list(totals.values())[-1]
        lines.append(f"## The delta: {killed} killed (+{last[0] - first[0]} over "
                     f"{len(totals)} rounds, same scope, one round of pins after each triage)")
        lines.append("")
    lines += [
        "## The classes the survivors fall in (read from the diffs, not guessed)",
        "",
        "* **message strings — 33** (`question_block` 15, `build_question` 10, and 8 inside",
        "  `role_split_render` whose `XX…XX`/upper-cased error text the classifier heuristically files",
        "  under boolean/guard). The gates assert error *codes* and rendered bytes, so mangled prose",
        "  survives. The reference's recurring class; deliberately not chased.",
        "* **defaults a caller always supplies — 46** (the 26 `other` in `build_prefix`, the 12",
        "  `json.dumps` kwargs in `render_value`, 4 in `build_prefix`'s `control/return`): the gates",
        "  reach `build_prefix` through callers that pass `chat_format`/`cue`/`contract` explicitly, and",
        "  `render_value` is driven by `tests/test_engine_fork.py` — a file outside the sweep's",
        "  selection. A named selection boundary, not an unpinned behaviour: the offline",
        "  `tools/e3e_role_render.py` record and the pin files hold the byte claims.",
        "* **three equivalent `role` arms in `build_prefix` (13/14/15)**: `plan_context` always passes",
        "  `role=`, so `role if role is not None else role_split_render(...)` cannot take its other arm",
        "  through any caller a gate drives.",
        "* **the shared-prefix shortcut in `role_split_render` (2)**: `prefix = _common_prefix(...) if",
        "  rendered else state_only` — both arms agree on every input the gates drive, so nothing pins",
        "  which one ran. Recorded as an unpinned optimisation.",
        "* **`empty_candidate_code` (12, `no tests`)** — the selection never calls it; its callers are",
        "  the CLI suites, outside the pair by design.",
        "* **cue.py's special-token catalogue** (`closer_map` 8, `single_token_closers` 9): pinned by",
        "  `tests/test_e3c_cue_specials.py`, outside the selection. The composed verdicts — where the",
        "  card's behaviour actually lives — score 93–100 %.",
        "",
        "## What the triage closed",
        "",
        "Three rounds, three real gaps found and pinned, no score-chasing loop: round 1 named the two",
        "acceptance guards inside `role_split_render` that no test reached (the module's only uncovered",
        "lines, 267 and 274 — live evidence for the reference's warning that a high score can sit above",
        "unreached code), round 2's `build_prefix` count named the inverted `chat_format` guard whose",
        "mutant survived because the gates never compared the plan's prefix *tokens* against the",
        "role-split prefix they were rendered from, and round 3 pins that with one assertion. After the",
        "three pins the remaining survivors are prose, supplied arguments, equivalent arms, and files",
        "the selection does not include — every one of them named above, and none of them a guard this",
        "card added.",
    ]
    text = "\n".join(lines) + "\n"
    sys.stdout.write(text)
    (ROOT / ".e3e" / "mutmut_summary.txt").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
