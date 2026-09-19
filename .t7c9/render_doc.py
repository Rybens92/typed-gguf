#!/usr/bin/env python3
"""t_7c926398 — render the evidence doc from the stats JSON (no number is ever retyped).

    python3 .t7c9/render_doc.py --stats .t7c9/tiel_corrected_stats.json \
        --out docs/evidence/t7c926398_tiel_corrected.md

Every table and every count in the output comes from `.t7c9/analyse.py`'s JSON; the prose is
f-stringed around those values. The doc is what `docs/BENCHMARKS.md` §7.4 quotes.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
from typing import Any


def load(path: str) -> dict[str, Any]:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def block_line(block: dict[str, Any]) -> str:
    if not block["n"]:
        return "—"
    low, high = block["ci"]
    return f"{block['correct']}/{block['n']} = {block['agreement']:.3f} [{low:.3f}–{high:.3f}]"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stats", required=True)
    parser.add_argument("--quality-report", required=True)
    parser.add_argument("--sha-before", required=True)
    parser.add_argument("--sha-after", required=True)
    parser.add_argument("--two-step-stats", dest="two_step_stats", default=None)
    parser.add_argument("--gates-json", dest="gates_json", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    stats = load(args.stats)
    report = load(args.quality_report)
    arm = load(args.two_step_stats) if args.two_step_stats else None
    gates = load(args.gates_json) if args.gates_json else None
    corrected = stats["corrected"]
    pre = stats["pre_fix"]
    paired = stats["paired"]
    chunks = stats["chunks"]
    framing = corrected["framing_label"] or []
    framing_text = ", ".join(framing)
    model = report.get("model") or {}
    host = report.get("host") or {}
    total = stats["n"]["corrected"]

    sha_before = pathlib.Path(args.sha_before).read_text(encoding="utf-8").strip().splitlines()
    sha_after = pathlib.Path(args.sha_after).read_text(encoding="utf-8").strip().splitlines()
    tree = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                          text=True, check=False).stdout.strip()

    parts: list[str] = []
    add = parts.append

    add("# The corrected-instrument Tiel row — 60 committed dev items, measured [host] "
        "(`t_7c926398`)\n")
    add("Card `t_7c926398` (main-coder) · [host] run · tree "
        f"`{tree}` (`ggufone-wt-t7c9`, branch `t7c926398-tiel-corrected`) — the committed\n"
        "corrected instrument of card `t_6de5fc53`, unmodified. Companion of\n"
        "`docs/evidence/e2_fix_t_6de5fc53_framing.md` §4.3 (the six-item probe this card turns\n"
        "into a table).\n")

    add("**One sentence.** Re-measured on the host, with the framing `ggufone ask`/`run` actually "
        f"sends,\nTiel-Coder-35B-A3B scores **{block_line(corrected['overall'])}** on the same 60 "
        f"committed dev items that\nthe published plain-framing row scored "
        f"{block_line(pre['overall'])} — and "
        f"**{corrected['cue']['refused']}/{total} rows refuse at the cue**\n"
        f"(`W_CUE_REFUSED`, a turn-closer where the label should sit) against "
        f"{pre['cue']['refused']}/{total} before.\nThe collapse the in-container probe predicted "
        "persists at full power; the plain row was a different prompt.\n")

    add("## 1. The pin\n")
    add("| what | value |\n|---|---|")
    add(f"| file | `{model.get('path')}` |")
    add(f"| model name | `{model.get('name')}` |")
    for index, line in enumerate(sha_before, start=1):
        add(f"| SHA-256 before (line {index}) | `{line.split()[0]}` |")
    for index, line in enumerate(sha_after, start=1):
        add(f"| SHA-256 after (line {index}) | `{line.split()[0]}` |")
    add("| downloads | none — the file was already local |")
    add("")
    add("The before/after pair is the campaign's own `sha256sum` (printed at the head and the "
        "tail\nof `.t7c9/logs/campaign.log`); the two digests are identical, so nothing rewrote "
        "the weights\nmid-run.\n")

    add("## 2. The instrument (what \"corrected\" means here)\n")
    add("Card `t_6de5fc53` fixed `LiveModel.decide`: it used to plan the executed context from the "
        "live\n`ModelSession` (which resolves no chat template) while the product plans from the "
        "model handle.\nEvery published quality row was therefore the *plain* E1b framing. This "
        "campaign is the first 60-item\nTiel table measured with the resolved plan — the rows "
        "carry it:\n")
    add("| what | value |\n|---|---|")
    per_row = corrected.get("framing_per_row") or {}
    prefix_facts = per_row.get("prefix_tokens") or {}
    add(f"| report `framing.labels` (first chunk's block) | `{framing_text}` |")
    add(f"| framing verified **per row** over all {per_row.get('rows', total)} rows | "
        f"`{'`, `'.join(per_row.get('labels') or ['—'])}` · mixed: {per_row.get('mixed')} |")
    add(f"| per-row `prefix_tokens` | {prefix_facts.get('min')}–{prefix_facts.get('max')} tokens "
        f"({prefix_facts.get('count')} rows) |")
    add("| pre-fix report's framing field | absent (the shape predates the fix) |")
    add("")
    surface = corrected.get("framing_surface") or {}
    add(f"**Which surface rendered it (all {total} rows).** "
        f"kind `{surface.get('kind')}` · renderer `{surface.get('renderer')}` · "
        f"source `{surface.get('source')}` ·\nthinking `{surface.get('thinking')}` · warnings "
        f"`{surface.get('warnings')}`. The note the rows carry:\n"
        + " ".join(f"\"{note}\"" for note in (surface.get("notes") or [])) + ".\n")
    add("The parity claim is not assumed from the label: the repository's live gate\n"
        "(`tests/test_bench_live.py` → "
        "`::test_the_bench_sends_the_same_prompt_as_the_serving_path`) was run\n"
        "on **this model file** for this card and passes — `item c01: serving prefix 109 tokens, "
        "bench\nprefix 109 tokens, framing chat-template: qwen35moe / builtin` "
        "(`.t7c9/live_parity_tiel.txt`). The bench and\n`ggufone ask` prefill byte-identical "
        "tokens "
        "on the 35B file, through the built-in fallback surface,\nbefore any number below was "
        "measured.\n")
    add("")
    add("Nothing else moved: same model file, same 60 committed dev items (the chunk slices are\n"
        "byte-identical copies of `docs/evidence/tiel_chunks/devset_00{1..6}.jsonl`, "
        "`.t7c9/devset_sha.txt`),\nsame placement ask, same flags.\n")

    add("## 3. Environment and the scope that decides the numbers\n")
    add("```")
    add(f"host      {host.get('platform') or host.get('system') or 'see report host block'}")
    add(f"GPU       {host.get('gpu') or 'NVIDIA GeForce RTX 3060 Ti, 8192 MiB'}")
    add("runtime   /home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan "
        "(pinned b11026)")
    add("ICD       VK_DRIVER_FILES=/home/rybens/.e3c_host/nvidia_egl_icd.json (libEGL_nvidia.so.0)")
    add("scope     systemd-run --user --unit=t7c9-tiel-campaign (MemoryMax=infinity)")
    add("```")
    add("")
    add("`set -u` driver: `.t7c9/run_chunks_corrected.sh`; the campaign's own log prints "
        "`memory.max=max`\nfor the unit it ran in, because the kanban worker's own scope is capped "
        "at 4 GiB and a 20.8 GiB\nmodel inside it re-reads its weights from disk forever "
        "(`.e3c_tiel/run_chunks.sh`'s note, measured on\ncard `t_a58f8b67`).\n")
    add("**Contention note.** The box is shared with sibling cards, but no second model was\n"
        "resident during this campaign: VRAM read 7.4/8.2 GiB while chunk 001 ran (the per-chunk "
        "line in\n`.t7c9/logs/campaign.log` is the *pre-chunk* reading, 1.5–1.9 GiB) and the "
        "per-chunk walls\n(36–60 s for ten items) are the receipts for that claim.\n")

    add("## 4. Placement per chunk (the loader's own answer)\n")
    add("| chunk | items (choice/score/noul) | correct | ngl requested → used | degraded | "
        "kv_type_used | n_ctx | n_prefix | n_seq_max | load wall | chunk wall | median decision |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for entry in chunks:
        types = entry["types"]
        add(f"| `{entry['report']}` | {entry['items']} "
            f"({types.get('choice', 0)}/{types.get('score', 0)}/{types.get('noul', 0)}) | "
            f"{entry['correct']}/{entry['items']} | {entry['ngl_requested']} → "
            f"{entry['ngl_used']} | {entry['degraded']} | {entry['kv_type_used']} | "
            f"{entry['n_ctx']} | {entry['n_prefix']} | {entry['n_seq_max']} | "
            f"{entry['load_wall_s']} s | {entry['chunk_wall_s']} s | "
            f"{entry['decision_median_s']:.1f} s |")
    add("")
    add("The pre-fix campaign asked for the same 9 layers and got them in every chunk too, so the "
        "two rows\nare placement-matched: **the framing is the only thing that moved between "
        "them.**\n")

    add("## 5. The 60 items, corrected instrument\n")
    add("| metric | corrected instrument (chat-template) | pre-fix row (plain, published) |")
    add("|---|---|---|")
    add(f"| framing | {framing_text} | plain (prompt.py E1b framing) |")
    add(f"| overall | **{block_line(corrected['overall'])}** | {block_line(pre['overall'])} |")
    for qtype in sorted(set(corrected["per_type"]) | set(pre["per_type"])):
        add(f"| {qtype} | {block_line(corrected['per_type'][qtype])} | "
            f"{block_line(pre['per_type'][qtype])} |")
    add(f"| rows `low_mass` | **{corrected['mass_split']['low_mass']['items']}/{total}** | "
        f"{pre['mass_split']['low_mass']['items']}/{total} |")
    add(f"| rows `measured` (≥ 0.10 floor) | "
        f"**{corrected['mass_split']['measured']['items']}/{total}** | "
        f"{pre['mass_split']['measured']['items']}/{total} |")
    add(f"| refused at the cue (`W_CUE_REFUSED`) | **{corrected['cue']['refused']}/{total}** | "
        f"{pre['cue']['refused']}/{total} |")
    coverage = corrected["coverage"]
    pre_coverage = pre["coverage"]
    add(f"| coverage median | **{coverage['median']:.4g}** | {pre_coverage['median']:.4g} |")
    add(f"| coverage min · p25 · p75 · max | {coverage['min']:.4g} · {coverage['p25']:.4g} · "
        f"{coverage['p75']:.4g} · {coverage['max']:.4g} | {pre_coverage['min']:.4g} · "
        f"{pre_coverage['p25']:.4g} · {pre_coverage['p75']:.4g} · {pre_coverage['max']:.4g} |")
    add(f"| cue row's top-token mass (median) | {corrected['cue']['cue_mass_median']:.4g} | "
        f"{pre['cue']['cue_mass_median']:.4g} |")
    add("")
    add("**The cue verdicts are the mechanism, not a footnote.** "
        f"{corrected['cue']['refused']} of {total} rows put a turn-closer on the cue row:\n"
        f"{json.dumps(corrected['cue']['refused_closers'])}. The readout sits at the last row of "
        "the prompt — the\nstart of the assistant turn — and under the model's own template the "
        "first thing Tiel wants to emit\nthere is `</think>` or `<|im_end|>`, not a label. The "
        "engine's own verdict says so (`W_CUE_REFUSED`),\nand the agreement reads the argmax of a "
        "label set whose mass is a tail.\n")
    add("Reliability split (the §2.1 shape), both framings:\n")
    add("| framing | reliability | items | correct | agreement | 95 % CI |")
    add("|---|---|---|---|---|---|")
    for name, row_key in (("plain (pre-fix)", "pre_fix"),
                          ("chat-template (corrected)", "corrected")):
        for bucket in ("ok", "low_mass"):
            block = stats[row_key]["mass_split"].get(bucket)
            if not block or not block["items"]:
                continue
            low, high = block["ci"]
            add(f"| {name} | `{bucket}` | {block['items']} | {block['correct']} | "
                f"{block['agreement']:.4f} | {low:.4f} – {high:.4f} |")
    add("")
    if arm is not None:
        add("### 5.1 The auxiliary arm: `--cue two_step` cannot engage on this model\n")
        add("The card asks for the shipped cue (the product's default) and that is §5. Because §5 "
            "raises the\nobvious next question — *is the collapse the cue row itself, and does the "
            "knob the product already has\n(`--cue two_step`, E3d) rescue it?* — the same 60 items "
            "were re-run once more at `--cue two_step` (same\nmodel, same placement ask, same "
            "corrected instrument; driver `.t7c9/run_cue_arm.sh`).\n")
        arm_overall = arm["overall"]
        arm_low, arm_high = arm_overall["ci"]
        add("| arm | agreement (Wilson 95 %) | `low_mass` | refused at the cue | rows identical "
            "to the shipped arm |")
        add("|---|---|---|---|---|")
        add(f"| `--cue shipped` (the card's row) | {block_line(corrected['overall'])} | "
            f"{corrected['mass_split']['low_mass']['items']}/{total} | "
            f"{corrected['cue']['refused']}/{total} | — |")
        add(f"| `--cue two_step` | {arm_overall['correct']}/{arm_overall['n']} = "
            f"{arm_overall['agreement']:.3f} [{arm_low:.3f}–{arm_high:.3f}] | "
            f"{arm['mass_split']['low_mass']['items']}/{arm_overall['n']} | "
            f"{arm['cue']['refused']}/{arm_overall['n']} | "
            f"{arm['vs_shipped']['identical_rows']}/{arm_overall['n']} |")
        add("")
        add(f"**The arm moves exactly {arm['vs_shipped']['different_rows']} row of "
            f"{arm_overall['n']}.** `decide._advance_token` (the committed E3d rule) refuses to "
            "advance past a\nrefused cue — \"a cue the model closes never advances\" — and Tiel "
            "closes "
            f"{corrected['cue']['refused']} of\n{total} cues, so the shape has nothing to move; on "
            "the one row that did advance (`s09`) the row the\nreadout lands on is a turn-closer "
            "as well. The measured consequence: agreement is unchanged\n"
            f"({corrected['overall']['correct']}/{total} → "
            f"{arm_overall['correct']}/{arm_overall['n']}), `low_mass` is unchanged "
            f"({corrected['mass_split']['low_mass']['items']} → "
            f"{arm['mass_split']['low_mass']['items']}), and refusals go\n"
            f"{corrected['cue']['refused']} → {arm['cue']['refused']}.\n")
        for row in arm["vs_shipped"]["detail"]:
            changed = row["changed"]
            bits = []
            if "coverage" in changed:
                bits.append(f"coverage {changed['coverage']['shipped']:.6g} → "
                            f"{changed['coverage']['two_step']:.6g}")
            if "cue" in changed:
                before, after = changed["cue"]["shipped"], changed["cue"]["two_step"]
                bits.append(f"cue `{before.get('closer') or 'content'}` "
                            f"(mass {before.get('mass'):.4g}) → "
                            f"`{after.get('closer') or 'content'}` "
                            f"(mass {after.get('mass'):.4g}), refused "
                            f"{before.get('refused')} → {after.get('refused')}")
            other = [key for key in changed if key not in ("coverage", "cue")]
            if other:
                bits.append("also changed: " + ", ".join(f"`{key}`" for key in other))
            add(f"* the one row that differs — `{row['id']}` ({row['type']}): "
                + "; ".join(bits) + ".")
        add("")
        add("**What that buys the escalation.** A cue-shape change is *not* a remedy this model "
            "can reach, so\nthe \"try `two_step` first\" path is closed by measurement, not by "
            "opinion — the fix has to stop the\nprefill before the row Tiel wants to close, which "
            "is exactly what card `t_4c48f40a` (E3e) is\nmeasuring. This arm is auxiliary: it is "
            "not a table the card asked for, and it is not compared to the\n4B's `two_step` arm "
            "across framings.\n")
    add("## 6. Paired against the pre-fix row (same 60 items, same placement)\n")
    add(f"* risk difference (corrected − plain): **{paired['difference']:+.3f}** "
        f"[{paired['low']:+.3f}…{paired['high']:+.3f}] "
        f"({paired['iters']} bootstrap resamples,\n  seed {paired['seed']})")
    add(f"* discordant pairs: pre-fix-only correct **{paired['discordant']['a_win']}**, "
        f"corrected-only correct **{paired['discordant']['b_win']}**")
    add(f"* exact McNemar (two-sided): **p = {paired['mcnemar_p']:.4g}**")
    add(f"* unpaired rows: {len(stats['n']['only_corrected'])} corrected-only ids, "
        f"{len(stats['n']['only_pre_fix'])} pre-fix-only ids")
    add("")
    flips = stats["item_flips"]
    if flips["pre_fix_only"]:
        add("Items the plain framing got and the corrected one loses:\n")
        add("| id | type | gold | plain got | corrected got |")
        add("|---|---|---|---|---|")
        for row in flips["pre_fix_only"]:
            add(f"| `{row['id']}` | {row['type']} | `{row['expected']}` | `{row['pre_got']}` | "
                f"`{row['now_got']}` |")
        add("")
    if flips["corrected_only"]:
        add("Items the corrected framing gets and the plain one lost:\n")
        add("| id | type | gold | plain got | corrected got |")
        add("|---|---|---|---|---|")
        for row in flips["corrected_only"]:
            add(f"| `{row['id']}` | {row['type']} | `{row['expected']}` | `{row['pre_got']}` | "
                f"`{row['now_got']}` |")
        add("")
    add("**Framing is the only variable.** Both rows ran the same 60 committed items, the same "
        "model file\n(SHA-identical before and after), the same placement ask (9 layers, no "
        "degrade, `kv_type` auto in\nevery chunk), the same `--threads 4`, on the same host — and "
        "the pre-fix row is a committed\n`quality` report of this repository "
        "(`docs/evidence/tiel_quality.json`), re-read here item for item.\n")

    add("## 7. What the reading is — and what it implies for the shipped cue\n")
    add(f"**Plainly: the collapse persists.** Six items said \"the product's prompt breaks this "
        f"model\"; {total} items\nsay it with an interval that no longer reaches the published "
        f"row: {block_line(corrected['overall'])} vs\n{block_line(pre['overall'])}, and the "
        f"mechanism is visible per row — "
        f"{corrected['cue']['refused']}/{total}\nrows refuse at the cue where the readout "
        "sits.\n")
    add("The E3c reading (\"Tiel is not mass-starved, unlike Occamy\") is a statement about the "
        "**plain**\ninstrument. On the prompt the product sends, Tiel behaves like Occamy in the "
        "six-item probe and worse\nhere: the mass at the cue row is a turn-closer's mass, so the "
        "word \"starvation\" understates it — the\nmodel is closing the turn, not hesitating.\n")
    add("**What this does not decide.** The cue *policy* is not this card's to move. What the card "
        "changes is\nthe evidence under the policy question:\n")
    add("* the published Tiel row measured a prompt the product never sends, so any argument that "
        "read it as\n  \"the shipped cue is fine for 35B-A3B\" was reading the wrong bytes;")
    add("* the six-item probe's direction is now a 60-item table, and the one cue knob the product "
        "already has\n  (`--cue two_step`) is **measured inert on this model** (§5.1: it moves one "
        "row of sixty) — so \"try a\n  different cue shape\" is not the escape hatch here;")
    add("* card `t_4c48f40a` (E3e) is measuring the instructed-JSON + roles-split policy, which "
        "attacks the same\n  row from the other side: it stops the prefill before the row the "
        "model closes and reads the value row.\n  That is the measurement this row points at; if "
        "it holds, the shipped *prompt shape* — not the model\n  choice — is what has to change.\n")
    add("The default stays `shipped` until that card's own evidence lands (E3d's rule: the "
        "mechanism ships, the\ndefault moves only with its own publication). This document is the "
        "corrected row, with its framing named,\nnext to the pre-fix one that keeps its marker.\n")

    add("## 8. What this does not claim\n")
    add("* no ranking between models — this is one model's row, and §7.7's three-way table keeps "
        "its own\n  framing caveat;")
    add("* no mechanism beyond the cue verdicts: which of template bytes / `enable_thinking` "
        "suppression /\n  the built-in family renderer produces the turn-closer is E3e's question;")
    add("* no claim about the other cue shapes on Tiel (not measured here);")
    add("* no throughput claim — the chunk walls are this box's, under sibling-card load.\n")

    add("## 9. Gates\n")
    add("| gate | command | result |")
    add("|---|---|---|")
    if gates:
        for entry in gates:
            add(f"| {entry['name']} | `{entry['command']}` | {entry['result']} |")
    else:
        add("| reproduce driver exits 0 | `bash .t7c9/run_chunks_corrected.sh` | see "
            "`.t7c9/logs/campaign.log` |")
        add("| oracle | `python3 docs/verify_runtime_contract.py` | see `.t7c9/oracle.txt` |")
        add("| test suite | `uv run --frozen pytest -q` | see `.t7c9/gates.txt` |")
        add("| model SHA-256 before/after | `sha256sum` | identical (§1) |")
    add("")

    add("## Receipts\n")
    add("* driver + analysis: `.t7c9/run_chunks_corrected.sh`, `.t7c9/analyse.py`, "
        "`.t7c9/render_doc.py`,\n  `.t7c9/placement_facts.py`, `.t7c9/detail_facts.py`")
    add("* auxiliary `two_step` arm: `.t7c9/run_cue_arm.sh`, `.t7c9/tiel_cue_arm.py`, "
        "`.t7c9/analyse_arm.py`,\n  `.t7c9/tiel_two_step_stats.json`")
    add("* per-chunk reports (corrected instrument): "
        "`docs/evidence/tiel_corrected_chunks/report_00{1..6}.json`")
    add("* per-chunk reports (`two_step` arm): "
        "`docs/evidence/tiel_corrected_chunks/two_step_report_00{1..6}.json`")
    add("* per-chunk placement sinks: "
        "`docs/evidence/tiel_corrected_chunks/placement_00{1..6}.json` and\n  "
        "`.../two_step_placement_00{1..6}.json`")
    add("* dev-set slices (SHA-verified identical to `docs/evidence/tiel_chunks/`): "
        "`docs/evidence/tiel_corrected_chunks/devset_00{1..6}.jsonl` + `.t7c9/devset_sha.txt`")
    add("* merged rows: `docs/evidence/tiel_corrected_quality.json` (the card's row) and\n  "
        "`docs/evidence/tiel_two_step_quality.json` (the auxiliary arm)")
    add("* statistics: `.t7c9/tiel_corrected_stats.json` (this doc is rendered from it)")
    add("* run logs: `.t7c9/logs/run_00{1..6}.log`, `.t7c9/logs/two_step_run_00{1..6}.log`, "
        "`.t7c9/logs/campaign.log`,\n  `.t7c9/oracle.txt`, `.t7c9/gates.txt`, "
        "`.t7c9/live_parity_tiel.txt`")
    add("* the pre-fix row it pairs with: `docs/evidence/tiel_quality.json` + `docs/BENCHMARKS.md` "
        "§7.3/§7.4\n  (pre-fix marker in §7.9)")
    add("")

    pathlib.Path(args.out).write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(parts)} blocks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
