#!/usr/bin/env python3
"""Assemble `docs/evidence/e3b_t_6952f0dd_label_policy.md` from the E3b run artifacts.

Everything the document states is read back from a file this campaign wrote:

* `.e3b/sweep.json`      — the 6-item variant sweep (coverage per cue x label, the cue rows,
                            the ranked readout, the `DecisionEngine` cross-check);
* `.e3b/after.json`      — the 20-item re-measure under the accepted policy (optional);
* `docs/evidence/e3_occamy_quality.json` — the *before* side (E3's published run);
* `docs/evidence/e2_quality.json`        — the 4B baseline of the published table;
* `docs/evidence/e3b_calibrate_accepted.txt` — `ggufone calibrate` on the accepted run.

    python3 tools/e3b_build_evidence.py            # writes the evidence doc + the tables
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ggufone.bench import compare  # noqa: E402

SPEC = importlib.util.spec_from_file_location("e3b", ROOT / "tools" / "e3b_label_policy.py")
e3b = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(e3b)


def load(path: pathlib.Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def cell(block: dict) -> str:
    if not block["n"]:
        return "—"
    low, high = block["ci"]
    return f"{block['agreement']:.3f} ({block['correct']}/{block['n']}) [{low:.3f}–{high:.3f}]"


def variant_ranking(record: dict, prefix: str = "shipped") -> list[tuple[float, int, str, str]]:
    pieces = [item["prefixes"][prefix] for item in record["items"]]
    rows: list[tuple[float, int, str, str]] = []
    for cue in record["cues"]:
        for label in record["label_variants"]:
            values = [piece["cues"][cue]["labels"][label]["coverage"] for piece in pieces]
            above = sum(1 for value in values if value >= record["mass_floor"])
            rows.append((sum(values) / len(values), above, cue, label))
    rows.sort(reverse=True)
    return rows


def prefix_pairs(record: dict) -> list[dict]:
    return [item for item in record["items"] if len(item.get("prefixes") or {}) > 1]


def comparison_section(after: dict | None) -> list[str]:
    if after is None:
        return ["No 20-item re-measure ran: the accepted policy is the shipped one, so the "
                "published E3 table stands unchanged (and this card changed no E3 number)."]
    e3 = load(ROOT / "docs/evidence/e3_occamy_quality.json")
    e2 = load(ROOT / "docs/evidence/e2_quality.json")
    lines: list[str] = []
    if e3 and after:
        table = compare.comparison(e3, after, labels=("Occamy shipped (E3)", "Occamy accepted"))
        lines += ["**Occamy before/after** (same 20 items, same box):", "",
                  compare.render_comparison(table), ""]
    if e2 and after:
        paired = compare.align(e2, after)
        table = compare.comparison(paired["baseline"], paired["challenger"],
                                   labels=("4B default (E2, CPU)", "Occamy accepted (E3b)"))
        lines += [f"**The published pairing re-rendered** ({paired['items']} paired items, "
                  f"dropped {paired['dropped']['baseline']} unpaired baseline row(s) and "
                  f"{paired['dropped']['challenger']} unpaired challenger row(s)):", "",
                  compare.render_comparison(table), ""]
    return lines


def build() -> str:
    sweep = load(ROOT / ".e3b/sweep.json")
    if sweep is None:
        raise SystemExit(".e3b/sweep.json not found — run the sweep first")
    after = load(ROOT / ".e3b/after.json")
    calibrate = (ROOT / "docs/evidence/e3b_calibrate_accepted.txt")
    model = sweep.get("model") or {}
    pieces = e3b.shipped_items(sweep)
    ranking = variant_ranking(sweep)
    best_mean, best_above, best_cue, best_label = ranking[0]
    floor = sweep["mass_floor"]
    ceiling = max(value for _, _, cue, label in ranking
                  for value in [row["cues"][cue]["labels"][label]["coverage"] for row in pieces])
    tops = [(item["prefixes"]["shipped"]["cues"]["shipped"].get("top_tokens") or [{}])[0]
            for item in sweep["items"]]
    top_pieces = {}
    for entry in tops:
        top_pieces[entry.get("piece", "?")] = max(top_pieces.get(entry.get("piece", "?"), 0.0),
                                                  float(entry.get("p_full") or 0.0))
    leading = sorted(top_pieces.items(), key=lambda pair: -pair[1])[0] if top_pieces else ("?", 0.0)
    if best_above == 0:
        verdict = (
            "No label rendering (leading space, casing, the two-step newline readout, the long "
            "description form) and no cue rewrite (blank line, an explicit cue naming the labels) "
            f"reaches the engine's {floor:.2f} coverage floor on this model: the best combination, "
            f"`{best_cue}` × `{best_label}`, has mean coverage {best_mean:.3e} and lifts "
            f"{best_above}/{len(pieces)} items above the floor. The reason is in the cue row "
            f"itself — `{leading[0]}` takes p ≈ {leading[1]:.5f} of the next-token mass on the "
            "shipped prompt, i.e. the model closes the assistant turn instead of answering, and a "
            "label rendering cannot change that. This is the negative result the card allows, and "
            "the live table is below.")
    else:
        verdict = (
            f"`{best_cue}` × `{best_label}` is the rendering that lifts the most items above the "
            f"engine's {floor:.2f} floor ({best_above}/{len(pieces)}, mean coverage "
            f"{best_mean:.3e}); every other combination stays below it. The numbers, the ranked "
            f"readout and the before/after comparison are below.")
    lines = [
        "# E3b — the `qwen35moe` label policy: candidate-mass coverage per variant",
        "",
        f"Card `t_6952f0dd` · branch `main` (this repo has no remote; the commits are local on the "
        f"shared tree) · Tier **M** · schema `{sweep['schema']}`",
        "",
        "**Question.** E3 (`t_a431be85`) published a paired comparison in which every Occamy "
        "answer was `low_mass` (20/20, no `measured` row at all) while the 4B was mostly "
        "`measured` on the same 20 items. This card asks whether the *rendering* of the "
        "candidate label is part of that, and picks a label policy for this family — or records "
        "the negative result.",
        "",
        "**Answer in one paragraph.** " + verdict,
        "",
        "## 1. The pin, the box, the placement",
        "",
        "| what | value |",
        "|---|---|",
        f"| model | `{model.get('name')}` — {model.get('bytes') or 0:,} bytes, arch "
        f"`{model.get('arch')}` |",
        f"| **SHA-256** | `{sweep.get('model_sha256')}` (hashed by this run, before the load) |",
        f"| runtime | `{sweep['runtime']}` (`GGUFONE_RUNTIME_DIR`, the pinned b11026 bundle) |",
        f"| backend / threads | `vulkan` / {sweep['threads']} (E3 §6.5: 4 beats 8 and 12 by ~2×) |",
        f"| `--gpu-layers` requested | {sweep['gpu_layers']} |",
        f"| placement the loader used | `{json.dumps(sweep['placement'])}` |",
        f"| model load | {sweep['load_ms']:.0f} ms |",
        f"| wall, whole run | {sweep['wall_s']:.1f} s |",
        "",
        "Every number in this document is tagged by construction **[container]**: the worker "
        "container sees `cpu.max = 2 CPU-seconds/s` and `memory.max = 8 GiB` against a 23 GB model "
        "(E3 §2.2), so one prefill *and* one decode batch each cost a full weight sweep from disk. "
        "No `[host]` re-run happened in this card.",
        "",
        "## 2. Method",
        "",
        "* the candidate label is **rendered** by `bench/labels.py`: `bare` (what the engine "
        "ships), `space` (`' billing'`), `caps` (`'Billing'`), `newline` (`'\\nbilling'`, the "
        "two-step readout) and `long` (the description the prompt shows — `'billing: payments, "
        "invoices and refunds'`, the level text for `score`, the criteria text for `noul`);",
        "* the cue line is rendered by the same module: `shipped`, `blank` (one empty line after "
        "the cue — the variant `docs/TEMPLATES.md` §4 measured to triple the label mass on the "
        "4B) and `explicit` (`Answer with exactly one of these candidate names: billing, …`);",
        "* **one model load, one context per item-prefix, one decode batch per prefix for all "
        "three cue variants** — each cue is a sequence forked from the item's prefix, which is the "
        "engine's own candidate protocol, so the cue variants share the sweep;",
        "* `coverage` is `readout.coverage_from_scale` on the cue row: the full-vocabulary mass of "
        "each label's first token, read *without* any extra forward pass (that is why 15 "
        "cue × label combinations cost 3 decodes per item);",
        "* the **ranked** readout (`--rank cue=label`) is the engine's decision — "
        "`readout.candidate_sequence_score` + `restricted_softmax` + `argmax_first` — with the "
        "candidate sequences decoded through a prefix-sharing trie (`labels.score_paths`, gated "
        "offline in `tests/test_e3b_labels.py`);",
        "* the **cross-check** runs the same item through `DecisionEngine` itself and compares "
        "coverage and winner, so a probe bug cannot be published as a model finding.",
        "",
        f"`{sweep['devset']}` — the first {len(pieces)} items ",
        f"({', '.join(f'{k} {v}' for k, v in sorted(sweep['counts'].items()))}), "
        "the same items E3 measured, so the before/after sides ask the same questions.",
        "",
        "## 3. Coverage per (cue, label) variant",
        "",
        *e3b.coverage_table(sweep),
        "",
        "### 3.1 Summary",
        "",
        *e3b.variant_summary(sweep),
        "",
        f"The best combination by mean coverage is `{best_cue}` × `{best_label}` "
        f"(mean {best_mean:.3e}, {best_above}/{len(pieces)} items at or above the floor); the best "
        f"single value anywhere in the sweep is {ceiling:.3e}, i.e. "
        f"{'above' if ceiling >= floor else 'still below'} the {floor:.2f} floor. ",
        "",
        "## 4. What the model wants to emit at the cue (why the label mass is where it is)",
        "",
        *e3b.top_token_block(sweep, cues=sweep["cues"]),
        "",
        "## 5. The prefix control: the template's own empty think block",
        "",
        *e3b.prefix_block(sweep),
        "",
        "## 6. The ranked readout, and the `DecisionEngine` cross-check",
        "",
        *e3b.ranked_block(sweep),
        "",
        *e3b.cross_check_block(sweep),
        "",
        "## 7. `ggufone calibrate` on the accepted run",
        "",
        "```",
        calibrate.read_text(encoding="utf-8").strip() if calibrate.exists() else "(not run)",
        "```",
        "",
        "## 8. Before/after on the paired item set",
        "",
        *comparison_section(after),
        "",
        "## 9. Honest limits",
        "",
        "1. **Container numbers.** One weight sweep per prefill *and* per decode batch is a "
        "property of the 8 GiB cgroup, not of the model; the coverage/agreement numbers do not "
        "depend on it, the wall times do.",
        "2. **Small n.** The variant sweep is a fixed 6-item subset (2 per question type) of "
        "chunk 001; the re-measure is 20 items. A 6-item subset can rank variants, it cannot "
        "estimate an agreement.",
        "3. **One runtime, one box, one quantization** (`b11026`, Vulkan + CPU offload, Q4_K_L).",
        "4. **The ranked readout is the engine's arithmetic driven by a probe**, not by "
        "`DecisionEngine`: the cross-check above is what makes the two comparable.",
        "5. **No E3-published number was changed** by this card; the before/after table is its own "
        "artifact.",
        "",
    ]
    if (ROOT / "docs/evidence/e3b_label_policy_tables.md").exists():
        lines += ["## 10. The generated tables", "",
                  (ROOT / "docs/evidence/e3b_label_policy_tables.md").read_text(encoding="utf-8"),
                  ""]
    return "\n".join(lines) + "\n"


def main() -> int:
    document = build()
    target = ROOT / "docs/evidence/e3b_t_6952f0dd_label_policy.md"
    target.write_text(document, encoding="utf-8")
    print(f"evidence: {target}")
    sweep = load(ROOT / ".e3b/sweep.json") or {}
    if sweep:
        tables = ROOT / "docs/evidence/e3b_label_policy_tables.md"
        tables.write_text(e3b.render_report(sweep), encoding="utf-8")
        print(f"tables: {tables}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
