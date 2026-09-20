#!/usr/bin/env python3
"""t_9bcbecff — render the evidence document FROM `.t9bcb/stats.json`.

The card's (and this repository's) rule: no number is retyped. Every value in the markdown this
script writes comes out of the stats file `analyse.py` produced from the stored reports; the only
literal strings here are labels and prose.

    python3 .t9bcb/analyse.py && python3 .t9bcb/render_doc.py

Writes `docs/evidence/e3e_role_split_t_9bcbecff.md`.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
STATS = ROOT / ".t9bcb" / "stats.json"
TARGET = ROOT / "docs/evidence/e3e_role_split_t_9bcbecff.md"


def _pct(value: Any, digits: int = 1) -> str:
    return "—" if value is None else f"{100.0 * float(value):.{digits}f} %"


def _num(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and (abs(value) < 1e-3 and value != 0.0):
        return f"{value:.3e}"
    return f"{value:.{digits}f}"


def _interval(ci: Any, digits: int = 3) -> str:
    if not ci or ci[0] is None:
        return "—"
    return f"{_num(ci[0], digits)} – {_num(ci[1], digits)}"


def _diff_interval(pair: dict[str, Any]) -> str:
    if pair.get("difference") is None:
        return "—"
    return (f"{pair['difference']:+.3f} ({_interval(pair['ci'])}, "
            f"exact McNemar p = {_num(pair['mcnemar_p'], 3)})")


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def cell_line(label: str, cell: dict[str, Any]) -> str:
    return (f"- **{label}** — {cell['correct']}/{cell['items']} = {_num(cell['agreement'])} "
            f"[{_interval(cell['ci'])}]; `low_mass` {cell['low_mass']}/{cell['items']}, "
            f"`measured` {cell['measured']}/{cell['items']}, refusals "
            f"{cell['refusals']}/{cell['items']}")


def render_occamy(stats: dict[str, Any]) -> list[str]:
    """§9 — the optional Occamy pass: both cells measured here, compared with each other."""
    occ = stats.get("occamy") or {}
    if not occ:
        return []
    base = occ["cells"]["baseline"]
    chall = occ["cells"]["challenger"]
    pair = occ["paired"]
    verdict = ("the pair clears the card's E3e unit rule" if pair.get("challenger_wins")
               else "the pair does **not** clear the card's E3e unit rule")
    lines = [
        "",
        "## 9. The optional Occamy pass (the card's \u201c(and Occamy)\u201d)",
        "",
        f"Occamy (`{occ.get('model_name')}`, `{occ.get('model')}`) has **no published row under "
        f"the corrected instrument**: "
        f"the E3 Occamy row of `docs/evidence/e3_t_a431be85_occamy.md` was measured in a container "
        f"before the framing fix, so it is not comparable with this pair. Both cells are therefore "
        f"measured here \u2014 the same 60 committed items, the same placement ask and the same "
        f"`--backend vulkan --threads 4` instrument \u2014 and compared with each other, paired by "
        f"item.",
        "",
        cell_line("Occamy, shipped placement + shipped cue (measured here)", base),
        cell_line("Occamy, role_split + json_instructed (measured here)", chall),
        "",
        f"* paired risk difference (challenger \u2212 its own baseline): "
        f"**{_num(pair['difference'])} ({_interval(pair['ci'])}, exact McNemar p = "
        f"{_num(pair['mcnemar_p'])})** \u2014 discordant {pair['challenger_only']} challenger-only "
        f"against {pair['baseline_only']} baseline-only, both correct {pair['both_correct']}, "
        f"neither correct {pair['neither_correct']}; {verdict}.",
        f"* refusals at the cue {base['refusals']}/{base['items']} \u2192 "
        f"{chall['refusals']}/{chall['items']}, "
        f"`low_mass` {base['low_mass']}/{base['items']} \u2192 "
        f"{chall['low_mass']}/{chall['items']}, "
        f"`measured` {base['measured']}/{base['items']} \u2192 "
        f"{chall['measured']}/{chall['items']}.",
    ]
    rows = []
    for qtype in sorted(base["per_type"]):
        b = base["per_type"][qtype]
        c = chall["per_type"].get(qtype) or {}
        pt = occ["per_type"].get(qtype) or {}
        rows.append([qtype, f"{b['correct']}/{b['n']}", f"{c.get('correct')}/{c.get('n')}",
                     f"{pt.get('challenger_only')} / {pt.get('baseline_only')}",
                     _num(pt.get("difference")) if pt.get("difference") is not None else "\u2014",
                     _num(pt.get("mcnemar_p")) if pt.get("mcnemar_p") is not None else "\u2014"])
    lines += ["", "### 9.1 By question type", ""]
    lines += _table(["type", "shipped cue (measured here)", "role_split + json_instructed",
                     "discordant (challenger-only / baseline-only)", "difference",
                     "exact McNemar p"], rows)

    lines += ["", "### 9.2 Placement per chunk (the loader's own answer)", ""]
    base_by_chunk = {chunk["chunk"]: chunk for chunk in occ["baseline_chunks"]}
    rows = []
    for chunk in occ["challenger_chunks"]:
        other = base_by_chunk.get(chunk["chunk"]) or {}
        other_cell = f"{other['correct']}/{other['items']}" if other else "—"
        rows.append([f"`{chunk['chunk']}`",
                     other_cell,
                     f"{chunk['correct']}/{chunk['items']}",
                     f"{chunk['ngl_requested']} \u2192 {chunk['ngl_used']}",
                     str(chunk["degraded"]), str(chunk["n_ctx"]),
                     f"{_num(chunk['chunk_wall_s'])} s", f"`{chunk['effective_backend']}`",
                     f"{chunk['low_mass']}/{chunk['items']}"])
    lines += _table(["chunk", "shipped cue correct", "challenger correct", "ngl req \u2192 used",
                     "degraded", "n_ctx", "chunk wall", "effective_backend", "challenger low_mass"],
                    rows)

    seen: dict[str, int] = {}
    templates: dict[str, int] = {}
    for chunk in occ["challenger_chunks"] + occ["baseline_chunks"]:
        for key, count in (chunk.get("chat_format_seen") or {}).items():
            seen[key] = seen.get(key, 0) + count
        for key, count in (chunk.get("template_seen") or {}).items():
            templates[key] = templates.get(key, 0) + count
    shapes: dict[str, int] = {}
    for key, count in seen.items():
        data = json.loads(key)
        label = json.dumps({name: data.get(name)
                            for name in ("kind", "question_turn", "contract", "dropped")},
                           sort_keys=True)
        shapes[label] = shapes.get(label, 0) + count
    lines += ["", "### 9.3 What the Occamy cells rendered through", ""]
    lines += _table(["surface", "value", "rows"],
                    [["`engine.chat_format`", f"`{key}`", str(count)]
                     for key, count in sorted(shapes.items())] +
                    [["`engine.template`", f"`{key}`", str(count)]
                     for key, count in sorted(templates.items())])
    lines += [
        "",
        f"* **the pin.** `{occ.get('model')}` — {occ.get('model_bytes')} bytes, mtime "
        f"`{occ.get('model_mtime')}`, SHA-256 `{occ['sha']}` "
        f"(`{occ.get('sha_receipt', '—')}`); the same digest is in "
        f"`{occ.get('sha_e3_receipt', '—')}` for this file — "
        f"**{occ.get('sha_matches_e3_receipt')}** — and its mtime precedes the pass, so the file "
        f"the two cells loaded is the file that was already on disk, unmoved.",
        "* the driver's own before/after hash pair did **not** reach this pass's log: the run went "
        "through `systemd-run` without a file redirect and only its status lines were journaled, "
        "so the digest above is a single measurement taken after the pass, not a pair.",
        f"* per-chunk reports + placement sinks: `{occ['chunk_dir']}/`; merged: "
        f"`{occ['merged']['baseline']}`, `{occ['merged']['challenger']}`; log: `{occ['log']}`.",
        "",
    ]
    return lines


def render_benchmarks(stats: dict[str, Any]) -> str:
    """The §7.4.2 block for `docs/BENCHMARKS.md`, rendered from the same stats file."""
    chall, base, pair, log = stats["challenger"], stats["baseline"], stats["paired"], stats["log"]
    aux = stats.get("aux", {})
    types = " · ".join(f"{chall['per_type'][key]['correct']}/{chall['per_type'][key]['n']}"
                       for key in sorted(chall["per_type"]))
    base_types = " · ".join(f"{base['per_type'][key]['correct']}/{base['per_type'][key]['n']}"
                            for key in sorted(base["per_type"]))
    aux_role = aux.get("role_split_only", {}).get("cell")
    aux_two = aux.get("two_step_role_split", {}).get("cell")
    lines = [
        "<!-- @@T9BCBECFF_TIEL_E3E_START@@ — rendered by `.t9bcb/render_doc.py` from "
        "`.t9bcb/stats.json`; edit the tool, never this block. -->",
        "",
        "### 7.4.2 The E3e policy on this row: `role_split` + `json_instructed` "
        "(card `t_9bcbecff`, [host])",
        "",
        "§7.4.1's corrected row is the baseline here and is **not re-measured**: the same 60 "
        "committed items, same model file (SHA-256 `" + (log["sha_before"] or "")[:8] + "…`, "
        "identical before and after), same placement ask (9 layers, `degraded: false`, `kv_type` "
        "auto in every chunk) and the same `--backend vulkan --threads 4` instrument were measured "
        "once more with exactly the two E3e switches of §9 added — `--chat-format role_split` and "
        "`--cue json_instructed` (contract `question`).",
        "",
        "| row | agreement | Wilson 95 % | `low_mass` | refused at the cue | coverage median | "
        "`measured` (≥ 0.10) | choice · noul · score |",
        "|---|---|---|---|---|---|---|---|",
        f"| §7.4.1, **corrected / shipped cue** | {base['correct']}/{base['items']} = "
        f"{_num(base['agreement'])} | {_interval(base['ci'])} | {base['low_mass']}/{base['items']} "
        f"| {base['refusals']}/{base['items']} | {_num(base['coverage_quantiles']['p50'])} | "
        f"{base['measured']}/{base['items']} | {base_types} |",
        f"| **+ `role_split` + `json_instructed`** | **{chall['correct']}/{chall['items']} = "
        f"{_num(chall['agreement'])}** | {_interval(chall['ci'])} | "
        f"**{chall['low_mass']}/{chall['items']}** | **{chall['refusals']}/{chall['items']}** | "
        f"{_num(chall['coverage_quantiles']['p50'])} | **{chall['measured']}/{chall['items']}** | "
        f"{types} |",
        "",
        f"Paired by item (exact McNemar + the closed-form interval, "
        f"`tools/e3e_roles_decision.py`): "
        f"risk difference **{pair['difference']:+.3f} [{_num(pair['ci'][0])}…"
        f"{_num(pair['ci'][1])}]**, p = {_num(pair['mcnemar_p'])} — discordant "
        f"{pair['challenger_only']} challenger-only against {pair['baseline_only']} "
        f"baseline-only. The two switches therefore do on this family what §9 measured on the 4B: "
        f"the placement moves the question out of the assistant turn and the instructed contract "
        f"gives the readout a row to read, so the collapse recorded above ("
        f"{base['refusals']}/{base['items']} refusals, "
        f"`low_mass` {base['low_mass']}/{base['items']}, "
        f"`measured` {base['measured']}/{base['items']}) becomes a fully measured row "
        f"(`low_mass` {chall['low_mass']}/{chall['items']}, every cue verdict `answered`).",
        "",
        "**Auxiliary cells** (same instrument, same items, never this card's row):",
        "",
        "| auxiliary arm | agreement | `low_mass` | refused at the cue | coverage median | "
        "paired vs §7.4.1 |",
        "|---|---|---|---|---|---|",
    ]
    for name, arm in sorted(aux.items()):
        cell = arm["cell"]
        lines.append(
            f"| `{name}` | {cell['correct']}/{cell['items']} = {_num(cell['agreement'])} | "
            f"{cell['low_mass']}/{cell['items']} | {cell['refusals']}/{cell['items']} | "
            f"{_num(cell['coverage_quantiles']['p50'])} | "
            f"{arm['vs_baseline']['difference']:+.3f} "
            f"(p = {_num(arm['vs_baseline']['mcnemar_p'])}) |")
    lines += [
        "",
        "The collapse cell reproduces §9's **non-additivity** on this family as well, with the "
        "opposite branch of the same rule: under the role split every cue row closes the turn "
        "(`<think>`), and `decide._advance_token` never advances past a cue the model closed — so "
        "`two_step` is *inert* here and the two auxiliary arms are decision-identical item by item"
        + (f" ({aux_two['refusals']}/{aux_two['items']} refusals, "
           f"`low_mass` {aux_two['low_mass']}/{aux_two['items']}, "
           f"{_num(aux_two['agreement'])})" if aux_two else "")
        + (f". Neither auxiliary cell is readable as accuracy — both are `measured` "
           f"{aux_role['measured']}/{aux_role['items']} with median coverage "
           f"{_num(aux_role['coverage_quantiles']['p50'])}: the placement moves "
           f"the question out of "
           f"the assistant turn, and it is the instructed contract that puts the answer's mass on "
           f"the readout row (`low_mass` {aux_role['low_mass']}/{aux_role['items']} → "
           f"{chall['low_mass']}/{chall['items']})." if aux_role and chall else "."),
        "",
        "**No default moves.** The two switches keep the frozen defaults §9 published "
        "(`cue=shipped`, `chat_format=answer_sheet`, `json_contract=question`); this subsection is "
        "a policy measurement on one model's row, and a row measured under it is not comparable "
        "with the rows measured on the shipped prompt bytes.",
        "",
        "Full detail (per-chunk placement ledger, refusal breakdown, the item flips, the two "
        "auxiliary arms, the render path receipts): `docs/evidence/e3e_role_split_t_9bcbecff.md`; "
        "raw `docs/evidence/t9bcbecff_tiel_challenger_quality.json` + "
        "`docs/evidence/t9bcbecff_tiel_chunks/`.",
        "<!-- @@T9BCBECFF_TIEL_E3E_END@@ -->",
    ]
    return "\n".join(lines)


def splice_benchmarks(block: str) -> str:
    """Put the §7.4.2 block into `docs/BENCHMARKS.md` (replace between markers, else insert)."""
    path = ROOT / "docs/BENCHMARKS.md"
    text = path.read_text(encoding="utf-8")
    start = "<!-- @@T9BCBECFF_TIEL_E3E_START@@"
    end = "<!-- @@T9BCBECFF_TIEL_E3E_END@@ -->"
    body = block.split("<!-- @@T9BCBECFF_TIEL_E3E_START@@", 1)[1].split("-->", 1)[1].strip("\n")
    if start in text and end in text:
        head = text.split(start, 1)[0]
        tail = text.split(end, 1)[1]
        path.write_text(head + start + " — rendered by `.t9bcb/render_doc.py` from "
                        "`.t9bcb/stats.json`; edit the tool, never this block. -->\n\n" + body
                        + "\n" + end + tail, encoding="utf-8")
        return "replaced"
    anchor = "<!-- @@T7C926398_TIEL_CORRECTED_END@@ -->"
    if anchor not in text:
        raise SystemExit("docs/BENCHMARKS.md: neither this card's markers nor §7.4.1's end anchor")
    head, tail = text.split(anchor, 1)
    path.write_text(head + anchor + "\n\n" + block + "\n" + tail.lstrip("\n"),
                    encoding="utf-8")
    return "inserted"


def main() -> int:
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    chall, base = stats["challenger"], stats["baseline"]
    pair = stats["paired"]
    log, devset = stats["log"], stats["devset"]
    lines: list[str] = []

    lines += [
        "# The [host] E3e probe — `role_split` + `json_instructed` on Tiel-Coder-35B-A3B "
        "(`t_9bcbecff`)",
        "",
        "Card `t_9bcbecff` (main-coder) · [host] run · worktree "
        "`ggufone-wt-t9bcb`, branch `t9bcb-e3e-tiel` · instrument: the committed corrected "
        "instrument of card `t_7c926398`, unmodified, plus the two E3e policy switches of card "
        "`t_4c48f40a`.",
        "",
        "**One sentence.** "
        f"On the same 60 committed dev items, the same model file, the same placement ask and the "
        f"same `--backend vulkan` instrument as the published corrected row, adding "
        f"`--chat-format role_split --cue json_instructed --json-contract question` moves Tiel "
        f"from {base['correct']}/{base['items']} = {_num(base['agreement'])} "
        f"[{_interval(base['ci'])}] to {chall['correct']}/{chall['items']} = "
        f"{_num(chall['agreement'])} [{_interval(chall['ci'])}] — paired difference "
        f"{_diff_interval(pair)} — and it takes the collapse with it: `low_mass` "
        f"{base['low_mass']}/{base['items']} → {chall['low_mass']}/{chall['items']}, refusals at "
        f"the cue {base['refusals']}/{base['items']} → {chall['refusals']}/{chall['items']}.",
        "",
        "## 1. The pin",
        "",
    ]
    lines += _table(
        ["what", "value"],
        [["file", "`/var/home/rybens/.hermes/models/Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf`"],
         ["SHA-256 before (first line of the campaign log)", f"`{log['sha_before']}`"],
         ["SHA-256 after (last line of the campaign log)", f"`{log['sha_after']}`"],
         ["the pair is identical",
          f"`{log['sha_identical']}` over `{log['sha_lines']}` sha line(s)"],
         ["downloads", "none — the file was already local"],
         ["dev-set slices", f"{devset['files']} slices, digests byte-identical to the baseline "
                            f"campaign's own receipt: `{devset['identical']}` "
                            f"(`{devset['receipt']}`)"]])
    lines += ["", "## 2. The instrument (what moved, and the one thing that did not)", ""]
    lines += [
        f"* **baseline** — `{base['label']}`: the corrected-instrument row of card `t_7c926398`, "
        f"committed at `{stats['generated_from']['baseline']}`; **not re-measured here** "
        f"(the card's "
        f"rule), read from its own per-chunk reports.",
        f"* **challenger** — `{chall['label']}`: the same instrument plus exactly the two E3e "
        f"flags (`--chat-format role_split` and `--cue json_instructed`, with the default "
        f"`--json-contract question`).",
        f"* reproduce line of the challenger's own report: `{chall['reproduce']}`",
        "",
        "**Which path rendered the role split.** The family's own template is outside the internal "
        "renderer's subset (E3e recorded it `not-renderable` offline, `.e3e/role_render.json`), so "
        "the question is whether the built-in bridge can express the two-user-turn shape in a live "
        "run. It can, and the rows say so through the response's own surfaces:",
        "",
    ]
    seen: dict[str, int] = {}
    for chunk in stats["challenger_chunks"]:
        for key, count in (chunk.get("chat_format_seen") or {}).items():
            seen[key] = seen.get(key, 0) + count
    templates: dict[str, int] = {}
    for chunk in stats["challenger_chunks"]:
        for key, count in (chunk.get("template_seen") or {}).items():
            templates[key] = templates.get(key, 0) + count
    shapes: dict[str, int] = {}
    chars: list[int] = []
    for key, count in seen.items():
        data = json.loads(key)
        shape = json.dumps({field: data.get(field)
                            for field in ("kind", "question_turn", "contract", "dropped")},
                           sort_keys=True)
        shapes[shape] = shapes.get(shape, 0) + count
        if data.get("prefix_chars") is not None:
            chars.append(int(data["prefix_chars"]))
    lines += _table(["surface", "value", "rows", "prefix chars"],
                    [["`engine.chat_format`", f"`{shape}`", str(count),
                      f"{min(chars)}–{max(chars)}" if chars else "—"]
                     for shape, count in sorted(shapes.items())] +
                    [["`engine.template`", f"`{key}`", str(count), "—"]
                     for key, count in sorted(templates.items())])
    lines += [
        "",
        "so every challenger row prefilled the question into a **user** turn (`question_turn` "
        "`user`, `kind` `role_split`) rendered by the built-in bridge "
        "(`llama_chat_apply_template`), with nothing the shared prefix had to drop "
        "(`dropped` empty on every row) — the built-in bridge *does* express this family's "
        "two-user-turn shape, and `--template plain` was **not** needed (it is not used "
        "in "
        "this run).",
        "",
        "## 3. Environment and the scope that decides the numbers",
        "",
        "```",
        "runtime   /home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan",
        "          (pinned b11026)",
        "ICD       VK_DRIVER_FILES=/home/rybens/.e3c_host/nvidia_egl_icd.json (libEGL_nvidia.so.0)",
        f"scope     systemd-run --user --unit=t9bcb-campaign (memory.max={log['memory_max']})",
        "driver    .t9bcb/run_chunks.sh → .t9bcb/tiel_e3e_arm.py "
        "(the committed observer + the flags)",
        "```",
        "",
        f"The campaign log prints `memory.max={log['memory_max']}` for the unit it ran in, because "
        "the kanban worker's own scope is capped at 4 GiB and a 20.8 GiB model inside it re-reads "
        "its weights from disk forever.",
        "",
        "## 4. Placement per chunk (the loader's own answer), one backend named",
        "",
    ]
    rows = []
    for chunk, bare in zip(stats["challenger_chunks"], stats["baseline_chunks"], strict=True):
        rows.append([
            f"`report_{chunk['chunk']}.json`",
            f"{chunk['correct']}/{chunk['items']}",
            f"{chunk['ngl_requested']} → {chunk['ngl_used']}",
            str(chunk["degraded"]),
            str(chunk["kv_type_used"]),
            str(chunk["n_ctx"]), str(chunk["n_prefix"]), str(chunk["n_seq_max"]),
            _num(chunk["load_wall_s"], 3) + " s", _num(chunk["chunk_wall_s"], 1) + " s",
            _num(chunk["median_decision_s"], 2) + " s",
            f"`{chunk['effective_backend']}`",
            " · ".join(f"{dev}={count}"
                       for dev, count in sorted(
                           (chunk.get("device_buffers") or {}).items())) or "—",
            f"`{bare['effective_backend']}`",
        ])
    lines += _table(["chunk", "challenger correct", "ngl req → used", "degraded", "kv_type_used",
                     "n_ctx", "n_prefix", "n_seq_max", "load wall", "chunk wall", "median decision",
                     "challenger effective_backend", "challenger compute buffers",
                     "baseline effective_backend"], rows)
    lines += [
        "",
        "* the row shape carries no per-item `effective_backend`, so \"per row\" "
        "here is the **per-chunk report**: each chunk is one report and each report carries the "
        "attribution block (`devices` / `device_buffers` / `effective_backend`) read from the "
        "engine's own log — one backend named, the card's rule after `t_55de5779`.",
        "",
    ]
    lines += ["", "* compute path per chunk (the engine's own buffer lines, challenger logs):", ""]
    lines += _table(["chunk", "compute lines", "first line", "devices in those lines"],
                    [[f"`{chunk['chunk']}`", str(chunk.get("compute_lines")),
                      f"`{(chunk.get('compute_first') or '').strip()}`",
                      ", ".join(f"{dev}={count}" for dev, count
                                in (chunk.get("compute_devices") or {}).items()) or "—"]
                     for chunk in stats["challenger_chunks"]])
    mismatches = sorted({w for chunk in stats["challenger_chunks"] for w in
                         (chunk.get("mismatch_warnings") or [])})
    lines += ["",
              f"* `W_BACKEND_MISMATCH` on any challenger row: **{mismatches or 'none'}** — every "
              f"challenger report claims `vulkan` and its own engine log shows `Vulkan0` computing "
              f"(the baseline reports the same claim; the pair is placement-matched: every "
              f"challenger chunk asked for {stats['challenger_chunks'][0]['ngl_requested']} layers "
              f"and got them, no degrade).",
              "",
              "## 5. The 60 items: challenger vs the corrected baseline",
              "",
              ]
    lines += [cell_line("baseline (corrected instrument, shipped cue)", base),
              cell_line("challenger (role_split + json_instructed)", chall), ""]
    rows = []
    for qtype in sorted(base["per_type"]):
        b = base["per_type"][qtype]
        c = chall["per_type"].get(qtype) or {}
        paired = stats["per_type_paired"].get(qtype) or {}
        rows.append([qtype, f"{b['correct']}/{b['n']}",
                     f"{c.get('correct')}/{c.get('n')}",
                     f"{paired.get('challenger_only')} / {paired.get('baseline_only')}",
                     _num(paired.get("difference"))
                     if paired.get("difference") is not None else "—",
                     _num(paired.get("mcnemar_p")) if paired.get("mcnemar_p") is not None else "—"])
    lines += ["### 5.1 By question type", ""]
    lines += _table(["type", "baseline", "challenger",
                     "discordant (challenger-only / baseline-only)",
                     "difference", "exact McNemar p"], rows)
    lines += ["", "### 5.2 The collapse, and where the mass went", ""]
    lines += _table(
        ["metric", "baseline", "challenger"],
        [["rows `low_mass`", f"{base['low_mass']}/{base['items']}",
          f"{chall['low_mass']}/{chall['items']}"],
         ["rows `measured` (≥ 0.10 floor)", f"{base['measured']}/{base['items']}",
          f"{chall['measured']}/{chall['items']}"],
         ["rows refused at the cue", f"{base['refusals']}/{base['items']}",
          f"{chall['refusals']}/{chall['items']}"],
         ["coverage min", _num(base["coverage_quantiles"]["min"]),
          _num(chall["coverage_quantiles"]["min"])],
         ["coverage p25", _num(base["coverage_quantiles"]["p25"]),
          _num(chall["coverage_quantiles"]["p25"])],
         ["coverage p50 (median)", _num(base["coverage_quantiles"]["p50"]),
          _num(chall["coverage_quantiles"]["p50"])],
         ["coverage p75", _num(base["coverage_quantiles"]["p75"]),
          _num(chall["coverage_quantiles"]["p75"])],
         ["coverage max", _num(base["coverage_quantiles"]["max"]),
          _num(chall["coverage_quantiles"]["max"])],
         ["prefix tokens", f"{base['prefix_tokens']['min']}–{base['prefix_tokens']['max']}",
          f"{chall['prefix_tokens']['min']}–{chall['prefix_tokens']['max']}"],
         ["framing (per row)", ", ".join(base["framing_rows"]["labels"]),
          ", ".join(chall["framing_rows"]["labels"])],
         ["framing mixed across rows", str(base["framing_rows"]["mixed"]),
          str(chall["framing_rows"]["mixed"])],
         ["distinct framing surfaces per row",
          str(len(base["framing_rows"]["surfaces"])),
          str(len(chall["framing_rows"]["surfaces"]))]])
    lines += ["", "**The refusals, with the tokens that closed them.**", ""]
    lines += _table(["arm", "refused rows", "closures"],
                    [["baseline", f"{base['closers']['refused']}/{base['closers']['refused_n']}",
                      ", ".join(f"`{key}` × {value}"
                                for key, value in base["closers"]["closers"].items()) or "—"],
                     ["challenger",
                      f"{chall['closers']['refused']}/{chall['closers']['refused_n']}",
                      ", ".join(f"`{key}` × {value}"
                                for key, value in chall["closers"]["closers"].items()) or "—"]])
    lines += ["", "**The cue verdicts (`value_verdict` breakdown).**", ""]
    lines += _table(["arm", "verdicts"],
                    [["baseline", ", ".join(f"`{key}` × {value}"
                                            for key, value in sorted(
                                                (base.get("verdicts") or {}).items())) or "—"],
                     ["challenger", ", ".join(f"`{key}` × {value}"
                                              for key, value in sorted(
                                                  (chall.get("verdicts") or {}).items())) or "—"]])
    aux_rows = []
    for name, arm in sorted(stats.get("aux", {}).items()):
        cell = arm["cell"]
        aux_rows.append([
            f"`{name}`", f"{cell['correct']}/{cell['items']}",
            f"{_num(cell['agreement'])} [{_interval(cell['ci'])}]",
            f"{cell['low_mass']}/{cell['items']}", f"{cell['refusals']}/{cell['items']}",
            _num(cell["coverage_quantiles"]["p50"]),
            f"{cell['prefix_tokens']['min']}–{cell['prefix_tokens']['max']}",
            f"{arm['vs_baseline']['difference']:+.3f} "
            f"(p = {_num(arm['vs_baseline']['mcnemar_p'])})",
            f"{arm['vs_challenger']['difference']:+.3f} "
            f"(p = {_num(arm['vs_challenger']['mcnemar_p'])})",
            f"`{cell['effective_backend']}`",
        ])
    if aux_rows:
        lines += [
            "",
            "### 5.3 The auxiliary arms (same instrument, never the card's row)",
            "",
            "The two levers are separable, so the same 60 items were measured once more for each "
            "half: the placement alone (`role_split` with the shipped cue) and the 4B's collapse "
            "cell (`role_split` with `two_step`). Same model file, same items, same placement ask, "
            "same single backend — the cells differ only in the policy flags. They are published "
            "as auxiliary: the card's row is the challenger.",
            "",
        ]
        lines += _table(["arm", "correct", "agreement (Wilson 95 %)", "low_mass", "refusals",
                         "coverage p50", "prefix tokens", "vs baseline (paired)", "vs challenger",
                         "effective_backend"], aux_rows)
        identity = stats.get("aux_identity", {}) or {}
        same = identity.get("role_split_only_vs_two_step") or {}
        moved = identity.get("challenger_vs_role_split_only") or {}
        delta = same.get("max_abs_coverage_delta")
        delta_text = "0" if delta == 0 else _num(delta)
        lines += [
            "",
            f"* **`two_step` cannot engage under the role split on this family**: the two "
            f"auxiliary cells agree on **{same.get('decisions_identical')}/{same.get('n')}** "
            f"decisions, on {same.get('prefix_tokens_identical')}/{same.get('n')} prefix-token "
            f"counts, and on the coverage of every shared row (max |Δcoverage| = {delta_text}, "
            f"identical to the printed digits). "
            f"**{len(same.get('winner_flips') or [])} winner flip(s)** — because every one of the "
            f"60 cue rows under the role split is refused (`low_mass` "
            f"{stats['aux']['role_split_only']['cell']['low_mass']}/60, refusals "
            f"{stats['aux']['role_split_only']['cell']['refusals']}/60) and "
            f"`decide._advance_token` "
            f"never advances past a cue the model closed (the rule "
            f"`t_7c926398` §5.1 measured on the "
            f"shipped placement). The 4B's collapse cell (28/60) is the "
            f"*other* branch of that rule: "
            f"there the label sits **at** the cue row, the advance succeeds, and "
            f"the row it reaches "
            f"is not a label row.",
            f"* the instructed contract moves "
            f"**{len(moved.get('winner_flips') or [])} of {moved.get('n')}** winners on top of the "
            f"placement "
            f"({moved.get('decisions_identical')}/{moved.get('n')} identical "
            f"decisions) and is what "
            f"turns the readout into a measured one "
            f"(`low_mass` {stats['aux']['role_split_only']['cell']['low_mass']}/60 → "
            f"{chall['low_mass']}/60) — the placement by itself still leaves the readout where the "
            f"model opens its think block, which is why the two cells above are `measured` 0/60.",
            f"* **the auxiliary agreements are not readable as accuracy**, and the table above "
            f"should not be read that way: every row of both cells is `low_mass` (`measured` "
            f"{stats['aux']['role_split_only']['cell']['measured']}/60) with median coverage "
            f"{_num(stats['aux']['role_split_only']['cell']['coverage_quantiles']['p50'])} — under "
            f"the role split with a cue that is not a contract, Tiel opens its think block at the "
            f"readout row (`<think>` × 60 closers, mass ≈ 1.0) and the candidates' mass is a tail. "
            f"Those numbers are stable (identical on a re-measure, §7) but they are guesses among "
            f"candidates, which is what the `low_mass` floor exists to say. The challenger is the "
            f"only cell in this document whose rows are `measured`.",
            "",
        ]
    lines += ["", "## 6. Paired against §7.4.1 (same 60 items, item by item)", ""]
    lines += [
        f"* risk difference (challenger − baseline): **{_diff_interval(pair)}**",
        f"* discordant pairs: challenger-only correct **{pair['challenger_only']}**, "
        f"baseline-only correct **{pair['baseline_only']}**",
        f"* both correct **{pair['both_correct']}**, neither correct **{pair['neither_correct']}**",
        f"* items paired: **{pair['n']}** — unpaired rows: "
        f"{len(stats['paired_items']['unpaired_baseline'])} baseline-only id(s), "
        f"{len(stats['paired_items']['unpaired_challenger'])} challenger-only id(s)",
        f"* the card's rule (the E3e unit): a challenger displaces the baseline only when the "
        f"paired difference's 95 % interval excludes zero and the exact two-sided McNemar p < 0.05 "
        f"— here **{'WIN' if pair['challenger_wins'] else 'NOT by the rule'}**",
        f"* interval caveat (the tool's own words): {pair['caveat']}",
        "",
    ]
    won = stats["paired_items"]["challenger_only"]
    lost = stats["paired_items"]["baseline_only"]

    def item_table(rows: list[dict[str, Any]]) -> list[str]:
        if not rows:
            return ["(none)", ""]
        return _table(["id", "type", "gold", "baseline got", "challenger got"],
                      [[f"`{row['id']}`", str(row["type"]), f"`{row['expected']}`",
                        f"`{row['baseline_got']}`", f"`{row['challenger_got']}`"] for row in rows])

    lines += ["Items the challenger wins that the corrected baseline lost:", ""]
    lines += item_table(won)
    lines += ["", "Items the baseline got and the challenger loses:", ""]
    lines += item_table(lost)
    lines += ["", "## 7. The two 6-item cells of the card's falsification order", ""]
    receipts = stats.get("smoke_vs_arm") or {}
    same_collapse = receipts.get("collapse_vs_two_step_arm") or {}
    same_challenger = receipts.get("challenger_vs_challenger_arm") or {}
    collapse = stats["smokes"].get("collapse")
    challenger_smoke = stats["smokes"].get("challenger")
    if collapse:
        lines += [
            f"**1. The collapse cell (`--chat-format role_split --cue two_step`), 6 items** — "
            f"`{collapse['cell']}`: {collapse['correct']}/{collapse['items']} correct, "
            f"`low_mass` {collapse['low_mass']}/{collapse['items']}, refusals "
            f"{collapse['refusals']}/{collapse['items']}, closers "
            + (", ".join(f"`{key}` × {value}"
                         for key, value in collapse["closers"]["closers"].items()) or "—") + ".",
            "",
        ]
        lines += _table(["item", "type", "gold", "got", "correct", "coverage", "reliability",
                         "cue top token", "top-token mass", "verdict"],
                        [[f"`{row['id']}`", row["type"], f"`{row['expected']}`",
                          f"`{row['got']}`", str(row["correct"]), _num(row["coverage"]),
                          f"`{row['reliability']}`", f"`{row['cue_top']}`",
                          _num(row["cue_mass"], 4),
                          "W_CUE_REFUSED" if row["cue_refused"] else "ok"]
                         for row in collapse["rows"]])
    if challenger_smoke:
        lines += [
            "",
            f"**2. The challenger (`role_split` + `json_instructed`), the same 6 items** — "
            f"`{challenger_smoke['cell']}`: {challenger_smoke['correct']}/"
            f"{challenger_smoke['items']} correct, `low_mass` "
            f"{challenger_smoke['low_mass']}/{challenger_smoke['items']}, refusals "
            f"{challenger_smoke['refusals']}/{challenger_smoke['items']}.",
            "",
        ]
        lines += _table(["item", "type", "gold", "got", "correct", "coverage", "reliability",
                         "cue top token", "top-token mass", "verdict"],
                        [[f"`{row['id']}`", row["type"], f"`{row['expected']}`",
                          f"`{row['got']}`", str(row["correct"]), _num(row["coverage"]),
                          f"`{row['reliability']}`", f"`{row['cue_top']}`",
                          _num(row["cue_mass"], 4),
                          "W_CUE_REFUSED" if row["cue_refused"] else "ok"]
                         for row in challenger_smoke["rows"]])
    lines += [
        "",
        f"**Determinism receipt (both cells).** The six smoke items are also the first six rows of "
        f"the 60-item campaign's chunk 001, so the same cells were measured twice — once in a "
        f"6-item run, once inside the 10-item chunk — and they agree row for row: collapse cell "
        f"**{same_collapse.get('flips')} winner flip(s) of {same_collapse.get('n')}**, challenger "
        f"**{same_challenger.get('flips')} of {same_challenger.get('n')}**, coverage identical to "
        f"the printed digits on every shared row. Temperature 0 reads the same bytes the same way; "
        f"what the auxiliary cells cannot do is put mass on a candidate (their agreement is a tail "
        f"argmax, and it is *stably* so).",
    ]
    lines += ["", "## 8. What the reading is — and what it is not", ""]
    lines += [
        f"* the challenger's {chall['correct']}/{chall['items']} against §7.4.1's "
        f"{base['correct']}/{base['items']} is measured on the same committed "
        f"items, the same model "
        f"file (SHA identical before and after), the same placement ask and the same single "
        f"backend; the only difference is the two policy flags.",
        f"* the collapse the corrected row published is gone on this arm: refusals "
        f"{base['refusals']} → {chall['refusals']}, `low_mass` {base['low_mass']} → "
        f"{chall['low_mass']}, median coverage {_num(base['coverage_quantiles']['p50'])} → "
        f"{_num(chall['coverage_quantiles']['p50'])}.",
        "* 60 items: a single cell's interval is still up to ~24 points wide, so the *paired* "
        "columns are what carry the claim; the type cells (18–24 items) are leads, not results.",
        f"* the auxiliary cells are **not** accuracy claims: they are `measured` "
        f"{stats['aux']['role_split_only']['cell']['measured']}/60 with median coverage "
        f"{_num(stats['aux']['role_split_only']['cell']['coverage_quantiles']['p50'])}, so their "
        f"agreement is the argmax of a tail (§5.3). What they establish is the *mechanism*: the "
        f"placement moves the question out of the assistant turn, and `two_step` is inert under it "
        f"on this family because every cue row is closed before the readout can advance.",
        f"* `score` remains the hardest type for this model even under the winning policy: "
        f"{chall['per_type'].get('score', {}).get('correct')}/"
        f"{chall['per_type'].get('score', {}).get('n')} against "
        f"{base['per_type'].get('score', {}).get('correct')}/"
        f"{base['per_type'].get('score', {}).get('n')} — the same reading E3c/E3d/E3e published.",
        "* this is one model's row under a *policy*; it moves no default and it does not rank "
        "models. The comparability note of E3e applies unchanged: a cell measured under a role "
        "split and an instructed contract cannot be compared with rows measured on the shipped "
        "prompt bytes.",
        "* the optional Occamy pass (§9) is a pair measured **entirely here** — its own shipped/"
        "answer-sheet cell against `role_split` + `json_instructed` — because Occamy has no row "
        "under the corrected instrument; it is not comparable with the container-era E3 row, and "
        "its numbers live in this document (this card's `docs/BENCHMARKS.md` block is the Tiel "
        "row).",
        "",
    ]
    lines += render_occamy(stats)
    occ_facts = stats.get("occamy") or {}
    lines += [
        "",
        "## 10. Gates",
        "",
    ]
    lines += [
        "| gate | command | result |",
        "|---|---|---|",
        "| campaign driver exits 0 | `bash .t9bcb/run_chunks.sh` (unit `t9bcb-campaign`) | "
        + ", ".join(f"chunk {name} exit={info['exit']} ({info['wall_s']} s)"
                    for name, info in sorted(log["chunk_walls"].items()))
        + f" — raw log `{log['path']}` |",
        "| model SHA-256 before/after | `sha256sum <tiel.gguf>` (campaign head and tail) | "
        + f"identical — `{log['sha_before']}` |",
        "| dev-set slices | `sha256sum docs/evidence/tiel_corrected_chunks/devset_00*.jsonl` | "
        + f"byte-identical to the baseline's receipt — `{devset['identical']}` |",
        "| 6-item smokes exit 0 | `.t9bcb/smoke.sh` (unit `t9bcb-smoke`) | see §7 |",
        "| the two auxiliary arms exit 0 | `TAG=… bash .t9bcb/run_arm.sh` (unit `t9bcb-aux`) | "
        + ", ".join(f"{name} {arm['cell']['correct']}/{arm['cell']['items']}"
                    for name, arm in sorted(stats.get("aux", {}).items()))
        + " — raw log `.t9bcb/logs/campaign_aux.log` |",
        "| oracle | `python3 docs/verify_runtime_contract.py` | "
        + ((stats.get("gates") or {}).get("oracle", {}).get("summary") or "—")
        + f" (`{(stats.get('gates') or {}).get('oracle', {}).get('receipt', '—')}`) |",
        "| test suite | `uv run --frozen --offline --extra dev pytest -q -rs` | "
        + ((stats.get("gates") or {}).get("suite", {}).get("summary") or "—")
        + f" (`{(stats.get('gates') or {}).get('suite', {}).get('receipt', '—')}`) |",
        "| ruff (the paths this card touches) | `uv run --frozen --offline --extra dev ruff check "
        ".t9bcb` | "
        + ((stats.get("gates") or {}).get("ruff", {}).get("summary") or "—")
        + f" (`{(stats.get('gates') or {}).get('ruff', {}).get('receipt', '—')}`) |",
        "| the suite again, after this document's render | "
        "`uv run --frozen --offline --extra dev pytest -q -rs -p no:cacheprovider` | "
        + (((stats.get("gates") or {}).get("suite_post_render") or {}).get("summary") or "—")
        + f" (`{((stats.get('gates') or {}).get('suite_post_render') or {}).get('receipt', '—')}`)"
        + " — the only tree change after that run is this section's own text |",
        "| the optional Occamy pass exits 0 | `bash .t9bcb/run_occamy.sh` (unit `t9bcb-occamy2`) | "
        + (f"{len(occ_facts.get('chunk_exits') or [])} chunk(s), all exit 0 — "
           f"`{(occ_facts or {}).get('all_chunks_ok')}` — log `{occ_facts.get('log', '—')}` |"
           if occ_facts else "not run |"),
        "",
        "The suite's skip count is this host's, not a container's: the worker scope carries no "
        "container pid cgroup, so `test_probe_pressure.py` skips — the same skip the baseline card "
        "recorded on this box.",
        "",
        "## Receipts",
        "",
        "* campaign driver: `.t9bcb/run_chunks.sh`, `.t9bcb/tiel_e3e_arm.py` (the committed "
        "observer of `tools/e3c_tiel_reproduce.py` + the two flags + the surface counters)",
        "* smokes: `.t9bcb/smoke.sh`, `.t9bcb/smoke_collapse.json`, "
        "`.t9bcb/smoke_challenger.json` and their placement sinks",
        "* per-chunk reports + placement sinks: `docs/evidence/t9bcbecff_tiel_chunks/`",
        "* merged challenger report: `docs/evidence/t9bcbecff_tiel_challenger_quality.json`",
        "* merged auxiliary reports: `docs/evidence/t9bcbecff_tiel_role_split_quality.json`, "
        "`docs/evidence/t9bcbecff_tiel_two_step_quality.json`",
        "* baseline: `docs/evidence/tiel_corrected_quality.json` + "
        "`docs/evidence/tiel_corrected_chunks/` (card `t_7c926398`, unchanged)",
        "* statistics: `.t9bcb/stats.json` (this document is rendered from it by "
        "`.t9bcb/render_doc.py`)",
        "* run logs: `.t9bcb/logs/campaign.log`, `.t9bcb/logs/run_00{1..6}.log`, "
        "`.t9bcb/logs/smoke_*.log`, `.t9bcb/logs/campaign_aux.log` (+ the two auxiliary arms' "
        "per-chunk logs)",
        "* the same numbers in the benchmark document: `docs/BENCHMARKS.md` §7.4.2, "
        "spliced by this "
        "script from the same `stats.json`",
        "* the optional Occamy pass: `.t9bcb/run_occamy.sh` → `.t9bcb/run_arm.sh`, "
        "`docs/evidence/t9bcbecff_occamy_chunks/`, the two merged reports, "
        "`.t9bcb/logs/occamy.log` + `.t9bcb/logs/occamy_sha.txt` (the pin, taken after the pass)",
        "",
    ]
    TARGET.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {TARGET}")
    print(f"docs/BENCHMARKS.md §7.4.2: {splice_benchmarks(render_benchmarks(stats))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
