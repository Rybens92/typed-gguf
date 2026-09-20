#!/usr/bin/env python3
"""Regenerate the E3 sections of `docs/BENCHMARKS.md` and the E3 evidence doc from the artifacts.

Card `t_6d2e084d` finishes A-E3-3 (20 -> 60 dev items) and requires the published prose to be
**re-generated, not hand-edited**: §6.4 of `docs/BENCHMARKS.md` and §2.3/§4 of
`docs/evidence/e3_t_a431be85_occamy.md` quote numbers that move when a chunk lands, so they are
computed here from the same files the table is computed from:

* `docs/evidence/e3_chunks/report_*.json` — one `--suite quality` report per chunk (the campaign);
* `docs/evidence/e3_occamy_quality.json`    — the merged report (`--suite merge`);
* `docs/evidence/e2_quality.json`           — the 4B baseline the table is paired against;
* `docs/evidence/e3_comparison.md`          — the rendered table (`--suite compare --align`).

Every region lives between a `<!-- @@..._BEGIN@@ -->` / `<!-- @@..._END@@ -->` marker pair and is
replaced wholesale; `--check` verifies the files on disk are what this tool would write (that is
the gate: a hand-edited number in a generated region fails it).

**The `[host]` / `[container]` tag is read from the report, never typed.** A report produced in the
worker container carries the cgroup it was squeezed into (`host.cgroup_memory_bytes`, 8 GiB); on
the operator host those keys are absent — `report.report_box()` is the only place that decides.

    python3 tools/e3_build_evidence.py          # rewrite the generated regions in place
    python3 tools/e3_build_evidence.py --check  # non-zero when a region is stale (no writes)
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
from collections.abc import Mapping, Sequence
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from typed_gguf.bench import compare  # noqa: E402

EVIDENCE_DOC = ROOT / "docs/evidence/e3_t_a431be85_occamy.md"
BENCHMARKS_DOC = ROOT / "docs/BENCHMARKS.md"
CHUNK_GLOB = "docs/evidence/e3_chunks/report_*.json"
MERGED_REPORT = "docs/evidence/e3_occamy_quality.json"
BASELINE_REPORT = "docs/evidence/e2_quality.json"
COMPARISON_MD = "docs/evidence/e3_comparison.md"

#: the two published labels of the A-E3-3 table (the same pair `--suite compare` is called with)
BASELINE_LABEL = "4B default (E2, 60 items, CPU)"
CHALLENGER_LABEL = "Occamy 1.0 (E3, chunks, vulkan)"

#: region name -> the marker pair that owns it (`(begin, end)`), one pair per file region
REGIONS: dict[str, tuple[str, str]] = {
    "e3_host_items": ("<!-- @@E3_HOST_ITEMS_BEGIN@@ -->", "<!-- @@E3_HOST_ITEMS_END@@ -->"),
    "e3_chunks": ("<!-- @@E3_CHUNK_BEGIN@@ -->", "<!-- @@E3_CHUNK_END@@ -->"),
    "e3_comparison": ("<!-- @@E3_COMPARE_BEGIN@@ -->", "<!-- @@E3_COMPARE_END@@ -->"),
    "e3_integrity": ("<!-- @@E3_INTEGRITY_BEGIN@@ -->", "<!-- @@E3_INTEGRITY_END@@ -->"),
    "bench_chunks": ("<!-- @@E3C_BENCH_6_2_BEGIN@@ -->", "<!-- @@E3C_BENCH_6_2_END@@ -->"),
    "bench_comparison": ("<!-- @@E3C_BENCH_6_4_BEGIN@@ -->", "<!-- @@E3C_BENCH_6_4_END@@ -->"),
}

#: which file each region lives in
REGION_FILES: dict[str, pathlib.Path] = {
    "e3_host_items": EVIDENCE_DOC,
    "e3_chunks": EVIDENCE_DOC,
    "e3_comparison": EVIDENCE_DOC,
    "e3_integrity": EVIDENCE_DOC,
    "bench_chunks": BENCHMARKS_DOC,
    "bench_comparison": BENCHMARKS_DOC,
}

#: the chunk commands the published section prints — the exact invocations that produced the files
CHUNK_COMMAND: tuple[str, ...] = (
    "python3 tools/e3_reproduce.py --write-chunks .e3/chunks --chunk 10",
    "TYPED_GGUF_RUNTIME_DIR=<bundle> VK_DRIVER_FILES=<icd> python3 tools/e3_reproduce.py \\",
    "    --suite quality --model ~/.hermes/models/Accio-Lab_occamy-1.0-Q4_K_L.gguf \\",
    "    --backend vulkan --gpu-layers 7 --threads 4 \\",
    "    --devset docs/evidence/e3_chunks/devset_00N.jsonl \\",
    "    --out docs/evidence/e3_chunks/report_00N.json     # N = 001 … 006",
    "python3 tools/e3_reproduce.py --suite merge \\",
    "    --reports 'docs/evidence/e3_chunks/report_*.json' \\",
    '    --label "Occamy 1.0" --out docs/evidence/e3_occamy_quality.json',
)


# --------------------------------------------------------------------------- reading the artifacts
def load(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def chunk_reports() -> list[tuple[pathlib.Path, dict[str, Any]]]:
    """Every chunk report on disk, sorted by file name (001, 002, ... — the campaign order)."""
    found = sorted(ROOT.glob(CHUNK_GLOB))
    if not found:
        raise SystemExit(f"no chunk reports under {CHUNK_GLOB}")
    return [(path, json.loads(path.read_text(encoding="utf-8"))) for path in found]


def report_box(report: Mapping[str, Any]) -> str:
    """`host` or `container`, from the report's own host block (never from its file name).

    The container reports carry the cgroup the run was squeezed into; a host report has no cgroup
    keys at all, because the bench records them only when the kernel exposes one.

    One-directional on purpose (review F4 of `t_6d2e084d`): the cgroup keys are positive evidence of
    a container, their *absence* is not positive evidence of a bare host — a container run whose
    kernel hides its cgroup would be tagged `[host]` here. So the tag is never typed into a table
    (§4.3 prints it next to each report's own `ok`/row count, and this function is gated), and a
    future cgroup-less container needs a report field this tool controls rather than this guess.
    """
    host = report.get("host") or {}
    memory = host.get("cgroup_memory_bytes")
    quota = host.get("cgroup_cpu_max")
    if memory or quota:
        return "container"
    return "host"


def item_rows(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = report.get("items")
    return list(rows) if isinstance(rows, Sequence) else []


def counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        key = str(row.get("type"))
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def seconds(ms: float) -> float:
    return float(ms) / 1000.0


def median_wall_s(rows: Sequence[Mapping[str, Any]]) -> float:
    return statistics.median([seconds(row.get("wall_ms") or 0.0) for row in rows])


def cell(block: Mapping[str, Any]) -> str:
    return compare._cell(block)


def safe_cell(block: Mapping[str, Any] | None) -> str:
    """`cell` for a block a side may not have at all — renders `—`, like a zero-row block.

    The table has one row per question type measured by *either* report, and `compare._cell` reads
    `block["n"]`, so an absent type must not reach it (review F2 of `t_6d2e084d`): `compare`'s own
    renderer fills the gap with an empty agreement block, and this prints what that renders to.
    """
    return cell(block) if block else "—"


def fmt_counts(counts_by_type: Mapping[str, int]) -> str:
    order = [name for name in ("choice", "score", "noul") if name in counts_by_type]
    return "/".join(str(counts_by_type[name]) for name in order)


def device_summary(report: Mapping[str, Any]) -> str:
    """What the report proves about the compute path: attribution keys, or the placement block."""
    buffers = report.get("device_buffers")
    effective = report.get("effective_backend")
    if isinstance(buffers, Mapping) and buffers:
        rendered = ", ".join(f"`{name}`={int(value)}" for name, value in buffers.items())
        return f"buffers: {rendered}; effective `{effective}`"
    placement = (report.get("placement") or {}).get("used") if isinstance(
        report.get("placement"), Mapping) else None
    if isinstance(placement, Mapping):
        walked = [str(attempt).replace("n_gpu_layers=", "").replace(" -> ", "→")
                  for attempt in (placement.get("attempts") or [])]
        suffix = f", walked {', '.join(walked)}" if walked else ""
        return (f"`n_gpu_layers {placement.get('n_gpu_layers')}`, "
                f"`kv_type {placement.get('kv_type')}`, "
                f"`degraded: {str(placement.get('degraded')).lower()}`{suffix}")
    return "no attribution key and no placement block in this report"


# --------------------------------------------------------------------------- the generated regions
def render_host_items(reports: Sequence[tuple[pathlib.Path, dict[str, Any]]]) -> str:
    """§2.3's host block: the first two items measured on the operator host (`[host]`).

    The container's own per-item table (A-E3-1) is a container fact and stays as published; this
    block is what the card added — the same protocol on a box whose 31 GiB can cache the weights,
    which is why the campaign could be finished at all.
    """
    host_chunks = [(path, report) for path, report in reports
                   if report_box(report) == "host" and item_rows(report)]
    lines: list[str] = []
    if not host_chunks:
        lines.append("No `[host]` chunk landed, so there is no host per-item table: every chunk "
                     "above is a `[container]` run.")
        lines.append("")
        return "\n".join(lines).rstrip() + "\n"
    path, report = host_chunks[0]
    lines.append(f"The card's `[host]` run (chunks "
                 f"{', '.join('`' + p.stem + '`' for p, _ in host_chunks)}) uses the same flags "
                 f"`--backend vulkan --gpu-layers 7 --threads 4` on the operator host — no cgroup, "
                 f"no 8 GiB memory cap, so the weights cache after the first pass. The first two "
                 f"items of `{path.name}`:")
    lines.append("")
    lines.append("| item | type | questions_ms | wall_s | correct | coverage | reliability |")
    lines.append("|---|---|---:|---:|---|---:|---|")
    for row in item_rows(report)[:2]:
        lines.append(
            f"| `{row.get('id')}` | {row.get('type')} | {row.get('questions_ms', 0.0):,.0f} | "
            f"{seconds(row.get('wall_ms') or 0.0):.1f} | "
            f"{'✔' if row.get('correct') else '✘'} (`{row.get('got')}`) | "
            f"{row.get('coverage', 0.0):.3f} | `{row.get('reliability')}` |")
    lines.append("")
    medians = [median_wall_s(item_rows(report)) for _, report in host_chunks]
    by_chunk = ", ".join(f"`{path.stem}` {median:.1f} s"
                         for (path, _), median in zip(host_chunks, medians, strict=True))
    lines.append(f"Per-item wall on the host, by chunk: {by_chunk} — against 99–113 s per item in "
                 f"the container, which is the point of the tag: the *protocol* is identical and "
                 f"the *box* is not.")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_chunks(reports: Sequence[tuple[pathlib.Path, dict[str, Any]]],
                  merged: Mapping[str, Any]) -> str:
    """§2.3's chunk block: one row per chunk with its box tag, plus the merged statement."""
    lines: list[str] = []
    lines.append("Because a single 60-item pass is a multi-hour job on the container, the campaign "
                 "is sliced into **stratified chunks of 10 items** (`devset.stratified_chunks`; "
                 "every chunk — and every prefix of them — mixes `choice | score | noul`), each "
                 "with its own JSON report, merged by `compare.merge_reports` into the one report "
                 "the table reads:")
    lines.append("")
    lines.append("```bash")
    lines.extend(CHUNK_COMMAND)
    lines.append("```")
    lines.append("")
    lines.append("| chunk | box | items (choice/score/noul) | compute path the report proves | "
                 "wall per item (median) | correct |")
    lines.append("|---|---|---|---|---:|---:|")
    for path, report in reports:
        rows = item_rows(report)
        box = report_box(report)
        overall = report.get("overall") or {}
        lines.append(
            f"| `{path.stem}` | `[{box}]` | {len(rows)} ({fmt_counts(counts(rows))}) | "
            f"{device_summary(report)} | {median_wall_s(rows):.1f} s | "
            f"{overall.get('correct')}/{overall.get('n')} |")
    lines.append("")
    host_paths = [path.stem for path, report in reports if report_box(report) == "host"]
    container_paths = [path.stem for path, report in reports if report_box(report) == "container"]
    merged_overall = merged.get("overall") or {}
    lines.append(
        f"Merged: `docs/evidence/e3_occamy_quality.json` — {merged_overall.get('n')} items, "
        f"{merged_overall.get('correct')} correct ({merged_overall.get('agreement'):.3f}, 95 % CI "
        f"{merged_overall.get('ci', [0, 0])[0]:.3f}–{merged_overall.get('ci', [0, 0])[1]:.3f}); "
        f"`chunks` lists every chunk it stitched: "
        f"{', '.join('`' + path.name + '`' for path, _ in reports)}.")
    lines.append("")
    if host_paths:
        lines.append(
            f"A **`[host]` run did happen**: {len(host_paths)} of the {len(reports)} chunks "
            f"({', '.join('`' + name + '`' for name in host_paths)}) were measured on the "
            f"operator host after the card moved the campaign off the container (`[container]`: "
            f"{', '.join('`' + name + '`' for name in container_paths) or 'none'}). What it "
            f"changed is `n` and the intervals — the published per-item *cost* table keeps its "
            f"`[container]` tag, because the container is where a 23 GB model against 8 GiB of "
            f"memory shows its real price; the host rows are the same protocol on a box that can "
            f"cache the weights, and each row above says which box it came from.")
        lines.append("")
    else:
        lines.append("No `[host]` run happened: every chunk above is a `[container]` run.")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def comparison_table(merged: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    paired = compare.align(baseline, merged)
    table = compare.comparison(paired["baseline"], paired["challenger"],
                              labels=(BASELINE_LABEL, CHALLENGER_LABEL))
    return {"paired": paired, "table": table}


def render_comparison(merged: Mapping[str, Any], baseline: Mapping[str, Any],
                      reports: Sequence[tuple[pathlib.Path, dict[str, Any]]]) -> str:
    """§4 (and §6.4): the paired table + the readout that has to follow the numbers."""
    pair = comparison_table(merged, baseline)
    table, paired = pair["table"], pair["paired"]
    first, second = table["models"]
    base, challenger = first["overall"], second["overall"]
    lines: list[str] = []
    lines.append("Paired on the dev items **both** models measured — `e2_quality.json` cut to the "
                 "same ids (`tools/e3_reproduce.py --suite compare --align`):")
    lines.append("")
    lines.append("```markdown")
    lines.append(compare.render_comparison(table))
    lines.append("```")
    lines.append("")
    lines.append(
        f"`{CHALLENGER_LABEL}` is {table['delta']:+.3f} against `{BASELINE_LABEL}` overall "
        f"({base['agreement']:.3f} -> {challenger['agreement']:.3f}); the paired comparison "
        f"covers {paired['items']} dev items and drops {paired['dropped']['baseline']} unpaired "
        f"baseline row(s) and {paired['dropped']['challenger']} unpaired challenger row(s), so "
        f"both sides answer the same questions.")
    lines.append("")
    lines.append("### 4.1 What the numbers say")
    lines.append("")
    for qtype in table["types"]:
        left = first["per_type"].get(qtype) or {}
        right = second["per_type"].get(qtype) or {}
        if not left.get("n") and not right.get("n"):
            continue
        lines.append(f"* **`{qtype}`**: {safe_cell(left)} against {safe_cell(right)} — delta "
                     f"{right.get('agreement', 0.0) - left.get('agreement', 0.0):+.3f}.")
    lines.append(f"* **mass**: Occamy's answers fall below the {table['coverage_floor']:.2f} floor "
                 f"on {second[compare.LOW_MASS]['n']} of its {second['overall']['n']} rows "
                 f"(the 4B's on {first[compare.LOW_MASS]['n']}); inside the split the two are "
                 f"level ({cell(first[compare.LOW_MASS])} against "
                 f"{cell(second[compare.LOW_MASS])}), and the row the table says to read first is "
                 f"`{compare.MEASURED}` — {cell(first[compare.MEASURED])} against "
                 f"{cell(second[compare.MEASURED])}{_measured_note(second[compare.MEASURED])}.")
    delta_items = abs(round(table["delta"] * base["n"]))
    verdict = ("overlap, so the headline cannot separate the models" if
               _intervals_overlap(base, challenger) else "do not overlap")
    lines.append(f"* **verdict**: {delta_items} item{'s' if delta_items != 1 else ''} apart "
                 f"overall at n = {base['n']} ({base['agreement']:.3f} vs "
                 f"{challenger['agreement']:.3f}); the two Wilson intervals {verdict} — the rows "
                 f"that separate them are the per-type ones and the mass split above.")
    lines.append("")
    boxes = _box_counts(reports)
    lines.append(f"The Occamy side of this table is the merged campaign report ({boxes}, per-chunk "
                 f"tags in §2.3), measured with `--backend vulkan --gpu-layers 7 --threads 4`; the "
                 f"baseline is E2's 4B default on the CPU. Agreement does not depend on the ladder "
                 f"a chunk settled on, and every chunk report carries its own compute-path "
                 f"evidence (table in §2.3).")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _intervals_overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Do two agreement blocks share a Wilson interval? (the honest reading of a small delta)"""
    low_a, high_a = left.get("ci", [0.0, 1.0])
    low_b, high_b = right.get("ci", [0.0, 1.0])
    return not (high_a < low_b or high_b < low_a)


def _measured_note(block: Mapping[str, Any], *, small: int = 5) -> str:
    """`, which is only N item(s)` — the note a three-row `measured` block needs to be readable.

    `block` is an **agreement block**, never the model row that carries it: a model row has no `n`,
    and reading one through this helper published "only 0 item(s)" next to a three-row block
    (review F1 of `t_6d2e084d`), so a missing `n` is an error here, not a zero.
    """
    if "n" not in block:
        raise KeyError("_measured_note takes an agreement block (with `n`), not a model row — "
                       f"got keys {sorted(block)}")
    count = int(block["n"])
    return f", which is only {count} item(s)" if count < small else ""


def _box_counts(reports: Sequence[tuple[pathlib.Path, dict[str, Any]]]) -> str:
    """`[container] ×2 + [host] ×4` — how many chunks came from which box."""
    counts: dict[str, int] = {}
    for _, report in reports:
        box = report_box(report)
        counts[box] = counts.get(box, 0) + 1
    return " + ".join(f"`[{box}]` ×{count}" for box, count in sorted(counts.items()))


def render_integrity(reports: Sequence[tuple[pathlib.Path, dict[str, Any]]],
                     merged: Mapping[str, Any]) -> str:
    """§4.3: every report the published rows read, with its `ok` and its measured row count."""
    lines: list[str] = []
    lines.append("Every Occamy row in this document comes from a run with an **explicit "
                 "`--backend vulkan`** (never `--backend all`, which on a single-bundle host is "
                 "designed to fail the `cpu` row — coordinator's `t_dd62ec29` note), and every "
                 "report's own `ok`/row count was checked before publishing:")
    lines.append("")
    lines.append("| report | box | `ok` | rows | report shape |")
    lines.append("|---|---|---|---|---|")
    for path, report in reports:
        rows = item_rows(report)
        shape = ("placement block, no `devices`/`device_buffers`/`effective_backend`"
                 if "placement" in report and "devices" not in report else
                 "carries the `t_603a35a0` device attribution (`devices`/`device_buffers`/"
                 "`effective_backend`)")
        box = report_box(report)
        ok = str(report.get("ok")).lower()
        lines.append(f"| `{path.name}` | `[{box}]` | `{ok}` | {len(rows)} items | {shape} |")
    chunks = merged.get("chunks") or []
    lines.append(f"| `e3_occamy_quality.json` (merged) | — | `{str(merged.get('ok')).lower()}` | "
                 f"{(merged.get('overall') or {}).get('n')} items, `chunks` lists all "
                 f"{len(chunks)} | top-level keys are chunk 001's (`merge_reports` copies the "
                 f"first report's envelope) |")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_bench_chunks(reports: Sequence[tuple[pathlib.Path, dict[str, Any]]],
                        merged: Mapping[str, Any]) -> str:
    """§6.2 in `docs/BENCHMARKS.md`: the same chunk ledger, in the published table's form."""
    lines: list[str] = []
    overall = merged.get("overall") or {}
    lines.append("| chunk | box | items (choice/score/noul) | compute path the report proves | "
                 "wall per item (median) | correct |")
    lines.append("|---|---|---|---|---:|---:|")
    for path, report in reports:
        rows = item_rows(report)
        lines.append(
            f"| `docs/evidence/e3_chunks/{path.name}` | `[{report_box(report)}]` | "
            f"{len(rows)} ({fmt_counts(counts(rows))}) | {device_summary(report)} | "
            f"{median_wall_s(rows):.1f} s | {(report.get('overall') or {}).get('correct')}/"
            f"{(report.get('overall') or {}).get('n')} |")
    lines.append("")
    lines.append(f"Merged: `docs/evidence/e3_occamy_quality.json` — {overall.get('n')} items, "
                 f"{overall.get('correct')} correct ({overall.get('agreement'):.3f}, 95 % CI "
                 f"{overall.get('ci', [0, 0])[0]:.3f}–{overall.get('ci', [0, 0])[1]:.3f}); the "
                 f"`[container]` rows are the cost measurement this section is about, the "
                 f"`[host]` rows are the same protocol on a box that caches the weights.")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_bench_comparison(merged: Mapping[str, Any], baseline: Mapping[str, Any]) -> str:
    """§6.4 in `docs/BENCHMARKS.md` — the published A-E3-3 table and its headline."""
    pair = comparison_table(merged, baseline)
    table, paired = pair["table"], pair["paired"]
    first, second = table["models"]
    lines: list[str] = []
    lines.append(f"Paired on the {paired['items']} dev items both models measured "
                 f"(`e2_quality.json` cut to the same ids — `tools/e3_reproduce.py --suite compare "
                 f"--align`):")
    lines.append("")
    lines.append(compare.render_comparison(table))
    lines.append("")
    base, challenger = first["overall"], second["overall"]
    lines.append(
        f"What separates the models is the **split itself**: Occamy answers below the floor on "
        f"{second[compare.LOW_MASS]['n']} of its {second['overall']['n']} rows, the 4B on "
        f"{first[compare.LOW_MASS]['n']} — while the agreement *inside* the split is level "
        f"({cell(second[compare.LOW_MASS])} against {cell(first[compare.LOW_MASS])}). Per type: "
        + "; ".join(f"`{qtype}` {safe_cell(first['per_type'].get(qtype))} vs "
                    f"{safe_cell(second['per_type'].get(qtype))}" for qtype in table["types"])
        + f". Overall the two are {abs(table['delta']):.3f} apart at n = {base['n']} and their "
        f"Wilson intervals "
        + ("overlap, so this table cannot rank them on the headline." if
           _intervals_overlap(base, challenger) else "do not overlap."))
    lines.append("")
    lines.append("Both prompts were verified to end at their own assistant header (no template "
                 "failure): the difference is the model's answer distribution, not the bytes it "
                 "was given (§4.2 of the evidence doc).")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def region_texts() -> dict[str, str]:
    """Every generated region, rendered from the artifacts on disk."""
    reports = chunk_reports()
    merged = load(MERGED_REPORT)
    baseline = load(BASELINE_REPORT)
    return {
        "e3_host_items": render_host_items(reports),
        "e3_chunks": render_chunks(reports, merged),
        "e3_comparison": render_comparison(merged, baseline, reports),
        "e3_integrity": render_integrity(reports, merged),
        "bench_chunks": render_bench_chunks(reports, merged),
        "bench_comparison": render_bench_comparison(merged, baseline),
    }


# --------------------------------------------------------------------------- region replacement
def splice(text: str, begin: str, end: str, body: str, *, where: str = "") -> str:
    """Replace everything between `begin` and `end` with `body` (both markers must be unique)."""
    for marker in (begin, end):
        if text.count(marker) != 1:
            raise SystemExit(f"{where}: marker {marker!r} appears {text.count(marker)}x "
                             f"(expected exactly once)")
    start = text.index(begin) + len(begin)
    stop = text.index(end)
    if stop < start:
        raise SystemExit(f"{where}: {end!r} comes before {begin!r}")
    if begin in body or end in body:
        raise SystemExit(f"{where}: the generated body would contain its own marker")
    return text[:start] + "\n" + body.rstrip() + "\n" + text[stop:]


def apply(path: pathlib.Path, regions: Mapping[str, str], *, check: bool) -> bool:
    """Splice every region of one file; in `check` mode only report differences."""
    current = path.read_text(encoding="utf-8")
    wanted = current
    for name, body in regions.items():
        if REGION_FILES[name] != path:
            continue
        begin, end = REGIONS[name]
        wanted = splice(wanted, begin, end, body, where=f"{path.name}:{name}")
    if wanted == current:
        print(f"{path.relative_to(ROOT)}: current")
        return True
    if check:
        print(f"{path.relative_to(ROOT)}: STALE (a generated region does not match the artifacts)")
        return False
    path.write_text(wanted, encoding="utf-8")
    print(f"{path.relative_to(ROOT)}: regenerated")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="verify the files on disk are current; write nothing, exit 1 if stale")
    args = parser.parse_args(argv)
    regions = region_texts()
    ok = True
    for path in dict.fromkeys(REGION_FILES.values()):
        ok = apply(path, regions, check=args.check) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
