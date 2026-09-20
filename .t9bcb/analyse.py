#!/usr/bin/env python3
"""t_9bcbecff — the [host] E3e probe's own statistics, read out of the stored reports.

The card's rule: **no number is retyped**. This script is the only thing that reads the campaign,
writes `.t9bcb/stats.json`, and `render_doc.py` renders the evidence document FROM that file.

Inputs (no model is loaded, no socket is opened):

* the campaign's per-chunk reports + placement sinks (`.t9bcb/run_chunks.sh` →
  `docs/evidence/t9bcbecff_tiel_chunks/`);
* the merged challenger report (`tools/e3_reproduce.py --suite merge`);
* the corrected-instrument baseline of card `t_7c926398` —
  `docs/evidence/tiel_corrected_quality.json`, plus its own per-chunk reports and
  placement sinks (`docs/evidence/tiel_corrected_chunks/`);
* the run logs (the scope's `memory.max`, the model SHA pair, the per-chunk walls, the
  compute-path line);
* the two 6-item smokes of the card's falsification order.

Statistics are the committed ones — `tools/e3e_roles_decision.py::cell_stats/pair_stats` (Wilson
interval, the exact two-sided McNemar test, the closed-form Wald interval on the discordant counts)
and `ggufone.bench.harness.percentile` — never a second implementation of the same interval.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys
import time
from collections import Counter
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import e3e_roles_decision as e3e  # noqa: E402  (the committed E3e statistics)

from ggufone.bench import harness  # noqa: E402

CHALL = ROOT / "docs/evidence/t9bcbecff_tiel_chunks"
CHALL_MERGED = ROOT / "docs/evidence/t9bcbecff_tiel_challenger_quality.json"
BASE = ROOT / "docs/evidence/tiel_corrected_chunks"
BASE_MERGED = ROOT / "docs/evidence/tiel_corrected_quality.json"
OUT = ROOT / ".t9bcb" / "stats.json"
CAMPAIGN_LOG = ROOT / ".t9bcb/logs/campaign.log"
SHA_LINE = re.compile(r"^([0-9a-f]{64})\s+(\S+)\s*$")
CHUNK_LINE = re.compile(r"^=== chunk (\d+) exit=(\d+) wall=(\d+)s")
MEMORY_LINE = re.compile(r"memory\.max=(\S+)")
COMPUTE_LINE = re.compile(r"compute buffer size is")
CHUNKS = ("001", "002", "003", "004", "005", "006")


def load(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rows_of(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(row) for row in report.get("items") or []]


def coverage_quantiles(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(row.get("coverage") or 0.0) for row in rows]
    if not values:
        return {"n": 0}
    return {"n": len(values), "min": min(values), "p25": harness.percentile(values, 25),
            "p50": harness.percentile(values, 50), "p75": harness.percentile(values, 75),
            "max": max(values)}


def framing_facts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The framing surface **per row** — the report-level block is the first chunk's only."""
    labels = sorted({harness.framing_label(row.get("framing") or {}) for row in rows})
    prefixes = sorted(int(row.get("prefix_tokens") or 0) for row in rows)
    surfaces = Counter(json.dumps({key: (row.get("framing") or {}).get(key)
                                   for key in ("kind", "renderer", "source", "family", "thinking")},
                                  sort_keys=True) for row in rows)
    return {"labels": labels, "mixed": len(labels) > 1,
            "prefix_tokens_min": min(prefixes, default=None),
            "prefix_tokens_max": max(prefixes, default=None),
            "prefix_tokens_all": prefixes,
            "surfaces": {key: value for key, value in sorted(surfaces.items())}}


def closer_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    refused = [row for row in rows if (row.get("cue") or {}).get("refused")]
    counter = Counter(str((row.get("cue") or {}).get("closer")) for row in refused)
    return {"refused": len(refused), "refused_n": len(rows),
            "closers": {key: value for key, value in sorted(counter.items())}}


def decision_ms(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(row.get("questions_ms") or 0.0) / 1000.0 for row in rows]
    if not values:
        return {"n": 0}
    return {"n": len(values), "median_s": harness.percentile(values, 50),
            "min_s": min(values), "max_s": max(values)}


def arm_cell(report: dict[str, Any], label: str) -> dict[str, Any]:
    """One arm: the committed `cell_stats` plus the facts the card's report line asks for."""
    rows = rows_of(report)
    cell = e3e.cell_stats(report)
    cell["label"] = label
    cell["kind"] = cell.get("kind")
    cell["coverage_quantiles"] = coverage_quantiles(rows)
    cell["framing_rows"] = framing_facts(rows)
    cell["closers"] = closer_counts(rows)
    cell["decisions"] = decision_ms(rows)
    cell["measured"] = sum(1 for row in rows if row.get("reliability") != "low_mass")
    cell["effective_backend"] = report.get("effective_backend")
    cell["device_buffers"] = report.get("device_buffers")
    cell["devices"] = report.get("devices")
    cell["mismatch_warnings"] = sorted({str(w).split(":")[0] for w in (report.get("warnings") or [])
                                        if "BACKEND" in str(w)})
    cell["ok"] = bool(report.get("ok"))
    cell["reproduce"] = (report.get("commands") or {}).get("reproduce")
    return cell


def paired_items(baseline: dict[str, Any], challenger: dict[str, Any]) -> dict[str, Any]:
    """The item-by-item split behind the paired counts (what moved, and to what)."""
    base_rows = {str(row["id"]): row for row in rows_of(baseline)}
    chall_rows = {str(row["id"]): row for row in rows_of(challenger)}
    shared = sorted(set(base_rows) & set(chall_rows))
    won: list[dict[str, Any]] = []
    lost: list[dict[str, Any]] = []
    for key in shared:
        base, chall = base_rows[key], chall_rows[key]
        entry = {"id": key, "type": base.get("type"), "expected": base.get("expected"),
                 "baseline_got": base.get("got"), "challenger_got": chall.get("got"),
                 "baseline_coverage": float(base.get("coverage") or 0.0),
                 "challenger_coverage": float(chall.get("coverage") or 0.0),
                 "baseline_reliability": base.get("reliability"),
                 "challenger_reliability": chall.get("reliability")}
        if chall.get("correct") and not base.get("correct"):
            won.append(entry)
        elif base.get("correct") and not chall.get("correct"):
            lost.append(entry)
    return {"n_shared": len(shared), "challenger_only": won, "baseline_only": lost,
            "unpaired_baseline": sorted(set(base_rows) - set(chall_rows)),
            "unpaired_challenger": sorted(set(chall_rows) - set(base_rows))}


def per_type_paired(baseline: dict[str, Any], challenger: dict[str, Any]) -> dict[str, Any]:
    base_rows = {str(row["id"]): row for row in rows_of(baseline)}
    chall_rows = {str(row["id"]): row for row in rows_of(challenger)}
    out: dict[str, Any] = {}
    for qtype in sorted({str(row.get("type")) for row in rows_of(baseline)}):
        ids = [key for key in sorted(set(base_rows) & set(chall_rows))
               if base_rows[key].get("type") == qtype]
        b = sum(1 for key in ids if chall_rows[key].get("correct") and not
                base_rows[key].get("correct"))
        c = sum(1 for key in ids if base_rows[key].get("correct") and not
                chall_rows[key].get("correct"))
        base_correct = sum(1 for key in ids if base_rows[key].get("correct"))
        chall_correct = sum(1 for key in ids if chall_rows[key].get("correct"))
        out[qtype] = {"n": len(ids), "baseline_correct": base_correct,
                      "challenger_correct": chall_correct,
                      "baseline_agreement": base_correct / len(ids) if ids else None,
                      "challenger_agreement": chall_correct / len(ids) if ids else None,
                      "challenger_only": b, "baseline_only": c,
                      "difference": (b - c) / len(ids) if ids else None,
                      "mcnemar_p": e3e.mcnemar_exact(b, c)}
    return out


def chunk_facts(directory: pathlib.Path, logs: pathlib.Path, *, prefix: str = "report_",
                sink_prefix: str = "placement_", log_prefix: str = "run_") -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for name in CHUNKS:
        report_path = directory / f"{prefix}{name}.json"
        sink_path = directory / f"{sink_prefix}{name}.json"
        log_path = logs / f"{log_prefix}{name}.log"
        report = load(report_path)
        sink = load(sink_path) if sink_path.exists() else {}
        placement = dict(sink.get("placement") or {})
        rows = rows_of(report)
        fact: dict[str, Any] = {
            "chunk": name,
            "report": str(report_path.relative_to(ROOT)),
            "sink": str(sink_path.relative_to(ROOT)),
            "items": len(rows),
            "correct": sum(1 for row in rows if row.get("correct")),
            "types": dict(sorted(Counter(str(row.get("type")) for row in rows).items())),
            "low_mass": sum(1 for row in rows if row.get("reliability") == "low_mass"),
            "refusals": sum(1 for row in rows if (row.get("cue") or {}).get("refused")),
            "prefix_tokens": framing_facts(rows)["prefix_tokens_all"],
            "framing_labels": framing_facts(rows)["labels"],
            "median_decision_s": decision_ms(rows)["median_s"],
            "effective_backend": report.get("effective_backend"),
            "device_buffers": report.get("device_buffers"),
            "mismatch_warnings": sorted({str(w).split(":")[0]
                                         for w in (report.get("warnings") or [])
                                         if "BACKEND" in str(w)}),
            "ok": bool(report.get("ok")),
            "ngl_requested": sink.get("gpu_layers_requested"),
            "ngl_used": placement.get("n_gpu_layers"),
            "degraded": placement.get("degraded"),
            "kv_type_used": sink.get("kv_type_used"),
            "n_ctx": sink.get("n_ctx"),
            "n_prefix": sink.get("n_prefix"),
            "n_seq_max": sink.get("n_seq_max"),
            "load_wall_s": sink.get("load_wall_s"),
            "chunk_wall_s": sink.get("wall_s"),
            "cue": sink.get("cue"),
            "chat_format": sink.get("chat_format"),
            "json_contract": sink.get("json_contract"),
            "chat_format_seen": sink.get("chat_format_seen"),
            "template_seen": sink.get("template_seen"),
        }
        if log_path.exists():
            text = log_path.read_text(encoding="utf-8", errors="replace")
            lines = [line.strip() for line in text.splitlines() if COMPUTE_LINE.search(line)]
            fact["compute_lines"] = len(lines)
            fact["compute_first"] = lines[0] if lines else None
            fact["compute_devices"] = dict(sorted(Counter(
                (re.split(r"\s+", line)[1] if len(re.split(r"\s+", line)) > 1 else "?")
                for line in lines).items()))
        facts.append(fact)
    return facts


def log_facts() -> dict[str, Any]:
    text = CAMPAIGN_LOG.read_text(encoding="utf-8", errors="replace")
    shas = [(match.group(1), match.group(2)) for match in
            (SHA_LINE.match(line) for line in text.splitlines()) if match]
    walls = {match.group(1): {"exit": int(match.group(2)), "wall_s": int(match.group(3))}
             for match in (CHUNK_LINE.match(line) for line in text.splitlines()) if match}
    memory = MEMORY_LINE.search(text)
    return {"memory_max": memory.group(1) if memory else None,
            "sha_before": shas[0][0] if shas else None,
            "sha_after": shas[-1][0] if shas else None,
            "sha_identical": bool(shas) and len({sha for sha, _ in shas}) == 1,
            "sha_lines": len(shas),
            "chunk_walls": walls,
            "path": str(CAMPAIGN_LOG.relative_to(ROOT))}


def devset_facts() -> dict[str, Any]:
    mine = (ROOT / "docs/evidence/t9bcbecff_tiel_chunks/devset_sha.txt").read_text(encoding="utf-8")
    theirs = (ROOT / ".t7c9/devset_sha.txt").read_text(encoding="utf-8")
    mine_lines = [line.split() for line in mine.splitlines() if line.strip()]
    their_lines = [line.split() for line in theirs.splitlines() if line.strip()]
    mine_digests = {line[0] for line in mine_lines}
    their_digests = {line[0] for line in their_lines}
    return {"campaign": [line[0] for line in mine_lines],
            "baseline": [line[0] for line in their_lines],
            "identical": mine_digests == their_digests,
            "files": len(mine_lines),
            "receipt": "docs/evidence/t9bcbecff_tiel_chunks/devset_sha.txt"}


def smoke_facts() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("collapse", "challenger"):
        report_path = ROOT / f".t9bcb/smoke_{name}.json"
        sink_path = ROOT / f".t9bcb/smoke_{name}_placement.json"
        if not report_path.exists():
            continue
        report = load(report_path)
        sink = load(sink_path) if sink_path.exists() else {}
        rows = rows_of(report)
        out[name] = {"cell": e3e.cell_label(report), "items": len(rows),
                     "correct": sum(1 for row in rows if row.get("correct")),
                     "refusals": sum(1 for row in rows if (row.get("cue") or {}).get("refused")),
                     "low_mass": sum(1 for row in rows if row.get("reliability") == "low_mass"),
                     "closers": closer_counts(rows),
                     "rows": [{"id": row.get("id"), "type": row.get("type"),
                               "expected": row.get("expected"), "got": row.get("got"),
                               "correct": bool(row.get("correct")),
                               "coverage": float(row.get("coverage") or 0.0),
                               "reliability": row.get("reliability"),
                               "cue_top": ((row.get("cue") or {}).get("closer")
                                           or f"token {(row.get('cue') or {}).get('token')}"),
                               "cue_mass": (row.get("cue") or {}).get("mass"),
                               "cue_refused": bool((row.get("cue") or {}).get("refused"))}
                              for row in rows],
                     "chat_format_seen": sink.get("chat_format_seen"),
                     "template_seen": sink.get("template_seen"),
                     "effective_backend": report.get("effective_backend"),
                     "device_buffers": report.get("device_buffers")}
    return out


def decision_identity(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """How much of two arms' *decisions* is the same, item by item (the freeze-probe vocabulary).

    Two arms with the same agreement can hide compensating flips, so an "inert" claim has to be
    checked per item: this counts the rows whose winner, correctness, reliability and prefix tokens
    are identical, and reports the largest coverage difference seen.
    """
    left_rows = {str(row["id"]): row for row in rows_of(left)}
    right_rows = {str(row["id"]): row for row in rows_of(right)}
    shared = sorted(set(left_rows) & set(right_rows))
    same_decision = sum(1 for key in shared
                        if left_rows[key].get("got") == right_rows[key].get("got")
                        and bool(left_rows[key].get("correct"))
                        == bool(right_rows[key].get("correct")))
    same_prefix = sum(1 for key in shared
                      if left_rows[key].get("prefix_tokens")
                      == right_rows[key].get("prefix_tokens"))
    deltas = [abs(float(left_rows[key].get("coverage") or 0.0)
                  - float(right_rows[key].get("coverage") or 0.0)) for key in shared]
    flips = [{"id": key, "left_got": left_rows[key].get("got"),
              "right_got": right_rows[key].get("got"),
              "left_correct": bool(left_rows[key].get("correct")),
              "right_correct": bool(right_rows[key].get("correct"))}
             for key in shared
             if left_rows[key].get("got") != right_rows[key].get("got")]
    return {"n": len(shared), "decisions_identical": same_decision,
            "prefix_tokens_identical": same_prefix,
            "max_abs_coverage_delta": max(deltas, default=None),
            "winner_flips": flips}


def smoke_vs_arm(smoke: dict[str, Any], arm: dict[str, Any]) -> dict[str, Any]:
    """The 6-item smoke's rows against the same items in the 60-item auxiliary arm.

    Same cell, same items, same instrument — measured twice (6-item run and 10-item chunk). Under a
    readout whose candidate mass is a tail the winner can be numerically fragile, so this is the
    receipt for "is the agreement of this cell readable at all", not a reproducibility check of the
    challenger (whose rows carry ~1.0 of their mass).
    """
    arm_rows = {str(row["id"]): row for row in rows_of(arm)}
    rows = []
    flips = 0
    for row in smoke.get("rows") or []:
        other = arm_rows.get(str(row["id"]))
        if other is None:
            continue
        same = (other.get("got") == row["got"]
                and bool(other.get("correct")) == bool(row["correct"]))
        flips += 0 if same else 1
        rows.append({"id": row["id"], "smoke_got": row["got"], "arm_got": other.get("got"),
                     "smoke_correct": row["correct"], "arm_correct": bool(other.get("correct")),
                     "smoke_coverage": row["coverage"],
                     "arm_coverage": float(other.get("coverage") or 0.0), "same": same})
    return {"n": len(rows), "flips": flips, "rows": rows}


def gates_facts() -> dict[str, Any]:
    """The gate receipts as the document must quote them (parsed, never retyped).

    `.t9bcb/gates.sh` writes the oracle output, the suite output and the gate run's stdout; this
    reads the summary lines out of those files instead of a human copying them into prose.
    """
    facts: dict[str, Any] = {}
    oracle_path = ROOT / ".t9bcb/oracle.txt"
    if oracle_path.exists():
        text = oracle_path.read_text(encoding="utf-8", errors="replace")
        summary = [line.strip() for line in text.splitlines()
                   if "failures" in line and "skip" in line]
        facts["oracle"] = {"receipt": ".t9bcb/oracle.txt",
                           "summary": summary[-1] if summary else None}
    suite_path = ROOT / ".t9bcb/gates.txt"
    if suite_path.exists():
        text = suite_path.read_text(encoding="utf-8", errors="replace")
        summary = [line.strip() for line in reversed(text.splitlines())
                   if re.search(r"\d+ (passed|failed)", line)]
        facts["suite"] = {"receipt": ".t9bcb/gates.txt",
                          "summary": summary[0] if summary else None}
    run_path = ROOT / ".t9bcb/logs/gates_run.log"
    if run_path.exists():
        text = run_path.read_text(encoding="utf-8", errors="replace")
        section = text.rsplit("== gate 2: ruff", 1)
        ruff_lines: list[str] = []
        if len(section) == 2:
            for line in section[1].splitlines()[1:]:      # drop the echo's own remainder
                stripped = line.strip()
                if stripped.startswith("== gate 3"):
                    break
                if stripped:
                    ruff_lines.append(stripped)
        facts["ruff"] = {"receipt": ".t9bcb/logs/gates_run.log",
                         "summary": " · ".join(ruff_lines[:3]) or None}
        exits = re.findall(r"exit=(\d+)", text)
        facts["exits"] = exits
    final_path = ROOT / ".t9bcb/logs/final_suite.txt"
    if final_path.exists():
        text = final_path.read_text(encoding="utf-8", errors="replace")
        summary = [line.strip() for line in reversed(text.splitlines())
                   if re.search(r"\d+ (passed|failed)", line)]
        facts["suite_post_render"] = {"receipt": ".t9bcb/logs/final_suite.txt",
                                      "summary": summary[0] if summary else None}
    return facts


def occamy_facts() -> dict[str, Any]:
    """The optional Occamy pass (the card's "(and Occamy)"): both E3e cells measured here.

    Occamy has no row under the corrected instrument, so this pair is measured *both ways* under
    one instrument and compared with itself — it is deliberately not comparable with the
    container-era E3 row of card `t_a431be85`.
    """
    merged_base = ROOT / "docs/evidence/t9bcbecff_occamy_base_quality.json"
    merged_e3e = ROOT / "docs/evidence/t9bcbecff_occamy_e3e_quality.json"
    if not (merged_base.exists() and merged_e3e.exists()):
        return {}
    base = load(merged_base)
    chall = load(merged_e3e)
    chunks = ROOT / "docs/evidence/t9bcbecff_occamy_chunks"
    log_path = ROOT / ".t9bcb/logs/occamy.log"
    shas: list[str] = []
    if log_path.exists():
        shas = re.findall(r"\b([0-9a-f]{64})\b", log_path.read_text(encoding="utf-8",
                                                                  errors="replace"))
    facts: dict[str, Any] = {
        "cells": {
            "baseline": arm_cell(base, "occamy: shipped placement + shipped cue (measured here)"),
            "challenger": arm_cell(chall, "occamy: role_split + json_instructed (measured here)"),
        },
        "paired": e3e.pair_stats(base, chall),
        "paired_items": paired_items(base, chall),
        "per_type": per_type_paired(base, chall),
        "baseline_chunks": chunk_facts(chunks, ROOT / ".t9bcb/logs",
                                       prefix="occ_base_report_",
                                       sink_prefix="occ_base_placement_",
                                       log_prefix="occ_base_run_"),
        "challenger_chunks": chunk_facts(chunks, ROOT / ".t9bcb/logs",
                                         prefix="occ_e3e_report_",
                                         sink_prefix="occ_e3e_placement_",
                                         log_prefix="occ_e3e_run_"),
        "merged": {"baseline": str(merged_base.relative_to(ROOT)),
                   "challenger": str(merged_e3e.relative_to(ROOT))},
        "chunk_dir": str(chunks.relative_to(ROOT)),
        "log": str(log_path.relative_to(ROOT)),
        "sha_lines": len(shas),
        "sha_identical": len(shas) >= 2 and len(set(shas)) == 1,
        "sha": shas[0] if shas else None,
        "all_ok": bool(base.get("ok")) and bool(chall.get("ok")),
    }
    model_meta = base.get("model") if isinstance(base.get("model"), dict) else {}
    facts["model"] = model_meta.get("path") or base.get("model_path") or None
    if not facts["model"]:
        first = chunks / "occ_base_report_001.json"
        if first.exists():
            facts["model"] = load(first).get("model_path") or None
    facts["model_name"] = pathlib.Path(str(facts["model"])).name if facts["model"] else None
    sha_receipt = ROOT / ".t9bcb/logs/occamy_sha.txt"
    if sha_receipt.exists():
        found = re.search(r"[0-9a-f]{64}",
                          sha_receipt.read_text(encoding="utf-8", errors="replace"))
        facts["sha"] = found.group(0) if found else facts["sha"]
        facts["sha_receipt"] = str(sha_receipt.relative_to(ROOT))
    model_path = pathlib.Path(str(facts["model"])) if facts["model"] else None
    if model_path is not None and model_path.exists():
        stat = model_path.stat()
        facts["model_bytes"] = stat.st_size
        facts["model_mtime"] = time.strftime("%Y-%m-%d %H:%M:%S",
                                             time.localtime(stat.st_mtime))
    e3_env = ROOT / "docs/evidence/e3_environment.json"
    if e3_env.exists() and facts.get("sha"):
        facts["sha_e3_receipt"] = str(e3_env.relative_to(ROOT))
        facts["sha_matches_e3_receipt"] = bool(
            facts["sha"] in e3_env.read_text(encoding="utf-8", errors="replace"))
    if log_path.exists():
        codes = [int(code) for code in
                 re.findall(r"=== chunk \d+ exit=(\d+)",
                            log_path.read_text(encoding="utf-8", errors="replace"))]
        facts["chunk_exits"] = codes
        facts["all_chunks_ok"] = len(codes) == 12 and all(code == 0 for code in codes)
    return facts


def main() -> int:
    challenger = load(CHALL_MERGED)
    baseline = load(BASE_MERGED)
    aux_specs = {
        "role_split_only": {
            "label": "aux: role_split + the shipped cue (the placement alone)",
            "merged": ROOT / "docs/evidence/t9bcbecff_tiel_role_split_quality.json",
            "tag": "aux_role_split_"},
        "two_step_role_split": {
            "label": "aux: role_split + two_step (the 4B's collapse cell)",
            "merged": ROOT / "docs/evidence/t9bcbecff_tiel_two_step_quality.json",
            "tag": "aux_two_step_"},
    }
    aux: dict[str, Any] = {}
    for name, spec in aux_specs.items():
        if not spec["merged"].exists():
            continue
        report = load(spec["merged"])
        aux[name] = {
            "label": spec["label"],
            "cell": arm_cell(report, spec["label"]),
            "vs_baseline": e3e.pair_stats(baseline, report),
            "vs_challenger": e3e.pair_stats(report, challenger),
            "per_type": per_type_paired(baseline, report),
            "chunks": chunk_facts(CHALL, ROOT / ".t9bcb/logs",
                                  prefix=f"{spec['tag']}report_",
                                  sink_prefix=f"{spec['tag']}placement_",
                                  log_prefix=f"{spec['tag']}run_"),
            "merged": str(spec["merged"].relative_to(ROOT)),
        }
    stats: dict[str, Any] = {
        "task": "t_9bcbecff",
        "generated_from": {
            "challenger": str(CHALL_MERGED.relative_to(ROOT)),
            "baseline": str(BASE_MERGED.relative_to(ROOT)),
            "challenger_chunks": [str((CHALL / f"report_{n}.json").relative_to(ROOT))
                                  for n in CHUNKS],
            "baseline_chunks": [str((BASE / f"report_{n}.json").relative_to(ROOT))
                                for n in CHUNKS],
            "campaign_log": str(CAMPAIGN_LOG.relative_to(ROOT)),
        },
        "challenger": arm_cell(challenger, "challenger: role_split + json_instructed"),
        "baseline": arm_cell(baseline, "baseline: corrected instrument, shipped cue"),
        "paired": e3e.pair_stats(baseline, challenger),
        "paired_items": paired_items(baseline, challenger),
        "per_type_paired": per_type_paired(baseline, challenger),
        "challenger_chunks": chunk_facts(CHALL, ROOT / ".t9bcb/logs"),
        "baseline_chunks": chunk_facts(BASE, ROOT / ".t7c9/logs"),
        "log": log_facts(),
        "devset": devset_facts(),
        "smokes": smoke_facts(),
        "aux": aux,
        "gates": gates_facts(),
        "occamy": occamy_facts(),
    }
    smokes = stats["smokes"]
    if "collapse" in smokes and "two_step_role_split" in aux:
        stats["smoke_vs_arm"] = {
            "collapse_vs_two_step_arm": smoke_vs_arm(
                smokes["collapse"],
                load(pathlib.Path(ROOT / aux["two_step_role_split"]["merged"]))),
            "challenger_vs_challenger_arm": smoke_vs_arm(
                smokes.get("challenger", {}),
                challenger),
        }
    else:
        stats["smoke_vs_arm"] = {}
    stats["paired"]["rule"] = e3e.UNIT if hasattr(e3e, "UNIT") else "paired"
    if "role_split_only" in aux and "two_step_role_split" in aux:
        aux_reports = {
            name: load(pathlib.Path(ROOT / arm["merged"])) for name, arm in aux.items()}
        stats["aux_identity"] = {
            "role_split_only_vs_two_step": decision_identity(
                aux_reports["role_split_only"], aux_reports["two_step_role_split"]),
            "challenger_vs_role_split_only": decision_identity(
                challenger, aux_reports["role_split_only"]),
        }
    else:
        stats["aux_identity"] = {}
    OUT.write_text(json.dumps(stats, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "written": str(OUT.relative_to(ROOT)),
        "challenger": {key: stats["challenger"][key]
                       for key in ("label", "correct", "items", "agreement", "ci", "low_mass",
                                   "measured", "refusals", "verdicts", "prefix_tokens")},
        "baseline": {key: stats["baseline"][key]
                     for key in ("label", "correct", "items", "agreement", "ci", "low_mass",
                                 "measured", "refusals", "verdicts", "prefix_tokens")},
        "paired": {key: stats["paired"][key]
                   for key in ("baseline", "challenger", "n", "both_correct", "neither_correct",
                               "baseline_only", "challenger_only", "difference", "ci",
                               "mcnemar_p", "challenger_wins")},
        "devset_identical": stats["devset"]["identical"],
        "sha_identical": stats["log"]["sha_identical"],
        "memory_max": stats["log"]["memory_max"],
        "aux": {name: {"label": arm["label"], "correct": arm["cell"]["correct"],
                       "items": arm["cell"]["items"], "agreement": arm["cell"]["agreement"],
                       "low_mass": arm["cell"]["low_mass"], "refusals": arm["cell"]["refusals"],
                       "difference": arm["vs_baseline"]["difference"],
                       "mcnemar_p": arm["vs_baseline"]["mcnemar_p"]}
                for name, arm in stats["aux"].items()},
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
