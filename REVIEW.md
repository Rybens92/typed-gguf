# E3 completion review — card `t_6d2e084d` (code-reviewer)

- **Verdict: REQUEST CHANGES** — one required correction (F1); everything else I checked reproduces.
- Reviewed head: `8c9e3e9` (branch `main`, shared tree), round 1 — artifact lens (cold diff read first).
- File note: this file previously held the E1a final review (`t_ee24bd7c`); that record stays in git at
  `05f3ee4` and in the E1a board thread.
- Environment: podman container (no GPU); the repo, all six chunk reports and the 24 GB model file are
  reachable. Gates run here: `uv run --frozen --extra dev pytest -q`, `ruff` (project config),
  `python3 tools/e3_build_evidence.py --check`, `sha256sum` on the model, and my own recomputation
  scripts (`/tmp/e3review/verify1.py`, `verify2.py`, `provenance_ast.py`).

## What I verified independently (not from the handoff)

**Requirement 1 — the merge covers every chunk that ran.**
- `e3_occamy_quality.json`: `ok: true`, 60 rows, ids unique, `chunks` = 6 × 10; my deep compare says
  `merged["items"] == concatenation(report_001..006["items"])` — true, element for element.
- every report on disk `ok: true` with 10/10 rows; no gap anywhere (the OOM lines in the logs are the
  degrade ladder, no lost row).
- `devset_001..006.jsonl` ids equal the committed 60-item dev set exactly (0 duplicates, 0 missing),
  and each chunk's devset ids equal its report's ids in order.

**Requirement 2 — the sections are generated, not hand-edited.**
- `tools/e3_build_evidence.py --check` → `current` for both files, exit 0 (I ran it myself).
- `tests/test_e3.py` → 28 passed, including the three new gates; the new gate re-renders every marked
  region and compares it to disk.
- The commit touches only `docs/*`, `tools/e3_build_evidence.py`, `tests/test_e3.py` — no production
  line. BENCHMARKS diff hunks are confined to §6.2/§6.4; the evidence-doc diff only replaces the old
  hand-written n=20 prose with the generated n=60 prose (nothing of substance dropped).

**Requirement 3 — the `[host]` statement and the tags.**
- §2.3/§6.2 state the `[host]` run happened (4 of 6 chunks, `report_003..006`) and what it changed
  (n and the intervals); the per-item *cost* table keeps its `[container]` lead-in.
- The tags match the reports' own facts, read directly: 001/002 carry
  `cgroup_memory_bytes=8589934592`/`cgroup_cpu_max=2.0` and a `placement` block, no `devices`;
  003..006 carry no cgroup keys and `devices`/`device_buffers {Vulkan0: 10, Vulkan_Host: 10}`/
  `effective_backend: vulkan`.

**Requirement 4 — pin, no downloads, no mutation.**
- I re-hashed `/var/home/rybens/.hermes/models/Accio-Lab_occamy-1.0-Q4_K_L.gguf` **in this container
  today**: `633ae57faf731e863cc3ba7cb75396a1b1e377191730e7b0d7294eff55cdf757` — identical to
  `e3c_sha256_receipt.json` (`identical: true`), to both receipts' before/after and to E3's pin; mtime
  `2026-09-18 09:18` predates the run. The host log records only local paths (no download).
- Tier M / mutation: no new production surface in this diff; the mutmut pair is owned by the sibling
  E3b sweep — consistent with the repo's own convention (pyproject comments; E1a precedent).

**Numbers — recomputed from the raw rows, not from the renderer.**
- An independent Wilson implementation reproduces every published cell: 4B 0.633 (38/60) [0.507–0.744],
  Occamy 0.517 (31/60) [0.393–0.638], delta −0.117, choice 0.750/0.625, noul 0.889/0.389, score
  0.222/0.500, low_mass 0.500 (6/12)/0.509 (29/57), measured 0.667 (32/48)/0.667 (2/3); `--align`
  drops 0/0; the intervals overlap.
- Chunk facts: per-chunk medians 113.2/99.5/49.2/42.7/47.0/44.6 s — the published "42.7–49.2 s/item
  against 99–113 s in the container" holds; the log's own boundaries (545/492/524/477 s, start
  15:55:21 CEST, done 16:29:19) match `e3c_host_run.log`; `report_003`'s first two rows match the §2.3
  table (`n07` 23,408 ms / 46.9 s / ✘(`no`) / 0.037; `c08` 28,577 ms / 49.1 s / ✔(`business_hours`) /
  0.052); `report_002`'s placement block says exactly what §2.3 prints.
- §4.4's core provenance claim is true: an AST walk over **all 13 revisions** of
  `src/ggufone/bench/suites.py` shows a report-level `placement` block is never written by
  `_run_quality` (only `_run_latency`, plus per-row fields in throughput/determinism).

## Suite state

`uv run --frozen --extra dev pytest -q` → `1 failed, 1116 passed, 44 skipped`; the one failure is
`tests/test_e3b_labels.py::test_a_table_with_any_sub_5e_05_value_prints_in_scientific_notation`, which
lives in the **27 uncommitted lines of the sibling card `t_6952f0dd`** in this shared tree — not in
this card's diff. With that file ignored: `1073 passed, 44 skipped, 0 failed` (total 1117 = the
implementer's 1074+43; one test moves between pass/skip on this box). `ruff` (project config) is clean
on both changed code files; the 213 repo-wide ruff hits all sit in untracked scratch trees
(`.e3b/mutants_old_1446`, `.e2e/…`, `mutants-e1c-backup`, `.e3c_scratch`).
(One earlier full-suite run showed 24 transient failures in the probe/runtime files; two consecutive
re-runs since are green apart from the sibling test. Container flake, not attributable to this diff.)

## Findings

### F1 — 🟠 MAJOR (required before this card can close)
`tools/e3_build_evidence.py:300` calls `_measured_note(second)` with the whole challenger model row,
so `block.get("n")` is always `0` and the **generated, published** §4.1 sentence reads:

> …against 0.667 (2/3) [0.208–0.939], **which is only 0 item(s)**.

The measured block is 3 rows (verified from the merged report), and the helper's own docstring says it
wants "the note a three-row `measured` block needs". The existing gates cannot catch this class: they
compare the generator's output to the files, and the file faithfully contains the generator's wrong
number. Minimum outcome: pass the measured block (`second[compare.MEASURED]`), regenerate the evidence
doc, keep `--check` green, and add a gate that pins the note's number to the actual measured-block size.

### F2 — 🟡 MINOR (implementer's call; cheap while the file is open)
`render_bench_comparison` (~`tools/e3_build_evidence.py:413`) renders
`cell(first["per_type"].get(qtype) or {})`; `_cell` reads `block["n"]`, so `{}` raises `KeyError` the
day one side lacks a whole question type. `render_comparison` already guards for that case. Not
reachable with today's six chunks.

### F3 — ⚪ NIT
§4.4's "the key exists only in the latency suite's report": true for the report-level block, but a
`placement` *field* also exists in the throughput/determinism rows (`row["placement"] = "n_gpu_layers=…"`).
Also, the merged report's top-level `placement` is chunk 001's alone (container, 0 layers) while its 60
rows now mix boxes — §4.3 documents this; half a sentence in §4.4 would spare a JSON-only reader.

### F4 — ⚪ NIT
`report_box()` decides `[host]` by "no cgroup keys". Correct for all six reports here (read from the
actual keys), but the tag is load-bearing for a published claim; a comment on the heuristic (or a
report field the tool controls) would make a future unconstrained-container run fail loudly instead
of silently tagging as `[host]`.

## Verdict

Approve is not available with F1 outstanding: the card's point is that published numbers come from the
artifacts, and the new generator prints one that does not. F2–F4 are optional notes. Everything else —
merge coverage, generated sections, the tags, the pin, the recomputed table, the suite — reproduces.
