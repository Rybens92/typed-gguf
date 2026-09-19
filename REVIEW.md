# E3 completion review — card `t_6d2e084d` (code-reviewer)

- **Verdict: APPROVE** — the round-1 required fix (F1) landed, is gated, and the gate is RED on the
  pre-fix tree; F2–F4 landed as described. Two ⚪ nits recorded below, neither blocking.
- Reviewed head: `13535a6` (branch `main`, shared tree), round 2 — **execution lens**: every handoff
  claim re-run or recomputed from the raw artifacts, not re-read.
- Round 1 (REQUEST CHANGES, F1) is preserved in git at `9ec0c0e`; the E1a review it replaced at
  `05f3ee4`.
- Environment: podman container (no GPU, HOME=/root). The project `.venv` is a dead host symlink
  here; I ran the suite with a scratch env (`uv run --frozen --extra dev` with
  `UV_PROJECT_ENVIRONMENT=/tmp/e3venv`, pytest 9.1.1). Scripts:
  `/tmp/probe_e3_round2.py`, `/tmp/inspect_shapes.py`, `/tmp/dump_devices.py`; the pre-fix scratch
  tree is `/tmp/e3rev2.XqL50D` (`git archive 9ec0c0e`).

## F1 (required) — fixed, and the fix is gated

- `tools/e3_build_evidence.py:316` now passes `second[compare.MEASURED]`; `_measured_note` raises
  `KeyError` on a block without `n` (`:349`) instead of defaulting to 0 — the exact mechanism that
  published the false count.
- The published §4.1 sentence (evidence doc line 290) now reads `…0.667 (2/3) [0.208–0.939], which
  is only 3 item(s).` I recomputed the `measured` block from the merged report with my own code:
  Occamy measured = **2/3 (0.667)**, so 3 is the block's own row count. ✔
- **RED reproduced independently**: on a scratch tree at `9ec0c0e` with only the new
  `tests/test_e3.py` (30 tests) swapped in, the new gate fails exactly as claimed —
  `assert 0 == 3` on a sentence reading `which is only 0 item(s)` (`1 failed, 29 deselected`). On
  `13535a6` the same gate passes.
- The new gate pins both the rendered and the *published* sentence to `models[1]["measured"]["n"]`
  (`tests/test_e3.py:330`), so the generator↔file comparison can no longer mask this class. ✔

## F2–F4 (optional round-1 notes) — landed

- F2: `safe_cell()` (`:143`) renders `—` for an absent/zero block and is used by **both** renderers
  (`:308` §4.1, `:437–438` BENCHMARKS §6.4); its gate passes. ✔
- F3: §4.4 now states the placement distinction precisely, and the source agrees — report-level
  `placement` is written only at `suites.py:235` (`_run_latency`); `_throughput_row` (`:569`) and
  `_determinism_row` (`:829`) write a per-row `placement` *string*. The merged envelope's top-level
  `placement` is chunk 001's alone (`used = {n_gpu_layers: 0, degraded: true, attempts: 7→oom,
  3→oom…}`, byte-equal to `report_001.json`'s block) while its 60 rows mix boxes. ✔
- F4: `report_box()`'s docstring documents the heuristic as one-directional. ✔

## Execution evidence (ran on `13535a6`)

- `python3 tools/e3_build_evidence.py --check` → both files `current`, exit 0. The reviewed diff
  equals the generator's output for every marked region (§2.3/§4.1/§4.3, BENCHMARKS §6.2/§6.4).
- `tests/test_e3.py` → **30 passed**; `ruff check tools/e3_build_evidence.py tests/test_e3.py` and
  `ruff check src tests tools` → `All checks passed!`.
- Full suite → **1121 passed, 44 skipped, 0 failed** (1165 collected) in this container. The handoff's
  host number (1122/43/0) reconciles exactly: of the 44 skips, 43 are `network`/`model`-marked (the
  conftest turns those into skips without `--run-network` on any box, `tests/conftest.py:70–75`); the
  one unmarked, environment-conditional skip is
  `test_runtime_contract.py::test_oracle_live_section_is_green_without_skips` — it runs on the host
  (installed runtime) and skips here (`_installed_runtime()` → None). Same tree, zero failures both
  sides.
- Fix-commit shape: 3 files, +87/−11; the evidence-doc delta is the one regenerated line plus the
  §4.4 prose paragraph (F3); `docs/BENCHMARKS.md` is a 0-byte diff between `9ec0c0e` and `13535a6`.
- Pin / requirement 4: re-hashed the 24 113 674 848 B model **in this container today** →
  `633ae57f…d757` = `e3c_sha256_receipt.json` before/after = E3's pin; mtime `2026-09-18 09:18`
  predates the run; receipt `before` 15:55:18, first chunk start 15:55:21; no
  download/curl/wget/pip/huggingface line in `e3c_host_run.log`; walls 545/492/524/477 s = the
  claimed 34 min, exit 0 per chunk.

## Independent recomputation (raw rows, own Wilson)

- Mass split: Occamy below the 0.10 floor on 57/60 (agreement 29/57 = 0.509), the 4B on 12 (6/12 =
  0.500); `measured` 32/48 vs 2/3; Wilson [0.383–0.634] / [0.254–0.746] / [0.525–0.783] /
  [0.208–0.939] — every cell matches the published sentence.
- Host wall per chunk 49.2 / 42.7 / 47.0 / 44.6 s recomputed from `wall_ms` → matches §2.3 exactly.
- Tags derived from the reports' own facts: 001/002 = 8 GiB cgroup + placement; 003–006 = no cgroup
  keys + `device_buffers {Vulkan0: 10, Vulkan_Host: 10}`, `effective_backend: vulkan`. ✔
- §4.4's "reduction predates this card" holds: `e2_quality.json` lacks every modern key
  (`budget`/`wall_ms`/`truncated`/`skipped`/`quick`/`devices`/`device_buffers`/`effective_backend`).

## Nits (⚪ — recorded, not blocking)

- **N1** The generated §2.3 sentence says "``chunks`` lists every chunk it stitched: `report_001.json`,
  …" — the merged report's `chunks` field entries carry `{items, model, reproduce}`, not file names
  (`compare.merge_reports`, `compare.py:111`). The mapping is 1:1 and order-true (my concatenation
  check holds element for element; 003–006 name their devset in `reproduce`), so the claim reads
  true, but a JSON-only reader cannot map entry→file from the field alone. A `path`-bearing entry
  (or "the campaign's six reports, in merge order") would make it exact.
- **N2** §4.4's parenthetical "`e2_quality.json` … has the same reduced shape" is loose: the baseline
  is *more* reduced than 001/002 — it lacks `placement` **and** `backend_selection` too. The
  paragraph's thesis (the reduction predates this card) stands.
- Interpretive note (not a finding): §4.2/§4.4 are prose and sit outside the marked regions by
  design; the card's requirement-2 concern (numbers that move with `n` must be generated) is fully
  covered by the six regions. The §4.4 edit in `13535a6` is prose maintenance, and its single
  machine fact ("degraded to 0 layers") recomputes true.

## Verdict

APPROVE. The reviewed change is exactly the round-1 fix set, every prior finding landed, the new gate
fails on the pre-fix artifact and passes on this one, the suite is green, and the pin is unchanged.
N1/N2 are wording notes for a future doc pass, not blockers.
