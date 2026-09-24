# E3 — Occamy 1.0 (35B-A3B, 23 GB) runs + comparison (card t_a431be85) — QA Report

Date: 2026-09-18 · Tier **M** (the card declares none) · Decision: **ship** — every card gate has
measured evidence; the one honest scope reduction (20 of 60 dev items for the comparison, because
the container cannot cache a 23 GB model) is stated in every place the table appears.

Commits: `76602e3` (evidence: pin, environment, placement, chunks, batch, comparison, tools, gates),
`15e89a9` (coverage 100 % on the new module + the Tier-M mutation sweep).

## Summary — risk-weighted

🔴 REQUIRES ATTENTION — **none open.**

🟡 WORTH CONSIDERING (4, all recorded, none blocks)

* **A-E3-3 was measured on 20 of the 60 dev items** (`docs/evidence/e3_occamy_quality.json`:
  `chunks: [10, 10]`; the table prints `n = 20`). Reason, measured: the worker container has
  `memory.max = 8 GiB` against 23 GB of weights, so one dev item costs ~2.5–4.5 min (a full weight
  sweep per prefill) — a 60-item pass is ~3–4.5 h *in this container*, and the box was shared with
  a sibling bench. **Cost to finish: ~40 min per 10-item chunk on a quiet box, or ~10 min per chunk
  on the host** (`tools/e3_reproduce.py --suite quality --devset .e3/chunks/devset_00N.jsonl` for
  N=3..6, then `--suite merge` + `--suite compare --align`; the published rows are untouched, `n`
  grows). Recommendation: **defer to the follow-up** unless the reviewer wants the fuller table now
  — the paired baseline (`--align`) makes the 20-item table valid, just wide (CI ±0.20).
* **The serving path's `engine.backend` still reads `cpu` while the Vulkan device computed** — the
  batch response says `backend: cpu` and its own log line says `Vulkan0 compute buffer size is
  363.5 MiB`. This is the label class the coordinator's audit opened (`t_603a35a0` fixed the bench
  path only). Risk: a reader of `ask`/`run` output can be misled about which device computed.
  Cost to fix: a serving-path attribution field, same pattern as the bench's `device_usage`.
  Recommendation: **a FIX card, not this one** (E3 is measurements + docs; the finding is in §3 of
  the evidence doc, with the log line as its proof).
* **Tier M mutation: 64.7 % (251 killed / 388 scored, 0 not-run) on `ggufone/bench/compare.py`** —
  soft threshold, survivors listed in `docs/evidence/e3_mutation_survivors.txt`. The clusters are
  `merge_reports` 57, `render_comparison` 22 (wording/formatting), `model_row` 20 (metadata), while
  the arithmetic the published table actually reads (`agreement_block`, `split_by_mass`,
  `is_low_mass`, `_cell`, `_delta`) carries 8 survivors in total, and every pinned example
  (Wilson interval, the 0.10 floor, the split partition) is still asserted. Counter-evidence that
  matters more than the score: the published table is **generated**, not hand-written, and its
  numbers were re-derived from the stored rows by an independent path (`tools/e3_reproduce.py
  --suite compare`) while writing the docs. Recommendation: **accept (Tier M, record-don't-loop)**;
  the merge/render branches are the next module's sweep scope if a reviewer wants >80 %.
* **The Vulkan ICD workaround is environment configuration, not repository content.** The container
  ships an ICD manifest pointing at a library that needs X11, so `ggufone`'s own bench prints
  `loaded Vulkan backend …` and then sees no device; every worker that needs the GPU must set
  `VK_DRIVER_FILES` to a manifest naming `libEGL_nvidia.so.0` (evidence doc §1.1). Risk: silent
  CPU-only "GPU" runs for the next worker. Cost to fix: one line in the worker image/profile.
  Recommendation: **infrastructure note to the coordinator** (also posted as a card comment).

🟢 ACCEPTABLE (no action)

* **A-E3-1** placement + timings: `n_gpu_layers 7/40` from `ggufone fit` at 5685 MiB free, the
  ladder's real answer recorded per report (`placement.used`, incl. the two `ErrorOutOfDeviceMemory`
  degradations), load 30.8 s, prefill 152.8 s / decision 105.2 s on the first item, 99–113 s
  medians per item afterwards. The reports carry the device attribution `t_603a35a0` added
  (`devices` / `device_buffers` / `effective_backend`).
* **A-E3-2** the 20-question batch: exit 0, 20/20 answers, no OOM; `waves 65` for `forks 80` at
  `n_seq_max 4` and `n_ctx 256` (the adaptation the gate asks about).
* **A-E3-3** the comparison itself: paired (same 20 ids both sides), 4B default 0.500 (10/20) vs
  Occamy 0.450 (9/20), per type and the `low_mass` split published; the honest headline is
  Occamy `low_mass` 20/20 vs the 4B's 3/20.
* **A-E3-4** `llama-bench`: `ngl 7 / t 4` = 1.71 pp64 / 0.28 tg8; `t 8` and `t 12` lose ~2×;
  CPU-only 0.87 / 0.09. Recommendation table published.
* **A-E3-5** `sha256` identical before and after the campaign (`docs/evidence/e3_sha256_*.txt`),
  file `mtime` before the card, no downloads directory change, no writer path touched.
* Gates: **991 passed / 41 skipped** offline (shared tree, includes every sibling's landed work),
  ruff clean (incl. 4 pre-existing E501s in a sibling's `tools/quick_breakdown.py` wrapped),
  `compare.py` 100 % line coverage, 25 E3 gates.

## Decision

🤔 DECISION: Ship or Fix?

**Option A: ship now** (recommended) — every gate has measured evidence; the one scope reduction is
labeled in `docs/BENCHMARKS.md` §6, the evidence doc and the JSON (`n = 20`, `chunks` list).
⚠️ accepted risks: the comparison table cannot separate two models one item apart (said so in the
doc), the serving-path backend label (🟡, FIX card), 137 soft-tier mutation survivors on
merge/render branches.
→ Overall risk: **LOW** for a measurement card; **MEDIUM** if someone reads the 20-item table as a
model ranking (mitigated by the `low_mass` split being the headline).

**Option B: finish the campaign on a quiet box (+40 min in-container, ~15 min on the host)** —
`n` grows to 40–60 and the CIs tighten; no metric on the existing rows changes.
→ Overall risk: LOW, at the cost of another long run.

**Option C: B + the serving-path attribution FIX** (a separate card) → the label class is closed in
both paths. → Overall risk: MINIMAL.

💡 Recommendation: **Option A** — the card's deliverable is measured runs + docs, and the docs say
exactly what was measured, on which compute path, and what the unresolved gap is. Options B/C
belong to follow-up cards.

## Confidence

📈 CONFIDENCE: 8/10

Increasing: every acceptance criterion A-E3-1…5 has a command and its output in
`docs/evidence/e3_t_a431be85_occamy.md`; the SHA-256 is byte-identical before/after; the comparison
is paired and generated; 991 offline gates pass with the change merged into the shared tree;
`compare.py` at 100 % line coverage; the placement is read from the loader's own answer, not from
the flags (and the coordinator's mislabelling class is explicitly re-checked in §4.3).

Decreasing: 20 of 60 dev items (stated everywhere, not hidden); per-item timings measured on a
shared box with a sibling bench (tagged `[container]`); the mutation score is Tier-M soft with 137
survivors in the new module; no `[host]` run of the E3 command exists yet.
