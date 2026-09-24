# E3b — the `qwen35moe` label policy — QA Report

Card `t_6952f0dd` · Tier **M** (declared by the E3b card's scheme; the card itself carries no
`Tier:` line, and this card's work is a probe + evidence campaign, so M is the honest read).
Date: 2026-09-19. Decision: **pending the reviewer** (this report is the reviewer's input, not a
self-approval).

## Summary

🟢 **ACCEPTABLE (shipped as the card's evidence)**

* the card's question is answered with live numbers: **no cue × label rendering reaches the
  engine's 0.10 floor on `qwen35moe`** — 0/6 items on all 15 combinations, best mean 3.359e-05
  (`explicit` × `bare`, i.e. even the cue that names the labels explicitly). The negative result
  the card allows is the measured one, not an inference.
* the *reason* is measured too: `<|im_end|>` holds 0.9976–1.00000 of the next-token mass at every
  cue — the model closes the assistant turn instead of answering. The template's own kept think
  block (the `kept` prefix control) lifts the label mass 45–100× and *still* stays orders of
  magnitude below the floor.
* the 6-item sweep is cross-checked against `DecisionEngine` itself on the shipped policy:
  |Δ coverage| = **8.37e-08** — the probe reads the same row with the same function.
* the 20-item re-measure reproduces E3's headline exactly: **`low_mass` 20/20** (E3: 20/20), and
  the alternative policy measured beside it (`shipped=newline`) is also 20/20.
* `ggufone calibrate` on the accepted run: **no question type clears the held-out gate** (too few
  fit rows at n=20) — recorded verbatim in the evidence doc.
* no E3-published number was changed; the before/after table is this card's own artifact.
* gates: 288 mutants, **244 killed / 40 survived = 85.9 %** (Tier-M soft threshold 80 % met, 0
  not-run, clean re-run after a stale-verdict trap — see below); `labels.py` line coverage
  **99 %** (1 miss: the trivial `__str__`); ruff clean on every file this card touched.

🟡 **WORTH CONSIDERING (documented, not blocking)**

* **Coverage is not comparable across the two execution paths.** E3's rows carry the
  `DecisionEngine`'s coverage (mean 2.838e-02 over the 20 items); this card's probe reads the same
  quantity from the *batched* cue row (mean 2.717e-06). Both are below the floor and both verdicts
  are `low_mass` on every row, so the finding is unchanged in either reading — but the
  before/after table says this in its own words instead of presenting two scales as one series.
  A future card that wants the engine-path number on all 20 items must re-run `--suite quality`
  (~1 h in-container), not read it off this probe.
* **40 mutation survivors**, concentrated in the formatting paths: `cue_line` 9,
  `suffix_with_cue` 7, `score_paths` 7, `label_texts` 6, `LabelsError.__init__` 5, `trie_levels`
  3, `label_coverage` 2, `_check_variant` 1. Several are provably equivalent mutants (e.g.
  `x != "explicit"` → `x is not "explicit"` on an interned variant name; `tuple(...)` →
  `list(...)` in a comparison), but a subset is only "not covered", not "equivalent" — the
  `score_paths` ones in particular can change the ranked readout's arithmetic. They are
  documented here as Tier-M findings for the reviewer per the gauntlet's no-return-loop rule.
  Round 1 of the sweep scored 79.2 % with 59 survivors; the difference is gates, not luck.
* **The ranked-readout path is probe-driven.** Its arithmetic is the engine's
  (`candidate_sequence_score` + `restricted_softmax` + `argmax_first`) and the trie's indexing is
  gated offline, but only the *shipped* policy got a live `DecisionEngine` cross-check. The
  agreement numbers of the 20-item re-measure therefore carry the probe's batch shape, not the
  engine's; the cross-check bounds that at 8.37e-08 on coverage, not on the winner.

🔴 **REQUIRES ATTENTION**

* none for this card. The full-suite run on the shared tree has 3 failures
  (`test_capability.py::test_build_number_reads_llama_cli`,
  `test_e3.py::test_the_published_e3_sections_are_currently_generated_from_the_artifacts`,
  `test_e3.py::test_the_merged_report_covers_every_committed_chunk`) — all three read sibling
  artifacts mid-flight (the E3-completion card `t_6d2e084d` was merging chunks while this ran) and
  none of them touches a file this card changed. Verified by inspection: each reads either the
  installed runtime's build number or the E3 merge state.

## Decision matrix

* **Option A (recommended for the reviewer): accept as the card's evidence.** The deliverable is a
  measured negative result plus the before/after table and the calibration verdict; every claim in
  the evidence document is generated from a stored artifact, and the numbers the card asked for
  are present. The 🟡 items are stated in the document itself (section 9) and in this report.
* **Option B: fix the `score_paths` survivors before accepting** (+~1 h: three gates over exact
  trie arithmetic). Buys tighter mutation score on the readout path; changes no number in the
  evidence document.
* **Option C: re-measure the E3-side 20 items through the engine path** (+~1–2 h in-container,
  competes with sibling cards for the GPU) to put both before/after series on one scale. Changes
  no verdict; changes the coverage magnitudes.

## Confidence

**7/10**

+ every published number is read back from a stored artifact by a gated generator;
+ the model file's SHA-256 (`633ae57f…d757`) matches E3's before/after pins — no weight mutation;
+ the negative result is an *order of magnitude* statement (best cell 1.29e-04 vs floor 0.10),
  not a near-threshold call that a better rendering could flip;
+ live cross-check against `DecisionEngine` (8.37e-08).
- the coverage scale difference (🟡) is real and is now documented rather than eliminated;
- 40 mutation survivors on non-critical paths, unpinned by design at Tier M;
- the 6-item variant ranking is a fixed subset (2 per type) — it ranks renderings, it cannot
  estimate an agreement.

## Environment findings (for the next card on this box)

* The worker container shares the host's RTX 3060 Ti and its **pids cgroup (256)** with sibling
  workers. Two launch failures were observed and are driver-handled now: `E_BACKEND_OOM` when a
  sibling holds the 8 GiB board, and `fork: Resource temporarily unavailable` near the pid cap.
  The fit ladder degrades (7 → 3 layers) instead of dying, which is why one sweep row says
  `degraded: true`.
* The re-measure's chunk 002 ran with `--gpu-layers 0` (placement `n_gpu_layers: 0`) after the
  board stayed busy; coverage is placement-independent, wall time is not (chunk 001 with 3 layers
  offloaded: 10 items in ~20 min; chunk 002 CPU-only: 10 items in ~17 min — the page cache matters
  more than the offload here).
* mutmut caches verdicts per mutant: after changing a gate file, a fresh `mutants/` tree is
  required or the score silently keeps the old verdicts (round 1 did exactly that and reported the
  same 79.2 % twice). `.e3b/mutmut_final.sh` renames the old tree instead of deleting it.
* The scratch trees this card leaves behind: `.e3b/` (runs, reports, logs, drivers, the stale
  mutant trees `mutants.stale-*`), all untracked by the repo.
