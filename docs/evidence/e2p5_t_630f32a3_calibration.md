# E2.5 — auto-calibration + routing: the fit, the gate, the router and the escalation

Card `t_630f32a3` · branch `main` (this repo has no remote; commits are local on the shared tree)
· Tier **M** (default — the card declares none) · report schema `ggufone.calibration/v1` +
`ggufone.calibration-store/v1` · evidence schema `ggufone.evidence.e2p5/v1`

Every claim below is a command plus its real output. The machine-readable artifacts live in
`docs/evidence/e2p5_*.json`; the published tables are `docs/BENCHMARKS.md` §5, and one command
regenerates each of them (`tools/e2p5_reproduce.py <calibrate|route|escalate>`).

Environment for every live number: pinned runtime
`/var/home/rybens/.hermes/runtime/b11026-linux-x64-cpu`, models
`Qwen3.5-0.8B-UD-Q4_K_XL.gguf` (558 772 480 B, arch `qwen35`, 24 layers, 0.51 GiB) and
`Spark-X2.5-4B-Q8_0.gguf` (4.37 GB, arch `spark2_5`, 36 layers, 4.07 GiB), the committed 60-item
dev set, and a scratch `GGUFONE_HOME` (`/work/e2p5-evidence/home`) — never the operator's store.
Container: 24 CPUs seen, **2 CPU-seconds/s** cgroup quota, no GPU (`/dev/dri` absent).

## 0. What landed

| area | files |
|---|---|
| calibration math: bins/ECE (the published bench definition), Wilson, agreement at a fixed threshold, power scaling, the ECE fit with an NLL guard, the canonical params hash | `src/ggufone/calibration/stats.py` |
| rows from a `--suite calibration` report, the deterministic per-type fit/holdout split, per-type fit + the acceptance gate + mode selection, the JSON table, the per-model store, the `--dry-run` renderer | `src/ggufone/calibration/calibrate.py` |
| the budget-aware router (candidate verdicts, the SPEC-2.4 conservative plan, the arch pre-flight, tiered ranking) and the escalation policy (selection, merge, log) | `src/ggufone/calibration/routing.py` |
| `ggufone calibrate` (live or `--from-report`), `--route auto`, `--escalate/--max-escalations/--escalation-model`, `--audit DIR`, the readout hook | `src/ggufone/cli.py`, `src/ggufone/engine/decide.py`, `src/ggufone/schema.py` |
| gates: `tests/test_calibration.py` (72) + `tests/test_routing.py` (28) | offline, model-free |
| one command per published table | `tools/e2p5_reproduce.py` |

## 1. Gate table (A-E2p5-1 … A-E2p5-8)

| gate | claim | evidence |
|---|---|---|
| **A-E2p5-1** | `calibrate` fits per-(model, question-type) parameters, writes `calibration.json`, `--dry-run` prints the table, applied at readout | live run §2 (table + store + `--dry-run`); the response carries `calibrated` and `calibration.{source,applied,model,params_hash,temperatures,confidence_modes}` (§3); `test_the_calibrate_command_*`, `test_a_calibrated_table_scales_the_readout_and_marks_the_response` |
| **A-E2p5-2** | accepted only if ECE (or the fixed-threshold agreement) improves on a held-out split; otherwise "no calibration applied" and nothing is stored | live run §2 rejected two of three types (`choice`, `noul`) and stored only the accepted one; `test_the_gate_accepts_...`, `test_a_type_that_is_already_calibrated_...`, `test_a_fit_that_does_not_survive_the_holdout_is_rejected`, `test_nothing_is_stored_when_no_type_was_accepted`, `test_a_rejected_fit_retires_a_previously_stored_table` |
| **A-E2p5-3** | all three modes measured in one report; the default stays `normalized_peak` unless a mode wins by a documented margin | the report carries per-mode `temperature/ece_*/eligible/selected` (§2 table); selection rule + margin pinned by 4 unit tests and by `test_the_live_rows_accept_exactly_one_type_and_switch_that_type_to_entropy`; the promoted statistic is what the response reports (`calibration.confidence_modes`) |
| **A-E2p5-4** | `--route auto` picks (alias, quant, `kv_type`, `n_ctx`, `n_seq_max`) inside the device budget, free-VRAM aware, checked against the fit plan, reason in `engine` | live route run §3; `test_the_router_picks_the_biggest_model_that_fits_the_budget`, `test_a_free_vram_report_smaller_than_the_card_shrinks_the_route`, `test_the_route_keeps_the_context_and_the_sequences_inside_the_budget`, `test_the_route_never_exceeds_the_fit_plan_it_was_asserted_against`, `test_route_request_picks_from_the_registry_and_explains_itself` |
| **A-E2p5-5** | opt-in, logged, bounded by `max_escalations` (default 1), improves the measured agreement on the dev set | live measurement §4; `test_escalation_has_to_be_asked_for`, `test_escalation_is_bounded_by_the_limit_and_ordered_by_confidence`, `test_escalation_replaces_only_the_flagged_answers`, `test_escalation_is_off_unless_the_request_asks_for_it` |
| **A-E2p5-6** | same set + same model ⇒ identical params hash | two live dev-set passes in one process → identical hash (§2); the digest covers only what the fit reads (`test_the_hash_ignores_the_fields_the_fit_never_reads`); `test_a_re_run_on_the_same_rows_hashes_identically` |
| **A-E2p5-7** | the router never routes a model to a runtime lacking its arch | `test_the_router_never_picks_a_model_no_installed_runtime_supports` (a `spark2_5` candidate with only the qwen-capable bundle is rejected, `E_MODEL_ARCH_UNSUPPORTED` when nothing survives); live: the CPU bundle claims both arches, so the live route exercises the acceptance side |
| **A-E2p5-8** | routing/escalation decisions in the response and in the audit log with `--audit DIR` | `engine.route` (full `to_dict`: winner, reason, per-candidate verdicts, budget, capability) and `engine.escalations` in the live responses (§3/§4); `write_audit` appends the record (`test_the_audit_log_records_the_route_the_calibration_and_the_escalations`), and the live route run echoes the audit record it wrote |
| exit codes | `calibrate` is no longer a stub | `test_the_engine_commands_are_no_longer_stubs[calibrate]` (moved out of the frozen list): no model → 2 `E_MODEL_NOT_FOUND`; rejected fit → 1 with "no calibration applied"; corrupt store → 2 `E_REGISTRY_CORRUPT` |

### 1.1 Quality gates (Tier M)

| gate | command | result |
|---|---|---|
| full suite (offline) | `uv run pytest -q -p no:randomly` | **858 passed, 38 skipped** (748 before this card; +110 new cases in `tests/test_calibration.py` + `tests/test_routing.py`) |
| lint | `uv run ruff check src tests tools` | clean |
| runtime contract | `uv run python docs/verify_runtime_contract.py` | `failures: 0  skips: 0` |
| coverage (new package) | `uv run pytest --cov=ggufone.calibration tests/test_calibration.py tests/test_routing.py` | **97%** (calibrate 97%, routing 98%, stats 96%) |
| mutation (Tier M, soft) | `uv run --extra dev --with mutmut mutmut run` over `src/ggufone/calibration` with both gate files as selection | **2008/2728 killed = 73.6%**, 720 survivors (list: `docs/evidence/e2p5_mutation_survivors.txt`) |
| live gates | the runs in §2–§4 | all exit 0; the 4B/0.8B runs are real model loads |

The mutation sweep hit this container's pid cap on its first pass (crashed at 1726/2728 with
`BlockingIOError`) and completed with `--max-children 2` — the number above is the completed run.
The survivors are concentrated in code whose *own* shape is the contract rather than a computed
value: the JSON (de)serialisers (`TypeFit.to_json`/`from_json`, `Table.to_json`/`from_json`,
`RouteStep.to_dict`, `response_fields`, 250+ mutants), the `--dry-run` renderer, and the human
reason strings. A mutation that renames a JSON key survives because both the writer and the reader
in the same test move together — the round trip stays green. The behavioural clusters worth
tightening later are `route` (114), `_plan_candidate` (37), `Table.apply` (20), `decision_of` (19),
`escalation_candidates`/`apply_escalation` (25) and `_write_store` (14); per Tier M this card does
not loop back on survivors — they are listed here for the reviewer's card.

## 2. The live calibration

Command (also the reproduce tool's `calibrate` subcommand):

```
GGUFONE_HOME=<scratch> GGUFONE_RUNTIME_DIR=<cpu bundle> \
  python3 tools/e2p5_reproduce.py calibrate \
    --model ~/.cache/llama.cpp/Qwen3.5-0.8B-UD-Q4_K_XL.gguf --threads 2 --repeat 2 \
    --rows-out docs/evidence/e2p5_rows_qwen08.json \
    --out docs/evidence/e2p5_calibration_qwen08.json
```

### What the run produced

`--out docs/evidence/e2p5_calibration_qwen08.json` (the table) +
`--rows-out docs/evidence/e2p5_rows_qwen08.json` (the raw rows — the artifact the fit actually
reads; re-fitting it reproduces the table exactly). Both passes of `--repeat 2` re-measured the
model and produced the **same `params_hash`**, i.e. the same parameters:

```
repeat 0  accepted=True  stored=True  params_hash=sha256:dd995c81…b1bb
repeat 1  accepted=True  stored=True  params_hash=sha256:dd995c81…b1bb
60 items · 611.5 s for the two passes (threads=2, cpu, runtime b11026-cpu) · created_at
2026-09-18T12:06:22Z
```

The per-type verdicts are the whole point of the gate — this is the §2 table:

| type | rows (fit/holdout) | selected mode | temperature | fit ECE | held-out ECE | verdict |
|---|---|---|---|---|---|---|
| `choice` | 16 / 8 | `normalized_peak` | 1.0 | 0.2451 → 0.2451 | 0.2266 → 0.2266 | **no calibration applied** — `margin` did improve the held-out ECE, but by 0.0031, i.e. below the 0.0050 margin a mode switch needs |
| `noul` | 12 / 6 | `normalized_peak` | 1.0 | 0.2627 → 0.2627 | 0.3843 → 0.3843 | **no calibration applied** — the fit chose the identity (the statistic is already honest) |
| `score` | 12 / 6 | **`entropy`** | **1.6475** | 0.1154 → 0.0695 | 0.4672 → **0.4479** | **applied** — the default statistic improved its *fit* split hugely (0.1906 → 0.0265) and lost the held-out split (0.3732 → 0.4400), so it was refused; `entropy` improved the held-out split by 0.0194 (≥ the 0.0050 margin) |

So `calibration.json` ends up holding exactly one accepted entry
(`accepted_types: ["score"]`), and the two types where the evidence did not support a correction
are left alone — including the `choice` type, where a 0.0031 "improvement" on 8 held-out items
was *not* enough to justify touching the readout.

Two honesty notes this table makes visible:

* the `choice`/`noul` refusals are refusals of a *proposed* parameter, not of the machinery: the
  same code accepted `score` in the same run;
* a per-type held-out split of 6–8 items is small (see §5).

### Reproducibility (A-E2p5-6), three ways

1. two live dev-set passes in one process → identical `params_hash` (above);
2. re-fitting the stored rows (`--from-report docs/evidence/e2p5_rows_qwen08.json`) → identical
   table and hash — pinned by `test_a_re_run_on_the_same_rows_hashes_identically` and by the live
   rows test;
3. the digest covers only what the fit reads, so a re-measurement that moves `coverage` or
   `reliability` cannot move the hash (`test_the_hash_ignores_the_fields_the_fit_never_reads`) —
   this was found the hard way: two live passes differed *only* in the coverage float, and the
   first version of the digest moved with it.

`--dry-run` prints the same table without writing anything (exit 0 either way); without
`--dry-run` the exit code is 0 when a table was stored and **1 when the gate refused everything**
(`no calibration applied` on stdout, nothing written — `test_the_calibrate_command_reports_when_nothing_was_stored`).

## 3. `--route auto`, end to end (A-E2p5-4/7/8)

Two live runs of `tools/e2p5_reproduce.py route` (a real registry, a real bundle, real decisions):

```
GGUFONE_HOME=<scratch> GGUFONE_RUNTIME_DIR=<cpu bundle> \
  python3 tools/e2p5_reproduce.py route \
    --register qwen-0.8b=~/.cache/llama.cpp/Qwen3.5-0.8B-UD-Q4_K_XL.gguf --current qwen-0.8b \
    --threads 2 --workdir <scratch>/work --audit <scratch>/audit \
    --out docs/evidence/e2p5_route_single.json

  ... (then) --register spark-4b=~/.hermes/models/Spark-X2.5-4B-Q8_0.gguf \
    --out /work/e2p5-evidence/e2p5_route_two.json
```

**Run 1 — one candidate** (`docs/evidence/e2p5_route_single.json`): the plan is
`alias=qwen-0.8b quant=null kv_type=f16 n_ctx=4096 n_seq_max=5 n_gpu_layers=0 backend=cpu`, and

```
engine.route.reason = "budget fit: qwen-0.8b kv=f16 n_ctx=4096 n_seq_max=5 on cpu — cpu placement
                       (cpu-only box, so the plan is sized against the system budget
                       (25596 MiB of RAM)); 0 alternative(s) rejected"
```

The decision itself came back calibrated — this is A-E2p5-1's readout hook, live:

```
"calibrated": true,
"calibration": {"source": "<scratch>/calibration.json", "applied": true,
                "model": "file:Qwen3.5-0.8B-UD-Q4_K_XL.gguf:558772480",
                "params_hash": "sha256:dd995c81…b1bb",
                "temperatures": {"score": 1.64755},
                "confidence_modes": {"score": "entropy"},
                "accepted_types": ["score"]}
```

and the per-answer effect is visible in the same response: the `score` answer's probabilities are
the flattened (T=1.6475) ones with the *entropy* confidence, while `choice`/`noul` — the two
refused types — go through the readout untouched (`Table.apply` returns the distribution unchanged
when a type has no accepted parameter: `test_applying_an_unknown_question_type_is_a_no_op`).

**Run 2 — two candidates** (same command plus `spark-4b`, no `--model`): the router picks
`spark-4b` (4.07 GiB of weights beats 0.51 GiB on a CPU-only box) and records the loser in the
plan:

```
"reason": "budget fit: spark-4b kv=f16 n_ctx=4096 n_seq_max=5 on cpu — cpu placement (cpu-only
           box, … (25596 MiB of RAM)); 1 alternative(s) rejected"
step[qwen-0.8b] = "rejected: budget fit: ranked below spark-4b — cpu placement (…) (522 MiB of
                   weights; device budget 30970 MiB)"
"calibrated": false          # the 4B has no stored table — nothing was invented for it
```

**The audit log** (`--audit <dir>` → `<dir>/audit.jsonl`, one JSON object per request, committed
as `docs/evidence/e2p5_audit.jsonl`) carries the same decisions: `ts`, `command`, the question
ids, `route.reason` + the full step list, `calibration` (source + params hash + temperatures) and
`escalations` (§4). One record is in the file, and the route run echoes it into the evidence JSON
under `audit`.

**Capability pre-flight (A-E2p5-7).** The live bundle claims both arches
(`capability.supports_arch` → `qwen35: True`, `spark2_5: True` for `b11026-linux-x64-cpu`), so the
live runs exercise the *acceptance* side; the refusal side is pinned offline: with a bundle that
claims only `qwen35`, a `spark2_5` candidate is rejected with `arch capability: … is not supported`
and routing picks the qwen candidate, and `E_MODEL_ARCH_UNSUPPORTED` is raised when nothing
survives.

## 4. Escalation on the dev set (A-E2p5-5)

Command:

```
GGUFONE_HOME=<scratch> GGUFONE_RUNTIME_DIR=<cpu bundle> \
  python3 tools/e2p5_reproduce.py escalate \
    --primary ~/.cache/llama.cpp/Qwen3.5-0.8B-UD-Q4_K_XL.gguf \
    --target  ~/.hermes/models/Spark-X2.5-4B-Q8_0.gguf \
    --rows docs/evidence/e2p5_rows_qwen08.json \      # reuse the calibration run's primary pass
    --limit 20 --threshold 0.5 --threads 2 --out docs/evidence/e2p5_escalation.json
```

`--rows` reuses the stored primary measurement, so only the escalated items hit the 4B (695.6 s on
the target for 20 items). The policy is the shipped one (`routing.escalation_candidates` +
`routing.apply_escalation`) and the decision rule that scores the result is the shipped one
(`routing.decision_of`), applied to rows that carry their own decision field.

```
escalation set   20 of 60 items (threshold 0.5, limit 20) — all s*/c* items with confidence
                 below 0.5 or reliability low_mass
whole dev set    32/60 (0.5333, ci [0.409, 0.654])  ->  34/60 (0.5667, ci [0.441, 0.684])
                 delta +0.0333  (+2 items)
fit split        21/40 (0.5250) -> 23/40 (0.5750)   delta +0.0500
held-out split   11/20 (0.5500) -> 11/20 (0.5500)   delta  0.0000
```

Every one of the 20 re-asks is in `engine.escalations.decisions` with its trigger (`low_confidence`
/ `low_mass`), the confidence that triggered it, `was` → `now` and `replaced: true`; nothing was
replaced without a log line, and the request-level log is what `--audit DIR` persists.

**Honest reading:** the delta is positive on the dev set but rests on **two items**, and it is
**zero on the held-out split** — i.e. this dev set supports "escalating the low-confidence answers
helped by 2 items here", not "escalation helps in general". `max_escalations` defaults to 1 in the
shipped request semantics (the measurement used 20 to bound the experiment, not to claim a
default); with the default, the same policy re-asks one item per request.

## 5. What this does not claim, and what it found

### 5.1 It does not claim the parameters generalise

The accepted `score` parameter (T=1.6475, `entropy`) rests on a **6-item held-out split**. It
survived the gate on this dev set and is reported as such; the report says nothing about other
prompts, other states or other languages. The two refused types are equally honest: `margin`
*proposed* a parameter for `choice` (held-out +0.0031) and it was refused for being below the
documented margin, not because the machinery could not see it.

**The smallest useful extension of the dev set** is +12 items per type (12/24/24 → 24/36/36,
i.e. 120 items total): that doubles the per-type held-out split to 12–16 items, which is the
smallest size where an ECE difference of ~0.02 is not dominated by one flip. Until then the gate's
verdicts are honest but noisy, and the honest answer to "is this model calibrated?" is "the
parameters here are; the next 12 items per type would show whether it matters".

### 5.2 It found a real bug in the E2 bench loader (not fixed here)

`ggufone bench` fails at model load on **any** box, GPU or not:

```
E_INTERNAL: AttributeError: 'Placement' object has no attribute 'kv_type'
  File ".../ggufone/bench/harness.py", line 380, in load          -> fit_plan=Placement(n_gpu_layers)
  File ".../ggufone/engine/session.py", line 250, in open_model   -> fit.degrade_ladder(fit_plan, facts)
```

`harness.LiveModel.load` passes its own `Placement(n_gpu_layers)` where `open_model` expects a
`fit.FitPlan`, and E1c's `degrade_ladder` then reads `plan.kv_type`. The E2 tables were measured in
a tree whose `session.py` predates `degrade_ladder` (0 hits for it in that tree's session, while
the shared tree has the call at line 250), so the two pieces were transplanted together without
ever running together; the coordinator filed it as card **`t_31b3943a`** (different card, different
owner — E2.5 does not fix it). Consequences here:

* E2.5 measures through the **serving path** (`open_model` + `ModelSession`, what `run`/`ask` use)
  — no E2.5 code path touches the bench loader, so `ggufone calibrate` works today;
* the same dev set gives different *absolute* agreement through the two load paths (0.8B: 32/60
  serving vs 28/60 bench). All E2.5 deltas are measured inside one path.

### 5.3 It found the parameters-hash trap

Two live passes of the same model produced identical probabilities, identical confidences and
identical decisions — and a **different digest**, because one `coverage` float and one
`reliability` label had moved (that pass ran an older `Row` shape). The first digest hashed the
whole row, so the parameters hash moved with data the fit never reads. Fixed: the digest covers
`id/type/expected/correct/probabilities` only (`test_the_hash_ignores_the_fields_the_fit_never_reads`).
Without this, "same set + same model ⇒ identical params hash" would have been false for reasons
nobody could see.

### 5.4 Scope boundaries the reviewer should know

* **No GPU on the measuring box.** The router's device branch (full offload, partial offload, the
  free-VRAM margin, the KV-floor fallback, the CPU fallback) is covered offline with injected
  budgets, and its CPU branch live. A device-placement run on the operator's box remains open.
* **Escalation needs an explicit target** (`--escalation-model REF`, or the next-ranked candidate
  when `--route auto` produced a ranking). There is no automatic second-model download, and no
  cascade: an escalated request is decided with escalation disabled.
* **The audit log is CLI-level** (`--audit DIR`), like `--out`. The request-level options
  (`route`, `escalate`, `max_escalations`) are part of the frozen `options` block, so a future
  HTTP/MCP surface inherits the routing and escalation semantics but must wire its own audit sink.
* **`confidence_mode: null` is the new wire default** (it used to be the literal
  `"normalized_peak"`). Behaviour is unchanged without a stored calibration — the engine resolves
  `null` to `normalized_peak` — but a caller that *echoed* the old default back is now explicitly
  pinning the statistic, which is the intended way to opt out of a promoted mode.
* **`--dry-run` does not write**, and a refused fit stores nothing: calibration can only be
  applied by a table the gate accepted, and the response always says whether it was applied
  (`calibrated`, `calibration.source`, `calibration.params_hash`).
