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
| **exit codes** | `calibrate` is no longer a stub | `test_the_engine_commands_are_no_longer_stubs[calibrate]` (moved out of the frozen list): no model → 2 `E_MODEL_NOT_FOUND`; rejected fit → 1 with "no calibration applied"; corrupt store → 2 `E_REGISTRY_CORRUPT` |

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

MEASURED_SECTION_4

## 5. What this does not claim, and what it found

MEASURED_SECTION_5
