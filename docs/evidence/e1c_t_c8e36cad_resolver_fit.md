# E1c — reasoning resolver + fit (templates, thinking suppression, `ggufone fit`, E2E)

Card `t_c8e36cad` · branch `main` (this repo has no remote; commits are local on the shared tree)
· Tier **M** (default — the card declares none) · evidence schema `ggufone.evidence.e1c/v1`

Every claim below is a command + a real output tail. Raw logs live next to this file in
`.e2e/t_c8e36cad-e1c/`; the machine-readable E2E record is `docs/evidence/e1c_e2e.json`, and the
reference document for the resolver is `docs/TEMPLATES.md`.

## 0. What landed

| area | files |
|---|---|
| the ordered template chain, the internal Jinja-subset renderer, thinking suppression, the family policy table | `src/ggufone/engine/template.py` |
| chat-template prompt assembly (the plain E1b framing stays as `--template plain`) | `src/ggufone/engine/prompt.py` |
| template resolution per request, `engine.template`, the fit cap on `n_ctx` | `src/ggufone/engine/decide.py` |
| `n_gpu_layers` from the plan, `kv_type`/`n_gpu_layers` in `SessionMeta` | `src/ggufone/engine/session.py` |
| the fit plan, its sources, the KV ladder, the cache, the RSS helpers | `src/ggufone/runtime/fit.py` |
| the chat-template ABI (fixed signature + `llama_chat_message` + `token_piece`) | `src/ggufone/runtime/ctypes_binding.py` |
| `fit`, `--no-fit`/`--fit-target`/`--fit-ctx`/`--no-fit-cache`, `--template`/`--thinking`, per-command `--help` | `src/ggufone/cli.py` |
| `options.template` / `options.thinking` | `src/ggufone/schema.py` |
| gates | `tests/test_templates.py`, `tests/test_fit.py`, `tests/test_fit_live.py`, `tests/test_cli_e1c.py`, `tests/conftest.py` (`GGUFONE_TEST_BLOCK_NET`), `tests/test_ctypes_binding.py` (chat-template signature pin) |
| tools | `tools/e1c_e2e.py` (A-E1c-8), `tools/e1c_offline_gate.py` (A-E1c-10), `tools/mutmut_driver.py` (container workaround) |
| docs | `docs/TEMPLATES.md` (A-E1c-9), `README.md` (E1c status + fit section) |

Environment for every live number: pinned runtime
`/var/home/rybens/.hermes/runtime/b11026-linux-x64-cpu`, pinned `XHToken/Spark-X2.5-4B-GGUF:Q8_0`
(4 375 021 152 B, sha256 `5c2c3c19…9dea2`) and `unsloth/Qwen3.5-0.8B-GGUF` UD-Q4_K_XL
(558 772 480 B). Container: 24 CPUs, 31 GiB RAM, no GPU (CPU placement, `n_gpu_layers=0`).

## 1. Gate table (A-E1c-1 … A-E1c-10)

| gate | claim | evidence (command → tail) |
|---|---|---|
| A-E1c-1 | chain, ordered + documented: GGUF renderer → built-ins (+`W_TEMPLATE_FALLBACK`) → user override → `E_TEMPLATE_UNRESOLVED` with the fix | `uv run pytest -q tests/test_templates.py` → `31 passed, 4 skipped`; both *real* templates render through step 1 (`test_a_real_gguf_template_renders_through_chain_step_one[spark2_5\|qwen35]`), step 2 renders a subset-rejected template through `llama_chat_apply_template` (`test_step_two_renders_through_the_runtime_builtin_table`), step 4's message names the construct **and** `--template` (`test_step_four_is_a_pinned_error_whose_message_carries_the_fix`); `docs/TEMPLATES.md` §1 |
| A-E1c-2 | the rendered prompt provably contains no think-opener | same run: `no_open_think()` is asserted on the rendered string **and** on the model's own detokenized ids (`test_the_real_prompt_has_no_think_token_on_the_real_vocabulary` → `"<think>" not in decoded`, `no_open_think(decoded) is True`, and the thinking-on control shows the check discriminates); Spark renders `<\|Bot\|></think>`, Qwen renders `<\|im_start\|>assistant` after the empty block is stripped |
| A-E1c-3 | post-cue degenerate output never affects the readout (synthetic logits fixture) | `test_post_cue_degenerate_output_never_affects_the_readout`: a 40-logit spike on a non-candidate token leaves `choice`/`probabilities`/`confidence` **identical** and only moves `coverage` (+`W_LOW_MASS`) — the diagnostic that is *supposed* to see the full-vocab mass |
| A-E1c-4 | `fit` returns the 9 documented fields; `source` is `llama-fit-params` when the binary ran, else `estimate` + `W_FIT_ESTIMATED`; cached per `(sha256, fingerprint)` | `uv run pytest -q tests/test_fit.py tests/test_cli_e1c.py` → `69 passed`; live: `test_the_pinned_default_model_gets_a_plan_from_the_binary` → `source: llama-fit-params`, and with no runtime handed in `source: estimate`, `warnings == ("W_FIT_ESTIMATED",)`; cache hit does not re-run the binary (`test_the_plan_is_cached_per_model_sha_and_host_fingerprint`) |
| A-E1c-5 | the plan is applied on load unless `--no-fit`; over-budget downgrades `kv_type` f16 → q8_0 → q4_0 with `W_KV_TYPE_DOWNGRADE` | `test_an_over_budget_plan_downgrades_kv_type_in_the_documented_order` + `test_a_tighter_budget_walks_the_ladder_down_to_q4_0` (ladder walked end to end); live: `test_the_plan_is_applied_on_load_unless_no_fit` → with the plan `engine.kv_type == "f16"`, `engine.n_ctx == fit.n_ctx` (cap wins over a 32768 request), with `--no-fit` `engine.kv_type == "auto"` and no `engine.fit` |
| A-E1c-6 | **[target]** the estimate cross-checks against measured load RSS ±20 % | `test_the_fit_estimate_cross_checks_against_measured_load_rss` → `est_total 4685 MiB delta_load 4215 MiB (-10.0%) delta_ctx 4251 MiB (-9.3%)` (an earlier run measured `+2.3 %`); assertion `within_tolerance(est_total, delta_ctx, 0.20)` |
| A-E1c-7 | fork equivalence holds for `qwen35` **and** `spark2_5`; `n_rs_seq=0` documented | `uv run pytest -q --run-network -s tests/test_engine_fork.py` → `fork vs sequential on qwen35: max \|delta\| = 0.000e+00 over 2 candidates` / `… on spark2_5: max \|delta\| = 0.000e+00` — measured at this head, i.e. **with** the chat template and the fit plan in the path; `docs/TEMPLATES.md` §6 documents `n_rs_seq = 0` |
| A-E1c-8 | E2E on the pinned default model: 4 documented example question sets, probabilities + confidence + timings recorded | `uv run python tools/e1c_e2e.py` → 4 runs, `docs/evidence/e1c_e2e.json` (table in §4) |
| A-E1c-9 | `docs/TEMPLATES.md` covers `spark2_5`, `qwen35`, `qwen35moe`, `k2-horizon` (source, thinking, label policy, caveats) | `docs/TEMPLATES.md` §4 (the table) + §8 (limits); `FAMILIES` in `engine/template.py` is the code copy and `test_every_documented_family_has_a_policy_row` pins all four rows |
| A-E1c-10 | no network in any decision path; all E1c tests run with network disabled | `uv run python tools/e1c_offline_gate.py` → `142 passed, 21 skipped (exit 0)`; with the live assets: `--run-network` → see §2. `tests/conftest.py` replaces `socket.socket`/`create_connection`/`getaddrinfo` with a raiser under `GGUFONE_TEST_BLOCK_NET=1` |

## 2. Headline receipts (verbatim)

```
$ uv run pytest -q                                  # offline canonical gate
590 passed, 33 skipped in 50.49s

$ uv run ruff check src tests tools
All checks passed!

$ uv run pytest -q --run-network -s tests/test_fit_live.py tests/test_templates.py \
      tests/test_engine_fork.py tests/test_ctypes_binding.py tests/test_cli.py
fork vs sequential on qwen35: max |delta| = 0.000e+00 over 2 candidates
fork vs sequential on spark2_5: max |delta| = 0.000e+00 over 2 candidates
warm prefill_ms = 0.000  cold prefill_ms = 1685.6
determinism sha256 = 24e81d2cc2b453a447c6a6ed028aa64eaed44ed7e43b1cdf9948ddbd8f680253
state round-trip max |delta| = 0.000e+00
waves: capped=16 single=8 max |delta| = 0.000e+00
live plan: {"n_gpu_layers": 0, "n_ctx": 4096, "kv_type": "f16", "n_seq_max": 8,
            "est_weights_bytes": 4369416192, "est_kv_bytes": 264241152,
            "est_total_bytes": 4912578560, "backend": "cpu", "source": "llama-fit-params"}
weights: tensor index 4369633280 vs binary 4369416192 (delta 217088 B)
estimate: kv 576 MiB for 4096 ctx tokens
est_total 4685 MiB  delta_load 4215 MiB (-10.0%)  delta_ctx 4251 MiB (-9.3%)
105 passed in 302.86s (0:05:02)

$ GGUFONE_TEST_BLOCK_NET=1 uv run python tools/e1c_offline_gate.py
142 passed, 21 skipped in 1.11s
network-disabled run: {'passed': 142, 'skipped': 21} (exit 0)
```

The template chain resolved every live request through **step 1** (`kind: gguf-renderer`,
`family: spark2_5`, `thinking: suppressed`, `warnings: []`) — the fallback path is exercised by
tests, not by the production runs.

## 3. Findings this card produced (beyond "tests pass")

1. **The `llama_chat_apply_template` binding was wrong** — seven arguments where the header
   (include/llama.h @ b11026:1222) has six, and a `char *` where the message *array* belongs.
   Calling it that way marshals garbage; the resolver would have "rendered" whatever happened to
   be in the registers. Fixed in `runtime/ctypes_binding.py` and pinned live by
   `tests/test_ctypes_binding.py::test_chat_template_signatures_match_the_header` (a bundle is
   needed because ctypes only exposes `argtypes` after `load_libraries()`).
2. **The KV estimate needs two numbers, not one.** SPEC 2.4's frozen planner charges 1 B/element
   for `q8_0` *and* `q4_0` (an upper bound — the oracle mirrors it). The fit plan instead uses the
   real ggml ratios (2, 34/32, 18/32) and the **unified** cache (`n_ctx` cells, not
   `n_ctx × n_seq_max`), because A-E1c-6 measures it against RSS. With the conservative numbers the
   plan would over-predict this model's load by ~50 % — outside the ±20 % target by construction.
   Both numbers are now documented side by side (`docs/TEMPLATES.md` §5, `runtime/fit.py`).
3. **`llama-fit-params` agrees with the GGUF tensor index to 0.005 %** (4 369 416 192 B vs
   4 369 633 280 B, Δ 212 KiB) on the pinned model — i.e. the estimate path and the binary path
   size the weights the same way; what differs is only the KV row, where the binary knows the
   hybrid layer layout and our formula charges every layer (252 MiB vs 576 MiB at 4 k ctx).
4. **The readout position is not where the pinned family wants to answer.** At the cue the model's
   top token is `\n` (logit ≈ 0, p ≈ 0.94); the candidate labels carry 1–3 % of the mass
   (measured, `docs/TEMPLATES.md` §4 "label policy"). `coverage` therefore reports `low_mass` on
   realistic states while the restricted softmax stays decisive (e.g. `queue: billing 0.92`,
   `refund yes 0.80`). Two candidate readout policies are recorded for E2 to test — a blank-line
   cue (`cue + "\n\n"` raised the label mass from 0.0114 to 0.0340 in the probe) and scoring the
   label *after* the model's own newline. E1c deliberately did **not** change the tip of the
   prompt: that would move every pinned E2E number, and the quality suite (E2) is the one that can
   measure whether it helps agreement.
5. **Wave accounting (coordinator question b).** `usage.waves` counts **decode batches**, not
   forks: one batch per question-group suffix decode plus one per extra candidate-token step, with
   groups of `n_seq_max − 1` candidates. `usage.forks` counts branches (one per candidate). The
   formula reproduces all four E2E runs exactly (recomputed from the recorded requests):

   | run | candidates per question | token lengths | forks (recorded/computed) | waves (recorded/computed) | decode steps |
   |---|---|---|---|---|---|
   | incident-triage | 4 / 5 / 2 | [2,2,2,1] / [1,1,1,1,1] / [1,1] | 11 / **11** | 4 / **4** | 14 |
   | support-routing | 4 / 4 / 2 | [2,2,1,3] / [1,1,1,1] / [1,1] | 10 / **10** | 5 / **5** | 14 |
   | release-readiness | 3 / 4 | [1,1,3] / [1,1,1,1] | 7 / **7** | 4 / **4** | 9 |
   | risk-assessment | 4 / 4 / 2 | [1,1,2,2] / [1,1,1,1] / [1,1] | 10 / **10** | 4 / **4** | 12 |

   So `waves=8` for `forks=5` is expected whenever the labels are multi-token or the branch count
   exceeds the wave cap (e.g. `n_seq_max=4` → groups of 3 + 2 → `1+maxlen−1` per group); it is
   **not** suffix-length grouping. The operator's observation is consistent with this rule.
6. **Vulkan pipeline cache (coordinator question a): there is nothing to amortize in the pinned
   build.** b11026's Vulkan backend creates compute pipelines with `VK_NULL_HANDLE` as the
   pipeline cache (`ggml/src/ggml-vulkan/ggml-vulkan.cpp` @ b11026, line 744:
   `device->device.createComputePipeline(VK_NULL_HANDLE, compute_pipeline_create_info)`), and the
   string table of the shipped `libggml-vulkan.so` has no cache-file path or env var. Consequence:
   a fresh CLI process pays the full pipeline/shader setup every time, and *no* configuration of
   the pinned bundle changes that. Recorded for E2 (it owns the benchmark and any upstream
   request); ggufone's own lever is the per-request `state_id` cache, which removes the prefill,
   not the Vulkan setup.
7. **Environment sensitivity of two pre-existing gates (not E1c's).** With
   `GGUFONE_RUNTIME_DIR` set *and* `$HOME` pointing at a home without the pinned model,
   `tests/test_runtime_contract.py::test_oracle_live_section_is_green_without_skips` and two
   `tests/test_runtime_fallback.py` cases fail — identically at the pre-E1c head `9f53315`
   (verified in a fresh worktree), so this card did not introduce it. Recorded, not fixed (out of
   E1c scope); `tools/e1c_offline_gate.py` deliberately does not include those files.

## 4. A-E1c-8: end-to-end on the pinned default model

`uv run python tools/e1c_e2e.py` (fit on, thinking suppressed, `threads=4`, CPU). Full record:
`docs/evidence/e1c_e2e.json` (request, answers, usage, timings, warnings, `engine.template`,
`engine.fit`). Every run resolved through **chain step 1** and used the **binary's** plan
(`source: llama-fit-params`, `kv_type: f16`).

| set | question | answer | p / score | confidence | coverage | reliability |
|---|---|---|---|---|---|---|
| incident-triage | area | `billing` | 0.522 | 0.363 | 0.010 | low_mass |
| | severity | score **0.457** | p = {0:0.62, 1:0.33, 2:0.04, 3:0.01, 4:0.00} | 0.519 | 0.057 | low_mass |
| | page | `noul` **0.581** | p = {yes:0.58, no:0.42} | — | 0.023 | low_mass |
| support-routing | queue | `billing` | 0.923 | 0.897 | 0.072 | low_mass |
| | urgency | score **1.393** | p = {0:0.10, 1:0.44, 2:0.43, 3:0.03} | 0.257 | 0.051 | low_mass |
| | refund | `noul` **0.802** | p = {yes:0.80, no:0.20} | — | 0.009 | low_mass |
| release-readiness | decision | `ship-migration` | 0.976 | 0.965 | 0.004 | low_mass |
| | confidence | score **0.548** | p = {0:0.51, 1:0.43, 2:0.05, 3:0.01} | 0.353 | 0.003 | low_mass |
| risk-assessment | risk | `skills` | 0.702 | 0.603 | 0.00002 | low_mass |
| | exposure | score **0.371** | p = {0:0.64, 1:0.35, 2:0.01, 3:0.00} | 0.523 | 0.235 | **ok** |
| | proceed | `noul` **0.683** | p = {yes:0.68, no:0.32} | — | 0.008 | low_mass |

Timings (this box, CPU, container quota):

| set | model_load_ms | prefill_ms | questions_ms | total_ms | forks | waves | decode steps |
|---|---|---|---|---|---|---|---|
| incident-triage | 703 | 14 704 | 22 193 | 36 901 | 11 | 4 | 14 |
| support-routing | 903 | 13 608 | 21 298 | 34 911 | 10 | 5 | 14 |
| release-readiness | 988 | 16 291 | 12 699 | 28 993 | 7 | 4 | 9 |
| risk-assessment | 654 | 16 985 | 21 104 | 38 094 | 10 | 4 | 12 |

Readings that matter: the answers are **sensible** for the states (payment incident → `billing`,
duplicate charge → refund `yes`, the release with a migration precedent → `ship-migration`), the
restricted distributions are decisive where the state is unambiguous (0.92 / 0.98), and `coverage`
is `low_mass` in 10 of the 11 questions — a property of *this model's* mass at the cue (finding 4),
reported rather than hidden (the one exception, `exposure`, reads `ok` at 0.235). Every run carried
the run-level warning `W_LOW_MASS`, and none carried `W_TEMPLATE_FALLBACK`.

Template + fit provenance for these runs, exactly as recorded: `engine.template` = `{kind:
gguf-renderer, family: spark2_5, thinking: suppressed, warnings: []}` — chain step 1 for all four
sets; the tool runs with `fit_enabled: true`, so `engine.kv_type: f16` is the plan's choice (the
`--no-fit` control in the live suite yields `auto`), and `engine.n_ctx: 256` is the *request-sized*
window (127/124/127/136 prefix tokens + question + margin) with the plan acting as a 4096-token
ceiling — the E2E's own engine record does not carry the `fit` surface, so the plan's
`source: llama-fit-params` for this host is evidenced in §2
(`test_the_pinned_default_model_gets_a_plan_from_the_binary`) and the application in §1 A-E1c-5
(`test_the_plan_is_applied_on_load_unless_no_fit`).

The whole run is one `prefill` per set (127/124/127/136 tokens) and 7–11 forks, i.e. the E1b
mechanics hold under the E1c template. Raw log: `.e2e/t_c8e36cad-e1c/logs/e2e_run1.log`;
machine-readable record: `docs/evidence/e1c_e2e.json`.

## 5. Limits / not covered here

* No GPU on this box (container): every live number is CPU placement. `n_gpu_layers` > 0 is
  exercised by unit tests (fake hosts) only; the RTX 3060 Ti numbers are the coordinator's
  **[recon]** and stay E2's.
* `qwen35moe` and `k2-horizon` have **no GGUF on this box**: their policy rows are documented from
  the published templates/TRL patch notes (tagged in `docs/TEMPLATES.md` §4) and pinned by unit
  tests, not by a live run.
* The E2E is correctness-and-shape evidence, not a quality benchmark: no labels, no agreement
  score (E2 owns the labeled dev set).
* **Mutation testing** (Tier M, soft threshold) is reported in §6.

## 6. Mutation testing (Tier M)

Runner: **mutmut 3.8** driven through `tools/mutmut_driver.py` (the container cannot re-apply
SELinux xattrs on the copied tree, so the driver disables `shutil._copyxattr` for the mutmut
process — the same workaround E1a documented). Scope: the two new modules
(`engine/template.py`, `runtime/fit.py`) with the E1c gates as the test selection; the config
change lives in `pyproject.toml` `[tool.mutmut]`.

RESULT_PLACEHOLDER
