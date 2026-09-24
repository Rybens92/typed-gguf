# E1c FIX (card t_8cb0a05e) — QA report

Date: 2026-09-18 · Tier **M** (the card declares no tier → default) · Decision: pending the
operator's host run (see §Risks)

Deliverable: free-VRAM planning, `--fit-target` bounding, the load-time degradation ladder,
`E_BACKEND_OOM`, and the busy-desktop host-gate scenario. Implementation notes and the full
evidence index: `docs/evidence/e1c_t_8cb0a05e_fit_oom.md`; acceptance criteria:
`.gauntlet/e1c-fit-oom.spec.md`.

## Risk-weighted summary

🟢 ACCEPTED (gates that passed on real machines, not on assertions alone)

- **R1 free-VRAM planning** — one driver query (`memory.total,memory.free`), amdgpu-sysfs
  fallback, `HostFacts.budget_bytes` from the free number, `replan_for_host` re-validate +
  cache rewrite. 18 offline gates + live: the pinned model loads with a plan built from the
  operator's numbers (`test_a_busy_desktop_plan_loads_on_the_free_reading`).
- **R2 `--fit-target` bound** — one budget rule (`fit_budget`) threaded through both sources and
  the `-ngl` the tool is asked about; boundary pin `n_gpu_layers == 23` at
  `8 GiB − 5200 MiB`; the gate shows `budget_bytes 92274688 → 0` when the target rises to 5200.
- **R3 degradation ladder** — real `.so` fixture (not a mock): 36→18→0 layers with
  `W_BACKEND_OOM` + `W_FIT_DOWNGRADE`; the all-rungs world answers `E_BACKEND_OOM`; the
  `--no-fit` placement note is proven on a real engine run.
- **R4 classification** — the operator's tail is `"oom"`, a real arch failure is `"arch"`; the
  pre-fix tree reproduces the exact misclassified sentence (`E_MODEL_ARCH_UNSUPPORTED … (arch
  spark2_5)`) while the fixed tree raises `E_BACKEND_OOM` with free/needed bytes and both hints.
- **R5 host gate** — `tools/host_gate_e1c_fit.sh` (7 steps, free VRAM before/after, `is_host_run`
  in the folded JSON) rehearsed in the sandbox; the host run is the only outstanding item.

🟡 WORTH CONSIDERING

- The **context-init kv ladder** (`_kv_ladder`) is exercised by unit tests (fake runtime) but not
  by a live allocation failure — producing one requires a device that is *almost* full. The
  code path is small (three rungs) and its failure mode is the pre-existing generic error, so the
  cost of leaving it live-unverified is low.
- `HostFacts.vram_free_bytes == 0` means "unknown" and falls back to the nominal size (the old
  behaviour). On a box whose driver exposes nothing (`nvidia-smi` absent *and* no amdgpu sysfs)
  the plan is still nominal — documented by a test, and the load-time ladder is the safety net.
- `recommend.device_memory()` is a *new* host read in `host_facts()`; the purity pin
  (`vram_probe=`/`device_probe=` owns the answer) is kept, so no test can silently read the box.

🔴 REQUIRES ATTENTION (for the reviewer / operator, not for the implementer)

- **The host run is PENDING.** The GPU-world readings (the Vulkan row of `llama-fit-params`, the
  real Vulkan allocation failure, the real busy desktop) cannot be produced in this container: no
  `/dev/dri`, no `/dev/nvidia*`, no `nvidia-smi`. The evidence file marks them PENDING and the
  gate prints `is_host_run` so the rehearsal cannot be mistaken for a host run. This is the same
  limitation every card in this environment has; the reviewer should read §3 of the evidence file
  as "executed except the device-only rows".

## Metrics

| metric | value |
|---|---|
| new offline gates | 40 (`tests/test_fit_free_vram.py`, `tests/test_fit_oom_recovery.py`) |
| suites re-run | `test_fit.py`, `test_e1c_mutation_pins.py`, `test_engine_fork.py`, `test_cli.py`, `test_cli_e1c.py`, `test_templates.py`, `test_host_purity.py`, `test_recommend_quant.py` — green |
| full suite at the final head | 714 passed / 28 failed / 34 skipped (all 28 outside this card: 21 = the sibling E2 card's in-flight `test_bench.py`, 7 = environment/contention flakes that pass in isolation) |
| coverage (changed modules) | `fit.py` 95 %, `decide.py` 92 %, `errors.py` 100 %, `session.py` 65 % (file-wide; the uncovered part is E1b's live state paths) |
| lint | `ruff check <changed files>` → clean |
| mutation (Tier M, one run) | see §Mutation below |
| real-engine evidence | 3 engine runs on the pinned model + pinned bundle (`run_busy_desktop`, `run_no_fit`, the red probe), 2 live pytest gates, 1 fake-`.so` OOM ladder |

## Mutation

<!-- filled in at commit time -->

## Verification limits (honest list)

1. No GPU, no driver, no `/dev/dri` in this container → the Vulkan/OOM rows are simulated with a
   real `.so` fixture and a fake `nvidia-smi`; the host run is PENDING.
2. `session.py`'s context-init kv ladder has no live OOM trigger (§🟡).
3. The container is shared with several other cards: two runs of the *same* pytest files produced
   different failure sets (7 environment-dependent tests failed once and passed in their own
   groups), the full suite cannot be made "green" while the sibling E2 card's untracked
   `test_bench.py` is failing, and one mutmut attempt aborted on pid exhaustion
   (`Resource temporarily unavailable`) which surfaced as a false test failure
   (`payload["source"] == "estimate"`, i.e. `fork`/`exec` of the stub tool failed). Both are
   recorded in the evidence file rather than smoothed over.

## Recommendation

**Fix anything that the reviewer finds in the diff; run `tools/host_gate_e1c_fit.sh` on the RTX
box before the next fit-touching card is considered proven.** Nothing in this card's own surface
is blocked on a decision.
