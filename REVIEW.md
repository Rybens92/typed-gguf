# E1a final review — card `t_ee24bd7c` (code-reviewer)

- **Verdict: PASS** — no blocking findings at the reviewed final head `1c20ad4`.
- Reviewers revision history (the head moved *during* this review — see §1):
  - `3f18890` — as-dispatched head; the duel's survivors (`m08`–`m11`) were verified to survive here:
    **B1 was blocking at this revision**;
  - `1c20ad4` — final head after card `t_83ee1eed` landed the B1/B2/B3 pins mid-review; the referee
    re-run (§2-G) shows every survivor killed. This is the revision the PASS applies to.
- Environment: podman container, **no GPU / no operator host**: `nvidia-smi` absent, `/dev/dri` absent,
  `/dev/nvidia*` absent. Python 3.11.15, uv 0.12.0, canonical gate `uv run pytest -q`.
  For the GPU-shaped runs the host fact was simulated the way the evidence file does it: a fake
  `nvidia-smi` first on `PATH` (`/tmp/rev/fakebin/nvidia-smi`), plus — for one reproduction only —
  a synthetic `/dev/dri/renderD128` + `/usr/share/vulkan/icd.d` that were **removed afterwards**
  (verified gone: `exists? False`).
- Inputs consumed: the diff `a4ab12e..1c20ad4`, `docs/evidence/e1a_baseline.json`
  (`e1a_fix_t_eae35404.host`), the adversarial report
  `docs/evidence/e1a_t_0fc576df_host_access_mutants.md`, the fight dir
  `/var/home/rybens/workspace/state/fights/e1a-t0fc576df/` (mutants + patches + witnesses).

---

## 1. Summary (what happened, in order)

1. At `3f18890` the reviewed state had a **verified, blocking test-strength hole**: the duel's `m11`
   (drop the injected `system` in `capability.backends`) survived the task-0 regression file, the full
   suite **and** the oracle (§2-F). `m08`/`m09` survived the task-0 file (killed only by `test_pins.py`),
   and `m10` survived the task-0 file *and* the whole suite **in a GPU-less world** (its only killer,
   `test_recommend_quant.py::test_host_budget_reads_meminfo`, needs a real GPU to bite).
2. While this review was running, card `t_83ee1eed` (created 21:22, same repo dir workspace) landed
   commit `1c20ad4` — add-only pins for exactly those four mutants in `tests/test_host_purity.py`
   (`+126/-1`, the one deletion being a re-used import line).
3. I re-ran the referee check myself on `1c20ad4` (fresh worktree, the duel's own patches applied,
   my own script): **m08, m09, m10 and m11 are all killed** by named node ids (§2-G), control green.
4. Canonical gates on `1c20ad4`: suite green in both worlds, oracle exit 0, the card's 7 named tests
   pass, pre-fix tree still red as required (§2-A..E).
5. Because the reviewed subject is a repo that is **actively being edited by a sibling live card**,
   the PASS is pinned to `1c20ad4`. Any later commit that touches `src/` or `tests/` invalidates the
   re-run for the changed files.

## 2. Evidence — command + output tail per claim

### A. Canonical suite on the final head (offline semantics: network tests are marked/skipped)
```
$ cd /var/home/rybens/workspace/ggufone && git log --oneline -1
1c20ad4 test(E1a t_83ee1eed): pin the duel's survivors — injected system (m11), unnamed machine (m08), omitted dri_nodes (m09), empty vram probe (m10)
$ UV_CACHE_DIR=/tmp/uvcache uv run pytest -q
333 passed, 12 skipped in 5.18s          EXIT=0
```

### B. Same suite in the GPU-shaped world (the host's shape, simulated)
```
$ PATH=/tmp/rev/fakebin:$PATH uv run pytest -q
333 passed, 12 skipped in 5.11s          EXIT=0
```

### C. Runtime-contract oracle
```
$ python3 docs/verify_runtime_contract.py
failures: 0  skips: 3                    EXIT=0
```
(the 3 skips are the runtime/model-absent ones in this container; section B live was covered by
`t_3831b7b3` and the coordinator's host run — §5)

### D. The two originally-failing tests + the 5 CLI tests using the same mapping
```
$ uv run pytest -q \
    tests/test_pins.py::test_detect_backend[probes2-vulkan] \
    tests/test_pins.py::test_host_variant_mapping[auto-linux-x86_64-linux-x64-cpu] \
    tests/test_cli_doctor_branches.py::test_init_text_mode_with_the_offline_cache \
    tests/test_cli_e1a.py::test_init_dry_run_json_prints_the_plan_and_writes_nothing \
    tests/test_cli_e1a.py::test_init_dry_run_text_says_so \
    tests/test_cli_e1a.py::test_init_from_offline_cache_with_a_poisoned_path \
    tests/test_cli_e1a.py::test_init_twice_is_idempotent
7 passed in 0.08s                        EXIT=0
```
(also verified on the pinned `3f18890` worktree: `7 passed`)

### E. The regression test genuinely FAILS on pre-fix code
Pre-fix tree = `a4ab12e`, detached worktree `/tmp/rev/prea4a`, final test file dropped in unchanged,
GPU-shaped world:
```
$ PATH=/tmp/rev/fakebin:$PATH python3 -m pytest -q tests/test_host_purity.py
11 failed in 0.14s                       EXIT=1
$ PATH=/tmp/rev/fakebin:$PATH python3 -m pytest -q          # whole pre-fix suite
18 failed, 254 passed, 11 skipped in 0.91s   EXIT=1
```
The 7 non-purity failures are **exactly** the coordinator's host set (5 CLI + 2 pins):
```
FAILED tests/test_cli_doctor_branches.py::test_init_text_mode_with_the_offline_cache
FAILED tests/test_cli_e1a.py::test_init_dry_run_json_prints_the_plan_and_writes_nothing
FAILED tests/test_cli_e1a.py::test_init_dry_run_text_says_so
FAILED tests/test_cli_e1a.py::test_init_from_offline_cache_with_a_poisoned_path
FAILED tests/test_cli_e1a.py::test_init_twice_is_idempotent
FAILED tests/test_pins.py::test_host_variant_mapping[auto-linux-x86_64-linux-x64-cpu]
FAILED tests/test_pins.py::test_detect_backend[probes2-vulkan]
```
(The 11 new tests also fail in a plain GPU-less world — same count, so the red is not an artifact of
the simulated GPU facts.)

### F. Duels' survivors at the as-dispatched head `3f18890` (my re-run, fight copies)
The fight `base/` is a byte-faithful copy of the reviewed tree for the relevant files
(`sha256 capability.py 05193de8…`, `pins.py 5ae4f0e1…`, `install.py 82b61ddd…`,
`finder.py 5926b87f…` all equal to HEAD; `tests/test_host_purity.py` equal to
`git show 3f18890:tests/test_host_purity.py`).
```
$ cd <fight>/mNN && python3 -m pytest -q tests/test_host_purity.py
m08 11 passed · m09 11 passed · m10 11 passed · m11 11 passed      (base: 11 passed)
$ cd <fight>/mNN && python3 -m pytest -q                            # full offline suite
m08 1 failed, 327 passed, 12 skipped   (tests/test_pins.py::test_probes_never_fall_back_to_the_real_host)
m09 1 failed, 327 passed, 12 skipped   (same test)
m10 328 passed, 12 skipped             <- SURVIVES the whole suite in a GPU-less world
m11 328 passed, 12 skipped             <- SURVIVES every gate
$ cd <fight>/base|m11 && python3 docs/verify_runtime_contract.py
failures: 0  skips: 3      (identical for base and m11 — the oracle does not catch m11)
$ python3 witnesses/w11.py base   ->  M11 capability.backends(<dir with libggml-cpu.so>, system=windows) -> []
$ python3 witnesses/w11.py m11    ->  M11 capability.backends(<dir with libggml-cpu.so>, system=windows) -> ['cpu']
```
Notes the attacker's report did not state:
- `m10`'s kill is **host-dependent**: on a GPU-less box the whole suite is green on the mutant,
  because `registry.recommend._query_nvidia_smi()` answers 0 there; only a real GPU makes
  `test_recommend_quant.py::test_host_budget_reads_meminfo` bite.
- `HEAD` `capability.backends()` **does** forward `system` (`capability.py:140`); the mutant is a
  plausible-fault probe, not a description of the shipped code. B1 is a test-strength finding.

### G. Referee re-run after `1c20ad4` — the survivors are dead (my script, fresh worktree)
```
$ git worktree add --detach /tmp/rev/head1c20 1c20ad4
$ bash /tmp/rev/killcheck.sh /tmp/rev/head1c20 <fight> /tmp/rev/kills
== m08 (patch applied) ==
FAILED tests/test_host_purity.py::test_an_unnamed_machine_is_a_caller_error_not_a_platform_machine_read
1 failed, 15 passed
== m09 (patch applied) ==
FAILED tests/test_host_purity.py::test_omitted_dri_nodes_never_list_the_real_dev_dri
1 failed, 15 passed
== m10 (patch applied) ==
FAILED tests/test_host_purity.py::test_an_empty_injected_vram_probe_never_reaches_the_real_driver
1 failed, 15 passed
== m11 (patch applied) ==
FAILED tests/test_host_purity.py::test_backends_answers_the_system_the_caller_named
FAILED tests/test_host_purity.py::test_backends_never_asks_this_host_for_a_system_the_caller_supplied
2 failed, 14 passed
== control: same file on unmutated HEAD ==
16 passed
```
The `m10` pin is trap-based (`monkeypatch.setattr(recommend, "_query_nvidia_smi", tripwire)`), so —
unlike the old killer — it bites in a GPU-less world too. That closes the host-dependence noted in §F.

Independent residual probe (mine, `/tmp/rev/residual_probe.py`, run on both the reviewed revision and
`m11`) confirms the injected fact wins in both directions after the fix:
```
capability.backends(distractor, system='linux')   -> ['cpu']        (m11: ['cpu']  — same, host is linux)
capability.backends(distractor, system='windows') -> ['vulkan']     (m11: ['cpu']  — the leak)
finder.library_glob('windows') with platform trapped -> '*ggml-*.dll'   (no host read)
capability.probe_symbols(system='windows') with platform trapped -> windows-shaped error, no host read
```

### H. Evidence-document check (`docs/evidence/e1a_baseline.json`)
- `e1a_fix_t_eae35404.host` exists with `round_1_host_result` (SIGABRT run) and
  `round_2_host_result` (all-green run: `init.exit 0`, `init.variant linux-x64-vulkan`,
  `init.fallback` = the pre-flight skip line, doctor 2, oracle 0 + `section_b_skips 0`, both suites 0,
  poisoned 0 / 0 shims), each with `recorded_by` + the source board comment, plus
  `container_corroboration` (worker `code-e2e t_3831b7b3`, raw logs under `.e2e/`).
- The sandbox section is present and **labelled**: `e1a_fix_t_eae35404.sandbox_host_simulated`
  (`vehicle`: podman container, no `/dev/dri`, no `/dev/nvidia*`, fake `nvidia-smi` on PATH; real
  downloads — e.g. the 30 294 625 B vulkan asset and the 4.38 GB model).
- Every live number in the file carries a `cmd` + a result/exit; the host block states explicitly that
  the raw host log dir `/home/rybens/.ggufone-host-gate-20260917T182117Z` is **not mounted** into the
  container, so it is quoted verbatim (no prose-only numbers; the limit is the mounting, not the
  reporting).
- Extra live-gate numbers that are command-backed in `.e2e/t_3831b7b3-host-gate/` (10/10 steps green,
  12 journeys, real 4.38 GB pull with sha check).

## 3. Findings by severity

### 🔴 B1 — CRITICAL — `capability.backends()` drops an injected `system` (m11 survives every gate)
- Found at `3f18890`; **closed at `1c20ad4`**; kill re-verified by me (§2-G).
- What it is: a caller stating `system="windows"` got Linux-shaped globs
  (`[]` vs `["cpu"]` on the witness). HEAD itself forwards the argument, so this was a hole in the
  *regression file*, not in shipped behaviour — but it is exactly the invariant E1a exists for
  ("a supplied fact must win"), and it was invisible to every gate.
- Fix landed (`1c20ad4`): `test_backends_answers_the_system_the_caller_named` (distractor bundle,
  both directions) + `test_backends_never_asks_this_host_for_a_system_the_caller_supplied` (tripwire
  on `pins.platform.system`). Both fail on the mutant, pass on HEAD.

### 🟠 B2 — MAJOR — omitted `machine` / omitted `dri_nodes` fell back to the real host (m08, m09)
- Found at `3f18890` (both survived the task-0 file in both worlds; killed only by `test_pins.py`);
  **closed at `1c20ad4`** (`test_an_unnamed_machine_is_a_caller_error_not_a_platform_machine_read`,
  `test_omitted_dri_nodes_never_list_the_real_dev_dri`); kill re-verified by me (§2-G).

### 🟠 B3 — MAJOR — `host_budget` fell through to the real driver on an empty injected probe (m10)
- Found at `3f18890`; the only killer was host-dependent (see §F), so in a GPU-less world the mutant
  survived the **whole suite**; **closed at `1c20ad4`** by a trap-based pin
  (`test_an_empty_injected_vram_probe_never_reaches_the_real_driver`), re-verified by me (§2-G).

### 🟡 M1 — MINOR — the older `vram` killer is still host-dependent (not blocking)
`tests/test_recommend_quant.py::test_host_budget_reads_meminfo` only detects an
`m10`-style fall-through on a box with a real `nvidia-smi`/GPU. The new pin makes the gate honest in
every world; consider giving the older test the same `_query_nvidia_smi` trap (one line) so the
detection does not depend on where the suite runs.

### 🟡 M2 — MINOR — two report-only host reads have no probe seam (not blocking)
`capability.platform_summary()` and `capability.host_expectation()` (`capability.py:390-403`, used by
`cli.py:220/251/374` for the `host`/`expected_backend` blocks) read the real machine with no injection.
That is their job (a report about the real box) and no test needs a foreign world there — but the
scope note in the module docstring should say so explicitly, since everything else in the detection
layer is injectable now.

### ⚪ N1 — NIT — wording of the adversarial headline
The duel's board comment says "`capability.backends()` gubi wstrzyknięty `system`", which reads as a
production bug; the mutant *is* the fault — HEAD forwards the argument. Keep the two apart in the fix
card so nobody goes looking for a production change.

### ⚪ N2 — NIT — evidence hygiene
`docs/evidence/e1a_t_0fc576df_host_access_mutants.md` is still **untracked** in the shared tree (the
duel wrote it after its last commit), and `state/groupchat/ggufone-e1.md` is dirty (the coordinator's
chat file — never stage it). The duel's register row is not yet in `state/fights.csv`; that plus the
referee register is card `t_83ee1eed`/@auditor territory. Coordinator: consider committing the report
before the workspace is cleaned.

## 4. Scope-creep / no-weakening review of `a4ab12e..1c20ad4`

- **Implementation was fixed, not the expectations.** The two originally-failing tests keep their
  assertions; across the whole range `tests/` has exactly **two deleted lines**, both import rewrites
  (`-from ggufone.runtime import pins`, `-from ggufone.runtime import finder`). Everything else in
  `tests/` is additions (`git diff --numstat a4ab12e..1c20ad4 -- tests/`: conftest 12/0,
  test_cli_doctor_branches 4/0, test_cli_e1a 36/0, test_host_purity 356/0, test_pins 117/1,
  test_probe_isolation 578/0, test_runtime_fallback 571/0, test_runtime_live 25/1).
- `tests/test_cli_e1a.py` / `test_cli_doctor_branches.py`: the autouse fixture now names the fake CPU
  machine those tests always assumed. Reviewed and accepted: the files test the *plan* mapping, the
  assertions are unchanged, and the GPU direction is covered by the new
  `test_init_dry_run_on_a_gpu_host_plans_the_pinned_cuda_bundle` plus the `test_pins.py` fake worlds.
- `pyproject.toml`: mutmut config only (test selection + `max_children 1 -> 8`); no pytest behaviour
  change.
- `runtime.lock`: pinned sha256 for `linux-x64-vulkan` / `linux-x64-cuda-12.8` + the new `system_libs`
  pre-flight map — addresses round-1 findings 1 and 2.
- `src/ggufone/runtime/isolated.py` + `probe_child.py` (new): the "one bundle per process" fix for the
  host SIGABRT. Reviewed for the security-relevant surfaces of the installer path: download/cache SHA-256
  verification still enforced (`E_SHA256_MISMATCH`), tar extraction uses `filter="data"`
  (py>=3.11.4; `TypeError` fallback documented), zip extraction pre-checks every member for traversal,
  `rmtree` targets are the plan's own staging/dest paths.
- No drive-by refactors, renames or reformatting spotted outside the touched modules.

## 5. Verification limits (honest list)

1. **No GPU / operator-host access from this worker** (same container class as the other cards):
   `nvidia-smi` absent, `/dev/dri` not creatable by `mknod` (EPERM), operator log dir not mounted.
   The card's "confirm green ON THE GPU HOST" criterion therefore rests on (a) the coordinator's
   quoted round-2 host run in the evidence file, (b) `t_3831b7b3`'s container corroboration of the
   same gate step-for-step on `eb1cde3`, and (c) my own green runs in the GPU-shaped world at the
   as-dispatched and final heads. I did **not** re-execute the suite on the RTX 3060 Ti myself.
2. The mutmut scores quoted by earlier cards were **not** re-run here (single-owner rule; the
   machine's pid cap makes full sweeps hostile). The gate that mattered for this review — the duel's
   11 hand-crafted mutants — **was** re-run by me, at both heads.
3. My GPU-shape simulation (a real `/dev/dri/renderD128` + `/usr/share/vulkan/icd.d` in the container)
   used during part of the review has been removed; `/dev/dri` and `/usr/share/vulkan` do not exist in
   the container now.
4. The workspace `.venv` was a dead symlink to a host interpreter; `uv run` rebuilt it in-container
   (gitignored, no effect on the repo).

## 6. Sign-off

At `1c20ad4`: canonical suite green in both worlds (`333 passed, 12 skipped`), oracle `failures: 0`
exit 0, the host's original 7 failures green, the regression file red on pre-fix code (`11 failed`)
and the whole pre-fix suite reproducing the host's `7 failed / 254 passed / 11 skipped` set, and all
four duel survivors killed with named node ids. **No blocking item remains. PASS.**

Reviewed head: `1c20ad4` — the tree was being edited by card `t_83ee1eed` during this run; if any
commit after `1c20ad4` touches `src/` or `tests/`, the kill re-run (§2-G) must be repeated for the
changed files.
