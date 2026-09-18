# E1c FIX (card t_e29734e6) — the fit gates stop reading the host's device state

Date: 2026-09-18 · Tier: M (no tier declared on the card → the default) · Tree: shared `main`
@ `7aa6ebb` + this card's commits · Vehicle: this container (RTX 3060 Ti visible to
`nvidia-smi`, held by sibling benches all afternoon: free VRAM measured 49–3133 MiB).

Card: `test_fit.py::test_the_cli_fit_command_returns_the_documented_json` asserted the **exact**
`warnings == ["W_FIT_ESTIMATED"]` list while the CLI read the real device. On the operator's busy
host (1631 MiB free of 8192) the same command legitimately warns `W_FIT_DOWNGRADE` /
`W_KV_TYPE_DOWNGRADE`, so the gate went red the moment the host stopped being quiet.

Raw logs for every number below: `.e2e/t_e29734e6-fit-cli-host-state/`.

## 1. The failure, reproduced digit for digit

The sandbox has the same card as the operator's host and a sibling benchmark was holding it:

```
# world at RED capture: 2026-09-18T14:09:05Z        (world_ambient_busy.txt)
# nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits (MiB)
8192, 161

tests/test_fit.py:349: AssertionError
E       AssertionError: assert ['W_KV_TYPE_D...IT_ESTIMATED'] == ['W_FIT_ESTIMATED']
E         At index 0 diff: 'W_KV_TYPE_DOWNGRADE' != 'W_FIT_ESTIMATED'
E         Left contains 2 more items, first extra item: 'W_FIT_DOWNGRADE'
1 failed, 906 passed, 41 skipped in 33.17s
```

Same assertion, same three-item list, same test line as the card's tail
(`red_ambient_suite_pre_fix.txt`). The extra skip versus the runs below is the sandbox's own
`HOME` (that capture predates the `~/.hermes/models` symlink the live gates need), not a world
difference — the 40-skip runs are the ones to compare.

## 2. The fix: every CLI fit gate pins the ONE host reader

`tests/conftest.py` grows one fixture, `pin_host_facts`: it replaces `fit.host_facts` — the single
function that reads the machine's RAM/device memory (the E1a rule, SPEC 2.2; nothing else in
`fit.py` reads the device) — with a named world, and **refuses to answer when the test also passes
explicit facts** (`meminfo_path=`, `device_probe=`, …), because two worlds in flight is a bug in
the test. This is the same pin `tests/test_fit_free_vram.py:284` and
`tests/test_fit_oom_recovery.py:490` already do inline; it is now named once for CLI-level gates.

* `tests/test_fit.py`
  * `test_the_cli_fit_command_returns_the_documented_json` pins an **idle** card and asserts
    `W_FIT_ESTIMATED` **is present** plus both downgrade marks **are absent**, and that
    `payload["host"]["vram_free_bytes"]` / `payload["host_fingerprint"]` are the pinned world's —
    a caller that bypasses the seam fails loudly instead of quietly reading the box.
  * new `test_the_cli_fit_command_warns_when_the_device_is_mostly_taken` pins **1.5 GiB free of
    8 GiB** (the band the operator's card reports under load) and requires *both*
    `W_KV_TYPE_DOWNGRADE` (the ladder moved) and `W_FIT_DOWNGRADE` (fewer layers than the nominal
    card), plus the free-memory note and the pinned host numbers.
  * the binary-path gate and the no-network gate pin too: they were reading the device for no
    reason (their assertions could not flip, but a fit gate on a GPU-less box must not depend on
    this box's mood).
* `tests/test_fit_live.py`: the three **plan-only** live gates take `host=_roomy_host()` (7 GiB
  free of 8 GiB) instead of `fit.host_facts()` — what they measure is the real binary and the real
  model, not the desktop.

No production line changed: the seam already existed and the CLI already went through it
(`cli.py` → `fit.host_facts()` in `_cmd_fit`/`fit_plan_for`, `plan_for_path` → `host or
host_facts()`). The fix is the *test's* determinism, stated here so a reviewer does not read it as
"loosened assertion": on the pinned world the new assertions are equivalent to the old exact list
(the estimate path can only carry those three codes) and strictly stronger about the host (the
fingerprint assertion pins the world).

## 3. Determinism: four worlds, two trees

`nvidia-smi` is replaced on `PATH` for the simulated worlds (`/work/shim-starved`, `/work/shim-roomy`
— `shutil.which` resolves to the shim, verified in each log header), so the *production* reader
(`recommend.device_memory`) reports that world; nothing about the function is mocked. Pre-fix runs
are the same tree with the three test files stashed (`git stash push -- tests/...`).

| world (`memory.total, memory.free`) | pre-fix tree | post-fix tree | log |
|---|---|---|---|
| ambient, 161 MiB free | **1 failed**, 906 passed, 41 skipped | — | `red_ambient_suite_pre_fix.txt` |
| ambient, 2073 MiB free | 908 passed, 40 skipped | 909 passed, 40 skipped | `red_ambient_pre_fix_suite.txt`, `green_ambient_busy_full_suite.txt` |
| shim "starved" 8192 / 1024 MiB | **1 failed**, 907 passed, 40 skipped | 909 passed, 40 skipped | `red_starved_world_pre_fix_suite.txt`, `green_starved_world_full_suite.txt` |
| shim "roomy" 8192 / 7168 MiB | 908 passed, 40 skipped | 909 passed, 40 skipped | `red_roomy_world_pre_fix_suite.txt`, `green_roomy_world_full_suite.txt` |

The pre-fix row that *passes* at 2073 MiB is the point of the card: before the fix the gate's
result was a function of the box's mood, not of the command (deterministic RED at 161/1024 MiB,
green at 2073 MiB), and after it the same 909/40 holds in every world.

A scratch probe (never committed — body kept in
`red_ambient_unpinned_probe_src.py.txt`, run in `red_ambient_unpinned_probe.txt`) writes the new
gate's assertions **with the pin removed** and fails on the ambient box for the right reason:

```
tests/test_zz_tripwire_scratch.py:51: AssertionError
E       assert 1326448640 == 1610612736        # the real card, not the pinned 1.5 GiB
```

### Requirement 4, literally: a run with free VRAM ≤ 2 GiB staying green

* The ambient world *was* ≤ 2 GiB for the whole campaign (161, 1275, 2073 MiB — the card itself
  measured 1631 MiB), and the post-fix suite is green in it.
* The starved world makes it deterministic instead of lucky: 1024 MiB free, **909 passed / 40
  skipped**, where the pre-fix tree fails.

## 4. The live plan-only gates (`--run-network`, real 4B model + real `b11026` vulkan bundle)

```
pre-fix, ambient at 1275 MiB free:  2 failed, 1 passed   (red_live_planonly_ambient_pre_fix.txt)
post-fix, ambient at 1275 MiB free: 3 passed             (live_planonly_ambient_post_fix.txt)
```

Pre-fix failure is the same shape as the card: `assert ('W_KV_TYPE_D...IT_ESTIMATED') ==
('W_FIT_ESTIMATED',)`. Post-fix the plan is still built by the real binary —
`source: llama-fit-params`, `n_gpu_layers 36`, `est_weights_bytes 4369416192` (the tensor index
agrees within 217 KiB) — only the *host* is pinned.

Measured, not changed: `test_the_live_answer_carries_the_template_and_the_fit_plan` (ambient
3133 MiB free) passes as committed — `live_template_fit_ambient.txt`; see §6 for why it is left
alone.

## 5. Sweep of `tests/` (requirement 3)

`tools/host_state_audit.py` (new, committed) lists every location of the three shapes; its raw
output is `sweep_host_state_audit_post_fix.txt`. Classification:

**Found and fixed (6 sites, all "free-VRAM reads" on a path with an assertion):**

| site | shape |
|---|---|
| `test_fit.py` `test_the_cli_fit_command_returns_the_documented_json` | **exact warning list** read from the device (the card) |
| `test_fit.py` `test_the_cli_fit_command_runs_a_binary_when_one_exists` | ambient host read (assertions could not flip) |
| `test_fit.py` `test_fit_never_touches_the_network` | ambient host read (same) |
| `test_fit_live.py` `test_the_pinned_default_model_gets_a_plan_from_the_binary` | ambient host read **+ exact `warnings == ()`** |
| `test_fit_live.py` `test_the_tensor_index_matches_the_binary_s_model_row` | ambient host read |
| `test_fit_live.py` `test_the_estimate_alone_still_answers_the_contract` | ambient host read **+ exact warning tuple** |

**Found, deliberately left (device tests by design — they load the real 4B model and need a card
with room):** `test_the_fit_estimate_cross_checks_against_measured_load_rss`,
`test_the_plan_is_applied_on_load_unless_no_fit`, `test_a_no_fit_run_keeps_the_requested_settings`,
`test_the_live_answer_carries_the_template_and_the_fit_plan`,
`test_a_busy_desktop_plan_loads_on_the_free_reading` (pins its own 1112 MiB world),
`test_the_log_capture_swaps_and_restores_the_real_handler`. Their *plans* come from the ambient
device (no `host=`), which is the honest way to measure the load on a box — except the template
gate, handled in §6.

**Found, different shape (not this card's bug class):**
`test_bench_live.py:78,193` read `harness.host_facts()`, which reports **cgroup** facts
(`cpu.max`, `memory.max`), not device memory; the values feed a skip reason and a printed line.
`tests/test_fit_free_vram.py`, `tests/test_fit_oom_recovery.py`, `tests/test_bench_placement.py`,
`tests/test_e1c_mutation_pins.py`, `tests/test_routing.py`, `tests/test_recommend_quant.py` and
`tests/test_host_purity.py` already build their worlds explicitly (`device_probe=`, `vram_probe=`,
`free_probe=`, `HostFacts(...)` literals) and were left untouched. Non-fit warning assertions
(`test_schema`, `test_templates`, `test_registry_store`) never touch host memory.

## 6. Gates

**Full offline suite on the committing tree** (fresh run, same command as the coverage run below):

```
909 passed, 40 skipped in 209.29s   (exit 0)
```

and the same 909/40 in all three worlds of §3 (`green_*_full_suite.txt`; re-measured on the
committing revision in the starved world: `green_starved_world_final_rev.txt`). `+1` versus the
pre-fix 948-test tree is the new busy-device gate; the skip count is identical (40) in every
post-fix run.

**Static analysis** (`uv run --extra dev ruff check .`): the three changed files are clean. The 12
remaining findings are pre-existing and none is in this diff: `E741` in
`.e2e/t_34abf324-e1b/logs/survivor-buckets.py`, five `E501` in
`.e2e/t_c8e36cad-e1c/logs/mutation_triage.py`, six `E501` in `tools/quick_breakdown.py`.

**Coverage** (`--cov=ggufone.runtime.fit`, `coverage_fit_module.txt`): the module the changed
gates exercise is at **97 %** (461 statements, 16 missed). The diff itself adds **no production
statement**, so "changed production lines" is 0/0 by construction; the three changed test modules
are executed whole by the suite (they are inside the 909).

**Mutation testing (Tier M, one run, soft).** Repo-documented fit sweep — `source_paths =
["src/ggufone/runtime/fit.py"]`, selection `tests/test_fit.py` + `tests/test_fit_free_vram.py` +
`tests/test_fit_oom_recovery.py`, `max_children = 2`, run through `tools/mutmut_driver.py`:

```
1884/1884 mutants, 3.76 mutations/s, exit 0, not-run 0
killed 1157   survived 707   no-tests 20        score killed/scored = 62.1 %
```

The E1c row for the same module was `1639 mutants … 69.3 %`, and the difference is **not** this
diff (which moves no production line): `fit.py` grew by 245 mutants from code later cards added —
`git log -S` puts `coerce_plan`, `planned_layers` and `degrade_ladder` in `8d4fc9f` (card
t_31b3943a), whose sweep mutated the *bench* package — and those symbols alone hold 247 of the 707
survivors (`coerce_plan` 212, `degrade_ladder` 29, `planned_layers` 6). Survivor mass by symbol:
`coerce_plan` 212, `ModelFacts.read` 47 + `read_tensor_index` 58, `replan_for_host` 51,
`estimate_plan` 37, `backend_oom_error` 34, `run_llama_fit_params` 26, `plan_for_model` 26,
`host_facts` 22, `plan_from_binary` 22, `_kv_from_budget` 18. The two numbers are therefore not
comparable as a trend, and no mutant of this diff exists to score.

**Which makes the targeted proof the one that matters** (`hand_mutation.txt`,
`mutation_lines_of_interest.txt`):

* hand mutations of the branches the changed gates pin, all **KILLED** by the changed selection:
  `M1` drop `W_FIT_ESTIMATED`, `M2` never warn `W_FIT_DOWNGRADE`, `M3` drop
  `W_KV_TYPE_DOWNGRADE`, `M4` invert the ladder trigger.
* from the tool run, the mutants whose mutated line *is* one of those branches inside
  `estimate_plan` (recovered by diffing each mutant's copy against its `__mutmut_orig` copy):
  `warnings.append("W_FIT_ESTIMATED")` 3/3 killed, `warnings.append("W_FIT_DOWNGRADE")` 3/3
  killed, `warnings.append("W_KV_TYPE_DOWNGRADE")` 3/3 killed.
* survivors found on *other* warning lines are pre-existing and named: the
  `"W_FIT_DOWNGRADE" not in warnings` / `"W_KV_TYPE_DOWNGRADE" not in warnings` **dedupe-key**
  string mutants in `degrade_ladder` (near-equivalent: the dedupe branch rarely runs) and
  `plan_from_binary`'s `W_KV_TYPE_DOWNGRADE` append, which
  `tests/test_e1c_mutation_pins.py::test_the_binary_plan_flags_a_downgrade_when_the_ladder_moved`
  does pin — that file is simply not in the fit sweep's selection (worth adding to the next
  sweep's pair, not worth changing this card's comparability for).

## 7. QA note — risk and recommendation

🔴 **Nothing blocks.** The card's acceptance is met with the box busy throughout: the CLI gate is
now a function of a pinned world (909/40 in the starved, ambient and roomy worlds versus
RED/GREEN depending on the minute before), the ≤2 GiB world is in the table, and the live
plan-only gates went 2-failed → 3-passed on the same box.

🟡 **Worth knowing (reviewer's call):**
1. *The fix is test-only.* No production line moved: the CLI already read the host through the one
   function (`fit.host_facts`), so the card's "inject/freeze the host facts" is a *test* seam, and
   the assertions were not weakened — the pinned idle world asserts the same thing the old exact
   list did, plus the host dict/fingerprint tripwire.
2. *The mutation score of the module went 69.3 % → 62.1 %*, and it is not attributable to this
   change (see §6: the module grew, the extra survivor mass sits in symbols added by later cards
   whose sweeps mutated other packages). If the reviewer wants a trend number, the honest
   follow-up is a sweep scoped to the symbols added after E1c, or adding
   `tests/test_e1c_mutation_pins.py` to the selection.

🟢 **Accepted, verified:** 6 host-reading sites pinned; 4/4 hand mutations killed; the 9 tool
mutants of the three pinned warning branches killed; ruff clean on the diff; 97 % coverage of the
module under test.

**Not verified here (and why), for the record:** the *device-loading* live gates
(`plan_is_applied_on_load…`, `no_fit_run…`, RSS cross-check, template+fit, busy-desktop-load,
log-capture) need a card with room and are device tests by design; the template+fit gate was
measured passing on the busy box (`live_template_fit_ambient.txt`), but its forbidden
`W_KV_TYPE_DOWNGRADE` is reachable on a host whose Vulkan device is *visible and partially held*
(this container logs `ggml_vulkan: No devices found`, so the load-time KV rung cannot be exercised
here). Changing that assertion without a quiet-device run would ship an unmeasured claim — the
vehicle for closing it is the operator's host during a busy window, or a follow-up card with
reserved GPU time.

