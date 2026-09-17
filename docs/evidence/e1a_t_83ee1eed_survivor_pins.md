# E1a FIX `t_83ee1eed` — the duel's four survivors, pinned (B1/B2/B3)

- Card: `t_83ee1eed` (code-tdd; defender side of duel `t_0fc576df`). **ADD-only, test file only** —
  no production change was required (B1 is a test-strength hole: HEAD `capability.py:138` *does*
  forward `system`; the mutant is the fault).
- Source report: `docs/evidence/e1a_t_0fc576df_host_access_mutants.md` (§6 B1/B2/B3); fight dir
  `/home/rybens/workspace/state/fights/e1a-t0fc576df/` (mutants, patches, witnesses, logs).
- Landed: `1c20ad4` on the shared tree (= the head final review `t_ee24bd7c` PASSed; `REVIEW.md` §2-G
  re-ran these kills independently).
- Runner for every number below: `uv run` (uv 0.12.0) + pytest 9.1.1, CPython 3.11.15, podman
  container, **GPU-less** (`nvidia-smi` absent, `/dev/dri` absent) — i.e. the duel's *world B*.
  The world-A (RTX 3060 Ti) re-run is the referee's (§4.5: only re-runs count) — see §6.

## 1. What landed

`tests/test_host_purity.py` only: `+126 / -1`, the single deletion being an import line rewritten
into three (`from ggufone.runtime import install, pins` →
`from ggufone.errors import RuntimeMissingError` + `from ggufone.registry import recommend` +
`from ggufone.runtime import capability, install, pins`). Fair-play §4.1 check:

```
$ git diff 3f18890..1c20ad4 -- tests/ | grep -E "^-" | grep -v "^---"
-from ggufone.runtime import install, pins
$ git diff --numstat 3f18890..1c20ad4 -- tests/test_host_purity.py
126     1       tests/test_host_purity.py
```

The five new kill assertions (exact node ids):

| mutant | node id (new) |
|---|---|
| m11 (B1) | `tests/test_host_purity.py::test_backends_answers_the_system_the_caller_named` |
| m11 (B1) | `tests/test_host_purity.py::test_backends_never_asks_this_host_for_a_system_the_caller_supplied` |
| m08 (B2) | `tests/test_host_purity.py::test_an_unnamed_machine_is_a_caller_error_not_a_platform_machine_read` |
| m09 (B2) | `tests/test_host_purity.py::test_omitted_dri_nodes_never_list_the_real_dev_dri` |
| m10 (B3) | `tests/test_host_purity.py::test_an_empty_injected_vram_probe_never_reaches_the_real_driver` |

Design notes (each pin is world-independent by construction, so it also bites on the operator's
GPU box and on a Windows dev box — the old killers did not):

* **B1** distractor bundle (`libggml-cpu.so` + `ggml-vulkan.dll`) asserted in **both** directions,
  plus a `platform.system` tripwire through the same call — either assertion alone is
  host-dependent (on a Windows host the mutant satisfies the "windows" one).
* **B2/m08** `platform.machine` tripwire + `host_variant("auto", system="linux")` must raise the
  caller error (`RuntimeMissingError`), never plan.
* **B2/m09** `detect_backend(system=…, has_nvidia_smi=False, icd_dir=<existing>)` on a world that
  really carries a render node answers `cpu`, plus a `DRI_DIR` Trap.
* **B3/m10** `host_budget(nvidia_smi=lambda: None)` stays `vram_bytes == 0` with
  `recommend._query_nvidia_smi` replaced by a tripwire — this is the pin that closes the
  host-dependence the reviewer flagged (`REVIEW.md` §2-F/§M1): on this GPU-less box the old killer
  `test_recommend_quant.py::test_host_budget_reads_meminfo` is green even on the mutant.

## 2. Protocol (fresh copies, not the fight dir's own copies)

Scripts + raw logs: `.e2e/t_83ee1eed-survivor-pins/` (`scripts/`, `logs/`).

```
$ git -C /workspace/ggufone archive HEAD | tar -x -C <copy>     # HEAD = 1c20ad4, 5 copies
$ ( cd <copy> && git apply /workspace/state/fights/e1a-t0fc576df/patches/mNN.patch )
patched: m08 / m09 / m10 / m11
```

Provenance — the mutated modules in my copies are byte-identical to the duel's manifest
(`MUTANTS.md`, first-16 sha256):

```
$ sha256sum <copy>/src/ggufone/... | cut -c1-16
m08 pins.py         aea6c1c5c3aa63c8   (fight copy + MUTANTS.md: aea6c1c5c3aa63c8)
m09 pins.py         8729be0fee3423c6   (fight copy + MUTANTS.md: 8729be0fee3423c6)
m10 recommend.py    a2f45d53898dde9f   (fight copy + MUTANTS.md: a2f45d53898dde9f)
m11 capability.py   04d555f57161afd4   (fight copy + MUTANTS.md: 04d555f57161afd4)
$ sha256sum patches/m08.patch … m11.patch
39a56a702590da7e… 70330fe0c950ed8a… cb99717b2cfaf4f1… 33664f93eddaca82…
```

Each copy carries a temporary `tests/_whoami_test.py` asserting the copy's own `src` is the import
source (the shared tree's editable install is the same package name); it passed in all five copies
(`logs/mNN.whoami.log`).

## 3. Raw results

### 3a. The named gate (`tests/test_host_purity.py`, updated) — PASS on HEAD, FAIL on each mutant

```
$ ( cd <copy> && uv run --project /workspace/ggufone pytest -q -p no:cacheprovider tests/test_host_purity.py -rf )
base: 16 passed in 0.62s
m08:  FAILED tests/test_host_purity.py::test_an_unnamed_machine_is_a_caller_error_not_a_platform_machine_read
      1 failed, 15 passed in 0.50s
m09:  FAILED tests/test_host_purity.py::test_omitted_dri_nodes_never_list_the_real_dev_dri
      1 failed, 15 passed in 0.50s
m10:  FAILED tests/test_host_purity.py::test_an_empty_injected_vram_probe_never_reaches_the_real_driver
      1 failed, 15 passed in 0.22s
m11:  FAILED tests/test_host_purity.py::test_backends_answers_the_system_the_caller_named
      FAILED tests/test_host_purity.py::test_backends_never_asks_this_host_for_a_system_the_caller_supplied
      2 failed, 14 passed in 0.32s
```

Attribution is single: each mutant fails exactly its own pin (m11 fails two, both its own). Tails in
`logs/{base,m08,m09,m10,m11}.gateA.log`; the failing assertions are the tripwire on
`platform.machine` (m08), `assert 'vulkan' == 'cpu'` (m09), the `_query_nvidia_smi` tripwire (m10),
`assert ['cpu'] == ['vulkan']` + the `platform.system` tripwire (m11).

### 3b. Full offline suite on the same copies

```
$ ( cd <copy> && uv run --project /workspace/ggufone pytest -q -p no:cacheprovider --ignore=tests/_whoami_test.py -rf )
base:  333 passed, 12 skipped
m08:   2 failed, 331 passed, 12 skipped   (new pin + tests/test_pins.py::test_probes_never_fall_back_to_the_real_host)
m09:   2 failed, 331 passed, 12 skipped   (new pin + the same test_pins.py tripwire)
m10:   1 failed, 332 passed, 12 skipped   (the new pin — the only killer in a GPU-less world)
m11:   2 failed, 331 passed, 12 skipped   (the two new B1 pins; before this fix m11 was 328 passed — zero failures)
```

### 3c. The five new node ids on unmutated HEAD

```
$ ( cd base && uv run --project /workspace/ggufone pytest -q -p no:cacheprovider -v <5 node ids> )
tests/test_host_purity.py .....                                          [100%]
5 passed in 0.14s                        (logs/base.new_nodes.log)
```

### 3d. The duel's own witnesses on my copies (behaviour delta)

```
$ python3 …/witnesses/w11.py <copy>
base [] · m08 [] · m09 [] · m10 [] · m11 ['cpu']      # reproduces the report's witness exactly

$ python3 …/witnesses/w10.py <copy>
base: vram_bytes=0 · m10: vram_bytes=0                # in THIS container: no delta — no nvidia-smi
```

3d is why m10 needs the trap-based pin: the duel's witness (and the old killer) only differ on a box
with a real driver; my pin fails the mutant on a GPU-less box too (`REVIEW.md` §M1 closes the same
host-dependence).

## 4. Canonical gates at `1c20ad4`

```
$ cd /workspace/ggufone && uv run pytest -q
333 passed, 12 skipped in 20.78s                         EXIT=0
$ uv run python docs/verify_runtime_contract.py
failures: 0  skips: 3                                    EXIT=0
$ uv run ruff check tests/test_host_purity.py
All checks passed!                                       EXIT=0
```

Runner note (honesty): the duel's base counts (`329 passed, 11 skipped`, hermes-venv runner,
pytest 7.4.3) are not the canonical runner's; in this container the canonical gate collects 340 and
its own pre-fix base is `328 passed, 12 skipped` (`--k` deselect of the five new node ids), i.e.
345 collected / 333 passed after the pins. The 12 skips are all network/model/runtime-absent marks
(§`-rs`), none of them a new pin.

## 5. Tier-M QA note (card declares no tier → default M)

- **Changed files:** `tests/test_host_purity.py` only (test-only; production behaviour unchanged by
  design — the card forbids a production change and HEAD already forwards `system`).
- **Mutation testing (single-owner rule):** the mutation campaign for this change-set *is* the duel —
  11 hand-crafted mutants, each with a witness, all byte-identical to `MUTANTS.md`. I did **not**
  re-run them as a "score" and did not run mutmut: the referee re-run is the auditor's (§4.5), and
  mutmut has no production line to mutate in a test-only diff. Reported as *exercised*, not as a
  fresh score: named gate now kills 11/11 (this box), 4/4 pinned survivors verified killed.
- **Coverage:** every new assertion executes on HEAD (5/5 pass) and is verified to fail on its
  mutant — the pins are the new code, and each one has a demonstrated kill. Package line coverage is
  unchanged (no `src/` diff).
- **Static analysis:** ruff clean on the changed file (0 new findings).
- **Risks:** (a) the world-A re-run is outstanding — external, referee-owned (§6); (b) two review
  minors (M1 older host-dependent vram killer; M2 report-only host reads without a seam) are
  untouched by this card because a commit touching `tests/`/`src/` after the reviewed head would
  invalidate `REVIEW.md`'s re-run (its §6) — routed to the coordinator in §6.
- **Recommendation:** accept — the four pinned survivors are killed by named node ids, the gate the
  duel named is now honest in every world, and no existing assertion was weakened.

## 6. Limits and handoff (what this file does NOT claim)

1. **World A (operator's RTX 3060 Ti) was not re-run here.** No GPU/`/dev/dri`, no operator host —
   the duel's own world-B limit. The referee (auditor) re-run is the only count that matters
   (§4.5); independent third-party evidence already exists at `REVIEW.md` §2-G (reviewer's fresh
   worktree, fight patches applied, same kills, control green), but that is a reviewer run, not the
   referee's.
2. **Register row not written.** `state/fights.csv` is the referee's ledger; a row with
   self-reported numbers would violate §4.5. Draft row (placeholder → the auditor's numbers):

   ```
   t_83ee1eed,mutation,/var/home/rybens/workspace/ggufone,2,attacker(ds),defender(glm),auditor,11,7,4,4,4,NA,NA,closed,,duel t_0fc576df: 4 survivors (m08/m09/m10/m11) pinned ADD-only in tests/test_host_purity.py @1c20ad4; referee re-run pending (this row to be confirmed by @auditor)
   ```

3. **Production leak stays latent.** No production caller injects a foreign `system` today
   (`doctor` uses `system=None`); the pin is the fix the card asked for. A future cross-platform
   caller now fails the gate instead of failing silently.
4. **Untracked source report.** `docs/evidence/e1a_t_0fc576df_host_access_mutants.md` was untracked
   at the reviewed head (`REVIEW.md` §N2); it is committed together with this file, unmodified.

## 7. Reproduction (container)

```bash
bash .e2e/t_83ee1eed-survivor-pins/scripts/setup_mutant_copies.sh   # fresh copies + patches + whoami test
bash .e2e/t_83ee1eed-survivor-pins/scripts/run_mutant_gate.sh      # gateA per copy
bash .e2e/t_83ee1eed-survivor-pins/scripts/run_mutant_suites.sh    # full suite per copy + 5 node ids on base
```
