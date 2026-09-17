# t_1b4632de — injected probes never touch the real host: verification log

Card: *Refactor ggufone detection so injected probes never touch the real host* (index 0 of the
E1a FIX decomposition of `t_eae35404`). Commit `e1ea5b0` on `main`, plus this file.

The seam itself landed with the parent card (`f3ae67d`: probes →
`ggufone.runtime.pins.HostProbes`, `current_host()` as the single real-host reader). This card
owns the regression that keeps it honest — and the pre-fix demonstration the reviewer asks for.

**This container has no GPU**: no `nvidia-smi`, no `/dev/dri`. The GPU host is therefore
*synthesised at the OS level*, which is exactly the difference between the sandbox the E1a
worker verified in (where the bug was invisible) and the operator's box (where 7 tests failed):

```
$ command -v nvidia-smi ; ls /dev/dri
which-exit=2
ls: cannot access '/dev/dri': No such file or directory
```

## 1. `uv run pytest -q` is green offline (acceptance 1)

```
$ uv run pytest -q
318 passed, 12 skipped in 3.85s          EXIT=0
```
307 passed / 12 skipped before this card; the 11 new tests are `tests/test_host_purity.py`.

## 2. Green on a host that *looks* like the operator's RTX 3060 Ti

Both readers the pre-fix code used are made real, not monkeypatched: an executable `nvidia-smi`
first on `PATH` and a real `/dev/dri/renderD128` node (created as root in this container, removed
again afterwards). The production path (`detect_backend()` with no arguments) then sees a GPU box.

```
$ cat fakebin/nvidia-smi
#!/bin/sh
echo 'NVIDIA GeForce RTX 3060 Ti, 8192 MiB'

$ mkdir -p /dev/dri && touch /dev/dri/renderD128

$ PATH=/work/t1b4/fakebin PYTHONPATH=$PWD/src .venv/bin/python -m pytest -q
318 passed, 12 skipped in 8.98s          EXIT=0
```

The two node ids the coordinator saw fail on the host, on that simulated box:

```
$ PATH=/work/t1b4/fakebin ... pytest tests/test_pins.py::test_detect_backend \
      tests/test_pins.py::test_host_variant_mapping -v
tests/test_pins.py::test_detect_backend[probes2-vulkan] PASSED                        [ 20%]
tests/test_pins.py::test_host_variant_mapping[auto-linux-x86_64-linux-x64-cpu] PASSED [ 60%]
============================== 15 passed in 1.70s ==============================
```

## 3. The new regression test FAILS on pre-fix code (acceptance 2)

Pre-fix tree = `a4ab12e` (the head the card names), cloned to `/work/t1b4/prefix-a4ab12e`, with the
final `tests/test_host_purity.py` dropped in **unchanged**, RTX world active:

```
$ PATH=/work/t1b4/fakebin ... pytest tests/test_host_purity.py -q
11 failed in 0.19s                       EXIT=1
```

and the whole pre-fix suite in that world reproduces the host report exactly:

```
$ PATH=/work/t1b4/fakebin ... pytest -q
18 failed, 254 passed, 11 skipped        EXIT=1
host (card):   7 failed, 254 passed, 11 skipped      <-- 254 + 11 match, 18 = 7 + our 11
```

The 7 — the exact set from the host, including the card's signature:

```
FAILED tests/test_cli_doctor_branches.py::test_init_text_mode_with_the_offline_cache
FAILED tests/test_cli_e1a.py::test_init_dry_run_json_prints_the_plan_and_writes_nothing
FAILED tests/test_cli_e1a.py::test_init_dry_run_text_says_so
FAILED tests/test_cli_e1a.py::test_init_from_offline_cache_with_a_poisoned_path
FAILED tests/test_cli_e1a.py::test_init_twice_is_idempotent
FAILED tests/test_pins.py::test_host_variant_mapping[auto-linux-x86_64-linux-x64-cpu]
FAILED tests/test_pins.py::test_detect_backend[probes2-vulkan] - AssertionErr...
...
E       AssertionError: assert 'cuda' == 'vulkan'
```

The 11 new tests that fail on that tree (this list is from the final file; the suite run above used
its first revision — same 11 tests, one renamed, so the counts are identical):

```
FAILED tests/test_host_purity.py::test_each_simulated_world_really_looks_like_that_box[cpu-cpu]
FAILED tests/test_host_purity.py::test_each_simulated_world_really_looks_like_that_box[vulkan-vulkan]
FAILED tests/test_host_purity.py::test_each_simulated_world_really_looks_like_that_box[cuda-cuda]
FAILED tests/test_host_purity.py::test_the_mapping_matrix_does_not_move_with_the_real_host[cpu]
FAILED tests/test_host_purity.py::test_the_mapping_matrix_does_not_move_with_the_real_host[vulkan]
FAILED tests/test_host_purity.py::test_the_mapping_matrix_does_not_move_with_the_real_host[cuda]
FAILED tests/test_host_purity.py::test_the_gpu_absent_world_never_yields_cuda
FAILED tests/test_host_purity.py::test_the_vulkan_world_with_supplied_probes_answers_vulkan
FAILED tests/test_host_purity.py::test_an_environment_naming_a_gpu_does_not_move_the_mapping
FAILED tests/test_host_purity.py::test_the_injected_path_reads_no_host_fact_from_the_environment
FAILED tests/test_host_purity.py::test_the_injected_path_reads_no_device_environment_and_runs_no_nvidia_smi
```

## 4. The card's two claims, on the exact expressions (acceptance 3)

`/work/t1b4/card-claims.py` (raw JSON kept beside it) evaluates them against four different hosts.

| world | `detect_backend(<vulkan probes>)` | `host_variant('auto', system='linux', machine='x86_64')` | `detect_backend()` (production) |
| --- | --- | --- | --- |
| plain sandbox (no GPU facts) | `vulkan` | `linux-x64-cpu` | `cpu` |
| simulated Vulkan host (`/dev/dri/renderD128` + real ICDs) | `vulkan` | `linux-x64-cpu` | `vulkan` |
| simulated RTX 3060 Ti (above + fake `nvidia-smi`) | `vulkan` | `linux-x64-cpu` | `cuda` |
| **pre-fix `a4ab12e`**, simulated RTX (control) | **`cuda`** | **`linux-x64-cuda-12.8`** | `cuda` |

So both claims hold on every host — including the most adversarial world available here, which is
the operator's own configuration — while the production path still reads the real machine
(`cpu` → `vulkan` → `cuda` as the box changes). The pre-fix control prints exactly the two
symptoms the card reports.

```
$ PATH=fakebin PYTHONPATH=/work/t1b4/prefix-a4ab12e/src ... prefix-claims.py
{ "claim_1_detect_backend_vulkan_probes": "cuda",
  "claim_2_host_variant_auto": "linux-x64-cuda-12.8", ... }
```

## 5. What the new file pins (`tests/test_host_purity.py`, 11 tests)

* three real-host worlds (cpu / vulkan / cuda) and a *non-vacuous* check that `current_host()`
  — the production reader — really sees each one;
* the whole mapping (detect_backend → host_variant → pinned asset → `plan_install`) under every
  world: the answer may not move with the box, and a GPU-absent world may never answer `cuda`;
* `PATH` is **replaced, not extended**, so the operator's real `nvidia-smi` cannot leak into the
  GPU-absent world (the "add a fake nvidia-smi to PATH" rehearsal cannot see that leak);
* no host-fact environment read (a recording `os.environ` spy). The assertion is "no key that
  could name a host fact", not "no read at all" — the harness itself reads the environment
  (a mutation run adds `MUTANT_UNDER_TEST` and re-reads the mapping), which the first version of
  this test learned the hard way under mutmut;
* no `nvidia-smi` subprocess: tripwires on `subprocess.run/Popen/check_output/check_call/call`.

## 6. Static + contract gates

```
$ ruff check src tests tools docs
All checks passed!

$ python3 docs/verify_runtime_contract.py
failures: 0  skips: 3                     EXIT=0
```
(the 3 skips are section B, which needs a runtime installed in this container; the live host run
is `t_3831b7b3`'s).

## 7. Tier-M note

This card's diff has **no production source** — `tests/test_host_purity.py` (new) and one line in
`pyproject.toml` (mutmut test selection). There is no mutation surface of its own; the Tier-M
score for the fix's source belongs to the parent card (66.0% over five modules, `e1a_qa.md`).

A scoped `mutmut run 'ggufone.runtime.pins*'` was started anyway. It swept **6604 mutants for
`pins.py` alone** (~3 h at this container's 256-pid cap) and was stopped at 148:
**127 killed / 20 survived / 1 other = 86.5% killed on the completed subset**. That is a partial
sample, **not a score**. Per-mutant kill evidence for re-introduced host reads belongs to
`t_0fc576df` (attacker).

## 8. Commit / push

```
$ git log --oneline -1
e1ea5b0 test(E1a t_1b4632de): the mapping matrix is a pure function of its probes

$ git remote -v
(no output)

$ git push
fatal: No configured push destination.
```

Committed locally on `main` (the branch every sibling E1a card works on). **There is no remote to
push to** — the repo is local-only, and `gh` is not installed in this container
(`bash: gh: command not found`).
