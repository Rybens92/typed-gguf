# E1a — adversarial verification: hidden host access on the injected-probes path

- Card: `t_0fc576df` (attacker). Artifacts under attack: the probe-injection refactor (`f3ae67d`),
  the injectable-probe regression `tests/test_host_purity.py` (`e1ea5b0`), the fallback surface
  (`68b0e8d`); tree root when the fight ran: `eb1cde3` (local `main`).
- Fight dir (mutants, patches, witnesses, raw logs): `/home/rybens/workspace/state/fights/e1a-t0fc576df/`.
- **Verdict: 11 hand-crafted mutants, every one with a witnessed behaviour delta.**

| gate (exact command below)                                             | killed | survived (of 11) |
|------------------------------------------------------------------------|--------|------------------|
| task-0 file `tests/test_host_purity.py` — world A (real RTX 3060 Ti)   | 7      | m08 m09 m10 m11  |
| task-0 file — world B (GPU-absent mount namespace)                     | 7      | m08 m09 m10 m11  |
| full offline suite — world A                                           | 10     | **m11**          |
| runtime-contract oracle (`docs/verify_runtime_contract.py`)            | 0      | m11 (output identical to base) |

- m08/m09/m10 survive the task-0 file but are killed by other suite files (evidence in §4).
- **m11 survives every gate in the repository → BLOCKING finding, §6-B1.**
- No mutant is equivalent: each has a witness input where the mutant's answer differs from HEAD's.
- The attack wrote only to fight copies; the repo under attack received no `src/`/`tests/` edit.

## 1. Environment and harness

The fight ran on the operator's RTX 3060 Ti host — the exact world the original bug needed:

```
$ nvidia-smi -L ; ls /dev/dri ; ls /usr/share/vulkan/icd.d | grep nvidia
GPU 0: NVIDIA GeForce RTX 3060 Ti (UUID: GPU-fde53142-3c4b-aea3-b86b-9929f8b242ca)
by-path  card1  renderD128
nvidia_icd.i686.json  nvidia_icd.x86_64.json
```
`/usr/sbin/nvidia-smi` (→ `/usr/bin/nvidia-smi`, ostree symlink) is a real executable; `PATH`
contains `/usr/sbin`. Real device env in the session: `VK_INSTANCE_LAYERS=…` only.

Runner: `/var/home/rybens/.hermes/hermes-agent/venv/bin/python` (Python 3.11.15, pytest 7.4.3),
no `ggufone` installed in it — each copy's `pyproject.toml` (`pythonpath = ["src"]`) makes the
copy's own `src` the import source (verified: the venv has no ggufone; every mutant effect below
proves the mutated module was the one imported).

Base (pristine HEAD copy) controls, all three gates green:

```
$ cd .../base && python -m pytest -q -p no:cacheprovider tests/test_host_purity.py
11 passed in 0.04s
$ cd .../base && python -m pytest -q -p no:cacheprovider
329 passed, 11 skipped in 16.45s
$ cd .../base && python docs/verify_runtime_contract.py
failures: 0  skips: 1        (exit 0)
```
(The repo's own baseline on this host via `uv run pytest -q tests/test_host_purity.py`: `11 passed`.)

World B (the card's "simulated GPU-absent world") is a mount namespace:
`unshare --map-root-user --mount bash tools/gate_worldB.sh <copy>` mounts a tmpfs over `/dev/dri`
and a non-executable empty file over `/usr/bin/nvidia-smi`, then runs the task-0 file. Proof lines
from every world-B run (`logs/m08.gateB.log`):

```
worldB: /dev/dri entries = 0
worldB: /usr/sbin/nvidia-smi not executable (masked)
shutil.which('nvidia-smi') -> None
Path('/usr/sbin/nvidia-smi').exists() -> True      # residual: the path still exists
/dev/dri renderD* -> []
```

## 2. Method (mutant-selection protocol, §8.1)

Fight copies: `tools/make_mutants.py <fight-dir>` copies `base/` per mutant and applies one exact
text edit (each anchor asserted to occur exactly once); the edit is diffed into
`patches/mNN.patch`; `MUTANTS.md` records a sha256 prefix per mutated module. Every mutant got:

- a witness script `witnesses/wNN.py <copy>` (run on base and on the mutant, output captured in
  `logs/base.witness.log` and `logs/all.witness.out`), and
- three gate runs, logs in `logs/mNN.<gate>.log`:
  - gateA: `cd <copy> && <runner> -m pytest -q -p no:cacheprovider tests/test_host_purity.py`
  - gateB: `unshare --map-root-user --mount bash <fight>/tools/gate_worldB.sh <copy>`
  - suite: `cd <copy> && <runner> -m pytest -q -p no:cacheprovider`
  - plus the oracle where a survivor needed the last gate checked:
    `cd <copy> && <runner> docs/verify_runtime_contract.py`

Witness and gate commands executed (the loop scripts used by all runs):
`bash tools/run_witnesses.sh <copy>`, `bash tools/gate_all.sh gateA|gateB|suite`,
`bash tools/summarize.sh` (kill-map, saved as `logs/summary.txt`).

## 3. Result matrix

`F` = task-0 file failures in that world; suite column = full offline suite failures; witness =
base → mutant (the secret witness proving the behaviour changed).

| # | zone / operator | witness (base → mutant) | gateA | gateB | suite |
|---|-----------------|-------------------------|-------|-------|-------|
| m01 | facts not supplied filled from the real host (`resolve_host`) | vulkan,linux-x64-cpu → cuda,linux-x64-cuda-12.8 | 6F | 6F | 9F |
| m02 | `shutil.which("nvidia-smi")` back in `HostProbes.detect_backend` | cpu → cuda | 5F | 5F | 15F |
| m03 | `os.environ` (`CUDA_VISIBLE_DEVICES`/`GGUFONE_BACKEND`) decides `detect_backend` | cpu → cuda | 3F | 3F | 3F |
| m04 | env-derived probe default (`VK_ICD_FILENAMES` → `icd_dir`) in `fake_host` | cpu → vulkan | 2F | 2F | 2F |
| m05 | caller's `icd_dir` replaced by the pinned `ICD_DIR` in `detect_backend` | cpu → vulkan | 1F | 1F | 3F |
| m06 | cached host answer sticks, order-dependent (`detect_backend` memo) | cuda,cpu → cuda,cuda | 5F | 5F | 9F |
| m07 | `plan_install` pre-flights the driver with a real `nvidia-smi` run | spawned=[] → spawned=[['nvidia-smi','-L']] | 1F | 1F | 1F |
| m08 | omitted `machine` filled from `platform.machine()` (`resolve_host`) | raises → linux-x64-cpu plan | 0 | 0 | 1F |
| m09 | omitted `dri_nodes` listed from the real `/dev/dri` (`HostProbes`) | cpu → vulkan | 0 | 0 | 1F |
| m10 | empty injected VRAM probe falls back to the real `nvidia-smi` (`host_budget`) | vram 0 → vram 8589934592 | 0 | 0 | 1F |
| m11 | injected `system` dropped → `library_glob()` asks the real platform | `system="windows"` → `[]` → `['cpu']` | 0 | 0 | 0 |

Killed-by attribution (exact failing node ids, from `logs/*.log`):

- m01: `test_the_mapping_matrix_does_not_move_with_the_real_host[vulkan|cuda]`,
  `test_the_vulkan_world_with_supplied_probes_answers_vulkan`,
  `test_an_environment_naming_a_gpu_does_not_move_the_mapping`,
  `test_the_injected_path_reads_no_host_fact_from_the_environment`,
  `test_the_injected_path_reads_no_device_environment_and_runs_no_nvidia_smi`
  (+ suite: `test_pins.py::test_host_variant_mapping[auto-linux-x86_64-linux-x64-cpu]`,
  `test_detect_backend[probes2-vulkan]`, `test_probes_never_fall_back_to_the_real_host`)
- m02: 5 of the same task-0 tests (min: `[cuda]` mapping) + 10 more suite tests in CLI/pins files.
- m03 / m04 / m07: only the task-0 file kills them (env-spy, tripwire) — the suite adds nothing.
- m05: task-0 `test_the_mapping_matrix_does_not_move_with_the_real_host[cpu]` with
  `AssertionError: assert 'cpu' == 'vulkan'` (tests/test_host_purity.py:127);
  suite adds `test_pins.py::test_probes_never_fall_back_to_the_real_host`,
  `test_current_host_is_the_only_reader_of_the_real_machine`.
- m06: all three `mapping[...]` params + `vulkan_world` + env-spy; suite adds 4 pins tests.
- m08: suite only — `tests/test_pins.py::test_probes_never_fall_back_to_the_real_host`
  `AssertionError: host access leaked: () {}` (tests/test_pins.py:224, the `platform.machine`
  tripwire) — the task-0 file never calls a synthetic world without a named machine.
- m09: suite only — same test, `AssertionError: host access leaked: /dev/dri.is_dir`
  (tests/test_pins.py:221, the `DRI_DIR` Trap) — the task-0 file never supplies an ICD path while
  omitting `dri_nodes`.
- m10: suite only — `tests/test_recommend_quant.py::test_host_budget_reads_meminfo`
  `assert 8589934592 == 0` (tests/test_recommend_quant.py:125) — the task-0 file does not cover
  `registry.recommend`.
- m11: **no gate fails**; the oracle output differs from base only in the printed repo path
  (`diff logs/base.oracle.log logs/m11.oracle.log` → the path line only).

## 4. Per-mutant records (operator, witness, raw result, verdict)

All paths relative to the fight dir. Runner path abbreviated `<runner>`. Each witness was run as
`<runner> witnesses/wNN.py <copy>` against `base/` and `mNN/`; the outputs quoted are verbatim
from `logs/base.witness.log` and `logs/all.witness.out`.

**m01 — KILLED (both worlds).** `pins.resolve_host`: `return fake_host(**given)` becomes
"base = current_host(); merge supplied facts over the real host's facts" — the classic pre-fix
bug, refactored. Witness: `M01 … -> cuda` / `linux-x64-cuda-12.8` (base: `vulkan` /
`linux-x64-cpu`). gateA `6 failed, 5 passed`; gateB identical; suite `9 failed`.

**m02 — KILLED.** `HostProbes.detect_backend`: `if self.has_nvidia_smi:` → `if self.has_nvidia_smi
or shutil.which(NVIDIA_SMI) is not None:`. Witness: `cpu` / `linux-x64-cpu` → `cuda` /
`linux-x64-cuda-12.8`. gateA `5 failed`; suite `15 failed` (CLI tests included — the leak moves
the whole plan).

**m03 — KILLED.** `HostProbes.detect_backend` gains `os.environ` fallback
(`CUDA_VISIBLE_DEVICES` or `GGUFONE_BACKEND` ⇒ `cuda`). Witness with
`CUDA_VISIBLE_DEVICES=0`: `cpu` → `cuda`. gateA `3 failed` (env test, env-spy, device-env
tripwire); suite `3 failed` — this file is the only killer.

**m04 — KILLED.** `fake_host` gains an env-derived default: `if not icd_dir: icd_dir =
os.environ.get("VK_ICD_FILENAMES", "")`. Witness with `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d`:
`cpu` → `vulkan`. gateA `2 failed` (env-spy reads `VK_ICD_FILENAMES`); suite `2 failed`.

**m05 — KILLED.** `HostProbes.detect_backend` stats the pinned `ICD_DIR` instead of the caller's
`icd_dir`. Witness with `icd_dir=/tmp/no-icd-here-xyz`: `cpu` → `vulkan`. gateA `1 failed`
(`mapping[cpu]` — the cpu-world instance is the one that exposes it); suite `3 failed`.

**m06 — KILLED.** `pins.detect_backend` memoizes the first (production) answer in a module global
and serves it to injected callers — cached host fact + order-dependent read. Witness:
`cuda, cuda` (base `cuda, cpu`). gateA `5 failed` (the first parametrized world already trips it);
suite `9 failed`.

**m07 — KILLED.** `install.plan_install`: when the caller did not state `has_nvidia_smi`, run
`subprocess.run(["nvidia-smi", "-L"], …)` before resolving. Witness (recorder on
`subprocess.run`): `spawned=[]` → `spawned=[['nvidia-smi', '-L']]`, same variant.
gateA `1 failed` — exactly `test_the_injected_path_reads_no_device_environment_and_runs_no_nvidia_smi`;
suite `1 failed` — the same test is the only killer.

**m08 — SURVIVED the task-0 file (both worlds); KILLED by the suite.** `pins.resolve_host` gains
`given.setdefault("machine", platform.machine())`: an unnamed arch in a synthetic world is
answered by the real machine. Witness: `RuntimeMissingError` (base) → `linux-x64-cpu` plan
(mutant). gateA `11 passed`, gateB `11 passed`; suite `1 failed` — the `test_pins.py` tripwire on
`platform.machine`.

**m09 — SURVIVED the task-0 file (both worlds); KILLED by the suite.** `HostProbes.detect_backend`
lists the real `DRI_DIR` when `dri_nodes` were not supplied. Witness:
`detect_backend(system=linux, has_nvidia_smi=False, icd_dir=/usr/share/vulkan/icd.d)`
`cpu` → `vulkan`. gateA `11 passed`, gateB `11 passed`; suite `1 failed` — the `test_pins.py`
`DRI_DIR` Trap.

**m10 — SURVIVED the task-0 file (both worlds); KILLED by the suite.**
`registry.recommend.host_budget`: `vram = probe() or _query_nvidia_smi()` — an injected probe
answering empty falls through to the real driver. Witness:
`host_budget(nvidia_smi=lambda: None)` → `vram_bytes=0` (base) vs `8589934592` (mutant — the real
RTX 3060 Ti, 8192 MiB). gateA `11 passed`, gateB `11 passed`; suite `1 failed` —
`test_host_budget_reads_meminfo` (`assert 8589934592 == 0`).

**m11 — SURVIVED EVERY GATE. BLOCKING.** `capability.backends`: `finder.library_glob(system)` →
`finder.library_glob()` — the injected `system` is dropped, `platform.system()` (this host: linux)
answers the glob. Witness: `capability.backends(<dir with libggml-cpu.so>, system="windows")` →
`[]` (base) vs `['cpu']` (mutant). gateA `11 passed`, gateB `11 passed`, suite `329 passed, 11
skipped`, oracle exit 0 with output identical to base except the printed repo path. This is the
parent card's own leftover lead (`finder.py::library_names/library_glob — platform.system() when
system= is not passed`), un-instrumented by any gate.

## 5. Zone coverage (§8.1 — every zone either has a mutant or a stated reason)

| zone (card method + parent leads) | mutant |
|-----------------------------------|--------|
| re-add `shutil.which` on the injected path | m02 |
| re-add `os.environ` access | m03 |
| re-add `/dev/dri` access | m09 (listing) + m05 (ICD path stat) |
| re-add `nvidia-smi` access | m07 (subprocess), m10 (vram probe seam) |
| cached host fact | m06 |
| environment-derived probe default | m04 |
| fallback to host for facts not supplied | m01 (kwargs), m08 (arch) |
| subtler: caching | m06 · order-dependent reads: m06 · fallback to environment: m03/m04 |
| parent lead: `registry/recommend.py::_query_nvidia_smi` | m10 |
| parent lead: `finder.library_names/library_glob` `platform.system()` | m11 |

Rejected candidate (no mutant crafted): instance-level caching inside `HostProbes` — it caches an
*injected* fact, not a host read, so it is outside this card's definition (and `test_pins.py`
already pins the re-`stat` behaviour at lines 264-269).

## 6. Blocking findings (card: "mark any SURVIVED mutant as BLOCKING for task 4 / index 4")

Task 4 = `t_ee24bd7c` (final review; its body: "Block on any critical/major finding or any
SURVIVED mutant from task 3").

**B1 — CRITICAL / BLOCKING: m11 survives the task-0 file, the full suite AND the oracle.**
`capability.backends()` still answers an injected `system` with this host's platform
(`finder.library_glob()` → `platform.system()`), i.e. a caller who states the platform
("windows") gets Linux-shaped globs. Witnessed delta: `[]` → `['cpu']`.
Reach today: no production caller injects a foreign system (`doctor` calls
`probe_runtime()` with `system=None`, and the default is the host platform by design), so the
leak is latent in the shipped CLI — but it is precisely the invariant E1a fixed ("facts supplied
must win; the real machine must not answer"), it will bite any cross-platform/CI caller, and it is
invisible to every gate. Fix = test pin, not a production change: e.g. in
`tests/test_host_purity.py` (defender may only add), assert with a distractor bundle
(`libggml-cpu.so` + `ggml-vulkan.dll`): `capability.backends(dir, system="linux") == ["cpu"]`
and `capability.backends(dir, system="windows") == ["vulkan"]`; or extend the tripwire test to
`pins.platform` reads through `capability.backends(system=…)`.

**B2 — BLOCKING relative to the named gate: m08 + m09 survive `tests/test_host_purity.py` in
both worlds.** Both are killed only by `tests/test_pins.py::test_probes_never_fall_back_to_the_real_host`
(tripwire/Trap), i.e. the task-0 regression does not pin the file's own docstring clause "facts
that were not supplied count as absent" for (a) an omitted `machine` and (b) omitted `dri_nodes`
with a supplied ICD. Recommend two added assertions to the file: a `platform.machine` tripwire
with `host_variant("auto", system="linux")` (expect the caller error, not a plan) and a
`DRI_DIR` trap with `detect_backend(system=…, has_nvidia_smi=False, icd_dir=<existing>)`.

**B3 — BLOCKING relative to the named gate: m10 survives `tests/test_host_purity.py`.** Killed
only by `tests/test_recommend_quant.py::test_host_budget_reads_meminfo`. `host_budget` lives in
`registry/recommend.py`, not in the detection modules, so the file's omission is defensible in
scope — recommend an explicit line in the file's docstring that the vram seam is owned by
`test_recommend_quant.py`, or add the one-line pin (`host_budget(nvidia_smi=lambda: None)` ⇒
`vram_bytes == 0`).

Nothing else survived: for every other channel the task-0 regression kills the mutant in both
worlds, and m01/m02/m05/m06 are additionally killed by `test_pins.py` and the CLI tests.

## 7. Evidence inventory / reproduction

- Mutants and patches: `m01 … m11/`, `patches/mNN.patch`, sha manifest `MUTANTS.md` (first-16
  sha256 of each mutated module), regeneration via
  `<runner> tools/make_mutants.py /home/rybens/workspace/state/fights/e1a-t0fc576df`.
- Witnesses: `witnesses/w01.py … w11.py`; raw outputs `logs/base.witness.log`,
  `logs/all.witness.out`, `logs/mNN.witness.log`.
- Gates: `logs/mNN.gateA.log`, `logs/mNN.gateB.log`, `logs/mNN.suite.log`,
  `logs/base.{suite,oracle}.log`, `logs/m11.oracle.log`, kill-map `logs/summary.txt`.
- World B: `tools/gate_worldB.sh`; world proof lines at the head of every `*.gateB.log`.
- The repo under attack was not touched: fight writes live under
  `/home/rybens/workspace/state/fights/e1a-t0fc576df/` plus this report and the bot-chat note.

## 8. Caveats

- Runner here is the hermes venv pytest (7.4.3, Python 3.11.15) with each copy's `pyproject.toml`
  providing `pythonpath=["src"]`; the repo's canonical gate is `uv run pytest -q`. The referee
  must re-run with the canonical gate — base counts to reproduce: `11 passed` (task-0 file),
  `329 passed, 11 skipped` (suite), oracle `failures: 0, skips: 1`.
- World B hide-set: `/dev/dri` (tmpfs) and the `nvidia-smi` binary (non-executable bind; the path
  still `exists()`, and `/dev/nvidia*` device nodes stay visible — neither channel is read by the
  artifact). The kill sets were identical in world A and B because the task-0 file simulates all
  three host worlds itself — that is the file's design and the reason both runs agree.
- Every mutant changes behaviour on the witness (no equivalent mutants); m11's witness is the
  "windows" injected system, which no existing gate exercises.
