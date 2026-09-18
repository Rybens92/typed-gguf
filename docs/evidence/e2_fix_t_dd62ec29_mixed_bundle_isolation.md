# E2 FIX — `bench --backend all` must not abort a host with two bundles (card t_dd62ec29)

Branch `main` (no remote; local commits on the shared tree) · Tier **M** (the card declares M; the
remedy stayed inside `bench/` and its gate file runs in ~1 s per mutant, so no upgrade) · evidence
schema `ggufone.evidence.bench-isolation/v1`

Box: the operator host's container (`cgroup quota 2.0` on a 24-CPU box; the E3 campaign owned the
GPU throughout these runs) with the GPU passed through as `/dev/nvidia0` + `/dev/dri/renderD128`
and `VK_DRIVER_FILES=/work/e3scratch/nvidia_egl_icd.json` (without it no Vulkan row can see a
device). Bundles: the installed Vulkan
`/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan` and the pinned
`linux-x64-cpu` asset of `runtime.lock` (sha256 `219cf1c7…`, verified by the neighbour card)
unpacked at `/work/t603-runtime/b11026-linux-x64-cpu`. Models: `Qwen3.5-4B-Q4_0.gguf` (the
neighbour card's, for comparability) and `Qwen3.5-0.8B-UD-Q4_K_XL.gguf` (the small one the live
gates use, so a run costs minutes on a box that is also benchmarking).

Commits (this card, on top of the tree's `main` at `1b7192f`):

| commit | what |
|---|---|
| `5ae3159` | `test`+`fix`: the 34-gate file and the remedy — `bench/isolation.py`, the throughput/determinism wiring, ruff |
| *(the commit that carries this document)* | `test`: the hardening assertions the Tier-M triage asked for (45 gates), the live gate, the mutmut pair retargeted, this document and `.e2e/t_dd62ec29-mixed-bundle-teardown/` |

Raw material for every claim below: `.e2e/t_dd62ec29-mixed-bundle-teardown/` (index in its
`README.md`). The RED control is a worktree of the parent commit (`1b7192f`), driven with the same
venv and the same environment, so only `src/` differs between a RED and a GREEN run.

## 0. What was wrong (reproduced here, then fixed)

`bench --suite throughput --backend all` on a host with **two** bundles measured one row per bundle
in **one** process. The report came out complete — and then the process died:

```
$ GGUFONE_RUNTIME_DIR=/work/t603-runtime/b11026-linux-x64-cpu \
  GGUFONE_BENCH_RUNTIME_DIR=/var/home/rybens/.local/share/ggufone/runtime \
    python -m ggufone bench --suite throughput --model …/Qwen3.5-0.8B-UD-Q4_K_XL.gguf \
      --backend all --runs 1 --threads 4 --sizes 64 --json      # parent tree, this box, this hour
…
}
double free or corruption (!prev)
Aborted (core dumped)                                              exit 134
```

(`logs/red_small.raw`; the neighbour card's own two runs in
`.e2e/t_603a35a0-backend-attribution/logs/{before,after}_mixed.raw` show the same tail on the 4B
model. The tail of a GREEN run is the report and nothing else — `logs/after_mixed.raw`.)

Two measured facts come with it, and they are why the audit's "one bundle per process" is the
remedy rather than a workaround:

1. **The second bundle's row was not what it said.** In the RED run the `vulkan` row (runtime_dir =
   the Vulkan bundle, `n_gpu_layers=-1`) carries `devices: []`, `effective_backend: null`,
   `warnings: ["W_BACKEND_MISMATCH"]` and `decision_ms: 2595` — host-CPU speed — while a
   *single-bundle* run of the same bundle on this box reports
   `devices: ["CPU_Mapped", "Vulkan0", "Vulkan_Host"]`, `effective_backend: "vulkan"` and
   `decision_ms: 519` (`logs/vulkan_alone.raw`). The engine log in the RED run even reports
   `llama_context: backend_ptrs.size() = 1` for that context (`logs/red_live_gate.txt`): the second
   `libllama` is talking to the **first** bundle's `libggml` (identical SONAMEs, `RTLD_GLOBAL`,
   one `Runtime` cached per directory). The crash is the same interposition seen from glibc's side;
   the *root cause* of the heap corruption is still a hypothesis, the *trigger* is not.
2. **The abort needs two directories, not two labels.** One bundle answering `cpu` *and* `vulkan`
   (the neighbour card's `opoffload` shape) exits 0/1 normally, and so does every single-backend
   run. Only a run whose selected backends resolve to two distinct directories dies.

Neither is fixed by making the report honest (that was card t_603a35a0, which flagged the row and
left the abort alone): a CI step that trusts the exit code still reads a complete report as a
crash, and E3-style mixed-host campaigns still cannot be scripted.

## 1. Requirement 1 — the remedy: one bundle per process

`bench/isolation.py` (new, stdlib-only, no `ggufone.runtime` import — A-E2-7 holds) decides and
executes the isolation; `suites.py` only calls into it:

* **the decision** (`isolation_needed`): two *distinct* directory keys (the same `Path.resolve()`
  `ctypes_binding._LOADED` caches under) for the backends this run will measure. One bundle under
  two labels stays in-process — that is the shape the mismatch gate was built on; a backend with no
  local bundle is never loaded at all.
* **the child** (`child_command`): the *documented* single-bundle path, not a private protocol —
  `python -m ggufone bench --suite <suite> --model <path> --backend <one> … --out <row.json>
  --json`. Only flags the run sets travel; a quick run states `--quick`; `--max-seconds` is
  deliberately absent (a row is one measurement unit and a unit that started always finishes).
* **the verdict** (`run_backend_child`): the parent publishes a row only when the child's answer
  corroborates the request — exit code `0` iff the child's own report says `ok`, a row for the
  requested backend, `runtime_dir` equal to the bundle this run selected, and an echoed `config`
  that matches the suite/model/runs/threads/kv-type/gpu-layers/scale flags. Anything else is
  **withheld**: the row carries `measured: false`, the reason, a `process` block with the child's
  exit code, and the report is `ok: false`.
* **the scratch data**: a healthy child's report is read and removed with its temp directory; a
  child that could **not** be verified keeps its directory (`row-<backend>.json/.stdout/.stderr`)
  and the row's reason names it — a crash is exactly when the child's own log is the evidence.
* **the record**: an isolated report carries `"isolation": {"one_bundle_per_process": true,
  "bundles": {…}, "reason": …, "suite": …}`, every row carries its `process` block, and the
  rendered markdown prints the note that explains both.

The alternative remedies were considered and rejected, with reasons:

* **(b) make the two copies coexist** (`RTLD_LOCAL` / `dlmopen` namespaces): the measured symptom
  says they already do not — the second bundle's engine emits no line of its own because it is
  bound to the first `libggml` — so the change would have to *prove* the interposition is gone, and
  CPython has no namespace API (`dlmopen` via ctypes, 16 namespaces, fragile). A wrong "they
  coexist now" would silently relabel rows again, which is the defect this family of cards is
  closing.
* **(c) refuse the second bundle in-process**: honest but it would make a *published* table unable
  to carry a real accelerator row in one command — exactly what E3's mixed-host campaign needs —
  and it would leave the row's numbers to a second, manual invocation. (a) keeps one command and
  gets *true* rows for both bundles.

Determinism (the other suite that measures every backend) takes the same path; the three
single-backend suites (`latency`, `quality`, `calibration`) measure one backend by construction and
never isolate; `--quick` cuts the measured list to one backend *before* the decision, so the quick
preset never pays for a child.

## 2. Requirement 2 — the exit code stays the report's

`cli._bench` returns `0 if report.get("ok", True) else 1` and is untouched. What changed is what a
crash *is*: a child that dies (the `double free` case included: a complete report **and** exit 134)
can no longer take the command down with it. Measured on this box, in this hour, with the same
command:

| run | tree | model | rows | exit |
|---|---|---|---|---|
| `logs/red_small.raw` | parent `1b7192f` | 0.8B | `cpu` ok, `vulkan` mismatch + no device log | **134** (SIGABRT, after the report) |
| `logs/after_mixed_small.raw` | this card | 0.8B | both measured in children, `vulkan` on `Vulkan0` | **0** |
| `logs/vulkan_alone.raw` | this card, one bundle | 4B | `vulkan` ok (`Vulkan0` buffers, 519 ms decisions) | **0** |
| `logs/after_mixed.raw` | this card | 4B | `cpu` measured; `vulkan` **withheld** — its child died at teardown | **1** |
| `logs/after_mixed_2.raw` | this card | 4B | `cpu` measured; `vulkan` reported `E_BACKEND_OOM` by the child itself | **0** |

Every exit code in that table is the report's (`0` ⇔ `ok: true`); the live gate asserts exactly
that equality on the real command, and the offline gates pin the four shapes of a broken child
(abort, no report, exit code contradicting its own report, a row for another backend/bundle/config).

Two of the five runs deserve their own line, because they are *box states*, not verdicts:

* `after_mixed.raw`: the 4B model under VRAM pressure (the E3 campaign was on the device). The
  `vulkan` child printed its whole report (`ok: true`, real `Vulkan0`/`Vulkan_Host` buffers in its
  stderr tail) and then died of SIGSEGV at teardown. The parent withheld the row, added
  `ISOLATED_CHILD_FAILED` to the report notes and exited 1. The same bundle, alone, on a free
  device, exits 0 (`vulkan_alone.raw`) — so this is a second, *single-bundle* defect of that
  bundle's teardown under memory pressure; this card's job was to stop it from killing the campaign,
  and it now cannot (finding F1 below).
* `after_mixed_2.raw`: the child's own fit ladder could not find 426 MiB of device memory
  (`E_BACKEND_OOM`, "the driver reports unknown free") and reported it as a row-level failure —
  which is what an in-process row does when a backend cannot run. The report is `ok: true` with the
  `vulkan` row `measured: false` **and its reason**, exit 0, exactly as SPEC 5 has always read.

## 3. Requirement 3 — the gates

`tests/test_bench_isolation.py` (45 gates, offline, model-free, ~1.9 s) is the remedy's contract:
the decision table, the exact child argv, all six verdict shapes, the scratch/keep rules, the
isolation record and notes, the wiring of both multi-backend suites (with a tripwire factory that
*proves* the parent never loads a model), a tripwire on `ctypes_binding.load_libraries` that proves
the command process dlopens **nothing** in an isolated run, the quick/single-bundle paths staying
in-process, and the CLI exit code following the report.

| gate | result |
|---|---|
| new file, **parent tree** (`1b7192f`, same venv/env) | **43 failed, 2 passed** — `logs/red_pretest.txt` (the two that pass are the CLI exit-code pins, which need no isolation) |
| new file, this tree | **45 passed** |
| `pytest -q --run-network tests/test_bench_isolation_live.py` (the real command, two real bundles) | **1 passed in 131.03 s** — `logs/live_gate.txt` |
| the same live gate on the **parent tree** | **1 failed** — the CLI exited `-6` (SIGABRT) where the gate requires 0/1 — `logs/red_live_gate.txt` |
| full offline suite (fresh, this tree) | **968 passed, 41 skipped** — `logs/green_full_suite.txt` |
| ruff | `All checks passed!` — `logs/ruff.txt` |
| coverage of the changed modules | `bench/isolation.py` **100 %**, `bench/suites.py` **99 %** (4 pre-existing gaps, unchanged) — `logs/coverage_changed.txt` |
| Tier-M mutation (soft), `bench/isolation.py` with `tests/test_bench_isolation.py` (the documented pair) | r1 72.6 % → r2 78.2 % → **r3 80.5 %** (see below) |

### The mutation rounds and their triage

The sweep is scored from mutmut's own artifacts (`tools/mutation_score.py`, never from the
progress display) and run **twice**: round 1 is the sweep as first written, round 2 is the same
scope after the triage's pins. A third round records the committed test file. The rounds are not a
loop — Tier M's contract is one sweep plus a triage; the numbers below are what the artifact says.

| round | scope | killed/scored | survivors | artifact |
|---|---|---|---|---|
| r1 | `bench/isolation.py`, `tests/test_bench_isolation.py` (43 gates) | **72.6 %** (387/533) | 146 | `logs/mutation_score.txt`, `logs/survivor_triage.txt` |
| r2 | same, after the triage's first pins (43 gates) | **78.2 %** (417/533) | 116 | `logs/mutation_score_r2.txt`, `logs/survivor_edits_r2.txt` |
| r3 | same, the committed 45-gate file | **80.5 %** (429/533) | 104 | `logs/mutation_score_r3.txt`, `logs/survivor_edits_r3.txt` |

Round 3 clears the 80 % bar the repository reports against, and its 104 survivors are the same
classes as round 2's — no survivor sits on a verdict: the row level (`measured`, `reason`,
`process.ok`, the withheld-gap shape), the exit-code comparison, the bundle/backend/config
verification, the isolation decision and the child argv are all killed. Triage of the survivors
per mutant (diff against its `__mutmut_orig` twin — the run's `.spans`/`.meta`, not
`mutmut show`, which is unreliable in 3.8):

* **message / prose** (mutmut's `XX…XX` string mangling, a `None` inside an f-string, a key name
  inside a sentence): this module's messages *are* half its product — the operator reads `reason`
  — and the gates pin their key phrases, not every word. The majority of the 116.
* **diagnostics pass-throughs**: `_unusable(backend, command, None, …)`-style argument mutants where
  only `stderr_tail`/`report`/`row` changes. The verdict (and the row's `measured`/`process` block)
  is unchanged; the killed variants of the same lines are the ones that move the verdict.
* **equivalents**: `report.get("ok", False/None/)` (the key is always present in a report),
  `timeout=None`, `env=None`, `mkdir(parents=…)`, `rmtree(ignore_errors=…)`,
  `SUITE_ECHO.get(…, None)` (non-isolating suites only), `isinstance(...) or True`, and the two
  mutants inside `_exit_desc`'s `# pragma: no cover` signal branch.
* **killed by the triage's pins** (each replayed against its own key — spliced into the live
  source, gate run, source restored; `logs/survivor_replays.txt`): the child's `python=None`
  (production's own path), the stdout diagnostic default, `_read_report`'s mapping guard, the
  "no `config` block" message, `_exit_desc`'s exact shapes, the two-size `--sizes` join, the
  `_tail` cap, the withheld-row gap row and the `isolation_record`/note keys. Six of them are in
  that log, each **KILLED** after the pin, plus one control that must survive
  (`report.get("ok", False)` — equivalent) and does: the harness is not failing everything it is
  handed.

The live gate is `model`-marked like the rest of `test_bench_live.py` and skips, with the reason,
on a box that has fewer than two bundles or no benchmarkable GGUF. It asserts the card's
requirements on the real thing: the exit code is `0`/`1` and equals the report's,
`--json` on stdout equals the report file, the isolation record names both bundles, every measured
row is a verified child row with a non-empty engine device set, and a row that could not run says
why (device too busy, or a child that died) instead of disappearing.

## 4. Requirement 4 — the operator box, both bundles installed

`logs/after_mixed.raw` (the card's command, this tree, 4B): exit **1**, report `ok: false`, and the
raw tail is

```
      "process": {
        "isolated": true,
        "exit_code": -11,
        "ok": false,
        "detail": "the isolated child exited -11 (SIGSEGV; 139 in a shell) while its own report
                   says `ok: true`: …"
```

with no `double free` line anywhere after the report — compared with `logs/red_small.raw`'s tail,
which is the report and then `double free or corruption (!prev)` / `Aborted (core dumped)`.
`logs/after_mixed_small.raw` (same command, small model, both rows measured) is the clean tail for
the doc's §2 table, and `logs/rendered_after_mixed_small.md` is what the renderer prints for it.

## 5. What was not changed, and the risks

* **The loader.** `ctypes_binding` still caches one `Runtime` per directory and still dlopens with
  `RTLD_GLOBAL`; nothing in this card claims two bundles can coexist in one process. The fix avoids
  the situation instead of explaining glibc's heap corruption (the root cause is still a
  hypothesis, §0.1) — if that explanation is ever wanted, this card's raws are the reproduction.
* **The published table shape.** Rows gained `process`, reports gained `isolation`, and the rendered
  tables gained a note; no column and no number changed. E3's regeneration (the neighbour card's
  F4) is unaffected in shape and now also produces true second-bundle rows.
* **`--max-seconds` and the budget.** A row is still one unit and the cap is still checked between
  units; the child is not given the cap, so a started row finishes exactly as before.
* **Cost.** An isolated row costs one interpreter start plus the model load it was already paying
  (~0.3 s of process overhead measured on the `--quick` path — under the preset's own target).
  `--quick`, single-bundle hosts and single-backend runs pay nothing: no child, no note, no key.
* **Residual risk (finding F1).** A *child* can still die at teardown on this box under VRAM
  pressure (`after_mixed.raw`): the run is then exit 1 with a named, withheld row instead of a
  killed campaign. That is a separate defect of the Vulkan bundle's teardown, not of the
  two-bundle interposition, and it is filed as its own card.

## 6. Findings for the coordinator

* **F1.** A single Vulkan-bundle process can SIGSEGV at teardown with a 4B model when the device is
  memory-starved (`after_mixed.raw`, child exit -11 after a complete report; `vulkan_alone.raw`
  exits 0 on a free device). Filed as its own card; this card contains it.
* **F2.** The second bundle's row, measured in its own process, now carries the device set its own
  engine log proves (`Vulkan0` + `Vulkan_Host` compute buffers, `effective_backend: "vulkan"`) —
  the expectation the neighbour card's F2 left open. `logs/after_mixed_small.raw` is the raw.
* **F3.** The E3 tables' per-row `effective` column now has a defensible value for *both* bundles
  of a mixed host in one command; the `isolation` block says which rows came from a child, so a
  later reader cannot mistake an isolated row for an in-process one.
* **F4.** `commands.reproduce` still does not echo `--sizes`/`--kv-type` (pre-existing, the
  neighbour card's F3); the raw commands are in `logs/runs.md`.

## 7. Files

`src/ggufone/bench/isolation.py` (new) · `src/ggufone/bench/suites.py` · `tests/test_bench_isolation.py`
(new) · `tests/test_bench_isolation_live.py` (new) · `pyproject.toml` (the Tier-M mutmut pair) ·
`.e2e/t_dd62ec29-mixed-bundle-teardown/` (raws, drivers, triage) · this document.
