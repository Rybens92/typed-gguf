# E2 FIX — a single Vulkan bundle's teardown SIGSEGV, root cause (card t_97f1bc93) — QA Report

Date: 2026-09-19 · Tier **M** (the card declares M) · Decision: **ship** — the card's three
requirements are met with the evidence below (reproduce + root cause; a fix that makes the exit
status the report's; a gate that runs under the pressure shape), and the residual risks are small
and stated.

Deliverable: `ggufone.runtime.teardown` + `cli.run` at the two process entry points. Implementation
notes, raws and the full index: `docs/evidence/e2_fix_t_97f1bc93_vulkan_teardown.md` and
`.e2e/t_97f1bc93-vulkan-teardown/`.

## Summary — risk-weighted

🔴 REQUIRES ATTENTION — **none open.** The two candidates were both resolved inside this card:

* *"is the crash ours?"* — no: the captured backtrace is the NVIDIA ICD's own exit handler
  (`libnvidia-eglcore` → `libnvidia-glvkspirv` from libc's `__run_exit_handlers`, fault address
  `0x18`) with no ggml/llama/ggufone frame, and the bundle's own `llama-bench` (4 runs, same model,
  same ICD) exits 0. But *ggufone's exit status* was still being written by that handler — so the
  remedy is ours: the process that has a bundle loaded now ends itself with the command's code.
* *"should the placement be refused at load time instead?"* — measured, rejected: the crashing runs
  fit, measured real device work and reported `ok: true`; a refusal would reject correct runs on a
  device whose free memory the driver reports as unknown. The unfittable regime already has a typed
  refusal (`E_BACKEND_OOM`).

🟡 WORTH CONSIDERING (2, recorded, none blocks)

* **`cli.run`'s `os._exit` line is not coverage-instrumentable** (a process that ends there writes
  no coverage data — that is the fix working). The branch is proven by the child-process gates'
  exit codes, and the callee (`end_process`) is 100 % covered in-process.
* **The live crash is intermittent** (4 crashes in 14 identical parent-tree runs; roughly one in
  three on a contended box). The live gate therefore bounds the outcome (exit 0/1, report readable,
  code == report's `ok`) instead of forcing the crash; the crash itself is on record in the parent
  tree's raws plus the deterministic offline proxy.

🟢 ACCEPTABLE

* Behaviour is additive for every existing caller: `cli.main` stays a pure function (the API and
  `tests/` are untouched), and a process that never loaded a bundle keeps the interpreter's normal
  shutdown — pinned both ways in `tests/test_cli_teardown.py`.
* A piped `--json` report survives the deliberate exit: both streams are flushed before `os._exit`
  (child-process gate: the usage text is on a piped stdout).
* The isolation layer above (`t_dd62ec29`, `t_57cc0179`) keeps its own guarantees; its recovery path
  is now the fallback for the narrower window where the ICD dies before the report is complete.

## What could go wrong

* **A user expects `atexit`/destructor side effects to run.** A bundle-loaded command no longer runs
  third-party exit handlers. The report file is written and closed during the command (not at exit),
  and both streams are flushed — the pinned answer a reader consumes is unaffected; a cleanup that a
  *driver* does at process exit (e.g. freeing device memory the OS reclaims anyway) is skipped, and
  skipping it is the point. Detectability: high (documented in `teardown.py` and the CLI docstring).
* **A future entry point forgets `run`.** The console script's target is pinned by a gate
  (`test_the_console_script_points_at_the_process_entry_point`) and `python -m ggufone` by
  `test_the_module_entry_point_runs_the_process_entry_point`; a new entry point would need its own
  pin — named here as follow-up awareness, not a defect.
* **`os._exit` hides a late flush bug elsewhere.** Nothing else in this change relies on flush;
  `contextlib.suppress` keeps a closed stream from changing the exit code, and that case is pinned.

## Metrics

| metric | value |
|---|---|
| new gates | 10 (7 child-process + 3 in-process); parent tree **6 failed, 1 passed** |
| full offline suite (integrated tree `0702a46`; the landed commits are byte-identical outside `pyproject.toml`'s mutmut comment block) | **1083 passed, 42 skipped, exit 0** |
| ruff | clean |
| coverage — `runtime/teardown.py` | **100 %** (13/13) |
| coverage — `__main__.py` | **100 %** (1/1) |
| coverage — `cli.run()` | 3/4 statements (the `os._exit` line is un-instrumentable by construction) |
| Tier-M mutation, `runtime/teardown.py` | **4/4 killed**, 0 survived, 0 pending |
| hand-mutation table (behavioural mutants mutmut cannot emit) | **7/7 KILLED**, 2 controls SURVIVED, byte-identical restores |
| live gate | fitting regime: exit 0, 176 s · starved regime: exit 1 + typed reason, 47 s |
| crash rate on the parent tree | **4 crashes in 14 runs** (0/1/139 mixes in `logs/*.summary`) |

## Decision

Ship. The defect is a third-party destructor rewriting a completed run's exit status; the fix makes
the process's exit status the command's, with the crash's own frames, a deterministic proxy, a
child-process gate file, a live gate in both device regimes, and a mutation story that covers the
tool's blind spot (7 hand mutants, each killed by a named gate). Accepted risks: the two 🟡 items
above — neither changes what a reader of a report consumes.
