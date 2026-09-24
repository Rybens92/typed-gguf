# E1c FIX (t_8cb0a05e) — `--fit` on a busy desktop: free-VRAM planning, bounded target, OOM ladder

Status: APPROVED (card body t_8cb0a05e is the acceptance contract; derived by the worker)
Date: 2026-09-18 | Criticality: HIGH (the default run path fails on the operator's box)
Tier: M (no `Tier:` line in the card → default M: TDD + coverage + static, one mutation run on the
      changed files, risk summary)

## Scenarios (each maps to a Required item of the card)

R1 — plan against FREE device memory at load time
- Given a host whose driver reports 8192 MiB total / 1112 MiB free (a desktop holds the rest),
  when a plan is produced or read from cache,
  then the budget is derived from the FREE number, the plan's device footprint ≤ budget and the
  cached plan is re-planned (never re-used blind) when the fresh free reading no longer fits it.
- Given the fingerprint is host identity (total VRAM), a free-VRAM change alone must NOT churn the
  cache key — re-validation, not re-keying, is the mechanism.

R2 — `--fit-target MiB` bounds the plan
- Given `--fit-target 5200` on the 8 GiB box, budget = free − 5200 MiB (0 when free is 1112 MiB),
  and the reported plan (both `estimate` and `llama-fit-params` sources) respects it; `n_gpu_layers`
  is the floored per-layer split for that budget, never the unconditional full offload.

R3 — automatic degradation on allocation failure
- Given a load that fails with the ggml/driver OOM text while `n_gpu_layers > 0`,
  when a CPU-possible placement exists,
  then the load is retried down the ladder fewer layers → smaller kv_type → CPU-only, the returned
  handle names the placement actually used, and `warnings` carries `W_BACKEND_OOM` (a real
  allocation failure happened) + `W_FIT_DOWNGRADE` (the plan was reduced). A hard failure is only
  reported when no CPU path can work.
- Given `--no-fit`, the engine surface says so explicitly ("fit disabled, CPU only"); the value
  `n_gpu_layers: 0` alone is not an explanation.

R4 — correct error classification
- Given the operator's exact tail (`Device memory allocation of size 1058982400 failed.` /
  `ErrorOutOfDeviceMemory` / `unable to allocate Vulkan0 buffer`),
  then the error code is `E_BACKEND_OOM` (never `E_MODEL_ARCH_UNSUPPORTED`), carrying the free and
  needed byte numbers, the plan that failed and the `--no-fit` / `--fit-target` hints.
- A genuine architecture failure keeps `E_MODEL_ARCH_UNSUPPORTED` and is NOT retried.

R5 — host-gate gap
- `tools/host_gate_e1c_fit.sh` runs the operator sequence with a busy-desktop precondition
  (driver reports low free VRAM) and a fake-allocation-failure world, writes raw tails + the
  free-VRAM number for every step, and is rehearsed in the sandbox; the host run is left PENDING
  with its expected shape. `--fit-target`-sensitive steps assert the plan changes.

## Non-goals
- No change to `recommend_quant`'s conservative planner semantics (E1a gate).
- No new third-party dependency (SPEC 2.11: core stays stdlib-only).

## Review notes
- No pre-existing pin had to be weakened or rewritten: the E1c pin
  `tests/test_e1c_mutation_pins.py::test_a_gpu_host_offloads_every_layer_in_the_binary_path` passes
  unchanged because the budget it passes (16 GiB) genuinely covers full offload — the bounded case
  is pinned next to it by a new test in `tests/test_fit_free_vram.py`. The other fit pins
  (`_gpu_layers`, the KV ladder, the budget boundary, the cache key) all still describe the fixed
  behaviour, which is why they were kept as the regression net rather than edited.
