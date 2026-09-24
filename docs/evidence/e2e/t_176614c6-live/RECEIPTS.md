# t_176614c6 — receipts for the three live `keep`/`fit` defects (a)(b)(c)

Everything below was run in this container on 2026-09-23, on the operator's box
(RTX 3060 Ti 8 GiB, `Spark-X2.5-4B-Q8_0.gguf`, runtime `b11026-linux-x64-vulkan`).

| what | where |
| --- | --- |
| product repo (fixed) | `/workspace/ggufone` — HEAD `59935db` + 4 commits (see "commits") |
| pristine tree for the before-receipts | `/work/t176614c6-before` (`git worktree add … HEAD`) |
| harness (scripts, logs, artifacts) | `/work/t176614c6-live` (nothing product-facing lives here) |
| launchers | `tg.py` (fixed tree) / `tg_before.py` (pristine tree), both `sys.path`-only, no install |

## (a) a swap kills the answer it is waiting for

`repro_a2.sh <before|after> <launcher>`: a warm host on the default key, then — 0.5 s apart —
`ask` on *that* key with a 12.8k-token state (a decision longer than `STOP_GRACE`) and a second
`ask` with `--n-ctx 32768` (a different host key ⇒ a swap). Receipts: `out/repro_a_before.log`,
`out/repro_a_after.log`, per-call `out/a_<label>_*.{json,exit,stderr,wall_ms}`
(read them with `keepinfo.py`).

| | before (pristine `59935db`) | after (fixed) |
| --- | --- | --- |
| A — the in-flight decision | `served_by: inline`, **31 019 ms** | `served_by: host`, **7 179 ms** |
| B — the swap | **exit 3**, `E_BACKEND_OOM … needed ~653 MiB; the driver reports 262 MiB free; tried 4 placement(s) down to CPU-only, none fit` | **exit 0**, 29 234 ms (waited for A's decision, then loaded and answered) |
| stderr of both | the OOM above | empty |
| later call on whatever was resident | exit 0, 15 848 ms | exit 0, 16 045 ms |

What the before column means: A's answer did not come from its warm host — the swap SIGKILLed that
host mid-prefill, A's client re-ran the whole thing inline (31 s vs the ~7 s a warm answer takes),
and the swap's own replacement host could not allocate at all, so **B never answered**. After: A's
answer comes from the host in 7.2 s and B answers too.

## (b) `ready` but cannot serve

- The recorded live case this card points at (`t_6a330eff`, finding b): a host whose record said
  `ready`, answering **every** request with an instant (~170–190 ms) `E_BACKEND_OOM`, until a human
  typed `keep stop`.
- I could **not re-build that state on demand today**, and I am not claiming otherwise:
  - `hog.py` (the "the desktop took the card" CUDA tenant) is refused by this container — `rc=201`
    with ~6.6 GiB free (recorded in the header of `repro_b2.sh`, log `out/b_hog.stderr`);
  - the second-tenant route SPEC 2.12 allows (a second data home, `repro_b3.sh`, `homeB/`) gave
    host B only ~1.1 GiB of the card, and the *resident* host A kept serving its requests in
    1 371 ms / 867 ms (`out/repro_b3.log`, `out/b3_*`). A host's context and compute buffers are
    allocated when it loads, so pressure that appears *after* the load does not produce the state:
    it needs the load-time estimate gap the E2E hit (a plan that fits the estimate but not the
    runtime's real per-request needs).
- What is pinned instead, offline and deterministic: `tests/test_keep_host.py::
  test_a_decision_that_cannot_allocate_leaves_the_ledger` drives the real `Server` with a decision
  that raises `BackendOomError` — before: the reply is the typed error **and the host stays up**
  (`assert not True`, "the host did not exit" → RED); after: the same typed reply, then the process
  exits by itself and `state.read_record` is `None`. The boundary is pinned too:
  `test_a_typed_error_that_is_not_about_the_device_keeps_the_host_serving` (`E_CTX_TOO_SMALL` is a
  property of that one request, so the host stays).

## (c) a busy reading becomes the box's permanent answer

`fit <4B> --json` at four moments, plus a copy of the plan cache at each step; the **entry** is the
file in `home/fit` every later `ask`/`run` reads. Receipts: `out/repro_c2.log` (before, artifacts
`out/c2_*`), `out/repro_c3.log` (after, artifacts `out/c3_*`).

| step | before: entry (`home/fit`) | after: entry | returned plan |
| --- | --- | --- | --- |
| 1 quiet `fit` | 36 layers, n_ctx 46 979 | 36 layers, n_ctx 55 974 | the policy answer |
| 3 `fit` while our own host is resident | **0 layers** (CPU), budget 1 570 766 848, `W_FIT_DOWNGRADE` | **36 layers** (the step-1 answer) | 0 layers for *that* reading (`W_FIT_DOWNGRADE`) |
| 4 `keep stop`, then `fit` again (~6.6 GiB free) | **still 0 layers** — the depleted plan stuck | **36 layers** — no stickiness | 36 layers |
| 5 `ask` on that entry | `n_gpu_layers: 0`, **25 756 ms** | `n_gpu_layers: 36`, **15 940 ms** |

The user-visible cost in the before column is the ~10 s/ask CPU penalty, and it persisted after
`keep stop` (shrink-only never climbs back) until the entry was deleted by hand.

## Gates

| gate | command (as run) | result |
| --- | --- | --- |
| RED on pristine HEAD | `.venv/bin/python -m pytest -q tests/test_keep_host.py tests/test_keep_client.py tests/test_context_v2.py` | **6 failed, 88 passed** (`red_all.log`) |
| RED, the updated `t_8cb0a05e` pin | same file, in `/work/t176614c6-before` | **1 failed** ("assert 0 == 36"), 16 passed (`red_free_vram.log`) |
| GREEN | same three files, fixed tree | **94 passed** (`green_all.log`) |
| GREEN, that pin | `tests/test_fit_free_vram.py` | **17 passed** (`green_free_vram.log`) |
| full offline suite (CI shape: `env -u PYTHONPATH … TYPED_GGUF_TEST_BLOCK_NET=1`, bundle stub) | `.venv/bin/python -m pytest -q -rs --timeout=120` | **1641 passed, 63 skipped, 0 failed** (`offline_suite2.log`) |
| full offline suite, same command on the pristine tree (a container-matched baseline) | in `/work/t176614c6-before` | **1634 passed, 63 skipped, 0 failed** (`offline_suite_pristine.log`) — +7 collected tests, same skips |
| docs gate | `.venv/bin/python -m pytest -q tests/test_public_docs.py` | **17 passed** |
| lint | `.venv/bin/ruff check src tests tools docs .github` | **All checks passed!** |

The 63 skips are the live/network gates plus six `test_wheel_install.py` skips whose own reason is
"the build backend is not in the local uv cache and this gate builds offline (SPEC A7)" — a
property of this container, not of the change. The card's reference figure (1 640 passed /
57 skipped) was measured in an environment where those six do run: 1 634 + 6 = 1 640 there,
1 634 + 7 new tests = **1 641** here, with the same 63 skips in both of this container's runs.

## (c), the other half of the hint: what the replacement plans against

The hint allows either "plan for the replacement against free + the outgoing host's footprint" or
"do not let a swap-degraded plan overwrite a better cached plan". The second is implemented
(`_may_replace_stored`), and the first turned out to be unnecessary *because of where the plan is
computed*: `ensure` stops the outgoing host (draining it, see (a)) and only then spawns the
replacement, whose own host resolves the fit plan at load time — against the memory that is
actually free by then. The before-receipt shows what went wrong instead: the swap's replacement was
planned while the *inline fallback* of the victim was holding the card (the (a) bug), so it came
out at 0 layers and `E_BACKEND_OOM`. In the after-receipt the same request is planned at
**36 layers / n_ctx 56 616** and answers (`out/repro_a_after.log`, `out/a_after_second.json`) —
i.e. the replacement lands on the GPU exactly when room exists after the release.

## Commits (`/workspace/ggufone`, on `main`, local — the coordinator pushes)

```
c5e54a3 chore(e2e): the t_176614c6 receipts — before/after live runs for (a)(b)(c)
0905c73 style: wrap the lines ruff flagged (E501) in the t_176614c6 changes
0b3c4d4 test(fit): the cache keeps the box's own answer (pin of t_8cb0a05e, updated for t_176614c6(c))
2f6eec3 fix(keep,fit): never let a victim's answer die, a dead host stay 'ready', a busy plan stick
b74c3da test(keep,fit): RED pins for the three live defects (a)(b)(c) — card t_176614c6
```
