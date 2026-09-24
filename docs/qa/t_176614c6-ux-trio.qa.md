# typed-gguf — the engine UX trio (a)(b)(c) — QA Report

Card: `t_176614c6` | Date: 2026-09-23 | Tier: M (condensed: risk summary + decision + confidence)
Tree: `/workspace/ggufone` on local `main` — `b74c3da` (RED pins) → `2f6eec3` (fix) → `0b3c4d4`
(the `t_8cb0a05e` pin moved to the new contract) → `0905c73` (ruff) → `c5e54a3`/`b22682d` (receipts).
Evidence: `.e2e/t_176614c6-live/RECEIPTS.md` (numbers, logs, artifacts), task comment `t_176614c6`.

## 🟡 WORTH CONSIDERING — (b) has no *live* after-receipt from this run

Everything about (b) is pinned offline, but the live "ready host that cannot serve" state could not
be rebuilt on demand in this container, so the fix is verified against the *state* (an injected
`BackendOomError` in the real `Server`) rather than against a re-observed occurrence.

- Risk: if the failure reaches the host as a *different* typed code than `E_BACKEND_OOM`, the new
  ledger-drop does not trigger and the host still sits `ready`.
- Why I believe it is covered: the card's recorded case is an `E_BACKEND_OOM` message from the
  ladder walk (`llama.cpp could not allocate device memory for the fit plan … tried N placement(s)
  down to CPU-only`), which is exactly `BackendOomError.code`; `UNSERVABLE_CODES` is a named set, so
  widening it later is a one-line change with its own pin.
- Cost to close the gap: needs a box where a *second* VRAM tenant can be created after a host is
  resident; this container refuses user CUDA allocations (`rc=201` with ~6.6 GiB free) and a second
  data home's host only ever took ~1.1 GiB, after which the resident host kept serving in ~1.4 s.
- Recommendation: **accept for this card** and let the next live E2E watch for the state; the
  boundary (a non-device typed error must *not* end the host) is pinned.

## 🟢 ACCEPTABLE — no action

- (a)/(c) before/after live receipts with comparable numbers (see RECEIPTS.md): the swap's victim
  answer went inline-31 019 ms → host-7 179 ms, the swap itself `exit 3 / E_BACKEND_OOM` →
  `exit 0`, and the sticky entry 0 layers → 36 layers (the `ask` after `keep stop` 25 756 ms →
  15 940 ms).
- RED first: 6 failed / 88 passed on the three touched gate files at pristine `59935db`; the updated
  `t_8cb0a05e` pin 1 failed / 16 passed. GREEN: 94 passed / 17 passed.
- Full offline suite in the CI shape: **1641 passed, 63 skipped, 0 failed** — against a
  container-matched pristine baseline of 1634 passed / 63 skipped / 0 failed on the same command
  (+7 collected tests, same 63 skips; the card's reference 1 640/57 differs only by the six
  `test_wheel_install.py` gates, which skip here because the uv cache has no hatchling offline).
- Docs gate 17/17, release gate untouched (version stays 0.2.2), ruff clean, one writer, no
  calibration or placement-ladder-contract edits.
- Tier-M mutation, one run, `keep/client.py` with `tests/test_keep_client.py` as the selection
  (`--max-children 2`, driver `.e2e/t_176614c6-live/mutmut_sweep.sh`, log
  `mutmut_sweep.log` in the same directory): **829 mutants, 506 killed = 61.0 %**, 317 survived,
  6 timeouts, **0 in the "no tests" bucket**. Soft threshold, and the score is deliberately *not*
  comparable to the 65.4 % published for this package (that sweep ran all four keep gate files;
  this one ran the (a) fix's own file). The drain has no behavioural survivor: the adjacent ones are
  equivalent mutants on pre-existing lines (`suppress(OSError)` → `suppress(None)`,
  `child = self.children.get(pid)` → `None`), which the *other* three gate files assert.
  `keep/host.py` and `runtime/fit.py` were not swept — one run is what Tier M asks and the pair
  cannot cover three modules inside this card's wall clock; their fixes carry RED/GREEN pins and
  (c)'s live before/after receipt.

## 📈 CONFIDENCE: 8/10

Increasing: every acceptance criterion of the card has either a live before/after receipt (a, c) or
an offline RED→GREEN pin against the production `Server` (b); the full published suite is green in
the CI shape with a container-matched baseline; the swaps and messages the card says must not change
(`t_6a330eff`/`t_9249bb0c` semantics: one host per data home, no second model copy, ledger drop on
an unservable placement, SIGKILL escalation still reachable for a wedged host) are all still pinned.

Decreasing: (b)'s live state was not re-observed; two of the three changed modules have no Tier-M
sweep; `DRAIN_GRACE` is expressed as "the client's own request ceiling", so a caller who lowers
`timeout` also shortens the drain (documented, and pinned by
`test_a_swap_gets_the_request_ceiling_to_drain_a_decision`).

## Metrics

| metric | before (`59935db`) | after |
| --- | --- | --- |
| gate files touched | — | 94 passed (was 6 failed / 88 passed) |
| full offline suite (CI shape, this container) | 1634 passed / 63 skipped / 0 failed | 1641 passed / 63 skipped / 0 failed |
| docs gate | 17/17 | 17/17 |
| ruff | clean | clean |
| mutation, `keep/client.py` (narrow selection) | not run | 506/829 = 61.0 % killed, 0 uncovered |
| live: victim's in-flight answer | inline fallback, 31 019 ms | served by the host, 7 179 ms |
| live: the swap's own call | exit 3, `E_BACKEND_OOM` | exit 0, 29 234 ms (drained) |
| live: cached plan after `keep stop` | 0 layers (CPU), `ask` 25 756 ms | 36 layers, `ask` 15 940 ms |

## Decision

Agent recommendation: **ship (Option A)** — the two gaps above are named, bounded, and both have a
follow-up route (a live E2E that can create a second VRAM tenant; a later Tier-M sweep of
`keep/host.py` + `runtime/fit.py`). No 🔴 item is open.

Human/reviewer decision: _pending — this card's report is the handoff._
