# t_16067777 — the E2E hardening batch (P1–P4): pins, fixes, and the two Tier-M sweeps

Card **t_16067777** carries four findings the E2E proposal (`docs/evidence/t_559ed8c8/`) raised
against the shipped 0.2.3 build. All four are behaviour, not cosmetics: they were written as RED pins
against the released code first, then fixed, then the two modules whose *logic* moved were swept for
mutants (Tier M: one run per changed module, soft threshold, no return loop). This directory is the
receipt: the RED transcript, the gate run, and every mutmut verdict the card stands behind.

| # | Finding | Fix (file) |
|---|---------|-----------|
| P1 | `runtime rollback` rebuilt a ten-key record, so the ~30 keys `update` had written (the probe's `backends`, `symbols_*`, the asset facts) were gone and `typed-gguf version` printed `backends unknown` | `src/typed_gguf/runtime/update.py::rollback` — the retained record is **merged** (`{**record, …}`), the keys that describe the *active* bundle rewritten for the bundle the record now names, `previous` cleared |
| P2 | transport errors never named the product (a GitHub asset leg could report a HuggingFace failure), and the User-Agent was the frozen literal `typed-gguf/0.1` | `src/typed_gguf/registry/hf.py` — `_translate(exc, repo, *, product=HUGGINGFACE)` names the product and its host (`PRODUCT_HOSTS`), `USER_AGENT` carries the packaged `__version__`; `runtime/install.py::_fetch` passes `product=GITHUB` |
| P3 | `keep stop` removed the record/socket/spec but left the host's `<digest>.log` behind | `src/typed_gguf/keep/state.py::clear_log` + `keep/client.py::Client.stop` calls it beside `clear_record` |
| P4 | the host-gate driver used the SDK's 10 s default, so the first decision of the day (cold Vulkan shader cache: 32.1 s measured) timed out three times on a box where the product is right | `tools/host_gate_serve_client.py` — `CLIENT_TIMEOUT = 180.0` passed to `TypeSafeClient(...)` |

## 1. RED → GREEN

Eight test functions went in (six before any source change, plus the two `registry/hf.py` leg pins that
landed *after* sweep (A) showed those lines alive — they are marked as such in the list), and one
existing serve test gained the P3 assertions. The RED run below therefore holds **seven** tests: the six
new pins that existed then plus `test_keep_stop_still_works_after_serving_and_leaves_nothing_behind`,
whose last assertion is P3's. Six of the seven were red at once;
`test_a_huggingface_failure_still_names_huggingface` passed already and must pass on both sides of the
fix — it is the guard that says the wording is *per leg*, so it cannot be satisfied by renaming
everything to GitHub.

    red_proposals.log        the first run: 6 failed, 1 passed (the failures are the four findings)
    red_reproduction.log     the resumed run's own RED/GREEN over the **whole** pin set (below)
    $ env -u PYTHONPATH -u TYPED_GGUF_HOME TYPED_GGUF_TEST_BLOCK_NET=1 \
        .venv/bin/python -m pytest -q -p no:randomly \
        tests/test_runtime_update.py tests/test_hf.py tests/test_serve.py tests/test_keep_client.py
    7 passed in 0.37s          (after the fixes; the file list above is the pin set)

The pins: `tests/test_runtime_update.py::test_rollback_keeps_the_probe_facts_the_update_recorded`
(P1: key-set superset, the probe facts, the active-bundle keys, the retained bundle's `installed_at`,
the report's own `home`/direction, and the CLI's `backends …` line),
`::test_the_github_asset_leg_names_github_and_sends_the_real_user_agent` (P2: the GitHub leg names
github.com, the HuggingFace leg still names huggingface.co, and both UAs carry `__version__`),
`tests/test_hf.py::test_the_user_agent_carries_the_packaged_version`,
`::test_a_huggingface_failure_still_names_huggingface`,
`::test_the_unreachable_download_leg_names_the_host_the_product_talks_to` (added *after* sweep A),
`::test_an_unknown_transport_failure_is_still_the_typed_download_error` (after sweep A),
`tests/test_keep_client.py::test_stop_takes_the_ledgers_log_with_it` (P3),
`tests/test_serve.py::test_keep_stop_still_works_after_serving_and_leaves_nothing_behind` (P3 through
the serve path) and `::test_the_host_gate_driver_raises_the_sdk_timeout_above_its_ten_second_default`
(P4, asserted by parsing the driver's AST so the value is pinned, not a string in a comment).

### The resumed run's own RED (the receipt the second claim rests on)

The card was reclaimed mid-run and resumed; the whole pin set was then driven RED and GREEN again from
the resumed session, on the real tree, so nothing in this receipt rests on a transcript the first
session wrote while the process was being killed:

    harness/red_cycle.sh     copies the six changed *source* files back to the pre-fix baseline (five
                             under `src/`, plus `tools/host_gate_serve_client.py` — the P4 fix lives in
                             `tools/`, and a RED that leaves it fixed cannot fail its own pin), runs the
                             nine pins, then restores the working tree under a `trap`. The baseline is
                             `HEAD~1` now that this card is committed (= 08a925e, the E2E head); it was
                             plain `HEAD` while the card was still uncommitted. `$1` overrides.
    red_reproduction.log     RED 7 failed, 2 passed  →  GREEN 9 passed
                             (the two that pass on both sides are the guards: HuggingFace keeps its
                             own wording, and an unknown transport still becomes `E_DOWNLOAD_FAILED`)

## 2. Gates (on the committed tree; `gates.log`)

    pytest   1793 passed, 59 skipped, 0 failed        (baseline before this card: 1785 passed, 59 skipped)
             (+8 test functions: 4 in test_hf.py, 2 in test_runtime_update.py, 1 in test_keep_client.py,
              1 in test_serve.py; the P3 assertion inside the existing serve test replaces none)
    ruff     All checks passed!                       (ruff check src tests tools docs .github)
    uv build typed_gguf-0.2.3.tar.gz + -py3-none-any.whl   (dist rebuilt, wheel installs)

`gates.log` here is the **resumed** session's capture (`harness/gates.sh`, which prints the host, the
same three commands and the tree under test) — re-run on the exact working tree this commit contains
(immediately before the commit, so its `tree under test` block still lists the receipts directory as
untracked), so the quoted counts are from the tree this commit contains, not from a pre-reclaim log.
The pins were RED/GREEN-verified against that same tree by `harness/red_cycle.sh`.

Offline as this repo's suite is run: `TYPED_GGUF_TEST_BLOCK_NET=1`, `TYPED_GGUF_BENCH_RUNTIME_DIR` at a
stub bundle dir (the four empty `lib*.so` names CI creates), and **no** `PYTHONPATH` / `TYPED_GGUF_HOME`
(an empty home reddens the bench gates).

## 3. The Tier-M sweeps

`tools/mutmut_driver.py run --max-children 2`, mutmut 3.8, offline env as above. Two runs, one per
changed module, each with the gate file that owns the surface — the pair convention the sweep config in
`pyproject.toml` documents (and which was restored to the serve pair afterwards; the paragraph for this
card records both runs' pairs and numbers).

| run | module (selection) | mutants | killed | survived | timeouts | "no tests" |
|-----|--------------------|--------:|-------:|---------:|---------:|-----------:|
| A round 1 | `registry/hf.py` (`tests/test_hf.py`) | 691 | 377 = **54.6 %** | 254 | 0 | 60 |
| A round 2 | 2 pins added, scoped re-run of `x__translate*` + `x__download_url*` + `x__download_file*` | 691 | 387 = **56.0 %** | 244 | 0 | 60 |
| B round 1 | `runtime/update.py` (`tests/test_runtime_update.py`) | 967 | 826 = **85.4 %** | 141 | 0 | 0 |
| B round 3 | 2 pins added, scoped re-run of `x_rollback*` | 967 | 830 = **85.8 %** | 137 | 0 | 0 |

Run (B) is the pair card t_ba767a2b run (A) swept (958 mutants, 819 killed = 85.5 %), so its numbers are
comparable: the `rollback` merge added 9 mutants and the two post-sweep pins moved `x_rollback` from 16
to 12 survivors. `mutmut_pins_delta.log` lists the exact 10 (A) + 4 (B) mutants the post-sweep pins
killed; both re-runs went through `run '<key>*'` because a second *full* `run` only reprints the cached
verdicts (`mutants/mutmut-stats.json`) at 0.00 mutations/second.

### What the survivors are

* **Run (A) — 244.** The selection is one file; the module's network-shaped surface can't run per
  mutant, so the big blocks are pre-existing branches whose assertions live in gates this sweep did not
  select: `download_url` (50, the resume/Range body), `model_info`/`model_info_from_snapshot`/
  `repo_files`/`search` (142, the shapers), `snapshot_path`/`token_paths`/`token` (22, path and env
  plumbing), `_headers` (3) and `download_file` (5). `tests/test_hf_network.py` is `network`-marked
  (skipped without `--run-network`, i.e. never available to a per-mutant run); the registry-store and
  CLI suites hold the rest.
* **Run (B) — 137.** Pre-existing: `run()`'s lifecycle and report strings (30), `_require_probe` (25),
  `fetch_releases` (21), `_stage` (12), `current_runtime` (11), `_stop_host` (7) — all asserted by the
  *other* update gate files (t_ba767a2b's note says the same). Nothing on P1's new merge is left alive.

### Findings this sweep produced (for the reviewer — pre-existing, not introduced here)

1. **`x_rollback__mutmut_73` is a real weak assertion.** `time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())`
   mutated to `time.strftime("%Y-%m-%dT%H:%M:%SZ", )` — CPython then uses *local* time. Hand-applied to
   the real source it **passes** every rollback pin, because the gate checks the stamp's shape
   (`re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", …)`), not its zone. Equivalent on this
   container (TZ=UTC), a real mislabel on a box at +02:00. Closing it needs a TZ-aware pin, not a
   code change.
2. **The `x_rollback` key-rename survivors (`variant`, `schema`, `libllama_sha256`, `installed_at`,
   ×2 each) are equivalent *under this fixture*.** The record is built as `{**record, …}` (P1), so a
   renamed override silently falls back to the value the `update` record already carried — identical
   because the fixture's retained and updated bundles share a variant and the same `libllama.so`
   bytes. A fixture whose two bundles differed would kill all of them (and only then would
   `record["libllama_sha256"] == sha256(home.dir / "libllama.so")` be checking a *recomputed* hash).
3. `x_rollback__mutmut_30` (`_stop_host(home, …)` → `_stop_host(None, …)`) survives because the tests
   inject a fake client, which makes `home` unused — equivalent under injection.
4. `x__headers`'s three and `download_file`'s five survivors are value/plumbing mutants of the header
   assembly and the disk precheck; the exact-value assertions for the bearer header live in
   `test_hf.py`, and a `download_file` mutant only the network-marked file can kill.

### How the two "did the pin really kill it?" checks were done

Where a pin was added *after* its sweep, the kill was proved on the real tree instead of by another
scoped run — apply the mutant by hand, run the pin, revert:

    # mutmut_61  previous.get("installed_at") -> previous.get(None)
    assert None == '2026-09-24T19:43:23Z'   # tests/test_runtime_update.py:923, RED, then GREEN after revert
    # mutmut_161 "home": str(home) -> str(None)
    AssertionError at tests/test_runtime_update.py:927, RED, then GREEN after revert

Both are also covered by the scoped re-run numbers above (the `x_rollback` group is what round 3 ran).

## 4. Reproduce

    cd /workspace/ggufone
    bash docs/evidence/t_16067777/harness/gates.sh        # → gates.log (fresh capture)
    bash docs/evidence/t_16067777/harness/red_cycle.sh    # → RED then GREEN over the 9 pins
    env -u PYTHONPATH -u TYPED_GGUF_HOME TYPED_GGUF_TEST_BLOCK_NET=1 \
        TYPED_GGUF_BENCH_RUNTIME_DIR=/tmp/offline-bundle-023 \
        .venv/bin/python -m pytest -q                      # the same suite by hand
    # sweep (A): pyproject source_paths = registry/hf.py, selection = tests/test_hf.py
    env -u PYTHONPATH -u TYPED_GGUF_HOME … uv run --extra dev --with mutmut \
        python tools/mutmut_driver.py run --max-children 2
    # sweep (B): source_paths = runtime/update.py, selection = tests/test_runtime_update.py
    # A re-run of the same pair must move `mutants/` aside first, or mutmut reprints cached verdicts.

`--extra dev` is load-bearing (an earlier card lost a run to `uv run --with mutmut` re-syncing the venv
without pytest); a sweep whose selection reads `docs/` must keep that file in `also_copy` — run (A)
needs one document (`docs/evidence/hf_spark_x2_5.json`), not the 4.3 GB tree, which is why `also_copy`
names the file rather than the directory.

## 5. Files

    red_proposals.log                    the RED run (6 failed, 1 passed) before any source change
    red_reproduction.log                 the resumed run: RED 7 failed, 2 passed → GREEN 9 passed
    gates.log                            pytest + ruff + uv build, captured on the committed tree
    harness/red_cycle.sh                 the swap-to-the-baseline / run / restore cycle behind it
    harness/gates.sh                     the three gate commands as `gates.log` ran them
    mutmut_hf_round1_results.log         sweep (A) round 1 survivors ("results" output, full list)
    mutmut_hf_round2_results.log         sweep (A) round 2, after the two post-sweep pins
    mutmut_update_round1_run.log         sweep (B) round 1 progress (967/967, 🎉 826 / 🙁 141)
    mutmut_update_round1_results.log     sweep (B) round 1 survivors
    mutmut_update_round2_results.log     sweep (B) after the `installed_at` pin (x_rollback 16 → 13)
    mutmut_update_round3_results.log     sweep (B) after the report pin (x_rollback 13 → 12)
    mutmut_pins_delta.log                the exact mutants the post-sweep pins killed, per run
