# t_a0fa2dc0 — a usable default after `models pull` (plain `ask`, no path hunting)

Card: *typed-gguf — UX: a usable default after `models pull`*. Tier: **M** (declared default; the
card's body carries no tier line). No push, no tag, no publish — commits stay local on `main`.

## 1. What the live smoke actually showed (before/after, real commands)

Raw receipts, verbatim: `docs/evidence/e2e/t_a0fa2dc0-default-model/before-receipt.txt`,
`after-receipt.txt`, `after-receipt-two.txt`. Every run is
`env -u PYTHONPATH uv run --extra dev typed-gguf …` on a scratch `TYPED_GGUF_HOME`.

Two facts the RED first had to establish, because the card's premise was half wrong:

* **[B4] a home that really pulled already has a default.** `models pull ggml-org/models --file
  tinyllamas/stories260K.gguf` (1.19 MB, real HF, sha verified against `lfs.oid`) writes
  `"current": "stories260k"` — `store.add_entry` defaults `current` to the first entry (SPEC 2.7).
  Bare `ask` then resolves and answers: `"model": "stories260k"`, `noul 0.599049`. So
  "nothing sets a default on pull" was not the bug; nothing had to change there.
* **[B1]/[B2] the two real dead ends.** A fresh home (empty registry) and a home whose registry has
  exactly one alias and **no** `current` both died on
  `E_MODEL_NOT_FOUND: None is not a registry alias, a path or a pulled model; known aliases: <none>
  (use …)` — exit 2. `None` is not what the caller asked for, and a one-entry registry is not
  ambiguous. `fit` (same resolver) said the same thing.

| # | home | before (c85ae33) | after (this card) |
|---|------|------------------|--------------------|
| B1/A1 | fresh, empty registry | `E_MODEL_NOT_FOUND: None is not a registry alias …` exit 2 | `…no model was named and the registry has no default to fall back on; known aliases: <none>. Run \`typed-gguf models pull\` … or \`typed-gguf models use <alias>\` …, or pass \`--model <alias\|path.gguf>\`` exit 2 |
| B2/A2 | one alias, `current: null` | same dead end, exit 2 | `model: requested <default> -> resolved stories260k (the registry's only alias)` + a real answer, exit 0 |
| B3/A3 | same home, bare `fit` | `E_MODEL_NOT_FOUND: None is not a registry alias or an existing file…` exit 2 | `model: stories260k` / `path: …` exit 0 |
| B4/A4 | registry with its default (a real pull) | resolves (exit 3: no runtime in that home) | unchanged, plus the note `resolved stories260k (the registry's \`current\`)` |
| A5 | two aliases, no `current` | — | still refuses, lists **both** aliases, names `models pull` / `models use` / `--model`; `models use stories15m` then answers |

## 2. The choice (DECISION SLACK), and where it sits in the SPEC

SPEC §2.7 defines the registry as `{alias, path, …}` plus the `current` default model; §2.8's
`models use <alias>` is the only verb that moves it. SPEC is **silent** on a registry that has
aliases and no `current`, so the card's own preference applies: *"`ask` falls back to the sole alias
when unambiguous"*. Implemented once, in `store.find_default(registry) -> Default | None`:

1. `current`, when it is set and still resolves (`source="current"`);
2. else the registry's **sole** alias (`source="sole"`) — one entry is an unambiguous choice, and a
   `registry.json` written before that key existed (or whose default was lost) must not be a dead end;
3. else `None`: two or more aliases with no `current` stay unresolved, because which model to run is
   the user's call, never a guess. The refusal text names the two remedies and `--model`.

`resolve(registry, None, use_current=True)` is the single caller-facing entry point, so `ask`, `run`,
`fit`, `calibrate` and `doctor` all inherit the same answer. `Registry.current` itself is **never**
rewritten by the fallback: it stays provenance (`Default.source`), not a silent write.

Two consequences elsewhere, both through that one call site:

* `doctor` reads `store.resolve(registry, None, use_current=True)` at `cli.py:452`. On the one-alias
  home it now reports `"model": {"alias": "stories260k", …}` (observed,
  `after-receipt-two.txt` [A6]); at `c85ae33` the same line was `None` and the report warned
  `no model in the registry; run models pull` (read from `git show c85ae33:src/typed_gguf/cli.py`);
* `serve`'s `GET /v1/models` (`api/http.py:248`) uses the same call, so it now lists the compat name
  `jev-latest` whenever a request for it resolves; an empty registry still lists `[]` (SPEC §2.9).
  Read from the code — the serve gates already cover the `current` path and stay green.

## 3. Nothing changed for a caller who names a model

* `--model <alias|path.gguf>` takes the same path as before (`_resolve_model` / `_resolve_model_ref`
  are untouched for a non-`None` ref); pinned in
  `tests/test_default_model.py::test_an_explicit_reference_is_never_rerouted_by_the_default`.
* the CLI's new provenance line is printed **only** for a call that named no model (and never for
  `--route auto`, which chooses the model itself), so an explicit caller's stderr is byte-identical:
  `test_an_explicit_model_prints_no_note`.
* `resolve(registry, None)` without `use_current` is still `None` (the pre-existing pin
  `tests/test_registry_store.py::test_resolve_by_alias_or_path`).
* the alias that answered was already visible (`response["model"]`); the new stderr line adds the
  *requested* half in the repo's own naming — `model: requested <default> -> resolved <alias> (why)`
  — so the resolution is visible without touching the response wire (stdout stays the response).

## 4. Pins (RED → GREEN)

`tests/test_default_model.py`, new, 17 gates. RED at `a94d773` (`12 failed, 4 passed`): no
`store.find_default`, `_resolve_model` raised on the sole-alias home, the no-model text still said
`None`, and no note existed. GREEN at `49a8746` (16 passed; the README gate at `250955f` → 17).

* `find_default`: current wins; sole alias when `current` is unset; stale `current` → sole; two
  aliases → `None`; empty → `None`.
* CLI resolution: bare `ask` request → the sole alias; bare `fit` ref → the sole alias; two aliases
  still refuse listing both; explicit refs untouched.
* the no-model error: names `models pull`, `models use` and `--model`, lists `<none>` / the aliases,
  and no longer says `None`.
* the provenance note: `<default> -> resolved <alias>`, `current` vs `only alias`, absent for
  `--model`, present for `run` too.
* the README's step 2 must say what the default is and name `models use <alias>`.

**Error-text pins updated:** none had to change — the two `E_MODEL_NOT_FOUND` texts that *were*
pinned anywhere in the suite are only asserted by code (`"E_MODEL_NOT_FOUND" in stderr`, e.g.
`tests/test_cli.py:283`, `tests/test_calibration.py:877`, `tests/test_cli_doctor_branches.py:207`)
and both keep their code and exit status. The text that changed is the `ref is None` branch, which no
test quoted before; it is now pinned by the two gates above (it would have been unpinnable prose).

## 5. Gates

* CI-shape suite (`docs`, `TYPED_GGUF_TEST_BLOCK_NET=1`, 4-empty-libs bundle stub, `TYPED_GGUF_HOME`
  unset): **1771 passed / 59 skipped / 0 failed** (baseline at `c85ae33`: 1754/59/0; the 17 new
  gates are exactly the difference). Targeted files:
  `test_default_model.py test_registry_store.py test_cli.py test_cli_e1a.py test_cli_doctor_branches.py
  test_fit.py test_calibration.py test_serve.py test_keep_cli.py test_e1c_mutation_pins.py test_routing.py`
  → 409 passed / 1 skipped.
* `env -u PYTHONPATH uv run --extra dev ruff check src tests tools docs .github` → **All checks passed!**
* `uv build` → `typed_gguf-0.2.3.tar.gz` + `typed_gguf-0.2.3-py3-none-any.whl`.
* docs gate: `tests/test_public_docs.py` green; the README's quickstart sentence stays true (one
  paragraph added, no new root entries).

## 6. Tier-M mutation sweep

Driver `docs/evidence/e2e/t_a0fa2dc0-default-model/mutmut_sweep.sh` (`--max-children 2`, one attempt, finished in ~27 s): pair `source_paths = ["src/typed_gguf/registry/store.py"]` +
`pytest_add_cli_args_test_selection = [tests/test_default_model.py, tests/test_registry_store.py,
tests/test_cli_e1a.py]`. `cli.py` is deliberately not under mutation (the standing convention: its two
hunks are wiring these gates drive end to end).

**317 mutants, 231 killed (72.9 %), 4 no-tests, 1 timeout, 81 survived.**
`docs/evidence/e2e/t_a0fa2dc0-default-model/mutmut_results.txt` lists every non-killed mutant. The
new surface is fully killed: a scoped re-run of `x_find_default*` reports **19/19 killed**
(`mutmut_find_default_killed.txt`). The 81 survivors are the module's pre-existing surface this
three-file selection does not drive — `load_registry` (19), `Entry.from_dict` (9), `data_home` (9),
`add_entry` (9), `_quarantine` (9), `remove_entry` (7), `save_registry` (6), `_atomic_write` (6),
`resolve`'s path-lookup arms (4, the `os.sep in ref or …` probe), `slugify` (3) — plus the 4
`no-tests` lines on `calibration_path` and 1 `add_entry` timeout. Tier M's threshold is soft; none of
those lines moved in this card.

## 7. What is left to the host

Nothing here needs a GPU or the 4B: the live leg ran in-container against real HF (a 1.19 MB and a
15 MB `ggml-org/models` file, `lfs.oid`-verified) and answered through the real engine. The
`Vulkan`/4B rung is unaffected by this change (no path in it touches the resolver's default), so no
host leg is claimed.
