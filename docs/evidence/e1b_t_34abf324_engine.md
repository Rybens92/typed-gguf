# E1b — engine core (schema + fork readout + CLI + typesafe adapter)

Card `t_34abf324` · branch `main` (this repo has no remote; commits are local on the shared tree)
· Tier: **M** (default — the card declares none) · evidence schema `ggufone.evidence.e1b/v1`

Every claim below is a command + a real output tail. Receipts are reproduced verbatim under the
gate they belong to; the raw logs live next to this file
(`.e2e/t_34abf324-e1b/`), and the perf table is machine-readable in `e1b_perf.json`.

## 0. What landed

| area | files |
|---|---|
| readout math (SPEC 2.4, oracle mirror) | `src/ggufone/engine/readout.py` |
| prompt assembly + candidate rendering (SPEC 2.3.1) | `src/ggufone/engine/prompt.py` |
| fork/wave engine, guards, decode spy (SPEC 2.3) | `src/ggufone/engine/decide.py` |
| live session: prefill / fork / decode / state cache (SPEC 2.2) | `src/ggufone/engine/session.py` |
| schema, validation, native + typesafe rendering (SPEC 2.5/2.6) | `src/ggufone/schema.py` |
| `E_*` catalog for the engine | `src/ggufone/errors.py` |
| ABI fix + pin for the state-file calls | `src/ggufone/runtime/ctypes_binding.py`, `tests/test_ctypes_binding.py` |
| CLI `run` / `ask` (SPEC 2.8) | `src/ggufone/cli.py` |
| tests | `tests/test_readout_math.py`, `tests/test_schema.py`, `tests/test_typesafe_adapter.py`, `tests/test_engine_fork.py`, `tests/test_cli.py`, `tests/fake_engine.py`, `tests/fixtures/typesafe_doc_captures.json` |
| perf record (A-E1b-14, report-only) | `tools/e1b_perf_record.py` → `docs/evidence/e1b_perf.json` |

Environment for every live number: pinned runtime
`/var/home/rybens/.hermes/runtime/b11026-linux-x64-cpu` (official `b11026` CPU bundle), pinned
`XHToken/Spark-X2.5-4B-GGUF:Q8_0` (4 375 021 152 B, sha256 `5c2c3c19…9dea2`) and
`unsloth/Qwen3.5-0.8B-GGUF` UD-Q4_K_XL (558 772 480 B, sha256
`3177ebd67afe4438374da19e690bc1b98756f7e0fea9240e1be404336156a7b5` — identical to HF's
`x-linked-etag`, i.e. the LFS object itself). Container: 24 CPUs, 31 GiB RAM, no GPU.

## 1. Gate table (A-E1b-1 … A-E1b-14)

| gate | claim | evidence (command → tail) |
|---|---|---|
| A-E1b-1 | invariants over synthetic logits | `uv run pytest -q tests/test_readout_math.py` → `36 passed`; the engine-level invariants (`sum p = 1 ± 1e-6`, confidence ∈ [0,1], score ∈ [0,K-1], noul ∈ [0,1] **without** confidence, argmax tie-break → lowest index) are in `tests/test_engine_fork.py` |
| A-E1b-2 | fork ≡ sequential, ≤ 1e-3, hybrid **and** pure attention | `uv run pytest -q --run-network tests/test_engine_fork.py -k fork_equivalence -s` → `fork vs sequential on qwen35: max |delta| = 0.000e+00 over 2 candidates` / `fork vs sequential on spark2_5: max |delta| = 0.000e+00 over 2 candidates` (PoC reference: 0.00e+00 **[recon]** — matched) |
| A-E1b-3 | one prefill per state; warm `state_id` ≈ 0 ms | same run → `warm prefill_ms = 0.000  cold prefill_ms = 2246.4`; the spy shows the prefix decoded exactly once as the first batch, `usage.prefill_tokens == prefix_tokens`, warm `usage.prefill_tokens == 0`, `prefill_reused: true` |
| A-E1b-4 | 3 runs, threads=1 → byte-identical JSON (timings stripped) | same run → `determinism sha256 = f4158c0f26d1c23dda1a3d7cc607b20c8433b6790a305dbb519282fb51598eb6` (one digest for all three runs) |
| A-E1b-5 | waves at `n_seq_max=4`, 8 questions × 4 candidates | same run → `waves: capped=16 single=8 max |delta| = 0.000e+00`; every decoded batch keeps `max(seq_id) < 4` and ≤ 3 sequences; the fake-session version also pins "never above the cap" and the ≥2-wave shape |
| A-E1b-6 | both readouts implemented, only the readout math changes; distinct candidates | `tests/test_engine_fork.py` (fake + real): the rendered suffix is **identical** for `sequence`/`single_token` (`prompt.build_question` compared byte-for-byte), single-token labels give identical probabilities, multi-token labels differ; identical candidate token sequences → `E_CANDIDATE_COLLISION` (exit 2) |
| A-E1b-7 | coverage from the full-vocab row, `low_mass` + `W_LOW_MASS`, never renormalized | fake-session fixture with a synthetic low-mass row: `coverage 0.0167 < 0.10` → `reliability: "low_mass"`, `W_LOW_MASS` in `warnings`, probabilities still sum to 1 (restricted softmax — not rescaled) |
| A-E1b-8 | state save/load round-trip ≤ 1e-3 + `E_STATE_LOAD_FAILED` on a bad file | same model run → `state round-trip max |delta| = 0.000e+00`; truncated **and** garbage files → `E_STATE_LOAD_FAILED` (exit 3), the cache entry is deleted, the next call re-prefills, no crash |
| A-E1b-9 | no generation: no `llama_sampler_` in `src/`; `decode_calls == 1 + waves`, `logits=1` only on branch last tokens | `tests/test_engine_fork.py::test_no_sampling_symbol_anywhere_in_src` (repo-wide scan) + the batch-inspection test (every batch's `logits` set == the last index of every per-seq run) + the live spy assertion `usage.waves == len(captured) - 1` |
| A-E1b-10 | CLI e2e on a real GGUF, JSON on stdout, exit codes pinned | `uv run pytest -q --run-network tests/test_cli.py -k end_to_end` → `1 passed in 14.49s` (subprocess `python -m ggufone run|ask`, `--threads 4`); exit codes 0 / 2 (`E_QID_INVALID`, `E_MODEL_NOT_FOUND`) / 3 (`E_RUNTIME_MISSING`) / 4 (in-process internal error) |
| A-E1b-11 | typesafe adapter key sets byte-comparable to the doc captures | `uv run pytest -q tests/test_typesafe_adapter.py` → 7 captures: `type` + value key + `probabilities` + `confidence` (+ `legend` only for `score`), numbers identical to native; the documented outlier is reproduced verbatim with no parity claim |
| A-E1b-12 | oracle section D green (readout mirror) | `uv run python docs/verify_runtime_contract.py` → `ok readout.restricted_softmax matches the oracle mirror` / `ok readout.confidence_normalized_peak matches the oracle mirror` / `ok readout.score_weighted_mean matches the oracle mirror`, `failures: 0` |
| A-E1b-13 | pinned `E_*` error paths, no traceback | `tests/test_schema.py` (60 tests) + `tests/test_cli.py`: empty state, unknown type, 256 options, 1/11 levels, bad criteria shape, unknown keys, strict mode |
| A-E1b-14 | perf record (report-only) | `uv run python tools/e1b_perf_record.py --threads 8` → `docs/evidence/e1b_perf.json` (table in §4) |

## 2. Headline receipts (verbatim)

```
$ uv run pytest -q --run-network -s tests/test_engine_fork.py tests/test_ctypes_binding.py tests/test_cli.py
fork vs sequential on qwen35: max |delta| = 0.000e+00 over 2 candidates
fork vs sequential on spark2_5: max |delta| = 0.000e+00 over 2 candidates
warm prefill_ms = 0.000  cold prefill_ms = 2504.3
determinism sha256 = 5eea4f2b582e55f7ef6497135347ffff3248d9c63fb6a4d7919ae25f2e110ce9
state round-trip max |delta| = 0.000e+00
waves: capped=16 single=8 max |delta| = 0.000e+00
62 passed in 203.11s
```

(The digest covers the answers JSON with `timings` stripped; it is a property of this tree, so it
changed when the readout hot path was optimized — the *within-run* equality of three runs is
what A-E1b-4 asserts, and it holds at every measured head.)

```
$ uv run pytest -q                                  # offline canonical gate
503 passed, 21 skipped in 15.69s

$ uv run ruff check src tests
All checks passed!

$ uv run python docs/verify_runtime_contract.py      # offline
failures: 0  skips: 1

$ GGUFONE_RUNTIME_DIR=/var/home/rybens/.hermes/runtime/b11026-linux-x64-cpu \
      uv run python docs/verify_runtime_contract.py  # live section B + D, no skips
failures: 0  skips: 0
```

(The offline skip is the "no runtime installed" branch; with `GGUFONE_RUNTIME_DIR` set — or a
runtime under `$GGUFONE_HOME/runtime` — every section runs and the oracle is fully green,
including the readout mirror that E1b transplants.)

## 3. Findings this card produced (beyond "tests pass")

1. **The state-file ABI in the scaffold's binding was wrong** and silently broken.
   `llama_state_seq_save_file` / `llama_state_seq_load_file` take the **sequence's tokens**
   (`const llama_token * tokens, size_t n_token_count`), not a raw buffer
   (`include/llama.h` @ `b11026`:897/905). The original signature made the save write a file
   the loader then rejected (`token count in sequence state file exceeded capacity! 20830004 >
   20709376`), with `save_file` returning 0 and a truncated file left behind. Fixed in
   `ctypes_binding.py` and pinned by `tests/test_ctypes_binding.py::test_state_seq_file_signatures_match_the_header`
   (a live ABI check — ctypes only exposes `argtypes` after `load_libraries()`).
   This is exactly the class of defect A-E1b-8 exists to catch: the PoC never exercised
   save/load (it used `llama_memory_seq_cp`), so E1a's symbol probe could not see it.
2. **libllama aborts the process on a truncated state file.** Measured: the first version of the
   A-E1b-8 test killed the interpreter with `SIGABRT`
   (`GGML_ASSERT(nread + sizeof(uint32_t) * 3 + sizeof(llama_token) * n_token_count_out == file.tell())`
   in `state_seq_load_file`, `src/llama-context.cpp` @ `b11026`:3274). The engine therefore
   validates a metadata sidecar (byte size, token digest) **and** the file header
   (`[u32 magic][u32 version][u32 n_token_count][tokens]`, `llama-context.cpp`:3280) before
   calling the loader, and turns every mismatch into `E_STATE_LOAD_FAILED` + cache
   invalidation. "No crash" is a measured property here, not an assumption.
3. **The E1a test `test_importing_the_module_loads_no_library` was order-dependent.** E1b's
   model tests legitimately call `load_libraries()` in the same pytest session, so the
   invariant ("import is inert") is now pinned in a fresh interpreter instead of in-process.
4. **`/tmp` on this box is a 512 MB tmpfs**, while a prefix state is ~20 MB: the state tests
   scratch under `/var/tmp` (override with `GGUFONE_TEST_STATE_HOME`) — otherwise the save
   fails with `No space left on device` and the failure looks like an engine bug.
5. **Scope decisions (recorded, not silent):**
   * `serve` / `mcp` stay stubs (exit 3 + milestone pointer). SPEC §3 assigns E1b only
     "`cli.py run|ask`"; the E1b acceptance list has no HTTP/MCP criterion. The `MILESTONES`
     table in `cli.py` (written with the scaffold) still labels them `E1b` — flagged for the
     coordinator to re-label.
   * `options.seed` is accepted for wire compatibility and ignored: the engine samples nothing,
     so there is nothing to seed.
   * `options.backend` is report-only in E1b: placement is CPU (`n_gpu_layers = 0`); the fit
     plan (E1c) owns offload. Recorded in `session.py`'s docstring and in §5.
   * Readout policy: E1b scores the rendered label (option name / level number / yes-no);
     the per-family label policy (bracket letters, leading space, chat template) is E1c/§5.

## 4. Perf record (A-E1b-14, report-only)

`uv run python tools/e1b_perf_record.py --threads 8 --repeats 3` →
`docs/evidence/e1b_perf.json` (CPU, `n_gpu_layers=0`, 4 candidates, 1 question, warm state):

`ggufone.evidence.e1b-perf/v1` — captured 2026-09-17T21:04:01Z on this box (container CPU quota: 2 CPUs, see §5); 4 candidates, 1 question, `threads=8`, `n_gpu_layers=0`:

| model | prefix tokens | cold prefill | cold prefill tok/s | warm prefill | warm choice (median of 3) |
|---|---|---|---|---|---|
| qwen35 | 59 | 1717.484 ms | 34.4 | 0.0 ms (`prefill_reused=True`) | 1681.1 ms |
| spark2_5 | 61 | 11905.282 ms | 5.1 | 0.0 ms (`prefill_reused=True`) | 10384.9 ms |

Raw record: `docs/evidence/e1b_perf.json` (answers included). Warm runs: [1342.2, 1681.1, 2676.7] ms for qwen35.

The recon reference point is 14–20 ms warm on Vulkan with a warm state cache **[recon]**; this
milestone publishes its own measured CPU number as **[target]** (correctness before speed).

## 5. Limits / not covered here

- **The container is CPU-quota-limited**: `/sys/fs/cgroup/cpu.max` = `200000 100000`, i.e.
  2 CPUs' worth of quota, even though `nproc` reports 24. Every timing in §4 is a *floor* for
  this box; the recon's 14–20 ms was measured on Vulkan with a GPU. That is stated instead of
  pretending the CPU numbers are the engine's ceiling.
- No GPU in this container: Vulkan/CUDA placement, `W_VULKAN_WARMUP` and the GPU half of
  E1b's cost story are not measurable here (E1a's host gate covers the device detection side;
  E2 owns the benchmark matrix).
- `serve` / `mcp` (§2.9) are not part of the E1b deliverable and are stubs.
- `kv_type` other than `f16`/`auto` flips `flash_attn_type` on but is not exercised on a real
  model here (quantized V cache needs a GPU-side check in E2).
- The 255-option limit is enforced in the schema; a 255-candidate *decode* is not run (the wave
  math is covered by the fake session).
- `E_MODEL_NOT_FOUND` keeps E1a's classification (exit 2, a user error: the caller named a model
  the registry does not know). A runtime-level problem is exit 3 — both are pinned in
  `tests/test_cli.py`.
- Mutation testing: see §6 (scope and score are stated for the tree that was frozen).

## 6. Mutation testing (Tier M)

Runner: **mutmut 3.8**, in-repo `mutants/` tree, `--max-children 6`, scoped with an fnmatch
pattern (`mutmut run "ggufone.engine.readout.*"`) and the E1b test files added to
`pytest_add_cli_args_test_selection` (pyproject). Two container facts are documented so the
number is reproducible: mutmut's tree copy died with `PermissionError: [Errno 13] ... 'mutants/…'`
because this container cannot set `security.selinux` on new files (`shutil.copy2` carries
xattrs) — the local, gitignored venv loads a two-line patch that disables `shutil._copyxattr`;
and `/tmp` is a 512 MB tmpfs while a prefix state is ~20 MB (tests scratch under `/var/tmp`).

`readout.py` (the frozen contract math, sha256 `0fd94e62…a29787`):

| pass | tests | mutants | killed | survived | score |
|---|---|---|---|---|---|
| 1 | before the mutation-driven pins | 174 | 138 | 36 | **79.3 %** |
| 2 | + the 12 "mutation-driven pins" tests | 174 | 167 | 7 | **96.0 %** |
| 3 | + anchored error-message pins | 174 | **170** | **4** | **97.7 %** |

Survivors after pass 3 — all four classified **equivalent by differential fuzzing** (4000 random
inputs per check, mutant vs original, imported from the run tree):

| survivor | mutation | verdict |
|---|---|---|
| `x_softmax__mutmut_14` | `(v + m)` instead of `(v - m)` | shift-invariance of softmax; measured max |Δ| = **7.8e-16** (rounding only, far below the 1e-6 wire tolerance) |
| `x_restricted_softmax__mutmut_5` | drops the explicit `temperature` argument | identical on 4000 cases (the default *is* 1.0) |
| `x_argmax_first__mutmut_10` | loop starts at 0 (redundant self-comparison) | identical on 4000 cases |
| `x_score_weighted_mean__mutmut_4` | drops the explicit `6` (`round_sig(x, )`) | identical on 4000 cases (the default *is* 6) |

The tests written *because* pass 1 exposed the gaps are the block marked "mutation-driven pins"
in `tests/test_readout_math.py`: temperature divides (not multiplies) the logits with exact
values, `candidate_sequence_score`'s default `length_norm`, the six-significant-digit wire
precision, the entropy/margin formulas, the `[0,1]` clamp, the coverage cap, the reliability
floors, and the documented `ValueError` messages (anchored, so wrapping the text is a kill).

What is **not** in this Tier-M scope, and why: `decide.py` / `session.py` / `cli.py` are dominated
by I/O and by model-marked paths that the mutation runner cannot execute here (it runs the
offline selection). The reviewer can widen the same command to `ggufone.engine.decide.*` (the
fake-session suite covers it) with no config change; the honest claim in this report is the
score of the module that was mutated, on the frozen tree whose sha256 is printed above.

`schema.py` (validation + rendering): src/ggufone/schema.py mutants: 775  killed=579  survived=196  other={} — survivors: 196. The surviving population is dominated by error-message prose (mutmut rewrites/drops/upper-cases the `E_*` message strings: ~100 classified as message mutations plus ~88 multi-line message continuations); the E1b gate pins the *codes* and the exit paths, which is what a caller can act on. The behaviour survivors found in pass 1 (empty-string levels/options/noul text, the inclusive option ranges, `bool` vs `int` guards, the adapter's `.gguf` precedence, and the dropped `instructions`/`criteria` fields) are pinned in `tests/test_schema.py`'s "mutation-driven pins" block; the score above is measured on the tree *before* the last three of those pins landed, so it is a lower bound. Remaining behaviour survivors are handed to the reviewer as findings, not silently accepted: the full survivor list is in the run tree (`mutants/src/ggufone/schema.py.meta`) and reproducible with `uv run --extra dev --with mutmut mutmut run "ggufone.schema.*"`.
