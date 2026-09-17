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
warm prefill_ms = 0.000  cold prefill_ms = 2246.4
determinism sha256 = f4158c0f26d1c23dda1a3d7cc607b20c8433b6790a305dbb519282fb51598eb6
state round-trip max |delta| = 0.000e+00
waves: capped=16 single=8 max |delta| = 0.000e+00
61 passed in 209.56s
```

```
$ uv run pytest -q
456 passed, 12 skipped in 20.47s

$ uv run python docs/verify_runtime_contract.py          # offline
failures: 0  skips: 1        (skip = no runtime installed in this env; see the live run below)

$ GGUFONE_RUNTIME_DIR=<rt> uv run python docs/verify_runtime_contract.py
failures: 0  skips: 1        (live section B green; the remaining skip is the absent Spark sha
                              re-check path, which is covered by E1a's host gate)
```

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

PERF_TABLE

The recon reference point is 14–20 ms warm on Vulkan with a warm state cache **[recon]**; this
milestone publishes its own measured CPU number as **[target]** (correctness before speed).

## 5. Limits / not covered here

- No GPU in this container: Vulkan/CUDA placement, `W_VULKAN_WARMUP` and the GPU half of
  E1b's cost story are not measurable here (E1a's host gate covers the device detection side;
  E2 owns the benchmark matrix).
- `serve` / `mcp` (§2.9) are not part of the E1b deliverable and are stubs.
- `kv_type` other than `f16`/`auto` flips `flash_attn_type` on but is not exercised on a real
  model here (quantized V cache needs a GPU-side check in E2).
- The 255-option limit is enforced in the schema; a 255-candidate *decode* is not run (the wave
  math is covered by the fake session).
- Mutation testing: see §6 (scope and score are stated for the tree that was frozen).

## 6. Mutation testing (Tier M)

MUTATION_SECTION
