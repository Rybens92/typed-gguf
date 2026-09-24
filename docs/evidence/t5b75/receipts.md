# Policy v2 (card t_5b754458) — receipts

Decision (user, 2026-09-20): option A — the measured-good policy becomes the product default.
`options.cue`: `shipped` → **`json_instructed`**; `options.chat_format`: `answer_sheet` →
**`role_split`**; `options.json_contract`: **`question`** (unchanged value, now actually used).
Grounds: E3e (`t_4c48f40a`) + audit (`t_57bd3db2`) + [host] probes (`t_9bcbecff`): Tiel 22/60 → 53/60,
Occamy 26/60 → 54/60, 4B 50/60 (inside the noise of the 51/60 best cell).

## 1. RED proof on the pre-v2 tree

    git stash push -- src/ggufone/schema.py src/ggufone/bench/harness.py src/ggufone/cli.py tools/e2_reproduce.py
    uv run --frozen --offline --extra dev pytest -q -rf -p no:cacheprovider tests/test_policy_v2.py

`9 failed, 4 passed, 1 skipped` (exit 1) — the source tree without the flip, the new file present.
Full log: `.t5b75/logs/red_pre_v2_full.txt`. The failing pins:

* `test_the_default_cue_is_the_instructed_one`
* `test_the_default_chat_format_is_the_role_split`
* `test_the_bench_and_the_cli_ask_for_the_v2_cell_by_default`
* `test_the_reproduce_tool_defaults_to_the_v2_cell`
* `test_a_no_flags_request_renders_the_v2_cell_byte_for_byte_plain`
* `test_a_no_flags_request_renders_the_v2_cell_byte_for_byte_chat_template`
* `test_the_no_flags_engine_publishes_the_v2_policy_and_the_value_row_verdict`
* `test_the_reproduce_line_names_the_pre_v2_cell_and_stays_silent_for_v2`
* `test_the_report_marks_the_pre_v2_policy_and_not_the_default`

(The first RED run of the same file, before the documents carried their marker too, was
`11 failed, 2 passed, 1 skipped` — `.t5b75/logs/red_pre_v2.txt`.)

## 2. GREEN

    uv run --frozen --offline --extra dev pytest -q -rs -p no:cacheprovider
    → 1345 passed, 48 skipped  (exit 0)   [.t5b75/logs/suite_green_final.txt]

The 48 skips are the `--run-network` live set, the repo's normal offline state (they are the tests
that need the network/model); nothing in this card's paths is skipped.

    python3 docs/verify_runtime_contract.py      → failures: 0  skips: 0   (exit 0)
                                                   [.t5b75/logs/oracle_after_flip.txt]

    ruff check src tests tools .t5b75 .t9bcb     → All checks passed!
    (the 86 hits of a bare `ruff check .` are all in other cards' scratch dirs: .e3e/.e2e/.e3c_tiel)

## 3. The re-measured 4B row (no policy flag at all)

    bash .t5b75/run_default_row.sh
      = `tools/e2_reproduce.py --suite quality --model <4B> --backend vulkan --threads 4 --items 60`
        with **no** `--cue` / `--chat-format` / `--json-contract`
    → 50/60 = 0.833 [0.720–0.906]  ·  docs/evidence/e2_quality_v2_t_5b754458.json
    compare with the published arm:  python3 .t5b75/compare_default_row.py
    → **60/60 items identical** on `prefix_tokens`, `got`, `correct`, `cue`;
       agreement 50/60 on both sides   [.t5b75/logs/identity_default_vs_arm.txt]

The live half of the parity pin, in the suite's own vocabulary:

    GGUFONE_RUNTIME_DIR=<b11026-linux-x64-vulkan> pytest -q --run-network \
        -k test_the_defaults_measure_the_published_v2_cell_on_the_4b tests/test_policy_v2.py
    → 1 passed in 5.08s              [.t5b75/logs/live_default_gate.txt]

## 4. The flag parity (offline, in the suite)

`test_a_no_flags_request_renders_the_v2_cell_byte_for_byte_plain` and
`..._chat_template` compare a request that names nothing with `{cue: json_instructed,
chat_format: role_split, json_contract: question}` spelled out: identical `prefix`, tail, JSON
opener and token ids, through the plain assembly and through a ChatML family template.

## 5. The documents

* `docs/BENCHMARKS.md`: the §-level policy marker in the header, §2's marker, the new **§2.3** (the
  default row, spliced from the report by `.t5b75/splice_benchmarks.py`), §3, §5, §6, §7.4.2 (Tiel),
  §8, §9 markers; "what stays pre-v2" is one blockquote in the header.
* `docs/TEMPLATES.md` §4: the cue-shape guidance and the E3e section now name the defaults.
* `docs/evidence/e3e_role_split_t_9bcbecff.md` §9: the Occamy pair marked policy v2
  (`.t9bcb/render_doc.py`, the renderer – the block is never hand-edited).
* CLI: `--cue` / `--chat-format` / `--json-contract` default to the schema constants and their help
  and docstrings say so; the bench's `reproduce:` line still prints a policy flag **only** when it is
  not the default, which is what keeps the published lines byte-identical.

## 6. What was *not* done

* No evidence artifact of an earlier campaign was re-written: the E3d/E3e tables keep their bytes and
  their stored `reproduce:` lines (recorded when the flags were the defaults); the live documents say
  which policy each row was measured under, and how to name it now.
* Latency / throughput / determinism / calibration (E2.5) rows stay pre-v2 by declaration — the
  optional E2-v2 campaign (option C) is what would re-measure them.
* No Tiel/Occamy default-run smoke on this box: the 35B files are 22+ GB against this container's
  8 GiB `memory.max`, and the card allows the explicit statement instead (§7.4.2 and the evidence
  doc §9 now carry it; the 4B default-run row is the live proof of the mapping).
