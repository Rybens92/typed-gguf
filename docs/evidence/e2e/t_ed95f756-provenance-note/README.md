# Card `t_ed95f756` — E2 provenance note: fresh re-verification of the crash claim

The note `docs/evidence/e2_provenance_note.md` and the pointer in `docs/BENCHMARKS.md` assert that the
recorded E2 `reproduce:` command crashes on the pre-fix shipping commit `4e1d549`. These files are
this card's own independent re-run of that assertion (the audit's runs are in
`state/fights/e2-provenance/logs/`, the fix card's committed repro in
`.e2e/t_31b3943a-bench-placement/`).

Container (`HERMES_KANBAN_WORKSPACE` = shared tree, docker backend); bundle from the container's own
runtime store: `/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan`
(`GGUFONE_RUNTIME_DIR`). The published table's `reproduce:` line is a CLI line, so the CLI form is
what was run — no bounded override, verbatim:

```bash
git clone /home/rybens/workspace/ggufone /work/e2prov-4e1d549 && git -C /work/e2prov-4e1d549 checkout 4e1d549
GGUFONE_RUNTIME_DIR=<bundle> uv run ggufone bench --suite latency \
    --model /var/home/rybens/.hermes/models/Spark-X2.5-4B-Q8_0.gguf \
    --backend auto --runs 5 --threads 4 --json
```

| tree | files | exit | tail |
|---|---|---|---|
| fresh clone @ `4e1d549` | `crash-4e1d549.err` / `.out` / `.exit` | **4** | `error: E_INTERNAL: AttributeError: 'Placement' object has no attribute 'kv_type'` |
| fresh clone @ `8d4fc9f` (post-fix control: `--suite quality --backend cpu --items 1 --runs 1 --threads 4`) | `control-8d4fc9f-quality.*` | **0** (48.7 s wall) | real report written; full row in `.out` |

Post-fix control, and the strongest single fact this card adds: the control's `c01` row is
**bit-identical to the published `docs/evidence/e2_quality.json` row** — every field, including all
four candidate probabilities, `confidence` and `coverage` (`c01-published-vs-control.txt`; the
published values were measured on a box with the same container-class CPU path). So on the fixed
tree the same bench path both *runs* and *lands on the published number*; only a different box
(the auditor's 24-core host) shifts the last digits by ~1e-14.

Crash log (verbatim tail, `.err`):

```
load_backend: loaded RPC backend from …/b11026-linux-x64-vulkan/libggml-rpc.so
ggml_vulkan: No devices found.
load_backend: loaded Vulkan backend from …/b11026-linux-x64-vulkan/libggml-vulkan.so
load_backend: loaded CPU backend from …/b11026-linux-x64-vulkan/libggml-cpu-haswell.so
error: E_INTERNAL: AttributeError: 'Placement' object has no attribute 'kv_type'
```

Wall time to the crash: 5.77 s wall for the whole CLI invocation (`crash-4e1d549-run3.err`, measured;
~1 s of that is interpreter start-up — fail-fast, the degradation ladder is built before the first
load attempt). Runs 2 and 3 (`crash-4e1d549-run2.err`, `-run3.err`) have a tail identical to run 1
(`diff` → identical); `git status` in the clone is clean before and after (nothing written outside
the clone).

No measured value in `docs/` was changed by this card: the diff is the note, the BENCHMARKS pointer
(and the one-line F3 caveat next to the determinism table).
