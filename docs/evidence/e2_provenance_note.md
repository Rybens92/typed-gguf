# E2 provenance note — the tables are genuine; the printed `reproduce:` recipe was broken on the shipping commits

**Verdict (audit card `t_78f5ea7a`, 2026-09-18).** Every value in `docs/BENCHMARKS.md` and
`docs/evidence/e2_*.json` is a real measurement; the `reproduce:` command printed inside those
reports **fails on the two commits that ship them** (`fff127e`, `4e1d549`). Recommendation, applied
here: **keep the numbers, record the caveat.** This note is documentation only — no measured value
was edited, and the diff that adds it touches `docs/` and this file.

Source of record: `state/fights/e2-provenance/scorecard.md` (in this working tree; raw run logs, the
163-check output and the worker's session receipts are in its `logs/`). This card's own re-runs are
in `.e2e/t_ed95f756-provenance-note/`.

## 1. The recipe was broken on the shipping commits

Verbatim recorded line (`docs/evidence/e2_latency.json → commands.reproduce`), run from a fresh clone
of `4e1d549` with a local bundle pointed at by `GGUFONE_RUNTIME_DIR`:

```
uv run ggufone bench --suite latency --model /var/home/rybens/.hermes/models/Spark-X2.5-4B-Q8_0.gguf \
    --backend auto --runs 5 --threads 4 --json
load_backend: loaded Vulkan backend from …/b11026-linux-x64-vulkan/libggml-vulkan.so
load_backend: loaded CPU backend from …/b11026-linux-x64-vulkan/libggml-cpu-haswell.so
error: E_INTERNAL: AttributeError: 'Placement' object has no attribute 'kv_type'      # exit 4
```

Fail-fast — the ladder is built before the first load attempt, so only the model header read has to
succeed (the audit measured ~1 s to the crash; this card's re-run took 5.8 s wall for the whole CLI
invocation, interpreter start-up included). Independently confirmed on three trees: the audit's two
fresh worktrees (`4e1d549`, `fff127e` — `logs/wt-*-crash.raw`), this card's fresh clone of `4e1d549`
(three runs, `.e2e/t_ed95f756-provenance-note/crash-4e1d549*.err`, `.exit` = 4), and the fix card's
own pre-fix repro (`.e2e/t_31b3943a-bench-placement/repro_published_cmd.err` + `.exit` = 4,
committed with the fix). `--suite quality --runs 1` on `fff127e` shows the same tail.

**Root cause.** The bench harness names its placement with a minimal
`harness.Placement(n_gpu_layers)` — the frozen dataclass at `src/ggufone/bench/harness.py:227`,
passed as `fit_plan=` in `LiveModel.load` (`harness.py:380` on `4e1d549`) — while the E1c fit fix
`5410e42` (04:50:35Z — *earlier* than the bench commit `fff127e`, 05:01:17Z) added a load-time
degradation ladder to `session.open_model` that reads `plan.kv_type`. The two halves were developed
from one shared tree by two cards and only ever met in the commits, where the missing normalisation
is an `AttributeError`. Fix `8d4fc9f` (`fit.coerce_plan`) is exactly that normalisation: **the
recorded commands work from `8d4fc9f` onward** — all seven distinct `reproduce:` lines exit 0
post-fix (bounded where the recorded form is a multi-hour campaign; scorecard §2).

## 2. Where the numbers actually came from

The campaign ran through `tools/e2_reproduce.py` on the worker's private clone `/work/e2repo` whose
engine was `c095c51` (the commit *before* the E1c fit fix) plus this card's bench surface — a tree
state **no commit contains**, which is why the published recipe cannot re-create the report on the
commit that publishes it. The suite code that produced the rows is the same code shipped in
`fff127e`; only the loader half differed. The runs are reconstructible from primary sources (session
export + per-run logs; scorecard §4).

## 3. Why the numbers are still trustworthy (re-generation is possible on demand)

The audit's bounded re-runs on the fixed tree (scorecard §3) reproduce the *payload* of the published
reports:

* `e2_quality.json` — per-item decisions identical for the compared items; the 4B's `c01`
  probabilities agree to ~13 significant digits (published `billing 0.7689847751553348` vs
  `0.7689847751553072`, Δ ≈ 4e-14);
* `e2_calibration.json` — the full 60 items re-run as published: agreement 28/60 identical, ECE
  `0.2859817561687615` vs `0.28598175616876126` (~15 digits), CI and correlation equally exact;
* **163 internal-consistency checks** on the published JSONs (summary ordering, row counts,
  `one_shot == serve + model_load`, `agreement == correct/n`, per-item probabilities summing to 1,
  ECE recomputed from the published bins to 1e-9 for all three modes) — the only flagged item is a
  p50 inversion at wave-scaling `N=2` 2.6 % below `N=1`, inside the contention the document itself
  describes;
* this card's own post-fix control (fresh clone of `8d4fc9f`, `--suite quality --backend cpu
  --items 1 --runs 1 --threads 4`, exit 0) reproduces the published `c01` row **bit-for-bit** — all
  four candidate probabilities, `confidence`, `coverage` and the decision
  (`.e2e/t_ed95f756-provenance-note/c01-published-vs-control.txt`). The audit's 24-core host agreed
  to ~4e-14; on the container-class CPU path the agreement is exact.

So re-generation is possible **on demand** on the fixed tree, and it changes no value modulo the
box. The *timings* are box-scaled by construction: the audit host (24 real cores) measured
0.18–0.63× the published container figures (2-CPU-seconds/s quota), with identical shape. Read every
timing row as "the 2-CPU container, that hour"; a re-run on faster hardware produces smaller
numbers, not better ones.

## 4. Determinism digests are *within-tree* witnesses

Internal determinism reproduced exactly (3 repeats per backend, `identical: true`; the digest is
`sha256` over the timings-stripped response body). The **digest value** is not comparable across
tree states: the published `cpu` digest `sha256:7ab32e58…` vs `sha256:a81f13b1…` on the fixed tree,
because the request payload is byte-identical (`diff` of the `.request` files → no difference) while
the response envelope gained fields after the fix (`engine.placement{note, kv_type, degraded,
attempts}`, `engine.fit`). Compare digests only within one tree state.

## 5. What to do with this

* Cite the tables as measurements of the pinned model + bundle **on the 2-CPU container**, not as
  re-runnable-by-one-command artefacts. The recipe works from `8d4fc9f` (and on every later commit,
  including HEAD).
* Regenerating the cheap suites (`quality` ×2 models, `calibration`; ≈8 min on the audit host) is
  available on demand; the latency/throughput tables stay as published, because re-measuring them
  would change the box for every row.
* Nothing here invalidates E2.5/E3: they build on the values, and the values stand.
