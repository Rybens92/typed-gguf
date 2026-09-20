# E2 provenance audit — scorecard (card `t_78f5ea7a`, auditor)

**Question (refute-or-confirm, mechanically):** can the committed E2 reports — `docs/evidence/e2_*.json`
and the tables in `docs/BENCHMARKS.md` — be reproduced from the commits that ship them?

**Answer in one line:** the recorded `reproduce:` command **crashes on both shipping commits
(`fff127e`, `4e1d549`)** — confirmed here on two fresh worktrees — but the *numbers themselves are
genuine*: bounded re-runs on the fixed tree reproduce the per-item data of `quality` and
`calibration` to floating-point precision, and 163 internal-consistency checks on the published
JSONs find no arithmetic defect. The gap is **provenance, not measurement**.

Audit window: 2026-09-18 13:02–13:45 UTC · operator host `localhost-live.home` · shared repo
`/home/rybens/workspace/ggufone` (HEAD moved during the audit: `a42aa13` → `d03b410` → `cc1f5da`).

---

## 0. Environment / noise context (every number below is "this box, this hour")

| item | value |
|---|---|
| host | `localhost-live.home`, Linux 7.2.4, x86_64, 24 CPUs seen, `/dev/dri` present (RTX 3060 Ti) |
| my worker cgroup | `cpu.max = max 100000` (no CPU cap), `memory.max = 4 GiB`, `memory.events`: 0 OOM over all runs |
| load average during runs | 6.0 – 13.8 (the box concurrently ran: the coordinator's host gate bench, card `t_f46cec41` benches, a mutmut sweep, `tsc`) |
| bundles | CPU b11026 copy used for the CPU measurements: `/tmp/e2-audit/runtime/b11026-linux-x64-cpu` (copied from the container overlay layer `…/containers/storage/overlay/867f972b…/diff/var/home/rybens/.hermes/runtime/b11026-linux-x64-cpu`); Vulkan b11026 at `~/.local/share/ggufone/runtime/b11026-linux-x64-vulkan` |
| models | Spark-X2.5-4B-Q8_0.gguf (4,375,021,152 B) · Qwen3.5-0.8B-UD-Q4_K_XL.gguf (558,772,480 B) |
| container the E2 campaign ran in (for comparison) | the same CPU bundle path, `cgroup_cpu_max 2.0`, 8 GiB, no GPU (`/dev/dri` absent) |

E2 note: the container's reporter wrote `cgroup_cpu_max 2.0`; on this host the cgroup files the
reporter reads are absent at `/sys/fs/cgroup/` root, so my re-run reports carry no quota field —
quoted here by hand instead.

---

## 1. Step 1 — the crash is real on both shipping commits (independent re-run)

Command (verbatim from `docs/evidence/e2_latency.json → commands.reproduce`), run from each
worktree with `GGUFONE_RUNTIME_DIR=<b11026 CPU bundle>`:

```
uv run ggufone bench --suite latency --model /var/home/rybens/.hermes/models/Spark-X2.5-4B-Q8_0.gguf \
    --backend auto --runs 5 --threads 4 --json
```

| tree | raw tail (log in `logs/`) | exit |
|---|---|---|
| `/tmp/e2-prov` @ `4e1d549` | `load_backend: loaded CPU backend from …/b11026-linux-x64-cpu/libggml-cpu-haswell.so` / `error: E_INTERNAL: AttributeError: 'Placement' object has no attribute 'kv_type'` (`logs/wt-4e1d549-crash.raw`) | **4** |
| `/tmp/e2-prov2` @ `fff127e` | identical tail (`logs/wt-fff127e-crash.raw`) | **4** |
| `/tmp/e2-prov2` @ `fff127e`, `--suite quality --runs 1` | identical tail (`logs/wt-fff127e-quality-crash.raw`) | **4** |

Fail-fast: both crash in ~1 s (the ladder is built before the first load attempt; only the model
header read has to succeed). The same command against the **post-fix** tree proceeds and runs the
suite (my bounded runs below, exit 0) — so the crash is a property of the shipped tree state, not of
the box.

**Why it crashes (root cause, verified in code):** the bench harness passes
`fit_plan=Placement(n_gpu_layers)` (`src/ggufone/bench/harness.py:383`). Commit `5410e42`
(the E1c fit fix, *earlier* than `fff127e`: 04:50:35Z vs 05:01:17Z) added a load-time degradation
ladder to `session.open_model`:

```python
walk = [step for step in fit.degrade_ladder(fit_plan, facts) …]   # session.py ~250
```

and `degrade_ladder` reads `plan.kv_type` (`fit.py`: `KV_DOWNGRADE_ORDER.index(plan.kv_type)`) —
`Placement` has only `n_gpu_layers`. Fix `8d4fc9f` (`fit.coerce_plan`) is exactly the missing
normalisation. The fix card's own repro is in `.e2e/t_31b3943a-bench-placement/repro_published_cmd.err`
(same tail, exit 4).

---

## 2. Step 2 — every recorded `commands.reproduce` against the current (post-fix) tree

All distinct reproduce commands in `docs/evidence/e2_*.json`, run post-fix (bounded where the
recorded form is a multi-hour campaign; bounded form noted):

| # | suite / model | recorded form (report) | run here | result |
|---|---|---|---|---|
| 1 | latency, Spark 4B | `--runs 5 --threads 4` | `--runs 1 --sizes 256 --threads 4` | **exit 0** (`main-latency-spark-bounded.json`, 6m31s) |
| 2 | latency, Qwen 0.8B | `--runs 5 --threads 2` | `--runs 1 --sizes 256 --threads 2` | **exit 0** (`main-latency-qwen-bounded.json`, 2m23s) |
| 3 | quality, Spark 4B | `--runs 1 --threads 4` (60 items) | `--items 6 --runs 1 --threads 4` | **exit 0** (`main-quality-spark-bounded.json`, 31s) |
| 4 | quality, Qwen 0.8B | `--runs 1 --threads 2` | `--items 6 --runs 1 --threads 2` | **exit 0** (`main-quality-qwen-bounded.json`) |
| 5 | calibration, Qwen 0.8B | `--runs 1 --threads 2` (60 items) | **as recorded** (all 60 items) | **exit 0** (`main-calibration-qwen.json`, 1m40s) |
| 6 | determinism, Spark 4B | `--backend all --runs 5 --threads 1` | **as recorded** | **exit 0** (`main-determinism-spark.json`, 2m07s) |
| 7 | throughput, Spark 4B | `--backend all --runs 5` | as recorded (`--runs 5`) + CPU-bundle variant (`--runs 1`) | **exit 0** (`main-throughput-spark.json`, `main-throughput-spark-cpubundle.json`) |

No suite fails post-fix. Two row-level caveats found on a *multi-bundle host* (do not affect the
container-published tables, but a reader re-running on a GPU box will hit them) — see Finding F4.

---

## 3. Step 3 — bounded re-run vs published values (deltas per metric)

### 3.1 latency, Spark 4B, CPU, threads=4 (published runs=5 vs my runs=1)

| metric (p50) | published (container, 2 CPU-s/s) | mine (this host) | ratio |
|---|---:|---:|---:|
| `model_load_ms` | 855.004 | 450.99 | 0.53× |
| prefill 256 ms / tok/s | 31,706.76 / 8.074 | 7,596.23 / **33.70** | 0.24× |
| per-question 2 candidates | 6,141.11 | 1,211.99 | 0.20× |
| per-question 4 candidates | 8,542.58 | 1,552.02 | 0.18× |
| per-question 10 candidates | 14,923.72 | 2,715.21 | 0.18× |
| warm cache `questions_ms` | 7,978.16 | 1,717.55 | 0.22× |
| serve / one-shot per request | 9,757.84 / 10,612.85 | 1,774.56 / 2,225.55 | 0.18× / 0.21× |
| wave scaling N=1 / N=16 | 7,594.45 / 72,488.11 | 1,186.38 / 20,370.77 | 0.16× / 0.28× |

**Shape is identical**: same rows, `waves`/`forks` = 2/2, 2/4, 2/10, `prefill_reused: true`,
`wave_accounting {groups:1, suffix_decodes:1, step_decodes:1, waves:2}`, and the same ~linear
growth per extra candidate/question. The ~4–5× value gap is the box: 24 real cores here vs the
container's 2 CPU-seconds/s quota (the same gap the campaign itself measured between its box and
the host in BENCHMARKS §3.4).

### 3.2 latency, Qwen 0.8B, CPU, threads=2

| metric (p50) | published | mine | ratio |
|---|---:|---:|---:|
| `model_load_ms` | 1,571.67 | 689.13 | 0.44× |
| prefill 256 ms / tok/s | 9,443.90 / 27.107 | 3,102.56 / **82.51** | 0.33× |
| per-question 2 / 4 / 10 | 833.89 / 1,125.09 / 2,221.64 | 527.13 / 758.71 / 1,358.59 | 0.63× / 0.67× / 0.61× |

### 3.3 quality — item-level comparison (the decisive check)

`--items 6` on Spark, same dev-set order, against the published per-item rows:

* decisions **identical** for all six items: c01 ✗, c02 ✓, c03 ✓, c04 ✗, c05 ✓, c06 ✗ (4B) and the
  same six for Qwen → `c01`–`c06` agreement matches the published per-item data exactly;
* `c01` (the 4B's wrong "billing" pick) probabilities reproduce to **~13 significant digits**:
  published `billing 0.7689847751553348, sales 0.11970672350554913, support 5.5553603820850974e-05,
  technical 0.11125294773529501`, confidence `0.6919797002071131`; mine
  `0.7689847751553072 / 0.11970672350556297 / 5.555360382085785e-05 / 0.11125294773530908`,
  confidence `0.6919797002070762` (difference ≈ 4e-14 — float noise, not a different measurement).

### 3.4 calibration, Qwen 0.8B, **full 60 items, `--runs 1 --threads 2` as published**

| field | published | mine |
|---|---:|---:|
| overall agreement | 0.4666666666666667 (28/60) | 0.4666666666666667 (28/60) |
| 95 % CI | [0.34627701540988, 0.5910679132886489] | identical |
| ECE | 0.2859817561687615 | **0.28598175616876126** |
| ECE by mode (entropy / margin / normalized_peak) | 0.2434526584871763 / 0.2621195703482921 / 0.2506391115624242 | 0.24345265848717618 / 0.2621195703482918 / 0.2506391115624241 |
| r(confidence, coverage) | 0.044047415722176256 | 0.04404741572217728 |
| per-item confidence (60 rows) | — | equal to ~15 significant digits, same correct/incorrect flags |

### 3.5 determinism — half reproduced, and the difference explained

* *Reproduced*: internal determinism — 3 repeats, one digest per backend (`identical: true`), and on
  this host also for `vulkan` (3 × `sha256:26eff145…`), which the container could not measure at all.
* *Not byte-equal*: the digest value differs for `cpu` (mine `sha256:a81f13b1…` vs published
  `sha256:7ab32e58…`). Evidence for the explanation: the **request payload is byte-identical**
  (`diff .request` published vs mine → no difference), while the fixed engine's response envelope
  gained fields (the loader now reports `engine.placement{note, kv_type, degraded, attempts}` and an
  `engine.fit` block — visible in `.e2e/t_31b3943a-bench-placement/ask_control.json`). The digest is
  `sha256` over the whole timings-stripped body, so it *must* move when the payload grows — even for
  identical decode outputs (which the quality/calibration FP-level match corroborates). Confidence:
  high that the decode is unchanged, medium on the exact byte cause (not proven byte-for-byte).

### 3.6 throughput — one honest CPU row, one row-labeling hazard

* Published `cpu` row (threads=1): prefill 6.756 tok/s, decision 8,011 ms, load 802 ms.
* Mine with the **CPU bundle**: 8.686 tok/s, 5,558 ms, 517 ms (`runs=1`) → 1.29× faster; consistent
  with the same box/quota difference (1 thread on a real core vs 2 CPU-s/s shared).
* Mine with the **Vulkan bundle present** (the default scan on this host): the row *labelled* `cpu`
  (placement `n_gpu_layers=0`) measured **587.9 tok/s** — GPU-class speed; the bundle's own
  `llama-bench -ngl 0` gives 582.66 tok/s, `-ngl 99` gives 2235.90 tok/s. So on a box with a Vulkan
  bundle, "cpu" ≠ CPU execution. This does **not** invalidate the published container row (the
  container had no Vulkan bundle or device → its `cpu` row is a true CPU measurement; the magnitudes
  are consistent with my CPU-bundle run).

### 3.7 internal consistency of the published JSONs (mechanical, 163 checks)

`logs/internal-consistency.txt`: for every summary (`n ≥ 2`) in all five reports plus the Qwen ones —
`min ≤ p50 ≤ p95 ≤ max` ✓; throughpout/quality/calibration/… row-count checks ✓; the arithmetic
identities (`one_shot == serve + model_load`, `saved == model_load`, `agreement == correct/n`,
per-type counts equal the per-item rows, per-item probabilities sum to 1, and **ECE recomputed from
the published bins equals the published ECE to 1e-9 for all three modes**) ✓. The single flagged
item is an expected-noise one: wave-scaling `N=2` p50 (7,394 ms) is 2.6 % *below* `N=1`
(7,594 ms) — a p50 inversion far inside the contention the document itself describes.

---

## 4. Where the numbers actually came from (primary evidence)

The E2 worker's session is exportable and was read directly
(`hermes -p code-tdd sessions export --session-id 20260918_055237_829f7e`; raw copy:
`/tmp/e2-audit/e2-session.jsonl`, command timeline: `logs/e2-session-cmd-timeline.txt`):

| time (UTC) | receipt | meaning |
|---|---|---|
| 03:52:34 | card `t_858c54d1` created | worker base = shared tree HEAD **`c095c51`** (the commit *before* the E1c fit fix) |
| 04:01:31 | `git clone -q /workspace/ggufone /work/e2repo` | the campaign home; clone takes **committed** state only |
| 04:01:40 | `cp src/ggufone/bench/{harness,suites,devset}.py … /work/e2repo/src/ggufone/bench/` | bench WIP moved into the clone |
| 04:05:31 | `TypeError: open_model() got an unexpected keyword argument 'n_gpu_layers'` | the clone's loader is the pre-fit-fix one |
| 04:05:41 | patch adds `class Placement` to the clone's `harness.py` | the shim that later breaks against the fixed loader |
| 04:05:44 | live run works (`resolve_fused_ops … graph_reserve …`) | on `c095c51`'s loader a `Placement` loads cleanly |
| 04:07:28 | `git log --oneline -2` → `64522ef …` / `c095c51 …` | clone base confirmed in-session |
| 04:07–07:37 | `/work/e2-scratch/run_live*.sh`, each `cd /work/e2repo` + `python3 tools/e2_reproduce.py --suite … --out …` | every published table was produced by the **tool** (same harness code) in the clone |
| 04:50:35 | shared tree gains `5410e42` (E1c fit fix) | the load-time ladder that reads `plan.kv_type` |
| 05:01:17 | shared tree gains `fff127e` (bench surface) | first commit where **both** halves coexist → crash |
| 05:54–10:04 | `e2_qwen_latency.json` … `e2_calibration.json` generated | reports written on the clone (no crash there) |
| 10:59:15 | `4e1d549` committed | evidence shipped with `reproduce:` lines that crash on it |

**Conclusion.** The published numbers are genuine measurements of the pinned model/bundle — the runs
really executed (llama.cpp decode logs in the session, per-run samples, per-item
probabilities/ECE that reproduce to FP precision) — but they were measured on a tree state
(*`c095c51` engine + bench surface*) that **no commit in the repository ever had**: the committed
combination pairs the new `Placement`-style bench harness with an E1c loader that expects a full
`FitPlan`. That is exactly why every `reproduce:` line fails on `fff127e`/`4e1d549` while the
per-item payloads still match a fresh run on `a42aa13`. The groupchat claim "the numbers may be
true but the reproducing command was broken" is now *confirmed and upgraded*: the numbers are
**verified genuine**, the command was **verified broken**, and the cause is identified.

---

## 5. Findings

Severity: BLOCKER (must fix before E3 sign-off) · IMPROVE (should) · NICE (optional).
Classification: EPISTEMIC / PROCEDURAL / COMPLIANCE / TOOLING.

| ID | sev | what happened (receipt) | class | proposed fix |
|---|---|---|---|---|
| **F1** | — (already fixed) | The shipping commits' own repro commands crash: `error: E_INTERNAL: AttributeError: 'Placement' object has no attribute 'kv_type'`, exit 4, on `fff127e` and `4e1d549` (`logs/wt-*-crash.raw`). Fix `8d4fc9f..a42aa13` repairs the path. | EPISTEMIC (bench harness and loader evolved in parallel from two cards sharing one tree) | none needed in code; the *record* must carry the caveat — see F2 |
| **F2** | **IMPROVE (blocking for the record, not for E3 code)** | `docs/BENCHMARKS.md` and the five JSONs present the tables as "one command regenerates each", but the shipping commits cannot run that command; the tree state they were measured on is not a commit. Without a note, E3 inherits a provenance gap it cannot detect. | PROCEDURAL | add a provenance note to `BENCHMARKS.md` §0 (exact text in §6.1); optionally regenerate the two cheap suites on the fixed tree (values are identical — §3.3/§3.4 — so the note alone is defensible) |
| **F3** | NICE | The determinism digest (`sha256:7ab32e58…` published vs `sha256:a81f13b1…` today) is presented as the byte-identity witness; it is not comparable across tree states because the response envelope grew (`engine.placement`, `engine.fit`). | PROCEDURAL | state next to A-E2-5 that the digest is a *within-tree* witness; compare digests only within one tree state (one line in BENCHMARKS/§3.3) |
| **F4** | IMPROVE | On a host with **two bundles**, row labels can lie: (a) with only the Vulkan bundle visible, the row labelled `cpu` (`n_gpu_layers=0`) measures 587.9 tok/s — GPU-class; the bundle's `llama-bench -ngl 0` independently gives 582.66 tok/s; (b) with a CPU bundle first and a Vulkan bundle also visible, `--backend all` produced a `vulkan` row (`n_gpu_layers=-1, degraded:false`) with **0 Vulkan / 9 CPU compute buffers** in the process log and CPU-class 9.11 tok/s (reproduced twice: `logs/thr-mixed2.raw`, `logs/main-throughput-spark-cpubundle.raw`). Container tables are unaffected (single bundle, no GPU). | TOOLING (loader/symbol resolution across two bundle copies; root cause not proven) | for host re-runs: one bundle per process (`--backend cpu` / `--backend vulkan` separately) — or have the engine detect a second runtime dir in-process and say so (proposal to the fix/E3 card, not applied here) |
| **F5** | NICE | The fixed loader's OOM ladder works and is reportable: a host `--backend vulkan` run today degraded `-1 → oom, 18 → oom, used 0 (kv_f16)` with `W_BACKEND_OOM/W_FIT_DOWNGRADE` under VRAM pressure, exit 0 (`probe-vulkan-only.raw`) — exactly the behaviour `t_8cb0a05e`/`t_31b3943a` designed for. | — | keep & replicate: host gate rehearsals should keep a degraded-row case |

---

## 6. Recommendation (one, decisive)

**Keep the published numbers; fix the record.** They are corroborated at FP precision and no cheaper
re-run would change a single value. Concretely:

1. **Apply the provenance note (F2)** — exact text, ready to paste into `docs/BENCHMARKS.md` §0:

   > **Provenance note (audit `t_78f5ea7a`, 2026-09-18).** These tables were measured *before* the
   > E1c fit fix (`5410e42`) met the bench harness: every suite ran through
   > `tools/e2_reproduce.py` on a private clone whose engine was `c095c51` with this card's bench
   > surface, i.e. a tree state that no single commit contains. As shipped (`fff127e`, `4e1d549`)
   > the printed `reproduce:` lines therefore fail with
   > `E_INTERNAL: AttributeError: 'Placement' object has no attribute 'kv_type'` (exit 4); they work
   > from `a42aa13` (the placement fix) onward, and a bounded re-run on that tree reproduces the
   > per-item quality/calibration payloads to float precision (auditor scorecard
   > `state/fights/e2-provenance/scorecard.md`). Read the timings as "the 2-CPU container, that
   > hour"; the numbers were not re-measured on the fixed tree.
2. **Optionally** regenerate the two cheap suites on the fixed tree (`quality` 60 items ×2 models,
   `calibration` ×1; ≈8 min on this host) and mark the old files `e2_*_pre-fix.json` — the values
   will be identical modulo the box, so this is bookkeeping, not measurement. The latency/throughput
   tables stay as published (re-measuring them on the host would change the box for every row).
3. **Do not** re-run the whole campaign for this audit's sake; E2.5/E3 build on the *values*, and
   the values stand.

Verdict per report (recorded recipe / values):

| report | recorded recipe on shipping commit | values on the fixed tree |
|---|---|---|
| `e2_latency.json` | **NOT REPRODUCED** (crash, exit 4) | shape reproduced; values = box-scaled (~0.2×) |
| `e2_qwen_latency.json` | **NOT REPRODUCED** (same path) | shape reproduced; ~0.33–0.63× |
| `e2_quality.json` | **NOT REPRODUCED** (crash) | **REPRODUCED** (item decisions + probabilities to ~1e-13) |
| `e2_qwen_quality.json` | **NOT REPRODUCED** (same path) | **REPRODUCED** (same decisions on the first 6 items) |
| `e2_calibration.json` | **NOT REPRODUCED** (crash) | **REPRODUCED** (full 60 items; ECE/CI/corr to FP precision) |
| `e2_determinism.json` | **NOT REPRODUCED** (crash) | internal determinism reproduced; digest value differs (payload grew — F3) |
| `e2_throughput.json` | **NOT REPRODUCED** (crash) | CPU row reproduced at 1.29× (CPU bundle); row-label caveat on multi-bundle hosts — F4 |

**No finding is BLOCKER.** E2's *code* is repaired and verified; only the *record* needs the caveat.
Nothing in this audit changes any published value.

---

## 7. What went well (keep & replicate)

* The fix card (`t_31b3943a`) reproduced and repaired the failure **before** this audit, with a
  pre-fix RED control and a post-fix GREEN on the same worktree — the audit only had to confirm it.
* The suites' reports are **self-describing** (config, host block, `reproduce:` line, per-item rows,
  wave accounting) — which is what made an FP-level comparison possible four hours after the fact.
* Payloads reproduce across tree states and boxes (per-item probabilities to 1e-13) — evidence that
  the engine's decision path is deterministic and that the published data was not massaged.
* Reports degrade honestly: the container's tables say `measured: false` + reason where no bundle
  exists, and the fixed loader now records OOM ladders (`attempts`, `W_BACKEND_OOM`) instead of
  failing (F5).
* The E2 worker kept a single session and a private clone, and its transcript shows the whole
  campaign — provenance was reconstructible from primary sources even with the container gone.

## 8. Open questions for @bots-coordinator

1. Does the coordinator's host-gate run recorded in `cc1f5da` need the same provenance tag (it ran
   *after* the fix, so it is clean — just confirm the note's wording covers it)?
2. For E3: should the "regenerate" option in §6.2 be executed now (8 min) or deferred to the card
   that first cites a stale row?
3. F4 (multi-bundle row labels) — is it worth a card of its own, or does it live as a note in the
   bench docs? It cannot be fixed by prose alone if `--backend all` is advertised on GPU hosts.

## 9. Reproducing this audit (commands, artifacts)

```bash
# crash (pre-fix trees; ~1 s each)
cd /tmp/e2-prov  && GGUFONE_RUNTIME_DIR=/tmp/e2-audit/runtime/b11026-linux-x64-cpu \
  uv run ggufone bench --suite latency --model /var/home/rybens/.hermes/models/Spark-X2.5-4B-Q8_0.gguf \
  --backend auto --runs 5 --threads 4 --json          # -> exit 4, E_INTERNAL

# bounded re-runs on the fixed tree (outputs in logs/)
GGUFONE_RUNTIME_DIR=/tmp/e2-audit/runtime/b11026-linux-x64-cpu uv run ggufone bench \
  --suite latency --model /var/home/rybens/.hermes/models/Spark-X2.5-4B-Q8_0.gguf \
  --backend auto --runs 1 --sizes 256 --threads 4 --json --out …/main-latency-spark-bounded.json
# quality --items 6 · calibration (60 items, as published) · determinism --backend all · throughput
```

Artifacts: `state/fights/e2-provenance/logs/` (raw stdout+stderr + JSON per run,
`internal-consistency.txt`, `e2-session-cmd-timeline.txt`, `e2-provenance-receipts.txt` — the quoted
session receipts, `e2-session-export.jsonl` — the full code-tdd session export), checker
`/tmp/e2-audit/check_reports.py` (also at `logs/../check_reports.py` if copied). All runs read-only
w.r.t. other profiles; nothing committed by me.
