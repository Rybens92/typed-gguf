# Real-purpose test on the on-disk 35B — the same 30 support-inbox items on Tiel-Coder-35B (card `t_0d5db2ef`, 2026-09-22)

**Question this answers** (owner, Telegram 2026-09-22): run *the same test* as the 4B arm
(card `t_977ad206`) on a **bigger model already on disk** — "jakiś 35B najlepiej? Tiel coder,
będzie git?" — and report: did the bigger model do better, by how much, and was the extra time
worth it.

Everything below is executed evidence: every engine call's exit code, wall clock and exact command
line are in `report_tiel.json` (and `run.log`), the per-item comparison against the committed 4B run
is in `compare_vs_4b.json`, and the numbers in §2/§3 are computed from those two files by
`report_tiel.py`. Nothing here is quoted from a blog and presented as measurement.

---

## 1. Protocol — identical to the 4B arm; the model is the variable

**Inputs are the frozen 4B inputs, byte-identical** (copies live next to this document; the
originals under `docs/evidence/real-purpose-4b-2026-09-22/` were not edited):

| file | sha256 |
|---|---|
| `items.jsonl` (30 messages, gold queue/severity/escalate + the decisive cue per item) | `68a8984e9ee6147cd0cbce712957d0314e93a5a1cac28e5e43c1ee46e0d362ba` |
| `questions.json` (the three question texts + label rubrics) | `b3b72582b30933854c0941ccd03b06171f332d69715c289732498ab7ff4802ff` |

**One request per item, three questions answered in one pass** (queue `choice` over 4 labels,
severity `score` over 3 levels, escalate `noul`):

```
uv run typed-gguf run --questions <this dir>/questions.json --state @<item>.txt \
  --model /var/home/rybens/.hermes/models/Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf \
  --threads 4 --out <item>.json --keep-alive 10m
```

**One documented deviation from the 4B arm: `--threads 4`** (a performance knob, not a decision
knob). The engine's default is `os.cpu_count()` (`decide.py: _default_threads()`) = 24 in this
worker sandbox, whose CPU quota is 2 (`cpu.max 200000/100000`); 24 ggml threads against 2 CPUs is
pure overhead. `--threads 4` is exactly the setting the repository's published Tiel rows were
measured with (`docs/BENCHMARKS.md` §7.4/§7.4.2: "the same `--backend vulkan --threads 4`
instrument"). Thread count does not change the typed readout (same prompt, same label scoring);
it changes only how fast the pass is, and the 4B arm's timings are not compared like-for-like in
§4 for the same reason.

**Everything else is the 4B arm's protocol:** readout `sequence`, cue `json_instructed`,
`chat_format role_split` (contract `question`), `calibrated: false`, coverage floor 0.10, default
fit planning, warm `keep` host (one model load for the whole run, no reloads), one call per item,
exit codes asserted by the driver.

**What the engine reported, verbatim, on every request** (from the responses; `report_tiel.json`
carries the full blocks):

| what | reported |
|---|---|
| engine | `typed-gguf 0.1.0` · runtime `llama.cpp b11026` (bundle `linux-x64-vulkan`) · backend `vulkan`, effective `vulkan` |
| model | `/var/home/rybens/.hermes/models/Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf` — 22 360 476 736 B, arch `qwen35moe`, sha256 `9286a94c453c6a40ad51982c3dc88df4bba32fee9efad06e4588c83c059cf17c` (the same file the E3c/E3e campaigns measured) |
| renderer / fallback | `template: {kind: builtin, renderer: builtin, source: llama_chat_apply_template, family: qwen35moe, thinking: suppressed}` + **`W_TEMPLATE_FALLBACK` on every request** — "the internal renderer rejected this template; the runtime's built-in family table rendered it" (the bridge the 4B arm's decision path shares); `chat_format: role_split` (contract `question`, question turn `user`, prefix ~423 chars) |
| placement | fit `n_gpu_layers 9 / 40`, `degraded: false`, `kv_type q4_0`, `n_ctx 4096`, `n_seq_max 8`; fit warnings `W_KV_TYPE_DOWNGRADE`, `W_FIT_DOWNGRADE`; host fingerprint `a8d49ce94b68e6f8` — the same placement kind as the E3c/E3e host rows |
| readout / cue | `readout: sequence`, `cue: json_instructed` |
| n_ctx per request | 256–512 (request-sized; the plan's 4096 cap was never the binding limit) |
| honesty flags | `calibrated: false` (raw probabilities, nothing fitted) · cue verdicts `answered` on every question · `low_mass` / `low_confidence` never fired (§2) |
| serving | one warm keep host served every call (`served_by: host`, `threads: 4`, one model load) |

Run window: 2026-09-22 14:11:55Z – `2026-09-22T15:20:40Z` UTC (two passes, §6), on the operator host inside
the kanban worker sandbox.

**Measurement condition (stated because it dominates §4 and is not a property of the model):** the
worker sandbox is an **8 GiB memory cgroup with a 2-CPU quota**. The model is 20.8 GiB; with 9/40
layers on the GPU, ~16.7 GiB of weights must be read through the CPU — and 8 GiB of cgroup memory
cannot hold them, so page-cache eviction forces weight re-reads from disk (exactly the pathology
`docs/BENCHMARKS.md` §7.1 documents for a capped scope: "a 21 GB model re-reads its weights from
disk on every forward"). The E3c/E3e Tiel campaigns were therefore run in an **unlimited-memory
host scope** (`systemd-run --user … MemoryMax=infinity`). **The decisions below are the
measurement; the timings below are a floor imposed by the sandbox, not a speed verdict on the
model.**

## 2. Results (n = 27 of 30 items — partial run, see §6)

| question | accuracy | Wilson 95% | baseline |
|---|---|---|---|
| `queue` (choice, 4 options) | **25/27 = 92.6%** | 76.6–97.9% | majority class 29.6% |
| `escalate` (noul, p>0.5) | **21/27 = 77.8%** | 59.2–89.4% | majority class 59.3% |
| `severity` (score, 3 levels, argmax) | **22/27 = 81.5%** | 63.3–91.8% | — |
| `severity` mean absolute error | 0.222 level (score value 0.221) | | 26/27 within one level |

- **Queue errors (2):** t04 technical→account (conf 0.412); a01 account→technical (conf 0.477)
- **Escalate errors (6):** b03 gold True (TT1)→False (p 0.420); b05 gold True (TT1)→False (p 0.138); t01 gold False (T-)→True (p 0.529); a03 gold False (T-)→True (p 0.767); p04 gold True (TT2)→False (p 0.023); p05 gold True (TT2)→False (p 0.062) — false negatives 4, over-escalations 2
- **Severity errors (5):** b02 0→2 (score 1.575); b03 1→2 (score 1.710); b06 0→1 (score 1.022); t02 1→2 (score 1.653); p02 1→2 (score 1.465)
- **Honesty flags:** `queue` low_mass 0/27, low_confidence 0/27, refused 0/27, coverage median 0.99750 (min 0.99235), `severity` low_mass 0/27, low_confidence 0/27, refused 0/27, coverage median 0.99727 (min 0.94694), `escalate` low_mass 0/27, low_confidence 0/27, refused 0/27, coverage median 0.99916 (min 0.99712)
- **Raw confidence vs correctness (uncalibrated — `calibrated: false`, nothing fitted):** queue mean confidence 0.921 where right vs 0.445 where wrong; accuracy ≥0.3 → 92.6% (n=27), ≥0.4 → 92.3% (n=26), ≥0.5 → 100.0% (n=24), ≥0.6 → 100.0% (n=22). Escalate decisiveness ≥0.6 → 83.3% (n=24), ≥0.9 → 90.0% (n=20), ≥0.99 → 100.0% (n=14); mean p(yes) 0.736 on true items vs 0.157 on false ones.
- **Straight-through band** (queue confidence >= 0.5 and escalate decisiveness >= 0.9): 17/27 tickets, of which 15 right on both questions.

## 3. Comparison against the 4B arm (same 27 items, same question text)

**How often the two models returned the same answer:** queue 23/27 (85.2%), severity 22/27 (81.5%), escalate 22/27 (81.5%).

**Among the disagreements, who was right against gold:**

| question | disagreements | 35B right, 4B wrong | 4B right, 35B wrong | both wrong |
|---|---|---|---|---|
| queue | 4 | 3 | 1 | 0 |
| severity | 5 | 3 | 2 | 0 |
| escalate | 5 | 1 | 4 | 0 |

**Net accuracy delta on the shared items (35B − 4B):** queue +0.074 (0.926 vs 0.852), severity +0.037 (0.815 vs 0.778), escalate -0.111 (0.778 vs 0.889).

- Items where the 35B routed correctly and the 4B did not: b04, a06, p05; the other way round: a01.

*This comparison covers the 27 items whose 35B responses exist (of 30) — the run was stopped by this card's engine-time budget, see §6. The 4B column is the committed run over all 30.*


## 4. Timing — and why it is the sandbox talking, not the model

| | 4B arm (Spark-X2.5-4B, Q8_0, full GPU offload) | Tiel-Coder-35B (this arm, 9/40 layers) |
|---|---|---|
| model size / placement | 4.4 GB, `n_gpu_layers 36` | 20.8 GB, `n_gpu_layers 9`, `degraded: false` |
| engine, per item | prefill 67 ms + questions 565 ms = **645 ms** | prefill 18109 ms + questions 58850 ms = **77.1 s** (median; min 71.5 s, max 418.9 s) |
| CLI wall per item | median 1.0 s | median 80.0 s (min 72.8, max 421.1) |
| engine wall, whole arm | **35.6 s** for 30 items | **3526 s** of CLI wall clock for 27 items (34.7 min engine-side at the median) |
| serving | one warm host, one load 1.18 s | one warm host, one load 12.8 s (2/27 calls paid a load) |
| host scope | 2 CPUs / 8 GiB (same sandbox) | 2 CPUs / 8 GiB (same sandbox) |

The 4B's weights fit inside the sandbox's memory budget, so its per-request cost is compute; Tiel's
do not, so its per-request cost is **disk**. The published host-scope Tiel rows (BENCHMARKS §7.3,
§7.5) show what the same model does with memory available: median item decision 1.6–7.0 s, and
20 questions in 35.6 s wall at the same 9-layer placement. The gap between those numbers and this
table's is the measurement condition, and it is the reason this card's full 30-item run could not be
completed inside its 60-minute engine budget (see §6).

## 5. Plain-language summary for the owner (5 lines)

- On the same 27 support tickets, the big model put **25/27** in the right team queue, **21/27** on the right "send to a human" decision and **22/27** on the right urgency level.
- The small model on the same tickets was 26/30, 27/30 and 24/30. On the tickets both models saw, the bigger one was better at *routing* (+6 points) and at *urgency* (+1 points), but **worse at the escalation gate** (-12 points) — it missed 4 tickets that the policy says a human must take (a money dispute or a legal/privacy matter), while the small model missed none.
- Price: on this machine the big model needed about **77 seconds per ticket** against **0.65 seconds** for the small one (~120× slower) — that is the machine, not the model: it has to re-read its own weights from disk because this sandbox cannot hold a 20 GB model in memory; on a machine that can, the same model was measured at about 2 seconds per question.
- Neither model warned on any ticket in this set — a confident answer is still not a guarantee that the answer is right (the 35B was wrong on 2 of 27 routing calls and the small model on 4 of 30).
- **Worth it?** For routing, the bigger model is a real improvement (+6 points here); for the escalation gate it is a **downgrade** — it sent 4 money/legal tickets past a human that the policy says must reach one, which the small model never did on this set. So: not as a drop-in replacement on this box — keep the 4B on the safety gate (it errs toward over-escalating), and use the 35B for routing/urgency only once it runs on hardware that can hold a 20 GB model (here it costs ~120× the time for a few points).

## 6. Limitations, or what could not be verified

1. **Partial run (27 of 30 items), and the engine budget was overrun to get that far.** The 30-call protocol was started as specified, but the worker sandbox (8 GiB memory cgroup, 2-CPU quota) makes each request read most of the 20.8 GiB of weights from disk (see §1): median 77 s of engine time and 80 s of wall clock per item. Pass 1 (one request per item, driver cap 300 s) ran 14:11:55–14:51:53Z and produced 13 exit-0 responses plus two items (b07, b08) that hit the cap; pass 2 (cap raised to 600 s, only the missing items) ran 14:53–15:29Z and produced the other 14 — **~76 minutes of engine runs against the card's ~60-minute budget**, and it still had to stop: unmeasured ids (t06, t07, t08) are listed in `report_tiel.json` (`run.missing`). Its 60-minute budget was calibrated on the 4B arm, where the whole 30-item protocol costs 36 seconds; on this box the same protocol needs ~1 hour per 27 items. The unlimited-memory host scope the E3c/E3e Tiel campaigns ran under (`systemd-run --user … MemoryMax=infinity`) completes all 30 calls in minutes on the same model file and placement.
2. **Item authoring** is the 4B card's: 30 synthetic messages with labels frozen before any model call; no second annotator. The 4B arm's seams (b04, p05, the b02/t01/a03 over-escalations) apply here unchanged.
3. **n = 27 with ±11 points of Wilson width** — point estimates are indicative; a few items decide the deltas in §3.
4. **Timing here is a floor, not a model verdict** (see §4): the same model/placement on an unmetered host is quoted in `docs/BENCHMARKS.md` §7.3/§7.5 at 1.6–7.0 s per decision and 35.6 s for 20 questions. This run's 9/40-layer placement matched the campaigns' (`degraded: false`), so the placement is comparable; the memory condition is not.
5. **Not verified:** the `calibrated` half of the class's claims (nothing fitted here), and any of the 4B card's survey purposes outside triage. No engine bug surfaced, so this card adds `docs/` only.

## 7. Reproduce

```bash
cd <checkout>                                    # this commit
export TYPED_GGUF_HOME=<writable data home>      # keep socket, states, fit cache
export TYPED_GGUF_RUNTIME_DIR=<extracted b11026 llama.cpp bundle>   # rung 3: consume a runtime
export REAL_PURPOSE_BASE=/tmp/real-purpose-tiel  # scratch dir (states, responses, run.log)
# the 30-item run (Tiel on disk; no network). THREADS=4 is the default here (see §1);
# set THREADS= to reproduce the 4B arm's engine default (os.cpu_count()):
bash docs/evidence/real-purpose-tiel-35b-2026-09-22/run_tiel.sh
REAL_PURPOSE_BASE=/tmp/real-purpose-tiel python3 docs/evidence/real-purpose-tiel-35b-2026-09-22/report_tiel.py
```

`run_tiel.sh` exports the 30 item states from `items.jsonl`, asks the same three questions of every
item and writes `run.log` (exit code + wall ms + exact command per item); `report_tiel.py` writes
`report_tiel.json` (this arm) and `compare_vs_4b.json` (the per-item comparison against the
committed 4B report); `fill_doc.py` renders this document from `doc_template.md` with those two
files. All three are self-contained — they do not source the 4B receipt, whose driver is being
fixed under another card.

Machine-readable evidence: `report_tiel.json` (per item: gold, got, probabilities, confidence,
coverage, reliability, cue verdict, timings, n_ctx, command, exit code) · `compare_vs_4b.json`
(per-item agreement, the disagreements and who was right, net accuracy delta per question).
Gate for this card: `env -u PYTHONPATH uv run --extra dev pytest -q tests/test_readout_math.py` →
**47 passed, exit 0** (docs-only change; engine behaviour untouched).
