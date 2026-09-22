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
knob). The engine's default is `os.cpu_count()` (`decide.py: _default_threads()`) = 24 threads,
which in the worker sandbox also had a 2-CPU cgroup quota (`cpu.max 200000/100000`) — 24 ggml
threads against 2 CPUs is pure overhead there. `--threads 4` is exactly the setting the
repository's published Tiel rows were measured with (`docs/BENCHMARKS.md` §7.4/§7.4.2: "the same
`--backend vulkan --threads 4` instrument"), and the host completion run below keeps it so both
scopes of this arm are comparable. Thread count does not change the typed readout (same prompt,
same label scoring); it changes only how fast the pass is, and the 4B arm's timings are not
compared like-for-like in §4 for the same reason.

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

Run window: **host completion 2026-09-22 15:32:03Z – `@@RUNEND@@` UTC, all 30 items** (this is the
run whose numbers §2–§5 quote), preceded by the capped-sandbox pass 14:11:55Z–15:29Z that measured
27 of 30 items and is kept in §4/§6 as superseded.

**Measurement condition (stated because it dominated the first pass and is not a property of the
model).** The kanban worker sandbox is an **8 GiB memory cgroup with a 2-CPU quota**. The model is
20.8 GiB; with 9/40 layers on the GPU, ~16.7 GiB of weights must be read through the CPU — and 8 GiB
of cgroup memory cannot hold them, so page-cache eviction forces weight re-reads from disk (exactly
the pathology `docs/BENCHMARKS.md` §7.1 documents for a capped scope: "a 21 GB model re-reads its
weights from disk on every forward"). That pass therefore cost ~77 s of engine time per item and
stopped at 27 items. The completion below was run **on the operator host directly, with no memory
cap** (same model file, same placement, same `--threads 4`): the weights stay resident, the same
protocol finished **30/30 in 3.9 minutes of CLI wall clock**. The decision numbers are the
measurement; the timing gap between the two scopes is the memory scope, not the model.

@@NUMBERS@@

## 4. Timing — host scope vs the capped sandbox

| | 4B arm (Spark-X2.5-4B, Q8_0, full GPU offload) | Tiel-Coder-35B (this arm, 9/40 layers, **host scope**) |
|---|---|---|
| model size / placement | 4.4 GB, `n_gpu_layers 36` | 20.8 GB, `n_gpu_layers 9`, `degraded: false` |
| engine, per item | prefill 67 ms + questions 565 ms = **645 ms** | @@TIMING65@@ |
| CLI wall per item | median 1.0 s | @@TIMINGWALL@@ |
| engine wall, whole arm | **35.6 s** for 30 items | @@TIMINGTOTAL@@ |
| serving | one warm host, one load 1.18 s | one warm host, one load (@@LOADS@@) |
| host scope | desktop host, unlimited memory | desktop host, unlimited memory (sandbox pass was 8 GiB cgroup → 77 s/item, superseded, §6) |

Both models are measured on the same desktop host, so this table is memory-unconstrained on both
sides; the placement differs (the 4B's 4.4 GB fit entirely on the GPU, Tiel's 20.8 GB took 9/40
layers). The host-scope Tiel median of 5.5 s of engine time per item sits inside the band the
published host-scope rows report (BENCHMARKS §7.3, §7.5: 1.6–7.0 s per decision, 20 questions in
35.6 s wall at the same 9-layer placement). The earlier sandbox pass — median 77 s of engine time
and 80 s of wall per item, stopping at 27 of 30 — was the same model, file and placement under an
8 GiB cgroup; that gap is the measurement condition, and removing the cap is what let the protocol
finish (§6).

## 5. Plain-language summary for the owner (5 lines)

@@PLAIN@@

## 6. Limitations, or what could not be verified

@@LIMITS@@

## 7. Reproduce

```bash
cd <checkout>                                    # this commit
export TYPED_GGUF_HOME=<writable data home>      # keep socket, states, fit cache
export TYPED_GGUF_RUNTIME_DIR=<extracted b11026 llama.cpp bundle>   # rung 3: consume a runtime
BASE=$HOME/coding-pipeline/t0d5-tiel-host         # scratch dir used for the host run (fresh base works too)
# the 30-item run (Tiel on disk; no network). THREADS=4 is the default here (see §1); the run
# needs a host with enough memory to keep the ~16.7 GiB of CPU-side weights resident — an
# 8 GiB cgroup stalls at ~77 s/item (that is the superseded sandbox pass, §6):
REAL_PURPOSE_BASE=$BASE bash docs/evidence/real-purpose-tiel-35b-2026-09-22/run_tiel.sh
REAL_PURPOSE_BASE=$BASE python3 docs/evidence/real-purpose-tiel-35b-2026-09-22/report_tiel.py
REAL_PURPOSE_BASE=$BASE python3 docs/evidence/real-purpose-tiel-35b-2026-09-22/fill_doc.py   # regenerates this doc
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
