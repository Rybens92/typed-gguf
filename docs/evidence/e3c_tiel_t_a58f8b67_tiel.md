# E3c — Tiel-Coder (Ornith-1.5-35B, 35B-A3B, 21 GB local) measured like Occamy 1.0

Card `t_a58f8b67` (main-coder) · host run · 2026-09-19 · artifacts committed in `2408fad`, QA in §11.

Tiel is the third model in the `qwen35moe` comparison: the E2 baseline (`4B default`, CPU, 60
items), E3's `Occamy 1.0` (24.1 GB, chunks, vulkan) and now Tiel-Coder (20.8 GB, chunks, vulkan) —
all three on the **same 60 committed dev items**, paired (`--align`, 0 unpaired rows).

## 1. The pin

| what | value |
|---|---|
| file | `/var/home/rybens/.hermes/models/Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf` |
| bytes · mtime | 22 360 476 736 (20.81 GiB) · 1788539361 |
| SHA-256 before | `9286a94c453c6a40ad51982c3dc88df4bba32fee9efad06e4588c83c059cf17c` |
| SHA-256 after | `9286a94c453c6a40ad51982c3dc88df4bba32fee9efad06e4588c83c059cf17c` (identical) |
| Occamy (untouched, verified before **and** after) | `633ae57faf731e863cc3ba7cb75396a1b1e377191730e7b0d7294eff55cdf757` · 24 113 674 848 B |
| GGUF facts | `qwen35moe` · `general.name = Ornith-1.5-35B` · 40 layers · 256 experts / 8 used · n_vocab 248 320 |

Receipts: `.e3c_tiel/sha256_before.txt`, `.e3c_tiel/sha256_after.txt` (the before-file carries a
note: a stray re-run five minutes into the after-capture overwrote its first line; the block above
is the transcription of the first capture, corroborated by the fit plan's own `model_sha256` at
17:20:46Z — `ggufone fit` reads the SHA itself — and by `e3c_sha256_receipt.json` for Occamy).

## 2. Environment

```
host      Linux-7.2.4-ogc3.1.fc44.x86_64 · 24 CPUs · RAM 33 548 615 680 B · cgroup quota n/a
GPU       NVIDIA GeForce RTX 3060 Ti, 8192 MiB, driver 615.71.09
runtime   /var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan (pinned b11026)
ICD       VK_DRIVER_FILES=/home/rybens/.e3c_host/nvidia_egl_icd.json (libEGL_nvidia.so.0 —
          E3 §6.2's recipe: `libGLX_nvidia.so.0` returns INITIALIZATION_FAILED without an X display)
scope     systemd-run --user --unit=e3c-tiel-campaign (MemoryMax=infinity, memory.max=max)
```

**The scope is not incidental — it is the measurement's first finding.** See §7.

## 3. Placement (deliverable 1)

`ggufone fit` (free-VRAM aware, `--print --json`, `.e3c_tiel/fit_tiel.json`):

```
n_gpu_layers 9 · n_ctx 4096 · kv_type q4_0 · n_seq_max 8
source llama-fit-params · budget_bytes 5 881 462 784
warnings [W_KV_TYPE_DOWNGRADE, W_FIT_DOWNGRADE]
note: "device memory bound: offloading 9/40 layers within 5609 MiB
       (--fit-target 1024 MiB, 6633 MiB free of 8192 MiB)"
host: vulkan · vram_free_bytes 6 955 204 608 of 8 589 934 592
```

What each chunk's **loader** actually used (the placement the quality suite's report shape does
not carry — E3 §4.3 — captured by `tools/e3c_tiel_reproduce.py`, an observer on
`handle.placement.to_dict()` that changes no flag, plan or measured number):

| chunk | ngl requested | **ngl used** | degraded | attempts | kv_type | warnings | load wall | chunk wall |
|---|---|---|---|---|---|---|---|---|
| 001 | 9 | **9** | false | `[]` | auto | `[]` | 29.3 s | 204.9 s |
| 002 | 9 | **9** | false | `[]` | auto | `[]` | 12.5 s | 141.0 s |
| 003 | 9 | **9** | false | `[]` | auto | `[]` | 24.9 s | 136.5 s |
| 004 | 9 | **9** | false | `[]` | auto | `[]` | 9.1 s | **957.7 s** |
| 005 | 9 | **9** | false | `[]` | auto | `[]` | 19.0 s | 178.3 s |
| 006 | 9 | **9** | false | `[]` | auto | `[]` | 26.9 s | 152.6 s |

**No degraded rung was taken in any chunk.** The E3 campaign degraded Occamy twice (7 → oom → 3 in
the container, and again in E3b's probe); Tiel at 9 layers fit both times it was asked. The engine
log's own line (`device_log` in each `placement_00N.json`):

```
load_tensors: offloading 8 repeating layers to GPU
load_tensors: offloaded 9/41 layers to GPU
load_tensors:   CPU_Mapped model buffer size = 16680.10 MiB
load_tensors:      Vulkan0 model buffer size =  4634.02 MiB
```

`kv_type` reads `auto` because a *benchmark* names its placement explicitly and lets the engine
choose the cache type (`harness.Placement(n_gpu_layers)`); the fit plan's `q4_0` is what `run`/`ask`
apply. `n_ctx` / `n_seq_max` are planned per request and recorded per chunk in the same files
(154–186 tokens of prefix+question+margin; `n_seq_max` 3–6 — the per-item questions are narrower
than the fit plan's global bound of 8).

Per-item decision cost, read off the 60 rows' own `questions_ms`: **median 5.0 s, min 1.2 s, max
65.2 s**. The max is chunk 004's: those ten items ran under a *neighbouring* 21 GB probe (a
sibling card's `e3c_cue_shapes.py` on Occamy, `.e3c/logs/occamy_c01_vulkan.log`) — see §9 — and its
30.7 s median is a contention artifact, not a model property. The other five chunks' medians are
1.6–7.0 s. Load wall per chunk is in the table above (9.1–29.3 s; the first two chunks also paid
the shader/pipeline compile).

## 4. Quality on the committed 60-item dev set (deliverable 2)

Raw per-chunk reports `docs/evidence/tiel_chunks/report_00{1..6}.json`, dev-set slices
`docs/evidence/tiel_chunks/devset_00{1..6}.jsonl` (byte-identical copies of E3's chunks —
SHA-verified equal, `.e3c_tiel/tiel_devset_sha.txt`), merged report
`docs/evidence/tiel_quality.json` (6 chunks · 60 items · every row `ok`).

Reproduce one chunk exactly as it ran:

```bash
GGUFONE_RUNTIME_DIR=/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan \
VK_DRIVER_FILES=/home/rybens/.e3c_host/nvidia_egl_icd.json \
python3 tools/e3c_tiel_reproduce.py --suite quality \
  --model /var/home/rybens/.hermes/models/Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf \
  --backend vulkan --gpu-layers 9 --threads 4 \
  --devset docs/evidence/tiel_chunks/devset_001.jsonl \
  --out docs/evidence/tiel_chunks/report_001.json \
  --placement-out docs/evidence/tiel_chunks/placement_001.json
```

Merged result (95 % Wilson):

| metric | n | correct | agreement | 95 % CI |
|---|---|---|---|---|
| overall | 60 | 31 | 0.517 | 0.393 – 0.638 |
| choice | 24 | 16 | 0.667 | 0.467 – 0.820 |
| noul | 18 | 7 | 0.389 | 0.203 – 0.614 |
| score | 18 | 8 | 0.444 | 0.246 – 0.663 |

Per chunk: 0.500 · 0.500 · 0.600 · 0.500 · 0.500 · 0.500 (no chunk is an outlier; the whole spread
is 1 item).

## 5. The mass split (deliverable 3 — input to `t_6952f0dd`)

**Tiel is NOT mass-starved like Occamy. Most of its answers land on the label strings.**

| | 4B default (E2) | Occamy 1.0 (E3) | **Tiel-Coder (E3c)** |
|---|---|---|---|
| rows `measured` (coverage ≥ 0.10) | 48/60 | **3/60** | **46/60** |
| rows `low_mass` | 12/60 | 57/60 | 14/60 |
| coverage median | 2.56e-01 | 2.34e-02 | **2.58e-01** |
| coverage min / max | 9.96e-03 / 8.94e-01 | 1.86e-03 / 2.00e-01 | 7.88e-03 / 8.20e-01 |

Tiel's coverage distribution is essentially the 4B's shape (median 0.258 vs 0.256), not Occamy's
starved one (median 0.023, and not a single row above 0.20). On the same items, on the same
shipped `bare` label policy, Occamy's answers open with a turn-closer / off-label token where
Tiel's open with the wire key.

**Policy recorded with these numbers:** the shipped rendering — `prompt.build_question` renders
bare wire keys (`billing`, `3`, `yes`), `readout.coverage_from_scale` reads that label's first
token, floor 0.10 (`OPTION_DEFAULTS["coverage_floor"]`). Nothing in the default label policy
changed for this campaign (E3b's `tools/e3b_label_policy.py` + `bench/labels.py` only *probe*
variants; `bare` is byte-identical to the shipped rendering by that card's own gate). The
label-policy sweep on Tiel, run for the same control E3b used, is in
`docs/evidence/tiel_label_policy_tables.md` (§8).

Interpretation for the label-policy card: Occamy's starvation is a *model behaviour*, not a
`qwen35moe` family property — a sibling checkpoint of the same architecture and vendor line puts
its mass exactly where a label policy needs it. Any policy that fixes Occamy must therefore be
justified as a rescue, not as a family-wide correction. §5.1 qualifies that sentence with a
measurement the reader must have before acting on it.

### 5.1 The split is shape-dependent — the same 20 items, read twice

The `run`-shaped batch (§6) asks the **same** questions as the bench suite — `batch_questions.json`
is byte-identical to `docs/evidence/e3_batch_questions.json`, and all 20 items' `criteria` /
`instructions` are equal to dev rows `c01`…`c20` (`.e3c_tiel/batch_vs_devset.py`, 20/20 identical).
The two shapes disagree completely about Tiel's mass:

| shape | items | `measured` | coverage min · median · max | cue row's argmax |
|---|---|---|---|---|
| bench (`--suite quality`, per item, no state) | c01–c20 | **15/20** (low_mass: c01, c02, c07, c10, c17) | 7.88e-03 · **2.33e-01** · 5.73e-01 | a real token on 17/20 (e.g. `14501`, `271`, `2054`), `248068`/`248069` on 3 |
| serving (`--suite batch`, one state, `readout: sequence`, 4 forks/question) | c01–c20 | **0/20** | 7.97e-07 · **6.42e-06** · 1.39e-03 | special token `248069` on 19/20 (mass 0.68–0.99), `<|im_end|>` on `c11` (0.50) |

Worst case vs best case, same item: `c03` reads 5.36e-01 (`ok`, bench) and 8.13e-06 (`low_mass`,
serving) — five orders of magnitude apart, and the model does not even pick the same option
(bench `mobile` 0.966, serving `backend` 0.845). `c01`/`c02`/`c07`/`c10`/`c17` are `low_mass` on
*both* shapes; the other 15 flip.

What the two shapes differ in — named, **not** attributed: a saved prefix state (`prefix_tokens`
109, one `state_id` for all 20 questions), `n_ctx` 256 (planned per request, not the fit plan's
4096), `readout: "sequence"`, four forks per question decoded as 40 waves, and one process for all
20 questions instead of one per item. Which of these turns Tiel's cue row away from the labels is
a controlled-probe question, and it belongs to the card that owns the cue shapes (`t_6c119626`,
`docs/evidence/e3c_cue_shapes.md`) — this card measures the pair of readings and stops there.

**Consequence for `t_6952f0dd` (the one sentence to carry over):** "Tiel answers where Occamy does
not" is true on the **bench** shape and false on the **serving** shape — on that shape *both*
35B-A3B models return 20/20 `low_mass`, Occamy at coverage 8.8e-09…2.2e-06 and Tiel at
8.0e-07…1.4e-03 (`docs/evidence/e3_batch.json` vs `.e3c_tiel/batch_response.json`). A label policy
justified as a family-wide correction is still wrong (the bench shape separates them); one
justified as an Occamy-only rescue would leave the serving shape's refusal in place for the
sibling too.

## 6. The 20-question batch (deliverable 4)

Command (`.e3c_tiel/run_batch.sh`, raw response `.e3c_tiel/batch_response.json`), run inside
`e3c-tiel-extras2.service` — an unlimited scope (the unit's own accounting: `21.1 G memory peak`,
see §9 for why that matters):

```bash
NSEQ=8 bash .e3c_tiel/run_batch.sh     # GGUFONE_RUNTIME_DIR / VK_DRIVER_FILES as in §4
```

| what | value |
|---|---|
| outcome | exit 0 · **20/20 answers** (`c01`…`c20`, none empty) · no OOM in the log |
| wall | **35.6 s** whole run (`timings.total_ms` 35 618.1) = load 9.85 s + prefill 7.88 s + questions 27.72 s |
| `usage` | questions 20 · forks 80 · **waves 40** · decode_steps 117 · input_tokens 1 175 · output_tokens 117 · prefill_tokens 109 |
| placement used | `{n_gpu_layers: 9, kv_type: q4_0, degraded: false, attempts: []}` — the fit plan's own rung, no degrade |
| context | `n_ctx` 256 · `n_seq_max` 8 (the fit plan's bound) · `prefix_tokens` 109 · one `state_id` (`sha256:c206645c…`) · `readout: "sequence"` |
| warnings | `[W_TEMPLATE_FALLBACK, W_LOW_MASS, W_CUE_REFUSED]` · `calibrated: false` |

`waves = 40` for `forks = 80`: the same adaptation E3 §6.3 measured on Occamy — with four forks per
question the engine groups them (20 questions × 2 groups = 40 decode batches) instead of decoding
all 80 at once. No OOM, no degraded placement, exit 0.

**The answers are not label answers, and the response says so.** The batch's own verdict is
`low_mass` on **20/20** rows (§5.1): the cue row's argmax is the special token `248069` on 19 rows
(mass 0.68–0.99) and `<|im_end|>` on `c11` (0.50, the one `W_CUE_REFUSED`). 20/20 answers exist as
objects with a `choice` field; what they lack is mass at the readout the engine uses to justify it.

Speed context, stated so it is not misread: the same 20 questions cost Occamy **3 633.6 s** in E3
(`--n_seq_max 4`, 65 waves) — a number E3 itself attributes to the 8 GiB container, which could not
keep 23 GB of weights resident (E3 §6.2). Tiel's 35.6 s is a *host* figure at `n_seq_max 8`; the
workload matches (`forks` 80, `decode_steps` 117, `input_tokens` 1 175) and the wall does not, so
this card claims no speed comparison between the two runs.

## 7. The threads probe (deliverable 5)

`llama-bench` from the same bundle, at the fitted placement (`-ngl 9`), E3's own sizes (`-p 64`,
`-n 8`, `-r 2`):

| threads | pp64 (t/s) | tg8 (t/s) |
|---|---|---|
| 4 | 5.33 ± 2.39 | 1.85 ± 0.60 |
| **8** | **10.91 ± 3.12** | 3.26 ± 0.32 |
| 12 | 10.29 ± 2.87 | **5.53 ± 0.86** |

**Receipt:** the probe's raw `llama-bench` tables are `.e3c_tiel/extras_logs/extras.log`
(sha256 `2eb77fd5def9a3c39118cb3ec6df111b27969ec5beb9f5793f14305b0e2f524d`), written by
`.e3c_tiel/run_threads.sh` inside `e3c-tiel-extras.service` (18:37:00–18:41:15). Each row is
`-r 2` samples of the same binary E3 used, on the same model file, one load per row.

**Opposite of Occamy** (E3 §6.5: `-t 4` won at 1.71 pp64 / 0.28 tg8, and 8/12 were *worse*). Tiel
scales with threads on this box: prompt processing doubles from 4 → 8, and decode keeps climbing
to 12. The campaign kept `--threads 4` for E3 comparability — and therefore reports Tiel's
*lower* bound.

## 8. Three-way table (deliverable 6)

`docs/evidence/e3c_tiel_three_way.md` (generated by `tools/e3c_tiel_table.py`, which reuses
`ggufone.bench.compare`), paired on the same 60 items:

```
| metric | 4B default (E2) | Occamy 1.0 (E3) | Tiel-Coder (E3c) |
|---|---|---|---|
| overall | 0.633 (38/60) [0.507–0.744] | 0.517 (31/60) [0.393–0.638] | 0.517 (31/60) [0.393–0.638] |
| choice  | 0.750 (18/24) [0.551–0.880] | 0.625 (15/24) [0.427–0.788] | 0.667 (16/24) [0.467–0.820] |
| noul    | 0.889 (16/18) [0.672–0.969] | 0.389 (7/18)  [0.203–0.614] | 0.389 (7/18)  [0.203–0.614] |
| score   | 0.222 (4/18)  [0.090–0.452] | 0.500 (9/18)  [0.290–0.710] | 0.444 (8/18)  [0.246–0.663] |
| low_mass | 0.500 (6/12) [0.254–0.746] | 0.509 (29/57) [0.383–0.634] | 0.571 (8/14) [0.326–0.786] |
| measured | 0.667 (32/48) [0.525–0.783] | 0.667 (2/3) [0.208–0.939] | 0.500 (23/46) [0.361–0.639] |
```

**Every interval overlaps. No ranking is claimed** — and three of the four pairwise deltas that a
reader might want are inside the noise:

- **Tiel vs 4B (overall): −0.117** (0.633 → 0.517) — same gap E3 measured for Occamy; intervals
  [0.507–0.744] and [0.393–0.638] overlap.
- **Tiel vs Occamy (overall): 0.000** — identical counts (31/60). Indistinguishable.
- The one *large* gap either way is `noul`: Tiel matches Occamy's −0.500 against the 4B
  ([0.203–0.614] vs [0.672–0.969] — these do not overlap), and both 35B-A3B models answer
  yes/no questions markedly worse than the 4B.

The honest reading: on this 60-item set Tiel and Occamy are the same model quality-wise
(0.517 both), while Tiel is *not* mass-starved — so the pair separates **how much a reader can
trust each number**, not who answers better.

## 9. The measurement-invalidating finding (must not be lost)

The first campaign attempt ran inside the kanban worker's own cgroup:

```
hermes-worker-kanban-t_a58f8b67-run-169.scope   memory.max = 4 294 967 296 (4 GiB)
```

A 21 GB model in a 4 GiB cgroup cannot keep its mmap resident: every forward re-reads weights
from disk. Measured: **608 s for one 10-item chunk** (vs 137–205 s in an unlimited scope) — the
run's own record is `.e3c_tiel/flawed_capped/placement_001.json` (`wall_s` 608.0, `load_wall_s`
62.7, `degraded: true`), the chunk that was kept — plus a `read_bytes` figure of **68 GB in 12
minutes** on the next chunk and the process parked in `folio_wait_bit_common` (page-fault wait);
those two are quoted from the run record in this card's comment thread, not from a committed file.
The second capped chunk is `.e3c_tiel/flawed_capped/report_001.json`; both are kept as a labelled
artifact **outside** the published evidence. They are a clean measurement of *what the cap does*,
and an unclean basis for model comparison.

The card's `[host]` instruction is therefore not just "not the container": it is **not the worker's
scope either**. E3's own host run escaped the cap implicitly (it ran from a session outside this
worker's scope). Reproducing the campaign requires:

```bash
systemd-run --user --unit=e3c-tiel-campaign --collect \
  --working-directory=/var/home/rybens/workspace/ggufone \
  bash /var/home/rybens/workspace/ggufone/.e3c_tiel/run_chunks.sh
```

The run's own log prints `cgroup=… memory.max=max` as its provenance.

Second contention event, visible in the numbers: chunk 004 took **957.7 s** while a neighbouring
worker ran its own 21 GB `qwen35moe` probe on the same box — two 21 GB models on a 32 GB host. The
contention receipt is the sibling's own log timestamps, which bracket chunk 004's window
(18:14–18:30): `.e3c/logs/occamy_s01.log` 18:13, `occamy_s02.log` 18:19, `occamy_items.log` 18:29.
Chunk 004's per-item wall is therefore not comparable to the other five chunks; the *agreement* is
unaffected (the decode is deterministic given the same items, and its 5/10 matches the rest), but
no latency claim should be read from chunk 004.

## 10. What is not claimed

- No download: the model file predates the card (mtime 2026-09-04); nothing wrote either `.gguf`
  (SHA identical before/after, §1).
- No ranking between the three models: every 95 % interval in §8 overlaps except the `noul` pair
  noted there.
- `low_mass` is a property of the *answer at the cue*, not of correctness: the `measured` rows in
  §8 are the ones a reader should trust (0.500 for Tiel on 46 rows — a coin flip).
- Chunk 004's wall time is a contention artifact (§9), not a model property.
- **No mechanism** for the bench/serving split in §5.1 — the two readings are measured, the cause is
  not; the cue-shape card owns that probe.
- **No speed comparison** between Tiel's batch (35.6 s, host, `n_seq_max 8`) and Occamy's (3 633.6 s,
  container, `n_seq_max 4`) — different box and different batch width.
- **No claim about the token id `248069`**: it is a single-token special in the 24804x–24806x block
  (Tiel's vocabulary: `248046` = `<|im_end|>`, `248044` = `<|endoftext|>`, `248045` = `<|im_start|>`
  — `.e3c_tiel/special_tokens.py`), it is not `<|im_end|>` (so the engine's cue verdict does **not**
  fire) and this card does not name it. A control token that holds ~0.9 of the cue row's mass while
  escaping `W_CUE_REFUSED` is a classification question for the closers list in
  `src/ggufone/engine/cue.py`, reported to that card rather than patched here.

## 11. QA — what was verified, by what

`python3 .e3c_tiel/qa_tiel.py` (read-only; exits non-zero on any failure) — **ALL CHECKS PASS**:

| check | result |
|---|---|
| six chunk reports | 10 rows each, every row `ok`, union **60 rows, 60 unique ids** |
| dev-set pairing | each chunk's `devset_00N.jsonl` ids == its report's ids; **6/6 byte-identical to E3's `e3_chunks/devset_00N.jsonl`** (sha256) |
| merge arithmetic | `tools/e3c_tiel_reproduce.py --suite merge --reports docs/evidence/tiel_chunks/report_*.json` re-run into `/tmp` reproduces the committed `docs/evidence/tiel_quality.json` **field for field** (report keys and all 60 rows' `correct`/`coverage`/`reliability`) |
| three-way table | the three `model_row`s recompute to §8's numbers (Tiel 31/60, Occamy 31/60, 4B 38/60); the tie is asserted, not eyeballed |
| batch | 20 ids, **20/20 non-empty answers**; `usage`/`timings` printed from the artifact (§6) |
| model pins after the campaign | Tiel `9286a94c…cf17c`, Occamy `633ae57f…df757` — **both identical to the before-capture** (`.e3c_tiel/sha256_after.txt`) |
| downloads | none: both files' `mtime` predate the card, the hashes are unchanged before **and** after, and no command in `.e3c_tiel/*.sh` fetches anything |

The QA scripts themselves are committed evidence, not scratch: `qa_tiel.py` (the gate above),
`chunk_ledger.py` (§3/§7 tables), `split_figures.py` (§5.1), `batch_facts.py`/`batch_peek.py` (§6),
`batch_two_models.py`/`batch_vs_devset.py` (§5.1 pairing), `special_tokens.py` (§10), `shape_diff.py`
and `token_probe.py` (the bench-vs-serving rows), `fit_dump.py` (the fit plan capture).

One provenance wrinkle, kept rather than tidied: `.e3c_tiel/sha256_before.txt` lost its first line to
a stray re-run of the *before* script five minutes into the after-capture (the job was killed
immediately; it read no hash). The block in that file is the verbatim first capture, and it is
corroborated twice — `ggufone fit`'s own `model_sha256` in `.e3c_tiel/fit_tiel.json` (created
2026-09-19T15:20:46Z, i.e. read from the file on disk) and, for Occamy, `docs/evidence/e3c_sha256_receipt.json`
(`identical: true`). Nothing in §1 rests on the damaged line alone.
