# E3c — Tiel-Coder (Ornith-1.5-35B, 35B-A3B, 21 GB local) measured like Occamy 1.0

Card `t_a58f8b67` (main-coder) · host run · 2026-09-19 · commit to follow this file.

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

Per-item wall: median ≈ 13–20 s over the 60 items (chunk 004's items ran under a *neighbouring*
21 GB probe — see §7).

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
justified as a rescue, not as a family-wide correction.

## 6. The 20-question batch (deliverable 4)

§8 below, after the run.

## 7. The threads probe (deliverable 5)

`llama-bench` from the same bundle, at the fitted placement (`-ngl 9`), E3's own sizes (`-p 64`,
`-n 8`, `-r 2`):

| threads | pp64 (t/s) | tg8 (t/s) |
|---|---|---|
| 4 | 5.33 ± 2.39 | 1.85 ± 0.60 |
| **8** | **10.91 ± 3.12** | 3.26 ± 0.32 |
| 12 | 10.29 ± 2.87 | **5.53 ± 0.86** |

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
from disk. Measured: **608 s for one 10-item chunk** (vs 137–205 s in an unlimited scope),
`read_bytes` **68 GB in 12 minutes** on the next chunk, and the process parked in
`folio_wait_bit_common` (page-fault wait). Two chunks were produced under that cap and are kept as
a labelled artifact **outside** the published evidence: `.e3c_tiel/flawed_capped/`. They are a
clean measurement of *what the cap does*, and an unclean basis for model comparison.

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
worker ran its own 21 GB `qwen35moe` probe on the same box (`.e3c/gufone…/tools/e3c_cue_shapes.py`,
`occamy_n01.log`) — two 21 GB models on a 32 GB host. Its per-item wall is therefore not
comparable to the other five chunks; the *agreement* is unaffected (the decode is deterministic
given the same items), but no latency claim should be read from chunk 004.

## 10. What is not claimed

- No download: the model file predates the card (mtime 2026-09-04); nothing wrote either `.gguf`
  (SHA identical before/after).
- No ranking between the three models: every 95 % interval in §8 overlaps except the `noul` pair
  noted there.
- `low_mass` is a property of the *answer at the cue*, not of correctness: the `measured` rows in
  §8 are the ones a reader should trust (0.500 for Tiel on 46 rows — a coin flip).
- Chunk 004's wall time is a contention artifact (§9), not a model property.
