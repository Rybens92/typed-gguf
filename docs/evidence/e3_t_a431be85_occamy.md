# E3 — Occamy 1.0 (35B-A3B MoE, Q4_K_L, 23 GB, local): runs and the 4B comparison

Card `t_a431be85` · branch `main` (this repo has no remote; the commits are local on the shared
tree) · Tier **M** (the card declares none) · report schema `ggufone.bench/v1`

**Scope (user, 2026-09-18, binding).** E3 runs *only* on Occamy 1.0. No Qwen3.6-35B-A3B download —
the artifact was already on disk (`t_23393cb8`, the card that would have downloaded it, is
archived). No run in this card downloads anything, and nothing writes a `.gguf` (the file hash is
re-checked after the campaign, §0).

Every number below is a command plus its real output. Machine-readable artifacts:
`docs/evidence/e3_environment.json`, `docs/evidence/e3_occamy_quality.json` (the merged run),
`docs/evidence/e3_batch.json` (the 20-question batch), `docs/evidence/e3_comparison.md` (the
published table) and the per-chunk reports `docs/evidence/e3_chunks/report_00*.json`.

## 0. The pin (A-E3-5)

| what | value |
|---|---|
| file | `/var/home/rybens/.hermes/models/Accio-Lab_occamy-1.0-Q4_K_L.gguf` |
| bytes | 24 113 674 848 (23 GiB) — `mtime` 2026-09-18 09:18, before this card started |
| **SHA-256** | `633ae57faf731e863cc3ba7cb75396a1b1e377190...` → see `docs/evidence/e3_environment.json` for the full digest |
| arch / size | `qwen35moe`, 40 layers, 24 102 685 184 weight bytes (`fit.ModelFacts`) |
| provenance | `Accio-Lab/occamy-1.0` (post-trained Qwen3.6-35B-A3B) via `bartowski/Accio-Lab_occamy-1.0-GGUF`; Apache-2.0 |
| runtime | pinned `llama-b11026-bin-ubuntu-vulkan-x64` bundle (the same build E1a/E1c pinned) |
| downloads | **none** — the model file predates the card; the runtime bundle was installed by `ggufone init` in E1a |
| weight mutation | **none** — `ggufone` has no code path that opens a `.gguf` for writing; the hash is re-verified after the campaign (§6) |

```bash
# the A-E3-5 command, run twice (before and after the campaign)
sha256sum /var/home/rybens/.hermes/models/Accio-Lab_occamy-1.0-Q4_K_L.gguf
```

Both runs are stored: `docs/evidence/e3_sha256_before.txt` (13:38 UTC, before the first load) and
`docs/evidence/e3_sha256_after.txt` (after the batch, ~16:00 UTC) — the two files are **byte
identical**, which is the "no weight mutation" half of the gate; the "no downloads" half is the
file's own `mtime` (2026-09-18 09:18, before this card) plus the downloads directory holding only
the E1a runtime tarball.

## 1. The box, and what the worker container really gets (A-E3-1)

| what | value |
|---|---|
| host | Ryzen 9 3900X (24 threads seen), 31 GiB RAM, RTX 3060 Ti 8 GiB (`driver 615.71.09`) |
| **container quota** | `cpu.max = 200000 100000` → **2 CPU-seconds/s**; `memory.max = 8 GiB`; `pids.max = 256` |
| runtime | pinned Vulkan bundle `~/.local/share/ggufone/runtime/b11026-linux-x64-vulkan` |
| GPU as the loader sees it | `Vulkan0: NVIDIA GeForce RTX 3060 Ti (8192 MiB, 5669 MiB free)`, `matrix cores: NV_coopmat2` |
| backend for every published Occamy row | `--backend vulkan` (forced), `--gpu-layers 7` |

The two container numbers are the ones a reader must keep in mind: **2 CPU-seconds/s** and
**8 GiB of memory**. The first makes the thread sweep a statement about the container (§5); the
second decides whether a 23 GB model can be cached at all (§2) — it cannot.

### 1.1 The GPU was reachable, but not through the mounted ICD

`--backend vulkan` resolved and llama.cpp printed `loaded Vulkan backend …`, which *looks* like a
working GPU — it is not. Under the CDI-mounted manifest
(`/etc/vulkan/icd.d/nvidia_icd.x86_64.json`, `library_path: libGLX_nvidia.so.0`) the loader reports

```
[Vulkan Loader] ERROR: loader_scanned_icd_add: Could not get 'vkCreateInstance' via
                'vk_icdGetInstanceProcAddr' for ICD /usr/lib64/libGLX_nvidia.so.0
ggml_vulkan: No devices found.
```

and a direct probe of the driver library says why:

```
vk_icdNegotiateLoaderICDInterfaceVersion(libGLX_nvidia.so.0) -> rc=-3 (INITIALIZATION_FAILED)
vk_icdNegotiateLoaderICDInterfaceVersion(libEGL_nvidia.so.0) -> rc=0
cuInit rc=0 cuDeviceGetCount rc=0 count=1 (CUDA driver 13040)
```

GLX needs an X display, EGL does not. One manifest in a writable directory is enough:

```bash
printf '{"file_format_version":"1.0.0","ICD":{"library_path":"/usr/lib64/libEGL_nvidia.so.0","api_version":"1.4.351"}}' \
  > /work/e3scratch/nvidia_egl_icd.json
export VK_DRIVER_FILES=/work/e3scratch/nvidia_egl_icd.json   # every [container] number below
```

This is environment configuration, not a repository change — but it belongs in the worker image,
otherwise every worker spends the same hour rediscovering that `loaded Vulkan backend` is not
evidence of a device.

## 2. Placement, and the wall this model hits (A-E3-1, A-E3-4)

### 2.1 What the E1c fit planner says (measured)

```
GGUFONE_RUNTIME_DIR=<bundle> VK_DRIVER_FILES=<icd> uv run ggufone fit \
    --model ~/.hermes/models/Accio-Lab_occamy-1.0-Q4_K_L.gguf --print --json
→ n_gpu_layers 7, n_ctx 4096, kv_type q4_0, n_seq_max 8
  "device memory bound: offloading 7/40 layers within 4661 MiB
   (--fit-target 1024 MiB, 5685 MiB free of 8192 MiB)"
  model_sha256 633ae57f…  host fingerprint e2f65aaa0226c162
```

7 of 40 layers is the whole of what 5.7 GB of free VRAM buys: Occamy's layers average ~600 MB
(24.1 GB / 40). The bench path pins that number explicitly (`--gpu-layers 7`), so every row below
is reproducible from its flags; the loader's own answer is recorded in each report
(`placement.used`) — **not** just the request, because a box with more than one bundle has
produced mislabelled bench rows before (coordinator's E2-provenance audit).

### 2.2 The wall: 8 GiB of cgroup memory against 23 GB of weights

An mmap'd GGUF is cached by the kernel *in the cgroup that faults it in*. This container's
`memory.max` is 8 GiB, so the model can never be resident, and every forward pass re-reads weights
from disk. Measured while one item was being answered:

```
/sys/fs/cgroup/memory.stat:  file 8.37 GB (file_mapped 8.28 GB) — the whole limit is page cache
/proc/<pid>/stat:            majflt 3.11e6 -> 3.23e6 in 8 s  (≈ 14 000 major faults/s ≈ 55 MB/s)
```

Occamy is a 40-layer MoE with ~3 B active parameters: a *prefill* touches essentially every
expert (one full ~21 GB weight sweep), and each decode step touches ~2 GB. That is the cost model
behind the per-item numbers below, and it is a property of the box, not of the flags.

### 2.3 Per-item timings (A-E3-1) and the chunked campaign

The first two dev items, measured through the bench path (`harness.LiveModel` + `DecisionEngine`)
with `--backend vulkan --gpu-layers 7 --threads 4`:

<!-- @@PER_ITEM_TABLE@@ -->

| item | type | prefill_ms | questions_ms | wall_s | correct | coverage | reliability |
|---|---|---:|---:|---:|---|---:|---|
| `c01` | choice | 152 825 | 105 163 | 258.6 | ✔ (`technical`) | 0.053 | `low_mass` |
| `c02` | choice | 72 956 | 74 249 | 147.2 | ✘ (`escalation` picked over `billing`) | 0.020 | `low_mass` |

`model_load_ms` = 30 846 (30.8 s, dominated by the 7-layer upload plus the first touched pages);
both items were answered with `placement.used = {n_gpu_layers: 7, kv_type: "auto", degraded:
false, attempts: []}` and the loader log's own device line
(`Vulkan0: NVIDIA GeForce RTX 3060 Ti`, `Vulkan0 compute buffer size is 362.2 MiB`), which is the
compute-path evidence the coordinator's E2-provenance note asks for next to the backend label.

The spread between `c01` (258.6 s) and `c02` (147.2 s) is the page cache: the first item faults in
weights the process has never touched, the second reuses whatever survived in the cgroup's 8 GiB.
`degrade=true` is exercised too — when the desktop's VRAM grew during the campaign, the ladder's
first attempt failed (`ggml_vulkan: Device memory allocation of size 949969664 failed →
ErrorOutOfDeviceMemory`) and the load settled one rung lower; every chunk report records what it
actually used (`placement.used`).

Because a single 60-item pass is a multi-hour job on this box, the campaign is sliced into
**stratified chunks of 10 items** (`devset.stratified_chunks`; every chunk — and every prefix of
them — mixes `choice | score | noul`), each with its own JSON report, merged by
`compare.merge_reports` into the one report the table reads:

```bash
python3 tools/e3_reproduce.py --write-chunks .e3/chunks --chunk 10
GGUFONE_RUNTIME_DIR=<bundle> VK_DRIVER_FILES=<icd> python3 tools/e3_reproduce.py \
    --suite quality --model ~/.hermes/models/Accio-Lab_occamy-1.0-Q4_K_L.gguf \
    --backend vulkan --gpu-layers 7 --threads 4 \
    --devset .e3/chunks/devset_001.jsonl --out docs/evidence/e3_chunks/report_001.json
python3 tools/e3_reproduce.py --suite merge --reports 'docs/evidence/e3_chunks/report_*.json' \
    --label "Occamy 1.0" --out docs/evidence/e3_occamy_quality.json
```

<!-- @@CHUNK_TABLE@@ -->

| chunk | items (choice/score/noul) | `placement.used` | wall per item (median) | correct |
|---|---|---:|---:|---:|
| `report_001` | 10 (4/3/3) | `n_gpu_layers 0`, `degraded: true` (`7 → oom`, `3 → oom`), `kv_type f16` | 113.2 s | 5/10 |
| `report_002` | 10 (3/4/3) | `n_gpu_layers 3`, `degraded: true` (`7 → oom`), `kv_type f16` | 99.5 s | 4/10 |

Both chunks loaded in ~31 s and ran 10 items in ~22 min (sum of `wall_ms`: 1354 s and 1325 s). Note the
placement: on the first chunk the desktop's VRAM was tight enough that the ladder walked all the way
down to **CPU-only**, on the second it stopped at 3 layers. That is the E1c degrade ladder doing its
job (`W_BACKEND_OOM` + `W_FIT_DOWNGRADE` in `placement.used.warnings`), and it is why the published
placement for this artifact has to be read from the report, not from the flags.

## 3. The 20-question batch (A-E3-2)

```bash
GGUFONE_RUNTIME_DIR=<bundle> VK_DRIVER_FILES=<icd> python3 tools/e3_reproduce.py \
    --suite batch --model ~/.hermes/models/Accio-Lab_occamy-1.0-Q4_K_L.gguf \
    --items 20 --n-seq-max 4 --threads 4 --backend vulkan --out docs/evidence/e3_batch.json
```

The batch is the *serving* path, not the bench path: 20 dev-set questions on one state (the first
item's), `n_seq_max` pinned to 4 so the engine must split the candidate branches into waves
(`planned_waves`) instead of decoding them in one batch. What the gate asks for is that it
*completes without OOM* and that the waves adapt; the agreement is measured by §4, not here.

<!-- @@BATCH_TABLE@@ -->

| what | value |
|---|---|
| wall (whole `ggufone run`) | **3 633.6 s** (60.6 min) — one load, one prefill, 65 decode batches |
| questions | 20 (a `choice`/`score`/`noul` mixture from the dev set) on the first item's state |
| `usage` | `prefill_tokens` 109 (= the shared state), `forks` 80, `decode_steps` 117, **`waves` 65**, `input_tokens` 1175, `output_tokens` 117 |
| placement used | `{n_gpu_layers: 3, kv_type: q4_0, degraded: false, attempts: []}` and the log's own `Vulkan0 compute buffer size is 363.5 MiB` |
| OOM | **none** — exit 0, 20/20 answers, `n_ctx` 256 |
| prompt | the GGUF's own template (`source='gguf:tokenizer.chat_template'`, family `qwen35moe`), `thinking: suppressed` |

`waves = 65` for `forks = 80` is the engine doing exactly what A-E3-2 asks: with `n_seq_max = 4`
there are 3 candidate slots per decode batch, so the 80 forks cannot be decoded at once — the
engine splits them into waves (80 candidate tokens / 3 per wave + one suffix decode per question)
instead of raising `E_SEQ_MAX_EXCEEDED` or OOM-ing. The 60 minutes are the same physics as §2.2:
65 decode batches plus one prefill on a 23 GB model whose weights cannot be cached in 8 GiB.

One label worth flagging for the FIX series: the response's `engine.backend` reads **`cpu`** while
the placement put 3 layers on the Vulkan device and the log proves a Vulkan compute buffer — the
same class of mislabelling the coordinator's E2-provenance audit found in the bench tables and
card `t_603a35a0` fixed there. In the *serving* path the field still comes from the fit plan's
`backend` rather than from the device the log shows.

## 4. The comparison: 4B default vs Occamy 1.0 (A-E3-3)

<!-- @@COMPARISON@@ -->
```markdown
| metric | 4B default (E2, 60 items, CPU) | Occamy 1.0 (E3, chunks, vulkan) | delta |
|---|---|---|---|
| overall | 0.500 (10/20) [0.299–0.701] | 0.450 (9/20) [0.258–0.658] | -0.050 |
| choice | 0.429 (3/7) [0.158–0.750] | 0.571 (4/7) [0.250–0.842] | +0.143 |
| noul | 1.000 (6/6) [0.610–1.000] | 0.167 (1/6) [0.030–0.564] | -0.833 |
| score | 0.143 (1/7) [0.026–0.513] | 0.571 (4/7) [0.250–0.842] | +0.429 |
| low_mass (below the floor) | 0.333 (1/3) [0.061–0.792] | 0.450 (9/20) [0.258–0.658] | +0.117 |
| measured (at or above the floor) | 0.529 (9/17) [0.310–0.738] | — | — |
```
(The published copy, with its generator line, is `docs/evidence/e3_comparison.md`; the numbers come
from `tools/e3_reproduce.py --suite compare … --align`, never from this file by hand.)

The table is the file `docs/evidence/e3_comparison.md`, generated by

```bash
python3 tools/e3_reproduce.py --suite compare --a docs/evidence/e2_quality.json \
    --b docs/evidence/e3_occamy_quality.json --align \
    --labels "4B default (E2, 60 items, CPU)" "Occamy 1.0 (E3, chunks, vulkan)"
```

and it is **paired**: `--align` cuts both sides to the 20 items Occamy actually measured (the
baseline's other 40 rows are dropped from *both* columns, so the two sides ask the same questions).

### 4.1 What the numbers say

<!-- @@COMPARISON_TEXT@@ -->

* **The two models are 0.05 apart overall on the 20 paired items** (4B 10/20, Occamy 9/20) — i.e.
  one item, well inside both Wilson intervals. The 20-item paired table cannot separate them; what
  it *can* say is where they differ, and that is not noise:
* **Occamy's answers are low-mass on every single item** (`low_mass` 20/20, i.e. the full-vocabulary
  mass the engine saw on the candidate tokens was under the 0.10 floor on all 20). The 4B on the
  same items is `low_mass` on 3/20, and its `measured` row (17 items, 9 correct, 0.529) is the one
  the table says to read first — Occamy has no `measured` row at all. The discrete decisions are
  still 9/20 correct, but they are taken on a distribution that barely assigns mass to the labels
  the protocol asks for.
* **The per-type split is not noise either**: `noul` 6/6 for the 4B against 1/6 for Occamy, while
  `score` goes the other way (1/7 vs 4/7). Five of Occamy's six noul answers are `no` with
  confidence 0.76–0.90 (the dev set's gold is `yes` for five of them), which is a systematic
  answer-style difference, not item-level luck.
* **This is model behaviour, not a rendering bug** (worth saying because the first hypothesis to
  check was a template failure): both GGUFs carry their own `tokenizer.chat_template`, both resolve
  through the chain's step 1 (`source='gguf:tokenizer.chat_template'`, no warnings), and both
  prompts end at their assistant header — `<|im_start|>assistant\n` for Occamy,
  `<|Bot|></think>\n` for the 4B (`tools/`-style probe, reproduced in §4.2). The difference is what
  the model does with that prompt, not what it was given.

### 4.3 Report integrity, the way the coordinator asked for it

Every Occamy row in this document comes from a run with an **explicit `--backend vulkan`** (never
`--backend all`, which on a single-bundle host is designed to fail the `cpu` row — coordinator's
`t_dd62ec29` note), and every report's own `ok`/row count was checked before publishing:

| report | `ok` | rows |
|---|---|---|
| `e3_chunks/report_001.json` | `true` | 10/10 items measured |
| `e3_chunks/report_002.json` | `true` | 10/10 items measured |
| `e3_occamy_quality.json` (merged) | `true` | 20 items, `chunks` names both |
| `e3_batch.json` (serving path) | exit 0 | 20/20 answers, `engine` + `usage` present |

The failure shape we *did* meet is the documented one for a busy GPU, and it did not cost a row:
`ggml_vulkan: Device memory allocation of size 949969664 failed → ErrorOutOfDeviceMemory` twice,
after which the E1c degrade ladder dropped the placement (7 → 3 → 0 layers) and the run continued
(`degraded: true` in each chunk's `placement.used`). No `SIGSEGV`, no `ok: false`, no exit 1 in any
E3 run.

### 4.2 The prompt check (offline, no model load)

```
$ uv run --frozen python /work/e3scratch/probe_template2.py
===== Occamy 1.0 (qwen35moe)
  arch=qwen35moe chat-ish keys=['tokenizer.chat_template'] template present=True
  resolution: source='gguf:tokenizer.chat_template' kind='gguf-renderer' family='qwen35moe' warnings=[]
  prefix len=483 tail='...<|im_end|>\n<|im_start|>assistant\n'

===== Spark-X2.5-4B (spark2_5)
  arch=spark2_5 chat-ish keys=['tokenizer.chat_template'] template present=True
  resolution: source='gguf:tokenizer.chat_template' kind='gguf-renderer' family='spark2_5' warnings=[]
  prefix len=569 tail='...<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|Bot|></think>\n'
```

Both prompts end where the engine's readout expects them to; the noul/`low_mass` asymmetry is in
the answers, not in the bytes. (That probe also caught its own first version's bug — feeding the
string `"None"` in as the template when a KV lookup missed — which is why §4.1 states the
`source=` line rather than a paraphrase.)

## 5. Threads and the routing recommendation (A-E3-4)

The card asks for the 4/8/12 thread question on the 24-thread 3900X. In *this container* the
question is answered by the quota before it is asked: `cpu.max = 2 CPU-seconds/s` means every
setting above 2 is oversubscription, and E2's five-point probe (§3.5 of `BENCHMARKS.md`) already
measured that `threads = cores_seen` is the worst setting on the box (24 threads: 5× slower
prefill, 20× slower decision on the 4B). The same sweep with the 23 GB MoE and a partial offload
(`llama-bench` from the pinned bundle, `-p 64 -n 8 -r 1`, ~2 min per row because each row loads
the model):

<!-- @@THREADS_TABLE@@ -->

| setting (`llama-bench`) | pp64 (tok/s) | tg8 (tok/s) | vs the best prefill |
|---|---:|---:|---:|
| `-ngl 7 -t 4` | **1.71** | **0.28** | 1.00× |
| `-ngl 7 -t 8` | 0.91 | 0.13 | 0.53× |
| `-ngl 7 -t 12` | 1.07 | 0.11 | 0.63× |
| `-ngl 0 -t 4` (CPU only) | 0.87 | 0.09 | 0.51× |

Two things are measured here and both matter for A-E3-4:

* **`threads = 4` wins**, and `8`/`12` lose by 2× — the same oversubscription shape E2 measured on
  the 4B (E2 §3.5: 24 threads was 5× slower than 4 for prefill). The 3900X has 24 threads; this
  container may use 2.
* **The 7 offloaded layers buy ~2× prefill and ~3× decode** (1.71 / 0.28 against 0.87 / 0.09
  CPU-only) — so on a GPU box it is worth offloading exactly what fits, and no more: the same
  `llama-bench` at `-ngl 7` on a 0.28 tok/s decode is ~3.6 s per token, which is what the per-item
  numbers in §2.2 look like from the outside.

### 5.1 The recommendation

| knob | recommendation | why (measured) |
|---|---|---|
| `n_gpu_layers` | **7** when ~5.7 GB of VRAM is free (`ggufone fit`), and expect the ladder to settle at **3** or **0** on a busy desktop | §2.2, §2.3: the desktop's 2.2 GB plus a sibling's bench triggered `ErrorOutOfDeviceMemory` at 7 and even at 3 |
| `kv_type` | `auto` for these requests; `q4_0` at `n_ctx 4096` is what `ggufone fit` recommends | KV is 94 MiB (4k/q4_0) vs 142 MiB (4k/f16) — irrelevant beside 23 GB of weights |
| `n_ctx` | keep the request small; the prefill is one weight sweep regardless of its length, and the KV is charged to the same 8 GiB | §2.2 |
| `threads` | **4** | the table above; and E2 §3.5 for the 4B |
| routing overall | **do not route this artifact to this container for interactive work.** It answers at 0.28 tok/s (decode) here because it cannot be cached; route it to a host/device where ≥24 GB is resident (the operator host's own 31 GiB), or run it as a batch job and accept ~2–3 min per decision | §2.2, §3, §5 |

The measured reason the artifact is slow here is not the GPU and not the flags: it is
`23 GB of weights against an 8 GiB memory limit`. On a host that can hold them, the same run's
per-item cost is bounded by compute (a few seconds), not by a disk sweep — which is what a
`[host]` re-run of §2's command would show.

## 6. Honest limits of this evidence

* **This is the worker container on the operator host, not a bare-host run.** Every number above is
  tagged by construction `[container]`: `cpu.max = 2 CPU-seconds/s` and `memory.max = 8 GiB`. The
  second is what makes the 23 GB model slow *here* — on a host whose 31 GiB can hold the weights,
  the same item is not a disk-IO-bound forward pass. So the per-item cost is an **upper bound** for
  this box; a `[host]` re-run of the same command (`tools/e3_reproduce.py --suite quality …`) is
  the honest way to get the floor. The quality numbers themselves (agreement, coverage, the
  comparison table) do not depend on it: they are read from the answers, and the placement only
  gates them through numerical noise.
* **The campaign is chunked, and the reports say which chunks landed** (`chunks` in
  `docs/evidence/e3_occamy_quality.json`): 2 chunks — **20 of the 60 dev items**. Every chunk is a
  complete run of its stratified subset; the table's `n` is that count, never 60. The remaining 40
  items are the same command with `--devset .e3/chunks/devset_00{3..6}.jsonl`, and
  `compare.merge_reports` accepts them without touching the published rows.
* **The box was shared while measuring**: a sibling card (`t_f46cec41`) ran its own `ggufone bench`
  in a different checkout (visible in `ps`: `/work/ggufone-t603/…`), and the desktop kept its
  ~2.2 GB of VRAM. Both are in the numbers: the `ErrorOutOfDeviceMemory` degradations (6.2) and the
  per-item spread (57 s–263 s). The agreement rows are immune to it; the timings are "this box,
  today", exactly what the `p50`/`p95` convention of E2 is for.
* **Agreement is report-only** (SPEC S-11): no minimum is claimed, no vendor score is implied, and
  a 20-item table cannot separate models that are one item apart — which is precisely why the
  per-type and `low_mass` rows are published next to the overall one.
* **`low_mass` is the headline, not a footnote.** Occamy's answers on this dev set are all below
  the engine's mass floor, so its discrete decisions are taken on a distribution that barely
  assigns probability to the labels the protocol asks for. E2's quality suite is report-only; the
  honest reading of §4 is "the default 4B is the better *router* for this short-label protocol on
  this box", not "the 23 GB model is worse at the task".
* **A-E3-6 (32k long-context smoke) is not in this card** — the card's gate list is A-E3-1…5; SPEC
  §5 E3's A-E3-6 belongs to the earlier, superseded scope (27B **and** 35B). At ~3 min per prefill
  sweep here, a 32k-token state is a `[host]` measurement, not a container one.
* **A structural finding worth a FIX card**: `llama_model_params` carries `tensor_buft_overrides`
  (`src/ggufone/runtime/ctypes_binding.py:72`) but nothing sets it, so the SPEC's "MoE expert
  offload via tensor buffer overrides where needed" is not reachable from `ggufone` today — the
  only offload knob is whole layers (`--gpu-layers`), which is why a 5.7 GB VRAM budget buys 7 of
  40 layers instead of "attention + shared experts on the GPU, experts on the host".
