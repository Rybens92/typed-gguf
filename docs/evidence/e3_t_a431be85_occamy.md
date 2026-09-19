# E3 — Occamy 1.0 (35B-A3B MoE, Q4_K_L, 23 GB, local): runs and the 4B comparison

Card `t_a431be85` · branch `main` (this repo has no remote; the commits are local on the shared
tree) · Tier **M** (the card declares none) · report schema `ggufone.bench/v1`

**Completion (card `t_6d2e084d`, 2026-09-19).** A-E3-3 was published on purpose at 20 of 60 dev
items; the four remaining chunks (`003`–`006`) ran on the operator host and the comparison below is
now the paired **60-item** table. §2.3, §4 and `docs/BENCHMARKS.md` §6.2/§6.4 are **regenerated from
the artifacts** by `tools/e3_build_evidence.py` (the same tool `--check` gates); the `[container]`
per-item cost table stays as published, and the `[host]` rows are tagged from the reports' own cgroup
facts, never typed.

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
with `--backend vulkan --gpu-layers 7 --threads 4` — **`[container]`**, the 8 GiB / 2 CPU-s/s
worker cgroup (the `[host]` side of the same measurement is below):

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

The chunks the card finished ran on the **operator host** (`[host]`), same protocol, same flags —
the differences are the box and the ICD manifest, and both are part of the recipe:

```bash
# the host run (card t_6d2e084d): one chunk per process, chunks 003..006
GGUFONE_RUNTIME_DIR=~/.local/share/ggufone/runtime/b11026-linux-x64-vulkan \
VK_DRIVER_FILES=~/.e3c_host/nvidia_egl_icd.json \
env -u VK_INSTANCE_LAYERS \
  python3 tools/e3_reproduce.py --suite quality \
    --model /var/home/rybens/.hermes/models/Accio-Lab_occamy-1.0-Q4_K_L.gguf \
    --backend vulkan --gpu-layers 7 --threads 4 \
    --devset docs/evidence/e3_chunks/devset_00N.jsonl \
    --out docs/evidence/e3_chunks/report_00N.json      # N = 003 … 006
```

`VK_DRIVER_FILES` points at the same one-line manifest §1.1 documents (`libEGL_nvidia.so.0`, which
does not need a display — the host shell has none either), and `VK_INSTANCE_LAYERS` is dropped
because this desktop session injects three implicit layers, one of which cannot resolve
`vkGetInstanceProcAddr`; none of them existed in the container. Nothing is downloaded: the model and
the pinned bundle are the ones on disk, and the artifact is re-hashed before and after —
`docs/evidence/e3c_sha256_before.txt` (15:55:18 CEST, three seconds before the first chunk loaded),
`e3c_sha256_after.txt` (after the campaign) and the machine-readable `e3c_sha256_receipt.json`, all
three digests equal to E3's published `e3_sha256_before.txt` (`identical: true`, 24 113 674 848 B).

<!-- @@E3_HOST_ITEMS_BEGIN@@ -->
The card's `[host]` run (chunks `report_003`, `report_004`, `report_005`, `report_006`) uses the same flags `--backend vulkan --gpu-layers 7 --threads 4` on the operator host — no cgroup, no 8 GiB memory cap, so the weights cache after the first pass. The first two items of `report_003.json`:

| item | type | questions_ms | wall_s | correct | coverage | reliability |
|---|---|---:|---:|---|---:|---|
| `n07` | noul | 23,408 | 46.9 | ✘ (`no`) | 0.037 | `low_mass` |
| `c08` | choice | 28,577 | 49.1 | ✔ (`business_hours`) | 0.052 | `low_mass` |

Per-item wall on the host, by chunk: `report_003` 49.2 s, `report_004` 42.7 s, `report_005` 47.0 s, `report_006` 44.6 s — against 99–113 s per item in the container, which is the point of the tag: the *protocol* is identical and the *box* is not.
<!-- @@E3_HOST_ITEMS_END@@ -->

The two container chunks settled differently, and that is the E1c degrade ladder doing its job
(`W_BACKEND_OOM` + `W_FIT_DOWNGRADE` in `placement.used.warnings`): the desktop's VRAM was tight
enough during the first chunk that the ladder walked all the way down to **CPU-only**, while the
second stopped at 3 layers — which is why the published placement for this artifact has to be read
from the report, not from the flags. The `[host]` chunks ran on the same 8 GiB device, with the
desktop session (and at times a sibling benchmark sharing the box) on it, so whether the 7 requested
layers fit is a per-chunk question there: the quality suite records the device attribution the
engine's own log proves (`devices` / `device_buffers` / `effective_backend`, card `t_603a35a0`)
rather than a ladder answer, and the chunk ledger below prints exactly what each report carries.

<!-- @@E3_CHUNK_BEGIN@@ -->
Because a single 60-item pass is a multi-hour job on the container, the campaign is sliced into **stratified chunks of 10 items** (`devset.stratified_chunks`; every chunk — and every prefix of them — mixes `choice | score | noul`), each with its own JSON report, merged by `compare.merge_reports` into the one report the table reads:

```bash
python3 tools/e3_reproduce.py --write-chunks .e3/chunks --chunk 10
GGUFONE_RUNTIME_DIR=<bundle> VK_DRIVER_FILES=<icd> python3 tools/e3_reproduce.py \
    --suite quality --model ~/.hermes/models/Accio-Lab_occamy-1.0-Q4_K_L.gguf \
    --backend vulkan --gpu-layers 7 --threads 4 \
    --devset docs/evidence/e3_chunks/devset_00N.jsonl \
    --out docs/evidence/e3_chunks/report_00N.json     # N = 001 … 006
python3 tools/e3_reproduce.py --suite merge \
    --reports 'docs/evidence/e3_chunks/report_*.json' \
    --label "Occamy 1.0" --out docs/evidence/e3_occamy_quality.json
```

| chunk | box | items (choice/score/noul) | compute path the report proves | wall per item (median) | correct |
|---|---|---|---|---:|---:|
| `report_001` | `[container]` | 10 (4/3/3) | `n_gpu_layers 0`, `kv_type f16`, `degraded: true`, walked 7→oom, 3→oom | 113.2 s | 5/10 |
| `report_002` | `[container]` | 10 (3/4/3) | `n_gpu_layers 3`, `kv_type f16`, `degraded: true`, walked 7→oom | 99.5 s | 4/10 |
| `report_003` | `[host]` | 10 (3/3/4) | buffers: `Vulkan0`=10, `Vulkan_Host`=10; effective `vulkan` | 49.2 s | 4/10 |
| `report_004` | `[host]` | 10 (4/3/3) | buffers: `Vulkan0`=10, `Vulkan_Host`=10; effective `vulkan` | 42.7 s | 4/10 |
| `report_005` | `[host]` | 10 (3/4/3) | buffers: `Vulkan0`=10, `Vulkan_Host`=10; effective `vulkan` | 47.0 s | 8/10 |
| `report_006` | `[host]` | 10 (7/1/2) | buffers: `Vulkan0`=10, `Vulkan_Host`=10; effective `vulkan` | 44.6 s | 6/10 |

Merged: `docs/evidence/e3_occamy_quality.json` — 60 items, 31 correct (0.517, 95 % CI 0.393–0.638); `chunks` lists every chunk it stitched: `report_001.json`, `report_002.json`, `report_003.json`, `report_004.json`, `report_005.json`, `report_006.json`.

A **`[host]` run did happen**: 4 of the 6 chunks (`report_003`, `report_004`, `report_005`, `report_006`) were measured on the operator host after the card moved the campaign off the container (`[container]`: `report_001`, `report_002`). What it changed is `n` and the intervals — the published per-item *cost* table keeps its `[container]` tag, because the container is where a 23 GB model against 8 GiB of memory shows its real price; the host rows are the same protocol on a box that can cache the weights, and each row above says which box it came from.
<!-- @@E3_CHUNK_END@@ -->

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

<!-- @@E3_COMPARE_BEGIN@@ -->
Paired on the dev items **both** models measured — `e2_quality.json` cut to the same ids (`tools/e3_reproduce.py --suite compare --align`):

```markdown
Agreement on the committed dev set, 95 % Wilson intervals; the mass split uses the engine's own verdict, or `coverage < 0.10` where a report predates it.

| metric | 4B default (E2, 60 items, CPU) | Occamy 1.0 (E3, chunks, vulkan) | delta |
|---|---|---|---|
| overall | 0.633 (38/60) [0.507–0.744] | 0.517 (31/60) [0.393–0.638] | -0.117 |
| choice | 0.750 (18/24) [0.551–0.880] | 0.625 (15/24) [0.427–0.788] | -0.125 |
| noul | 0.889 (16/18) [0.672–0.969] | 0.389 (7/18) [0.203–0.614] | -0.500 |
| score | 0.222 (4/18) [0.090–0.452] | 0.500 (9/18) [0.290–0.710] | +0.278 |
| low_mass (below the floor) | 0.500 (6/12) [0.254–0.746] | 0.509 (29/57) [0.383–0.634] | +0.009 |
| measured (at or above the floor) | 0.667 (32/48) [0.525–0.783] | 0.667 (2/3) [0.208–0.939] | +0.000 |

`Occamy 1.0 (E3, chunks, vulkan)` is worse than `4B default (E2, 60 items, CPU)` by -0.117 overall (0.633 -> 0.517); the `measured` row is the one to read first.
```

`Occamy 1.0 (E3, chunks, vulkan)` is -0.117 against `4B default (E2, 60 items, CPU)` overall (0.633 -> 0.517); the paired comparison covers 60 dev items and drops 0 unpaired baseline row(s) and 0 unpaired challenger row(s), so both sides answer the same questions.

### 4.1 What the numbers say

* **`choice`**: 0.750 (18/24) [0.551–0.880] against 0.625 (15/24) [0.427–0.788] — delta -0.125.
* **`noul`**: 0.889 (16/18) [0.672–0.969] against 0.389 (7/18) [0.203–0.614] — delta -0.500.
* **`score`**: 0.222 (4/18) [0.090–0.452] against 0.500 (9/18) [0.290–0.710] — delta +0.278.
* **mass**: Occamy's answers fall below the 0.10 floor on 57 of its 60 rows (the 4B's on 12); inside the split the two are level (0.500 (6/12) [0.254–0.746] against 0.509 (29/57) [0.383–0.634]), and the row the table says to read first is `measured` — 0.667 (32/48) [0.525–0.783] against 0.667 (2/3) [0.208–0.939], which is only 0 item(s).
* **verdict**: 7 items apart overall at n = 60 (0.633 vs 0.517); the two Wilson intervals overlap, so the headline cannot separate the models — the rows that separate them are the per-type ones and the mass split above.

The Occamy side of this table is the merged campaign report (`[container]` ×2 + `[host]` ×4, per-chunk tags in §2.3), measured with `--backend vulkan --gpu-layers 7 --threads 4`; the baseline is E2's 4B default on the CPU. Agreement does not depend on the ladder a chunk settled on, and every chunk report carries its own compute-path evidence (table in §2.3).
<!-- @@E3_COMPARE_END@@ -->

### 4.3 Report integrity, the way the coordinator asked for it

<!-- @@E3_INTEGRITY_BEGIN@@ -->
Every Occamy row in this document comes from a run with an **explicit `--backend vulkan`** (never `--backend all`, which on a single-bundle host is designed to fail the `cpu` row — coordinator's `t_dd62ec29` note), and every report's own `ok`/row count was checked before publishing:

| report | box | `ok` | rows | report shape |
|---|---|---|---|---|
| `report_001.json` | `[container]` | `true` | 10 items | placement block, no `devices`/`device_buffers`/`effective_backend` |
| `report_002.json` | `[container]` | `true` | 10 items | placement block, no `devices`/`device_buffers`/`effective_backend` |
| `report_003.json` | `[host]` | `true` | 10 items | carries the `t_603a35a0` device attribution (`devices`/`device_buffers`/`effective_backend`) |
| `report_004.json` | `[host]` | `true` | 10 items | carries the `t_603a35a0` device attribution (`devices`/`device_buffers`/`effective_backend`) |
| `report_005.json` | `[host]` | `true` | 10 items | carries the `t_603a35a0` device attribution (`devices`/`device_buffers`/`effective_backend`) |
| `report_006.json` | `[host]` | `true` | 10 items | carries the `t_603a35a0` device attribution (`devices`/`device_buffers`/`effective_backend`) |
| `e3_occamy_quality.json` (merged) | — | `true` | 60 items, `chunks` lists all 6 | top-level keys are chunk 001's (`merge_reports` copies the first report's envelope) |
<!-- @@E3_INTEGRITY_END@@ -->

The failure shape we *did* meet is the documented one for a busy GPU, and it did not cost a row:
`ggml_vulkan: Device memory allocation of size 949969664 failed → ErrorOutOfDeviceMemory` in the
container chunks (the ladder dropped the placement 7 → 3 → 0 layers and the run continued,
`degraded: true` in `placement.used`), and the same `ErrorOutOfDeviceMemory` on the host, where the
allocation that fails is the first offload attempt on a device the desktop already fills. No
`SIGSEGV`, no `ok: false`, no exit 1 in any E3 run.

### 4.4 Report provenance, and the one thing that does not line up

`report_001.json` and `report_002.json` do **not** have the key shape the committed bench path
writes: they carry a `placement` block that no revision of `suites._run_quality` in this repo writes
(the key exists only in the latency suite's report) and lack `budget`/`wall_ms`/`truncated`/
`skipped`/`quick` plus the `devices`/`device_buffers`/`effective_backend` attribution the same
card's QA note says they carry (`docs/evidence/e2_quality.json`, the 4B baseline, has the same
reduced shape — so the reduction predates this card). Their rows are the published container
measurements and nothing in this card changed them; the four `[host]` chunks are written by
`tools/e3_reproduce.py` from the committed tree, so the merge mixes both shapes — the merged
report's envelope is chunk 001's, because `compare.merge_reports` copies the first report — and §4.3
prints the per-report shape in its table rather than papering over it. The two files are left as
published: re-generating them is a decision for the coordinator, not a completion card.

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
