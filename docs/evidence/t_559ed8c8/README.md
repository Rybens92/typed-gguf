# t_559ed8c8 — E2E: the serve wave end-to-end (installed wheel → serve + SDK; update + rollback)

Head `3d8e088e22379680e185ffa3a718214eb9c01fca` (working tree clean), `env -u PYTHONPATH uv build` →
`typed_gguf-0.2.3-py3-none-any.whl` (sha256 `18521f290ce434f73701abf7c5a25e8720b1b23c0d7580668c5310822b6721fe`),
`uvx ruff check src tests tools docs .github` → *All checks passed!*.
Everything below runs a wheel installed into a fresh venv with the repo **not** on `sys.path`
(`imports from the repo checkout: False`), against a real llama.cpp b11026 runtime tree and the real
0.8B GGUF that are on this box.

## Report (the 12 lines)

1. **STORY 1 — SHIP.** Fresh-venv install, `version`, `serve --help` fine; the real `serve` answered a
   real `typesafe-sdk==0.7.1` client over loopback: cold 6.76 s, warm 2.28 s, both `200 served_by=host`,
   the gate's own `verify()` PASS on the real model. `logs/01-clean-install.log`, `logs/serve_long_story.log`,
   `logs/serve_long.log`, `logs/sdk_body.{cold,warm}.json`.
2. **STORY 1 extra — A-E5-4 on the real model.** served `noul 0.574913` == `ask` `noul 0.574913`, same
   warm-host pid 6069, same usage `128/2`. `logs/same_engine.log`.
3. **STORY 2 — SHIP.** `runtime update --check`: b11026 → b99999 (18.41 MB), record byte-identical;
   `runtime update`: record moves to `b99999-linux-x64-cpu`, `previous` = b11026, probe
   `symbols_ok:true backends:[cpu,rpc] expect_backend:cpu`, doctor green, second `--check` says
   "already at b99999"; `runtime rollback` returns to b11026, doctor green.
   `logs/update_check_leg.log`, `logs/update_story.log`, `logs/update.json`, `logs/rollback.json`.
4. **STORY 2 kill -9 — SHIP.** Throttled download `kill -9`d at 3 145 728 B (exit 137):
   `runtime.json` byte-identical (`7ac69aa5…c3bc27`), no `b99998` bundle, no `.pending-*` staging, no
   surviving process; the retry completed and the rollback returned. `logs/kill_leg.log`.
5. **STORY 3 — SHIP.** 13 negative legs (wrong SDK, §2.9 422/404/413 + the 30 s idle bound, rung-1
   refusal, `E_RUNTIME_MISSING`, offline, no matching asset, 404/truncated/mismatched asset, release
   API 500, no-`previous` rollback, CUDA pre-flight) — every refusal leaves `runtime.json`
   **byte-identical**, with the code and the status the SPEC's tables name.
   `logs/negatives_serve.log`, `logs/negatives_update.log`, `logs/negatives_update_b.log`.
6. **The fix this card had to re-verify.** `--backend cuda` is refused by `install._preflight_reason`
   **before a byte is fetched**: `E_RUNTIME_SYMBOLS … skipped the 168.81 MB download`, and the fixture's
   request log shows the release list only — no asset GET (`U8`).
7. **The §2.9 bounds (t_ba767a2b).** A >1 MiB body with **no body sent** is answered `413` in 0.000 s on
   both route families (`E_BODY_TOO_LARGE` / `too_large`); a half-sent request is closed after **30.0 s**.
8. **Leaks.** After `keep stop`: no `host.json`/`spec.json`/`*.sock`, no host/serve process, port 8088
   closed; the sentinel `Authorization` value appears **0** times in the server log.
9. **PROPOSALS (non-blocking, details + repro below):** P1 the rollback record loses the installed-probe
   keys (`version` then prints `backends unknown`); P2 GitHub download failures are worded
   "HuggingFace returned HTTP 404" and send `User-Agent: typed-gguf/0.1`; P3 `keep stop` leaves
   `<key>.log` behind; P4 the SDK's default 10 s timeout vs a cold shader cache (already documented in
   SPEC §2.9/§6 — the gate driver needs a `timeout=` on a box like this one).
10. **HOST-ONLY (4 legs, exact commands below):** the live GitHub release + `init`'s own download; the
    4B/GPU serve gate; SPEC §5 (a) and (b) as written.
11. **Hygiene.** No GitHub request ever left this box: the whole update story was driven through a local
    HTTPS proxy that stands in for `api.github.com`/`github.com` under a local CA (`SSL_CERT_FILE`), with
    the *real* `runtime.lock`, the real retag rule and real TLS verification — `/etc/hosts` was **not**
    modified (the sandbox refused it; the proxy route needs no system change). No push. The only repo
    writes are this directory; the card's data home is `/workspace/e2e-t_559ed8c8/`.
12. **VERDICT: SHIP** — no BLOCKING finding.

## How the two stories were driven (fixture design)

* **Data home** (`/workspace/e2e-t_559ed8c8/home`): a *real* llama.cpp b11026 tree (the copy the repo's
  own `docs/evidence/t07b5/` kept) installed as `runtime/b11026-linux-x64-cpu` with a `typed_gguf.runtime/v1`
  record, plus the real `Qwen3.5-0.8B-UD-Q4_K_XL.gguf` (arch `qwen35`, `Q4_K_M`, 558 772 480 B) registered
  as `qwen3.5-0.8b`. `harness/setup_home.py`. Real CPU/Vulkan inference works in this container
  (lavapipe: `24 layer(s) offloaded, kv_type=f16`).
* **Fake release, real client path** (`harness/fixture_proxy.py` + `harness/make_fixtures.py`): an HTTPS
  `CONNECT` proxy that terminates TLS with a leaf cert for `api.github.com`/`github.com` signed by a local
  CA the client trusts via `SSL_CERT_FILE`. Nothing in the app is patched or monkeypatched: `runtime
  update` does its real `urllib` GETs (with TLS verification), the real `runtime.lock` retag rule
  (`llama-b11026-bin-ubuntu-x64.tar.gz` → `llama-b99999-…`) and the real download/extract/probe/switch
  path. The release list mirrors the API's order (a milestone `v0.5.0` release with only `nightly-tag.txt`,
  a newer `b100001` that is asset-bearing but not for this host, then `b99999`).
* **Scenarios** (re-read per request): `releases: ok|no_target_asset|http_500|wrong_digest`,
  `asset: ok|missing|short`, `throttle_ms` (the slow download the `kill -9` leg needs).
* **Serve side**: `venv/bin/typed-gguf serve --keep-alive 600|0`; the client is the repo's
  `tools/host_gate_serve_client.py` (unchanged, its own `verify()`) — and `harness/sdk_client_long.py`,
  which imports that same tool's `STATE`/`build_questions()`/`verify()` and only passes an explicit
  `TypeSafeClient(timeout=600)` (see P4).

## Story 1 — the receipts in detail

```
### command: python3 -m venv venv && venv/bin/pip install <wheel>
Successfully installed typed-gguf-0.2.3
### command: venv/bin/python src/import_probe.py
typed_gguf.__file__: /workspace/e2e-t_559ed8c8/venv/lib/python3.11/site-packages/typed_gguf/__init__.py
imports from the repo checkout: False       EXIT=0
### command: venv/bin/typed-gguf version     -> typed-gguf 0.2.3                       EXIT=0
### command: venv/bin/typed-gguf serve --help
usage: typed-gguf serve --host IP --port N --format native|typesafe --keep-alive <dur|0>   EXIT=0
```
`logs/01-clean-install.log` (full transcript, incl. the wheel's sha256 and `pip show`).

Cold/warm through the pinned SDK (`logs/serve_long_story.log`, server log lines verbatim):

```
GET /health 200 served_by=- 0.1ms                -> {"model": null, "warm": false}
POST /v1/systemone 200 served_by=host 6761.5ms   (cold; SDK client 6.76 s)
POST /v1/systemone 200 served_by=host 2282.4ms   (warm; SDK client 2.28 s)
   ok: 3 typed answers, keys and ranges conform (the gate's own verify())
PASS: both calls were answered through the official SDK, typed.
{"model": "qwen3.5-0.8b", "answers": {"renew_style": {"type":"choice","choice":"exec_sponsor_plan",
 "confidence":0.284188,"probabilities":{...}}, "risk": {"type":"score","score":2.30962,
 "legend":{"0":"healthy","1":"watch","2":"at risk","3":"lost cause"},"probabilities":{...}},
 "escalate": {"type":"noul","noul":0.610267}}, "usage": {"input_tokens":481,"output_tokens":16}}
```
Teardown (`logs/serve_long_story.log`): `kill -TERM` → `serve exit=143`, port closed; `keep stop` →
`stopped the warm host (pid 4135) and cleaned up its record`; ledger then holds **only** the host's
`<key>.log`; `keep status` → `no warm host`; no process matches `keep_host|serve --host`.

**A caveat that is not a bug, but worth knowing** (also in P4): on the *first* run of the day the same
mixed body took 32.1/28.6/28.5 s server-side (cold Vulkan shader cache) and `tools/host_gate_serve_client.py`
gave up — `TypeSafeAPITimeoutError: Request timed out (timeout=10.0)`, three attempts, `logs/serve_story.log`.
SPEC §2.9 already documents this ("a CPU box or a cold shader cache can exceed 10 s", §6 **[recon]**) and
names the remedy (the client raises its timeout), which is exactly what `harness/sdk_client_long.py` does.

## Story 2 — the receipts in detail

`logs/update_check_leg.log`:

```
### command: typed-gguf runtime update --check
current b11026 (build 11026, linux-x64-cpu) at .../runtime/b11026-linux-x64-cpu
target  b99999 (build 99999, linux-x64-cpu) 18.41 MB -> .../runtime/b99999-linux-x64-cpu
nothing changed (--check prints the plan only)                       EXIT=0
runtime.json  718022423a0d6ce0ac45d8e5b442cb62bb0346adfe8862491795149cb5e5bfa1  (before AND after)
downloads/ empty; runtime/ holds only b11026-linux-x64-cpu (no .pending-*)
```
`logs/update.json` (verbatim, abridged): `updated:true`, `from {tag b11026, build 11026, variant linux-x64-cpu}`,
`to {tag b99999, dir …/runtime/b99999-linux-x64-cpu}`, `asset {name llama-b99999-bin-ubuntu-x64.tar.gz,
size 18407198, sha256 5bcec9a2…}`, `probe {build 11026, symbols_ok true, symbols_probed true,
missing_symbols [], backends [cpu, rpc], expect_backend cpu}`, `previous {dir …b11026…, tag b11026}`,
`host_stopped {stopped false, pid null, reason "no host"}`, `home …`. Record after: `update_from`,
`previous`, `updated_at`, `tools` retargeted into the new dir; doctor then reports
`runtime.present …/b99999-linux-x64-cpu`, `runtime.build b11026` (the bundle's own probe), `runtime.sha_recorded ok`,
`model.arch qwen35 supported`; `version` prints the new dir; `--check` → `already at b99999 … nothing changed`
(EXIT 0). Rollback: `rolled back to b11026 …; b99998 is kept on disk — `typed-gguf runtime update` returns to it`,
record `rolled_back_from {tag b99999}`, `previous: null`, doctor green again.

`logs/kill_leg.log` (throttle 100 ms per 64 KiB):

```
### command: typed-gguf runtime update --tag b99998 --json   (throttled, then SIGKILL)
partial download: home/downloads/llama-b99998-bin-ubuntu-x64.tar.gz.part 3145728 bytes
cli exit after kill -9: 137 (137 = SIGKILL)
runtime.json 7ac69aa557fe6146f3b922b623f5271cd6dd43e123def61b69e2e25527c3bc27 (before AND after)
RECORD BYTE-IDENTICAL: yes
runtime/  b11026-linux-x64-cpu  b99999-linux-x64-cpu        <- no b99998, no .pending-*
downloads/ llama-b99998-…tar.gz.part (the resumable partial stays, by design)
typed-gguf version -> runtime b11026 (linux-x64-cpu) at …/b11026-linux-x64-cpu
### command (retry, unthrottled): typed-gguf runtime update --tag b99998 --json   EXIT=0
```

## Story 3 — the negative paths (each row is a quoted receipt)

Serve (`logs/negatives_serve.log`):

| leg | request | result |
|---|---|---|
| N1 | gate client `--expect-sdk 9.9.9` | `REFUSING TO RUN: this gate measures typesafe-sdk 9.9.9; the venv has 0.7.1…` EXIT **3** |
| N2 | `+ "extra_key": 1` | **422** `{"type":"extra_forbidden","loc":["body","extra_key"],…}` |
| N3 | body `not json at all` | **422** `json_invalid`, `loc ["body"]` |
| N3b | no `model` | **422** `missing`, `loc ["body","model"]`, `Field required` |
| N3c | `"model": "no-such-alias"` | **422** `value_error` … `E_MODEL_NOT_FOUND: … known aliases: qwen3.5-0.8b` |
| N3d | score with one level | **422** `value_error` … `E_SCORE_LEVELS … 2..10 levels` |
| N3e | `GET /v1/nope` | **404** `{"detail": "Not Found"}` |
| N4 | `Content-Length: 2097152`, **no body sent** | **413** in **0.000 s**; `/v1/systemone` → `too_large`, `/v1/decide` → `E_BODY_TOO_LARGE` |
| N5 | half a request, then silence | connection closed after **30.0 s** |
| N6 | sentinel `Authorization: Bearer E2E-SENTINEL-AUTH-efe9` | **0** occurrences in the server log |

Update/rollback (`logs/negatives_update.log`, `logs/negatives_update_b.log`) — every leg printed
`runtime.json byte-identical: YES`:

| leg | command | result (EXIT) |
|---|---|---|
| U1 | `TYPED_GGUF_RUNTIME_DIR=/opt/managed-runtime … runtime update --check` | `E_UPDATE_UNAVAILABLE: … managed outside typed-gguf …` (2) |
| U2 | `TYPED_GGUF_HOME=<empty> … runtime update` | `E_RUNTIME_MISSING: no llama.cpp runtime installed under …` (3) |
| U3 | `HTTPS_PROXY=http://127.0.0.1:1 … runtime update --check` | `E_DOWNLOAD_FAILED: cannot reach https://api.github.com/… (URLError: [Errno 111] Connection refused); nothing was changed` (3) |
| U4 | release list with no matching asset (`--tag b99997`) | `E_UPDATE_UNAVAILABLE: tag b99997 carries no bundle under this host's pinned asset name (llama-b11026-bin-ubuntu-x64.tar.gz); … never guesses a name` (2) |
| U5 | asset 404 | `E_DOWNLOAD_FAILED: … (HuggingFace returned HTTP 404)` (3) — see P2 |
| U6 | truncated body (6 135 732 of 18 407 198 B) | `E_DOWNLOAD_FAILED: … expected 18407198 bytes, got 6135732 (the download is incomplete; re-run to resume)` (3) |
| U7 | advertised digest ≠ bytes | `E_SHA256_MISMATCH: … expected ffff…, got 520a9021…; the partial file was kept at …part` (3) |
| U8 | `runtime update --backend cuda` | `E_RUNTIME_SYMBOLS: pre-flight: the pinned linux-x64-cuda-12.8 bundle links libcudart.so.12, libcublas.so.12, libcuda.so.1, which this host cannot load …; skipped the 168.81 MB download …; nothing was changed` (3) — fixture log: release list only, **no asset GET** |
| U9 | release API 500 | `E_DOWNLOAD_FAILED: cannot reach https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=20 (HTTPError: HTTP Error 500: Err); nothing was changed` (3) |
| U10 | `runtime rollback` with no `previous` | `E_UPDATE_UNAVAILABLE: … records no previous bundle to roll back to …; nothing changed` (2) |

After the whole pass the home still holds the same active runtime (`version` → b11026), the three
bundles and no `.pending-*`; the only residue is the resumable `.part` of U7 (by design — the code says so
in its own message).

## PROPOSALS (non-blocking — out of this card's scope; repro included)

**P1 — `runtime rollback` rebuilds a 10-key record, so the installed-probe facts are lost.**
After `rollback`, the record keys are `['build','dir','installed_at','libllama_sha256','previous',
'rolled_back_at','rolled_back_from','schema','tag','variant']` — the 30 keys the update wrote (incl.
`backends`, `tools`, `probe_warnings`, `backend_requested`, `asset*`, `symbols_ok`, `rung`, `url`) are
gone, and `typed-gguf version` then prints `backends unknown` (compare the update leg's
`backends cpu, rpc`). SPEC §2.8 only promises "all existing record keys stay" for *update*; nothing says
rollback must keep them, and `doctor` tolerates the gap — so this is information loss, not breakage.
Repro: `logs/update_story.log` step 9 vs step 3; `logs/negatives_update.log` head.

**P2 — the GitHub download inside `runtime update` speaks HuggingFace.** `E_DOWNLOAD_FAILED: … :
HuggingFace returned HTTP 404` (U5) and the asset request carries
`User-Agent: typed-gguf/0.1 (+https://github.com/Rybens92/typed-gguf)` while the release API sends
`typed-gguf/0.2.3` (`src/typed_gguf/registry/hf.py:38` hardcodes `0.1`). Same for the U6/U7 wording
(`_translate`). Repro: `grep -n 'HuggingFace returned' logs/negatives_update.log`; the fixture log lines
in the same file show both User-Agents.

**P3 — `keep stop` leaves the host's `<key>.log` in the ledger dir.** A-E5-5 says "no host/socket/ledger
entry is left behind"; the record (`host.json`), the spec and the socket do go, the log (762 B → 3 090 B)
stays. Almost certainly intentional (diagnostics), but it is the one thing a strict reading of A-E5-5
would trip over. Repro: `ls home/keep/` after `keep stop` (both serve logs).

**P4 — the SDK's 10 s default vs a cold shader cache.** Documented in SPEC §2.9/§6, so not a product bug;
what the card should carry forward is that `tools/host_gate_serve_client.py` cannot pass on a box whose
first decision exceeds 10 s (exactly what happened here, `logs/serve_story.log`), while the same body
passes once warm (`logs/serve_long_story.log`). Suggestion: give the gate client an explicit
`TypeSafeClient(timeout=…)`.

## HOST-ONLY (what this container could not run, and the command for it)

1. **The live GitHub release** (no GitHub request was made here by design): SPEC §5 block (b) —
   `uv run typed-gguf doctor --json | tee /tmp/upd-before.json`, `uv run typed-gguf runtime update --check`,
   `uv run typed-gguf runtime update`, `uv run typed-gguf doctor --json | tee /tmp/upd-after.json`,
   `uv run typed-gguf runtime rollback && uv run typed-gguf doctor --json | tee /tmp/upd-back.json`.
2. **`init`'s own download legs** (the 30.9 MB vulkan / 168.8 MB cuda assets): `uv run typed-gguf init`
   on a box whose `runtime.json` records what was installed — the fixture here installed the runtime by
   hand, so `init`'s download was never exercised.
3. **The 4B + GPU serve gate** (SPEC §5 block (a)): `uv run typed-gguf serve --host 127.0.0.1 --port 8088 &`
   then `TYPESAFE_API_KEY=local TYPESAFE_BASE_URL=http://127.0.0.1:8088 /tmp/ts-venv/bin/python
   tools/host_gate_serve.py`; this card used the 0.8B on lavapipe (no GPU in the container).
4. **Two-concurrent-request serialization on the real host**: only observed indirectly (the SDK's three
   retried `systemone` calls were answered one at a time, `200 served_by=host`, in arrival order —
   `logs/serve_story.log`); the pinned offline gate covers it on the fake host.

## Re-run recipe

```
cd /workspace && rm -rf e2e-t_559ed8c8 && mkdir -p e2e-t_559ed8c8      # the scratch dir this card used
# wheel:  cd /workspace/ggufone && env -u PYTHONPATH uv build
# venvs:  python3 -m venv e2e-t_559ed8c8/venv  && venv/bin/pip install dist/typed_gguf-0.2.3-py3-none-any.whl
#         python3 -m venv e2e-t_559ed8c8/sdkvenv && sdkvenv/bin/pip install 'typesafe-sdk==0.7.1'
# then, in order (all scripts are in harness/):
bash src/install_receipt.sh          # story 1, leg 0
env -u PYTHONPATH venv/bin/python src/setup_home.py /workspace/e2e-t_559ed8c8/home    # the fixture home
bash src/make_certs.sh               # local CA for api.github.com/github.com
env -u PYTHONPATH venv/bin/python src/make_fixtures.py                                 # releases + bundles
venv/bin/python src/fixture_proxy.py &                                                 # the GitHub stand-in
bash src/serve_long_story.sh         # story 1 (real model, real SDK)
bash src/same_engine.sh              # A-E5-4
bash src/update_check_leg.sh && bash src/update_story.sh && bash src/kill_leg.sh        # story 2
bash src/negatives_serve.sh && bash src/negatives_update.sh && bash src/negatives_update_b.sh  # story 3
```
The update commands need `TYPED_GGUF_HOME=…/home`, `SSL_CERT_FILE=…/ca/ca.pem` and
`HTTPS_PROXY=http://127.0.0.1:8443` in the environment (every script sets them itself). With the proxy
down, `runtime update` talks to the real GitHub — which this card never wanted.

## Scratch state left behind (not in the repo)

`/workspace/e2e-t_559ed8c8/` — `venv/`, `sdkvenv/`, `home/` (the fixture data home),
`home-clean/`, `fixtures/`, `ca/` (throwaway CA + leaf, 3-day validity), `logs/` (copied here),
`src/` (the harness, copied here). `docs/evidence/t07b5/home/runtime/…` in the repo is the source of the
runtime tree the fixture hardlinked — untouched. One incidental host-side effect: running
`env -u PYTHONPATH uv run --no-sync ruff` recreated the repo's (untracked, gitignored) `.venv`;
`git status --porcelain` is 0 lines and no tracked file changed.
