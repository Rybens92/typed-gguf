# t_eff926f9 — an out-of-tree install reads its own pins (`typed-gguf` v0.1.0 FIX)

Captured on the shared box, 2026-09-20. The card: `uvx --from …`, `uv tool install` and a plain
venv install all answered

```
E_RUNTIME_MISSING: /tmp/runtime.lock not found; run from the repository root or set TYPED_GGUF_LOCK
```

for every command. Root cause: `pins.default_lock_path()` walked up from the *package*, and an
installed package has no repository above it (the fresh-install acceptance that passed ran from the
clone root — the one cwd where the bug cannot show). Remedy: the wheel now carries the lock
(`<pkg>/data/runtime.lock`, copied in by the build) and the lookup order is
`$TYPED_GGUF_LOCK` → nearest above → packaged copy → `./runtime.lock`.

## The repro, before and after

### Before — a7cd2e7, wheel installed as a tool, run from an empty directory

```
$ typed-gguf version
error: E_RUNTIME_MISSING: /work/ffx/neutral3/runtime.lock not found; run from the repository root or set TYPED_GGUF_LOCK
exit=3

$ typed-gguf init --dry-run
error: E_RUNTIME_MISSING: /work/ffx/neutral3/runtime.lock not found; run from the repository root or set TYPED_GGUF_LOCK
exit=3

$ typed-gguf doctor --json
error: E_RUNTIME_MISSING: /work/ffx/neutral3/runtime.lock not found; run from the repository root or set TYPED_GGUF_LOCK
exit=3
```

### After — the same three commands, the same neutral cwd

```
$ uvx --from <checkout> typed-gguf version
typed-gguf 0.1.0
  runtime  not installed (`typed-gguf init`)
  home     /work/ffx/uvx-home
exit=0

$ uvx --from <checkout> typed-gguf init --dry-run --json
{
  "dry_run": true, "rung": "prebuilt", "backend": "cuda",
  "variant": "linux-x64-cuda-12.8",
  "asset": "llama-b11026-bin-ubuntu-cuda-12.8-x64.tar.gz",
  "size": 168811114,
  "sha256": "5b2d30d7a5e448fbe0aceda360c8f9ed2949aa1734e94db078e6d0b722521e2b",
  …
}
exit=0        # the plan's size + sha256 are the pinned row the wheel carries

$ uvx --from <checkout> typed-gguf doctor --json
{
  "status": "failures", "exit_code": 1,
  "checks": [{"id": "runtime.present", "status": "fail",
              "detail": "no runtime installed under /work/ffx/uvx-home/runtime; run `typed-gguf init` (no compiler needed)"}, …],
  "runtime": {"pinned_tag": "b11026", …}
}
exit=1        # the documented code for "broken: nothing installed yet", not a lock failure
```

### …and a real install, out of tree (what the README one-liner now claims)

```
$ uvx --from <checkout> typed-gguf init --backend cpu        # empty cwd, no checkout above it
installed: True
variant: linux-x64-cpu
dir: /work/ffx/uvx-init-home/runtime/b11026-linux-x64-cpu
backend: cpu
working_backend: cpu
asset: llama-b11026-bin-ubuntu-x64.tar.gz
asset_sha256: 219cf1c726bae1da4289b96a6378314d5485c6bc74c43891a4203e30906afb06
asset_verified: True
libllama_sha256: 8d72d777d94069455c812fc49a7156ee0b7cc5d73f8c9451292bd08ae0070975
bytes_fetched: 16855810
build: 11026                  # == the pinned tag
symbols_ok: True              # the isolated probe child dlopened the bundle
rung: prebuilt
exit=0                        # 4.3 s, 16.86 MB
```

## The artifact

```
BEFORE (a7cd2e7) wheel:    49 members
AFTER  (t_eff926f9) wheel: 50 members  →  + typed_gguf/data/runtime.lock
```

## The gates

| gate | command | result |
|------|---------|--------|
| full suite | `uv run --extra dev pytest -q` | **1392 passed, 48 skipped**, exit 0, 62.8 s, 0 pid-pressure skips |
| the new wheel gate | `pytest -q tests/test_wheel_install.py` | 8 passed (6 artifact gates + 2 harness pins) |
| the new unit pins | `pytest -q tests/test_pins.py` | 43 passed |
| oracle | `python docs/verify_runtime_contract.py` | `failures: 0  skips: 0`, exit 0 |
| lint | `ruff check src tests tools docs .github` | clean |
| mutation (Tier M) | mutmut 3.8, `src/typed_gguf/runtime/pins.py` + `tests/test_pins.py` | 344/421 killed = **81.7%** (two rounds below) |

Run the suite with `HOME=/work/agent-home UV_CACHE_DIR=/work/.uv-cache` and **no** `TYPED_GGUF_HOME`
(or one whose home carries a bundle): the bench gates resolve runtimes from the data home, and a
`TYPED_GGUF_HOME` pointing at an empty home makes `tests/test_bench_prompt_parity.py` fail for a
reason that has nothing to do with this card (measured: same tree, one env var, red → green).

### The box this gate runs on

`tests/test_wheel_install.py` spawns `uv` and the installed CLI, and this box's pid cgroup fills
*while* a run is in flight (measured at the start of this card: 6 ERRORS in one suite run, `uv`
panicking with `OS can't spawn worker thread: Resource temporarily unavailable (os error 11)`).
Both of its fixtures now classify a `uv` failure: a full cgroup skips through the conftest's
pressure path (the `pid cgroup has no fork headroom` prefix plus the counter the session-finish hook
reads, so the run prints the banner and cannot exit 0), and only a real build error fails the gate.

```
$ TYPED_GGUF_TEST_PID_HEADROOM=250/256 pytest -q tests/test_wheel_install.py
2 passed, 6 skipped … PID PRESSURE: 6 fork gate(s) skipped        exit=1
$ pytest -q tests/test_wheel_install.py                            # roomy box
8 passed                                                          exit=0
```

The two harness pins spawn nothing (only the six artifact gates carry `needs_fork`), so the path
they guard stays measurable on the starved box that motivated them — and both were shown to be
load-bearing by hand probes on a clone (the classification returns `None` → the first fails; the
skip loses its prefix/counter → the second fails; baseline green in both cases).

`tests/test_wheel_install.py` builds the real wheel (`uv build --wheel --offline`), installs it with
`uv tool install --offline` into a temp tool env, and runs `version` / `init --dry-run` /
`doctor --json` from a *neutral* cwd that carries a **decoy** lock (tag `b00042`, every asset size
+1): a pass cannot come from the cwd fallback either. Two more nodes break the install on purpose
(delete the packaged copy; typo `$TYPED_GGUF_LOCK`) and assert the `E_RUNTIME_MISSING` text names
every path that was searched — the old text is asserted *absent*.

**The one-liner's own fetch** (`uvx --from git+https://github.com/Rybens92/typed-gguf …`, the URL the
README's clone line already names) could not be exercised here, and the reason is the remote, not the
code: the checkout has **no remote configured**, and GitHub answers the README's URL with a refusal.

```
$ git -c credential.helper= ls-remote https://github.com/Rybens92/typed-gguf
fatal: could not read Username for 'https://github.com': terminal prompts disabled     exit=128

$ GIT_CONFIG_GLOBAL=<a store helper carrying this sandbox's token> \
      uvx --from git+https://github.com/Rybens92/typed-gguf typed-gguf version
   Updating https://github.com/Rybens92/typed-gguf (HEAD)
error: Failed to resolve `--with` requirement
  Caused by: Git operation failed … exit status: 128
    remote: Repository not found.
    fatal: repository 'https://github.com/Rybens92/typed-gguf/' not found          exit=1
```

(The URL stays public in every log line — the credential came from a `store` file, never a URL or a
command line, and the receipt has no token in it.) So the receipts above install from a **local
path**, which is the same PEP 517 build + `uv tool install` the one-liner performs once a clone or a
fetch lands; what remains unmeasured *here* is uv's `git fetch` step, and it stays unmeasured until
the project has a published remote.

## The mutation sweep (Tier M)

`[tool.mutmut]` points at `src/typed_gguf/runtime/pins.py`, selection `tests/test_pins.py` (the pair
is documented in `pyproject.toml`, including why the artifact gate is deliberately out of it: a
`uv build` per mutant). Two rounds on the shared box, `--max-children 2`, same 421 mutants both
times:

| round | killed | survived | what changed |
|-------|--------|----------|--------------|
| 1 | 338/421 = 80.3% | 83 | the first run of the new pair |
| 2 | **344/421 = 81.7%** | 77 | six new pins for the survivors that sat on the new surface |

Round 2 differs from round 1 in exactly **six** verdicts, all `survived → killed`, all in the new
code: `x_packaged_lock_path__mutmut_3` (the `package_file` seam read `__file__`), the four
`x_located_lock` default-argument slips (`environ=`/`cwd=` dropped, so a caller's injected world
would have read the real environment or the real cwd), and `x_missing_lock_message__mutmut_13` (the
one-path-per-line join). Every one of the other 415 verdicts is identical between the two runs —
which is what makes the numbers trustworthy on a box where a starved fork can look like a kill.

The 77 survivors were classified by artifact triage (`.meta` + the inlined copy's `_orig` twins,
never `mutmut show`): the bulk are pre-existing `host_variant`/`accelerator_of`/`detect_backend`
rows (their gates live in other files) and casing/padding mutants of the `E_RUNTIME_MISSING` prose,
plus type/`or`-collapse mutants in the big `load_lock` parse body — none of them on this card's
surface. Not chased: Tier M's threshold is soft and none of them can make an out-of-tree install
read the wrong pin.

**Hand probes for the artifact gate** (its kill surface is the one the sweep's selection leaves
out). Two survivors' twins were spliced into a clone by hand and the *wheel* test run against them:

```
probe A  packaged_lock_path loses `data/`            → FAIL: assert 'b00042' == 'b11026'
probe B  the cwd rung promoted above the packaged copy → FAIL: assert 'b00042' == 'b11026'
baseline (unmutated tree, same node)                  → 1 passed
```

`b00042` is the decoy lock the gate plants in the neutral cwd, so both failures are the wheel
reading something other than the copy it ships — the packaged path and its precedence are
load-bearing in the artifact gate too, not only in the injection tests.

## Decisions

- **`force-include`, not a committed second copy** (requirement 2). `runtime.lock` is data at the
  repo root (SPEC 4) and the repo's rule is "data is never repo content", so a tracked
  `src/typed_gguf/data/runtime.lock` would be a second copy that can drift. Hatchling's
  `force-include` maps the root file into the wheel at build time: one source, nothing to keep in
  sync. Two fast gates pin the two spellings together (`tests/test_pins.py` reads `pyproject.toml`
  and `pins.PACKAGED_LOCK_RELATIVE`; `tests/test_wheel_install.py` compares the shipped bytes with
  the root file).
- **`$TYPED_GGUF_LOCK` stays authoritative** (it is used as-is, and a path that does not exist is an
  error rather than a quiet fallback to the packaged copy): a silent fallback would leave the
  caller's pin looking in use when it is not. The message now lists what was searched.
- **`./runtime.lock` is kept, last.** It is what the old text meant by "run from the repository
  root", so dropping it would change behaviour beyond this card; the packaged copy now wins over it,
  which is what makes a stray lock in the cwd harmless for an installed tool.
