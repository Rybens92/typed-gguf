# Raw material — card t_a696ce02 (the pid-pressure gate)

The flake: this container's pid cgroup is small and shared (`pids.max = 256`), so a busy box
answers `fork()` with `EAGAIN`; the runtime-probe gates spawn real children and reported the
refusal as a probe finding (`['probe_failed', 'no_asset'] == ['loader_error', 'no_asset']`).

## The rig (`rig/`)

| file | what it does |
|---|---|
| `setup.sh` | private clone + container venv + `pytest-randomly` (not in the locked `dev` extra) |
| `seed_matrix.sh` | the flake on the real box: the three probe files under `--randomly-seed=N`, with `pids.current` before/after each run |
| `pidtrace.py` + `trace.sh` | per-test `pids.current` (is the suite leaking children, or is the pressure external?) |
| `inject.py` + `inject_run.sh` | the deterministic "starved box": `GGUFONE_TEST_DENY_SPAWN=N\|always` makes every spawn raise the kernel's `EAGAIN`. This is how the 25 `needs_fork` tests were *measured* rather than guessed |
| `matrix.sh` / `matrix2.sh` | the run matrix (real vs forced cgroup reading × real vs injected denial) |
| `red.sh` | RED on the landed parent: a detached worktree at `8474802` with only the new test files |
| `mark.py` | inserts `@pytest.mark.needs_fork` above the measured tests |
| `retarget.py` + `sweep.sh` | the Tier-M sweep of `pressure.py` + `isolated.py` (private clone only: the shared `pyproject.toml` is a sibling's WIP) |
| `gates.sh` | ruff + coverage over the probe/isolated gate files |

## The logs (`logs/`)

| file | what it is |
|---|---|
| `baseline_seed_matrix.log` | **pre-fix**, the three flaky files on the real box, seeds 1-6: 43 passed at `pids.current` 107-151, 2-6 mislabelled failures at 244-254 |
| `pidtrace.log` / `pidtrace_run.txt` | the per-test reading: flat plates across our own tests, jumps only when a sibling ticks the cap |
| `inject_always_3files.txt`, `inject_always_isolation_capability.txt` | the measured `needs_fork` set (25 tests) — every spawn denied |
| `matrix.txt` | rows A-E: injected denial + the real box (including the capped full-suite run and three green random-order runs) |
| `matrix2.txt` | rows A2-D2: the same rows with the cgroup reading forced (`GGUFONE_TEST_PID_HEADROOM`) |
| `red_parent.txt` | **RED**: the new gate file on the parent code (3 failed / 10 passed) + the filed flake under injection (8 failed) |
| `ruff.txt`, `coverage.txt` | static + coverage over the changed modules |
| `mutation.txt`, `sweep_driver.log` | the Tier-M mutmut sweep and its driver output |

## Reproduce the headline rows

```sh
export HOME=/work/agent-home UV_CACHE_DIR=/work/.uv-cache TMPDIR=/tmp
cd <clone> && uv venv .venv && uv pip install -p .venv pytest pytest-timeout ruff pytest-randomly

# the box: reading + the retry budget absorbing a transient denial (green)
GGUFONE_TEST_PID_HEADROOM=0/256 GGUFONE_TEST_DENY_SPAWN=3 \
    PYTHONPATH=<rig-dir> .venv/bin/python -m pytest -q -p no:randomly -p inject \
    tests/test_runtime_fallback.py tests/test_runtime_install.py tests/test_runtime_contract.py

# a starved box: named skips, one loud gate, exit status forced non-zero
GGUFONE_TEST_PID_HEADROOM=250/256 .venv/bin/python -m pytest -q
```
