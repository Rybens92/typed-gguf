# Receipt — review `t_c7d5e33d` (the serve wave): the CI finding #2 reproductions

Verdict and the full analysis live in `REVIEW.md` (root of this checkout, section "MAJOR-1"/"MAJOR-2").
This directory carries the raw material for the py3.11 CI finding so the fix card (`t_dab7a690`) can
re-check it after `/tmp` is gone.

## What is here

| file | what it is |
|---|---|
| `par-2-c.log`, `lw6-3.log`, `lw9-3-c.log`, `lw9-4-a.log` | the four red runs of the committed sub-gate (`TYPED_GGUF_TEST_BLOCK_NET=1 pytest -q tests/test_keep.py tests/test_keep_host.py tests/test_keep_client.py tests/test_keep_cli.py`) at the reviewed head — each reads `ERROR at setup of test_a_host_that_dies_mid_decision_falls_back_inline_once` + `ResourceWarning: subprocess <pid> is still running` + `108 passed, 1 error`, i.e. the CI's exact shape (pids 7425, 12610, 17569, 17906) |
| `probe_fixture_wait2.py` | the mechanism proof: a synthetic replica of the `make_client` teardown shape. `kill()` without `wait()` + the Popen becoming unreachable during a later test → `PytestUnraisableExceptionWarning: Exception ignored in: Popen.__del__`; with `wait()` → green |
| `leakwatch6.py` | the instrumentation used to see which tests leave children to the fixture's blanket kill (report lines `CHILDREN | DEL | SPAWN`) |
| `probe_runtime_cli.sh` | the §2.8 refusal / `--check` probes (empty home, rung-1, rollback-without-previous, dead-proxy `--check`) |
| `probe-rt-live.out` | the live default `runtime update` on a box whose detection says cuda but which lacks `libcudart.so.12`: 169.49 MB downloaded, then `E_RUNTIME_SYMBOLS`, exit 3, `runtime.json` byte-identical (the pre-flight gap) |

## How to re-run

```bash
cd "$(git rev-parse --show-toplevel)"
# 4-way concurrency is what trips it; ~4/68 concurrent vs 0/46 sequential on the review box
for round in 1 2 3 4; do for slot in a b c d; do
  ( TYPED_GGUF_TEST_BLOCK_NET=1 .venv/bin/python -m pytest -q \
      tests/test_keep.py tests/test_keep_host.py tests/test_keep_client.py tests/test_keep_cli.py \
      > /tmp/m1-$round-$slot.log 2>&1 ) & done; wait; done
grep -l "108 passed, 1 error" /tmp/m1-*.log
# mechanism, no repo needed:
python -m pytest -q docs/evidence/t_c7d5e33d/probe_fixture_wait2.py   # from a checkout root
```
