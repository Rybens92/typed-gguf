# Mutation sweep — context sizing v2 (card `t_ca1d4231`)

Tier **M**, one run, soft threshold, repo convention (`pyproject.toml` `[tool.mutmut]`, driven by
`tools/mutmut_driver.py`):

```
env -u PYTHONPATH TYPED_GGUF_TEST_BLOCK_NET=1 TYPED_GGUF_BENCH_RUNTIME_DIR=/tmp/typed-gguf-offline-bundle \
  timeout 1700 uv run --extra dev --with mutmut python tools/mutmut_driver.py run --max-children 2
```

- `source_paths = ["src/typed_gguf/runtime/fit.py"]` — the module this card rewrites;
- test selection `tests/test_context_v2.py` (the card's own gate file, 31 tests) +
  `tests/test_fit.py` (the module's fast contract file);
- mutmut 3.8, baseline clean, `pending 0`.

## Result (the run's own tally line)

```
2186/2186  killed 1244  no-tests 127  timeout 0  survived 815
killed / scored = 1244 / (1244 + 815) = 60.4 %
```

For comparison, the module's previous sweeps: `1884` mutants at **62.1 %** (card t_e29734e6, the
same fit pair) and `1639` at 69.3 % (E1c). The mutant count grew with the module (+302 lines in
this card: the policy, the SWA KV model, the two new fields).

*Soft threshold, stated honestly:* this is a repo-idiomatic fit sweep, not a full-sweep claim. The
survivor mass sits where the historical ones did — the module's pre-existing branches the fast
selection does not drive (`ModelFacts.read`, `host_facts`, `run_llama_fit_params`, `_gpu_layers`)
plus the new policy's own reporting paths (the `W_CTX_BELOW_STANDARD` / note wording, the
`ctx_limit` labels). The card's *behavioural* contract is covered by the 31 gate tests + AC-16 live;
the sweep is the Tier-M safety net, and its survivors are equivalent-mutant candidates for a
fuller sweep, not evidence of a wrong answer.

**Timing note (honest):** the sweep generated its mutants at 13:31 against the tree as of commit
`11505f4`. Two lines landed after it (`plan_from_binary`'s `W_CTX_BELOW_STANDARD`, found from the
live shrink receipt) — `tests/test_context_v2.py::test_ac9_the_binary_path_below_the_standard_carries_the_same_warning`
is their gate and is part of this sweep's selection, but their own mutants were not scored in this
run.

Raw log (not committed, `/tmp` is wiped): `/work/t_context_v2/logs/mutmut_fit.txt`.
