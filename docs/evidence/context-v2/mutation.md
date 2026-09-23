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

## Re-run on the fix head (card `t_9542037f`) — the coverage gap above, closed

Same pair, same driver, same flags; the tree is now the v2 **final head** `31bb3f4`. `mutants/` was
moved aside before this run (a first invocation had kept mutmut's cached verdicts for the older
tree — its `mutmut-stats.json` still named `git_commit 5e3f834` — and was discarded), so all
**2267** mutants were executed against the head; this run's own `mutmut-stats.json` records
`git_commit 31bb3f4b2f7f50baef4d9aff663a615311b445ae`. Baseline:
`pytest -q tests/test_context_v2.py tests/test_fit.py` → `72 passed`; the sweep's clean-tests phase
reported no failure line, and the tally's leading `2267/2267` is mutmut's `not_checked 0`.

### Result (the run's own tally line)

```
2267/2267  🎉 1335 🫥 127  ⏰ 0  🤔 0  🙁 805  🔇 0  🧙 0
```

```
killed 1335  no-tests 127  timeout 0  survived 805
killed / scored = 1335 / (1335 + 805) = 62.4 %
```

**Delta vs the `11505f4` sweep:** mutants `2186 → 2267` (**+81**), killed `1244 → 1335` (**+91**),
survived `815 → 805` (**−10**), **60.4 % → 62.4 % (+2.0 pts)**. The prior fit numbers quoted above
stand (`1884` @ 62.1 % on `t_e29734e6`, `1639` @ 69.3 % on E1c).

**Coverage-gap closure.** The timing note above is why the `11505f4` tally was a snapshot: its
mutants were generated at 13:31 against that tree, leaving the two late `plan_from_binary` lines
(`4b7867f`, the §5.4 warning on the *table* path) and the cache-policy fix (`45abd7f`:
`is_policy_call`, `cache_entry_is_stale`, the `cacheable` paths) unscored. Measured against the
68 added/changed lines, this sweep scored **84 mutants — 51 killed, 33 survived**; no line of the
module's changed surface is unscored any more.

### Survivor triage — the cache policy (the newly changed lines)

The 33 survivors on the changed lines fall into three clusters:

1. **Equivalent candidates (32).** Four are `kv_type` *default* mutants on the two new `def` lines
   (`"auto"` → `"XXautoXX"` / `"AUTO"`, `fit.py:1084` and `:1102`): both helpers are called from
   `plan_for_model` only, and both of its call sites pass every knob explicitly, so the mutated
   defaults are dead on every reached path. The other 28 drop or `None` one keyword of the
   `is_policy_call` / `cache_entry_is_stale` / `estimate_plan` calls (`:1120-1121`, `:1152-1153`,
   `:1157-1158`); none of them can change an answer on the only path that reaches them, because a
   *policy* call is by definition `n_ctx None`, `kv_type auto/None`, `min_ctx None`,
   `budget_bytes None`, `n_seq_max DEFAULT_N_SEQ_MAX`, `fit_target_mb DEFAULT_FIT_TARGET_MB` —
   exactly what the callee's own signature supplies when the keyword is dropped. (Coverage note,
   not a defect: a direct no-knob call to either helper would kill the four default mutants and
   the `*_is_stale` keyword mutants; production never makes that call.)
2. **Reporting/wording (0).** The historical cluster is absent on this surface: all three
   `W_CTX_BELOW_STANDARD` literal mutants (`:1004`) are killed.
3. **Real semantic gap (1) — `fit.py:1001`.** `if not pinned and int(n_ctx) < policy_target(model):`
   → `<=` **survived**. Witness: a *non-pinned* table plan at exactly the standard —
   `plan_from_binary(model, host, table=..., n_ctx=32768, pinned=False)`, the shape a roomy box
   reaches through `plan_for_model`'s `n_ctx=preliminary.n_ctx` — carries no warning in the code
   and `W_CTX_BELOW_STANDARD` in the mutant, which also breaks the "both paths speak one policy"
   rule AC-9 exists for. Its gate
   (`test_ac9_the_binary_path_below_the_standard_carries_the_same_warning`) drives `4096`, so the
   *equality* boundary of the new line is unpinned. Reported, not fixed — this card is report-only.

For the record: §5.7's `return now.n_ctx > cached.n_ctx` (`:1122`) is **killed** in this sweep (the
`>` → `>=` mutant), so the "strictly greater" rule is scored by this selection; the hand-made
`+1`-style variant the review noticed is not a mutmut mutant and is not part of this tally.

Raw log (not committed): `/work/t_context_v2/logs/mutmut_fit_v2head_fresh.txt`; the discarded
cached-verdict invocation is `/work/t_context_v2/logs/mutmut_fit_v2head.txt`.
