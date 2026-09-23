# Evidence — context sizing v2 (card `t_ca1d4231`)

Live receipts for `docs/SPEC-context-v2.md` on this box (`RTX 3060 Ti 8192 MiB`, `b11026`
linux-x64-vulkan, model `Spark-X2.5-4B-Q8_0`, `--fit-target 1024`). Free device memory moves
between runs, so the plan's `n_ctx` does too (the spec says so in §0.4: 53 511 at 6 760 MiB).

| file | what it proves |
| --- | --- |
| `fit_before_default_4096.json` | the pre-v2 `fit --json` (no flags): 4 096 @ f16 — the starting point |
| `fit_v2_default_grown.json` | the v2 default: **n_ctx 49763, kv_type q8_0, standard_n_ctx 32768, ctx_limit 'grown', n_gpu_layers 36, source llama-fit-params, warnings ['W_KV_TYPE_DOWNGRADE']** |
| `fit_v2_default_shrunk.json` | `--fit-target 5200` (a budget that cannot hold the standard): 4 096 @ q4_0, `ctx_limit 'shrunk'`, `W_CTX_BELOW_STANDARD` — graceful shrink, live |
| `fit_v2_pin_32768.json` | `--n-ctx 32768` now really plans 32 768 (the live regression flipped) |
| `ask_before_4096.txt` | a 5 988-token request against the 4 096 cap -> `E_CTX_TOO_SMALL` (exit 3) |
| `ask_v2_6k_default.json` | the same request with no pins: answered, `engine.n_ctx` 50 688, `engine.fit.n_ctx` 50 620 — the load dropped q8_0 -> q4_0 at context init (`placement.kv_type` q4_0 vs `engine.fit.kv_type` q8_0; the JSON's top-level warnings are `[W_KV_TYPE_DOWNGRADE, W_FIT_DOWNGRADE]`) |
| `ask_v2_6k_state_tokens.txt` | the state's token ids (5 988 tokens) that make the request ~6 k |
| `kv_cli_32768_f16.txt` | a live `llama-cli` load at 32 768 f16: 1 152.00 MiB (9 layers) + SWA, 1 024 cells |
| `ac6_check.py` / `.out` | §3's formula recomputed standalone against the four pinned bytes (`OK` x4) |
| `v2_plan_probe.py` / `.out` | §3/§8's arithmetic on this box: today-vs-real KV table, max fit per rung |

Commands (verbatim) are the `# cmd:` first lines; the receipts are small on purpose — the logs
themselves stay out of git.

**One measured deviation from AC-16's exact equality:** the engine reports the *runtime's* context
(`llama_n_ctx`, `engine/session.py:519`), and llama.cpp pads the KV to a 256-cell block: asked
50 620, loaded 50 688 (+68 cells, the §8.6 "the runtime may allocate slightly more" risk, absorbed
by the fit-target margin) — the request was answered at that size.
