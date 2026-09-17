# E1b raw evidence — card `t_34abf324` (ggufone engine core)

Frozen tree: `0858c43` (plus this directory's commit). Reproduce with:

```bash
export GGUFONE_RUNTIME_DIR=/var/home/rybens/.hermes/runtime/b11026-linux-x64-cpu
uv run pytest -q                                             # 503 passed, 21 skipped
uv run pytest -q --run-network tests/test_engine_fork.py tests/test_ctypes_binding.py tests/test_cli.py
python3 docs/verify_runtime_contract.py                      # failures: 0, skips: 0 with the runtime set
uv run python tools/e1b_perf_record.py --threads 8 --repeats 3
uv run --extra dev --with mutmut mutmut run "ggufone.engine.readout.*"
uv run --extra dev --with mutmut mutmut run "ggufone.schema.*"
```

| file | what it is |
|---|---|
| `logs/e1b_model_final2.log` | the `--run-network` run at the final head (62 passed): fork equivalence on qwen35 + spark2_5, warm prefill, determinism digest, state round-trip, waves, CLI e2e |
| `logs/mutation-readout.txt` | readout.py mutation report (174 mutants, 170 killed, 97.7%; 4 survivors) |
| `logs/mutation-schema.txt` | schema.py mutation report (775 mutants, 579 killed, 74.7%; survivor list) |
| `logs/mutation-report.py` | the script that reads `mutants/**/*.meta` and prints killed/survived |
| `logs/equivalent-fuzz.py` | differential fuzzing that classified the 4 readout survivors as equivalent |
| `logs/survivor-buckets.py` | buckets the schema survivors (message-text vs behaviour) |
| `logs/perf-record.txt` | raw stdout of the A-E1b-14 perf record (machine-readable copy: `docs/evidence/e1b_perf.json`) |

Report with the gate table and the receipts: `docs/evidence/e1b_t_34abf324_engine.md`.
