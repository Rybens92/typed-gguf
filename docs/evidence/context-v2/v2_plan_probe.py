#!/usr/bin/env python3
"""SPEC-context-v2 §3/§8 arithmetic, recomputed for the box (Appendix A's `v2_plan_probe.py`).

Run from the repo root:  uv run python v2_plan_probe.py [MODEL]

Prints today's formula vs the measured-formula KV table, the largest n_ctx per rung under the live
budget, and the v2 answers the policy returns. It is a *report*, not a test: the pins live in
`tests/test_context_v2.py` (AC-6) and `ac6_check.py`.
"""
from __future__ import annotations

import dataclasses
import sys

from typed_gguf.runtime import fit

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/var/home/rybens/.hermes/models/Spark-X2.5-4B-Q8_0.gguf"
FIT_TARGET_MB = fit.DEFAULT_FIT_TARGET_MB

model = fit.ModelFacts.read(MODEL)
host = fit.host_facts()
budget = fit.fit_budget(host, FIT_TARGET_MB)
print(f"model: {{'path': {model.path!r}, 'arch': {model.arch!r}, 'n_ctx_train': "
      f"{model.n_ctx_train}, 'weights_MiB': {model.weights_bytes / fit.MIB:.1f}, "
      f"'layers': {model.n_layer}, 'kv_head': {model.n_kv_head}, "
      f"'key': {model.key_len}, 'value': {model.value_len}}}")
print(f"swa: {{'window': {model.sliding_window}, 'n_swa_layers': {model.n_swa_layers}, "
      f"'n_global_layers': {model.n_global_layers}, 'has_swa': {model.has_swa}}}")
print(f"host: {{'vram_free_MiB': {host.vram_free_bytes / fit.MIB:.0f}, "
      f"'budget_MiB': {budget / fit.MIB:.0f}}}")

plain = dataclasses.replace(model, sliding_window=0, n_swa_layers=0, n_global_layers=0)
print("\nKV MiB at n_ctx (the pre-v2 all-layer formula vs the SWA-aware one):")
print("   n_ctx  " + "  ".join(f"{kv}_today {kv}_real" for kv in fit.KV_DOWNGRADE_ORDER))
for n_ctx in (4096, 32768, 65536, 131072):
    today = [fit.kv_bytes(plain, n_ctx, kv) / fit.MIB for kv in fit.KV_DOWNGRADE_ORDER]
    real = [fit.kv_bytes(model, n_ctx, kv) / fit.MIB for kv in fit.KV_DOWNGRADE_ORDER]
    print(f"  {n_ctx:6d}  " + "  ".join(f"{a:10.1f} {b:10.1f}" for a, b in zip(today, real)))

print("\nlargest n_ctx this box holds per KV (policy target 32768, window cap applied):")
for kv in fit.KV_DOWNGRADE_ORDER:
    raw = fit.max_fit_n_ctx(model, kv, budget)
    cap = min(raw, model.n_ctx_train) if model.n_ctx_train else raw
    total = model.weights_bytes + fit.kv_bytes(model, cap, kv) + fit.OVERHEAD_BYTES
    print(f"  {kv:>5}: raw {raw:7d}  window-capped {cap:7d}  "
          f"(total {total / fit.MIB:.0f} MiB of {budget / fit.MIB:.0f} MiB)")

print("\nv2 semantics (executed by this checkout):")
for kwargs, label in (({}, "no flags"), ({"n_ctx": 32768}, "--n-ctx 32768"),
                      ({"n_ctx": 65536}, "--n-ctx 65536"), ({"n_ctx": 8192}, "--n-ctx 8192")):
    plan = fit.estimate_plan(model, host, **kwargs)
    print(f"  n_ctx {plan.n_ctx:7d}  kv_type {plan.kv_type:>5}  "
          f"standard_n_ctx {plan.standard_n_ctx}  ctx_limit {plan.ctx_limit!r}  "
          f"warnings {list(plan.warnings)}   <- {label}")
print(f"\nnote: a report, not a receipt — the live `fit --json` runs are in docs/evidence/context-v2/. "
      f"Free device memory moves between runs ({host.vram_free_bytes / fit.MIB:.0f} MiB here).")
