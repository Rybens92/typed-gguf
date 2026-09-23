#!/usr/bin/env python3
"""AC-6's four pinned byte values, recomputed from the SPEC-context-v2 §3 formula.

Self-contained on purpose (stdlib only, no repo import): the check re-derives the numbers the
spec pins and compares them to the pins, so a wrong formula cannot pass by importing itself.

    kv_bytes(n_ctx, kv) = n_global * b * n_ctx + n_swa * b * (window + n_ubatch)
    b(kv)              = n_kv_head * (key_len + value_len) * bytes_per_element(kv)
"""
KV_BYTES_PER_ELEMENT = {"f16": 2.0, "q8_0": 34 / 32, "q4_0": 18 / 32}
N_LAYER, N_KV_HEAD, KEY_LEN, VALUE_LEN = 36, 4, 256, 256
WINDOW, N_SWA, N_GLOBAL, N_UBATCH = 512, 27, 9, 512


def kv_bytes(n_ctx: int, kv_type: str, *, n_ubatch: int = N_UBATCH) -> int:
    per_layer = round(N_KV_HEAD * (KEY_LEN + VALUE_LEN) * KV_BYTES_PER_ELEMENT[kv_type])
    return N_GLOBAL * per_layer * n_ctx + N_SWA * per_layer * (WINDOW + n_ubatch)


PINS = ((4096, "f16", 264241152), (32768, "f16", 1321205760),
        (32768, "q4_0", 371589120), (131072, "q4_0", 1390804992))

failed = 0
for n_ctx, kv_type, want in PINS:
    got = kv_bytes(n_ctx, kv_type)
    ok = got == want
    failed += not ok
    print(f"PIN{'' if ok else ' MISMATCH'} ({n_ctx}, {kv_type!r}) {got} want {want} "
          f"{'OK' if ok else 'FAIL'}")
print("all four OK" if not failed else f"{failed} pin(s) FAILED")
raise SystemExit(1 if failed else 0)
