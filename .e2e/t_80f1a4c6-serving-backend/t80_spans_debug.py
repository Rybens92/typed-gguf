"""Debug: are the spans and meta keys in sync? (card t_80f1a4c6 helper)"""
import json
import pathlib

base = pathlib.Path("/work/t80serve/mutants/src/ggufone/engine")
for name in ("session", "decide"):
    spans = json.loads((base / f"{name}.py.spans").read_text())["spans"]
    codes = json.loads((base / f"{name}.py.meta").read_text())["exit_code_by_key"]
    span_keys = set(spans)
    meta_keys = set(codes)
    print(f"{name}: spans={len(span_keys)} meta={len(meta_keys)} "
          f"spans-not-in-meta={len(span_keys - meta_keys)} meta-not-in-spans={len(meta_keys - span_keys)}")
    missing = sorted(span_keys - meta_keys)[:3]
    for key in missing:
        print("   span key missing from meta:", key, spans[key])
    none_in_spans = [k for k in span_keys if codes.get(k) is None]
    print(f"   spans keys with None verdict: {len(none_in_spans)}")
