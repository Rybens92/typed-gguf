"""ctypes structs + function bindings for the pinned llama.cpp ABI

Milestone: E1a.

Transplant docs/evidence/poc-ctypes-20260917.py verbatim: the struct field order is
verified against b11026. Mandatory call order (SPEC 2.2, PoC pitfalls):
    CDLL(libggml) -> ggml_backend_load_all_from_path(rt) -> llama_backend_init()
    -> load model -> init context with kv_unified=True.
"""
