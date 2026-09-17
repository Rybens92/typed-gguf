#!/usr/bin/env python3
"""PoC: Jev-owy trik bezpośrednio na libllama.so przez ctypes (bez kompilacji, bez llama-cpp-python).

Test na hybrydowym Qwen3.5-0.8B GGUF:
  1) prefill (state + schema) raz na seq 0
  2) fork stanu: llama_memory_seq_cp(0 -> 1), (0 -> 2)
  3) jeden batched decode: sufiks pola na seq 1 i sufiks pola na seq 2 (logits=1 na ostatnich tokenach)
  4) odczyt logitów per gałąź (llama_get_logits_ith), restricted softmax po kandydatach A-D
  5) TEST RÓWNOWAŻNOŚCI: to samo pole liczone sekwencyjnie (świeży kontekst) == wynik gałęzi
"""
import ctypes as C
import math, os, sys, time

LIBDIR = "/tmp/llamalib/llama-b11026"
MODEL  = "/home/rybens/.cache/llama.cpp/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"

ggml = C.CDLL(os.path.join(LIBDIR, "libggml.so"), mode=C.RTLD_GLOBAL)
lib = C.CDLL(os.path.join(LIBDIR, "libllama.so"), mode=C.RTLD_GLOBAL)

llama_token  = C.c_int32
llama_pos    = C.c_int32
llama_seq_id = C.c_int32

class llama_batch(C.Structure):
    _fields_ = [("n_tokens", C.c_int32),
                ("token",   C.POINTER(llama_token)),
                ("embd",    C.POINTER(C.c_float)),
                ("pos",     C.POINTER(llama_pos)),
                ("n_seq_id",C.POINTER(C.c_int32)),
                ("seq_id",  C.POINTER(C.POINTER(llama_seq_id))),
                ("logits",  C.POINTER(C.c_int8))]

class llama_model_params(C.Structure):
    _fields_ = [("devices", C.c_void_p),
                ("tensor_buft_overrides", C.c_void_p),
                ("n_gpu_layers", C.c_int32),
                ("split_mode", C.c_int32),
                ("load_mode", C.c_int32),
                ("lazy_mode", C.c_int32),
                ("main_gpu", C.c_int32),
                ("tensor_split", C.POINTER(C.c_float)),
                ("progress_callback", C.c_void_p),
                ("progress_callback_user_data", C.c_void_p),
                ("kv_overrides", C.c_void_p),
                ("vocab_only", C.c_bool),
                ("check_tensors", C.c_bool),
                ("use_extra_bufts", C.c_bool),
                ("no_host", C.c_bool),
                ("no_alloc", C.c_bool),
                ("load_mtp", C.c_bool)]

class llama_context_params(C.Structure):
    _fields_ = [("n_ctx", C.c_uint32),
                ("n_batch", C.c_uint32),
                ("n_ubatch", C.c_uint32),
                ("n_seq_max", C.c_uint32),
                ("n_rs_seq", C.c_uint32),
                ("n_outputs_max", C.c_uint32),
                ("n_outputs_max_per_seq", C.c_uint32),
                ("n_threads", C.c_int32),
                ("n_threads_batch", C.c_int32),
                ("ctx_type", C.c_int32),
                ("rope_scaling_type", C.c_int32),
                ("pooling_type", C.c_int32),
                ("attention_type", C.c_int32),
                ("flash_attn_type", C.c_int32),
                ("rope_freq_base", C.c_float),
                ("rope_freq_scale", C.c_float),
                ("yarn_ext_factor", C.c_float),
                ("yarn_attn_factor", C.c_float),
                ("yarn_beta_fast", C.c_float),
                ("yarn_beta_slow", C.c_float),
                ("yarn_orig_ctx", C.c_uint32),
                ("defrag_thold", C.c_float),
                ("cb_eval", C.c_void_p),
                ("cb_eval_user_data", C.c_void_p),
                ("type_k", C.c_int32),
                ("type_v", C.c_int32),
                ("abort_callback", C.c_void_p),
                ("abort_callback_data", C.c_void_p),
                ("embeddings", C.c_bool),
                ("offload_kqv", C.c_bool),
                ("no_perf", C.c_bool),
                ("op_offload", C.c_bool),
                ("swa_full", C.c_bool),
                ("kv_unified", C.c_bool),
                ("samplers", C.c_void_p),
                ("n_samplers", C.c_size_t),
                ("ctx_other", C.c_void_p)]

# ---------- bindings ----------
lib.llama_backend_init.argtypes = [];             lib.llama_backend_init.restype = None
lib.llama_model_default_params.restype = llama_model_params
lib.llama_context_default_params.restype = llama_context_params
lib.llama_model_load_from_file.argtypes = [C.c_char_p, llama_model_params]
lib.llama_model_load_from_file.restype = C.c_void_p
lib.llama_init_from_model.argtypes = [C.c_void_p, llama_context_params]
lib.llama_init_from_model.restype = C.c_void_p
lib.llama_model_get_vocab.argtypes = [C.c_void_p]; lib.llama_model_get_vocab.restype = C.c_void_p
lib.llama_vocab_n_tokens.argtypes = [C.c_void_p];  lib.llama_vocab_n_tokens.restype = C.c_int32
lib.llama_model_n_layer.argtypes = [C.c_void_p];   lib.llama_model_n_layer.restype = C.c_int32
lib.llama_model_meta_val_str.argtypes = [C.c_void_p, C.c_char_p, C.c_char_p, C.c_size_t]
lib.llama_model_meta_val_str.restype = C.c_int32
lib.llama_tokenize.argtypes = [C.c_void_p, C.c_char_p, C.c_int32,
                               C.POINTER(llama_token), C.c_int32, C.c_bool, C.c_bool]
lib.llama_tokenize.restype = C.c_int32
lib.llama_get_memory.argtypes = [C.c_void_p];      lib.llama_get_memory.restype = C.c_void_p
lib.llama_memory_seq_cp.argtypes = [C.c_void_p, llama_seq_id, llama_seq_id, llama_pos, llama_pos]
lib.llama_memory_seq_cp.restype = None
lib.llama_decode.argtypes = [C.c_void_p, llama_batch]; lib.llama_decode.restype = C.c_int32
lib.llama_get_logits_ith.argtypes = [C.c_void_p, C.c_int32]
lib.llama_get_logits_ith.restype = C.POINTER(C.c_float)
lib.llama_free.argtypes = [C.c_void_p]; lib.llama_free.restype = None
lib.llama_model_free.argtypes = [C.c_void_p]; lib.llama_model_free.restype = None

def check(rc, what):
    if rc != 0:
        print(f"!! {what} zwrocilo {rc}"); sys.exit(1)

ggml.ggml_backend_load_all_from_path.argtypes=[C.c_char_p]
ggml.ggml_backend_load_all_from_path(LIBDIR.encode())
print("[0] backendy ggml zaladowane")
lib.llama_backend_init()
mp = lib.llama_model_default_params()
t0 = time.time()
model = lib.llama_model_load_from_file(MODEL.encode(), mp)
if not model: sys.exit("!! model load fail")
print(f"[1] model zaladowany w {time.time()-t0:.2f}s  (cpu, n_gpu_layers={mp.n_gpu_layers})")

vocab = lib.llama_model_get_vocab(model)
n_vocab = lib.llama_vocab_n_tokens(vocab)
n_layer = lib.llama_model_n_layer(model)
arch = C.create_string_buffer(64)
lib.llama_model_meta_val_str(model, b"general.architecture", arch, 64)
print(f"    arch={arch.value.decode()}  layers={n_layer}  vocab={n_vocab}")

def tokenize(text: str, add_special=False):
    b = text.encode()
    arr = (llama_token * (len(b) + 64))()
    n = lib.llama_tokenize(vocab, b, len(b), arr, len(arr), add_special, True)
    if n < 0: raise RuntimeError("tokenize fail")
    return list(arr[:n])

def decode_tokens(tokens, pos0, seq_id, logits_last=False):
    n = len(tokens)
    tok  = (llama_token * n)(*tokens)
    pos  = (llama_pos   * n)(*[pos0 + i for i in range(n)])
    nsid = (C.c_int32   * n)(*[1] * n)
    seq_arrays = [(llama_seq_id * 1)(seq_id) for _ in range(n)]
    seqp = (C.POINTER(llama_seq_id) * n)(*[C.cast(a, C.POINTER(llama_seq_id)) for a in seq_arrays])
    lg   = (C.c_int8 * n)(*[1 if (logits_last and i == n - 1) else 0 for i in range(n)])
    batch = llama_batch(n, tok, None, pos, nsid, seqp, lg)
    rc = lib.llama_decode(ctx, batch)
    if rc != 0: raise RuntimeError(f"decode rc={rc}")
    return n

# ---------- kontekst testowy: n_seq_max=4 (3 galezie + zapas) ----------
cp = lib.llama_context_default_params()
cp.n_ctx = 2048; cp.n_batch = 512; cp.n_ubatch = 512; cp.n_seq_max = 4
cp.kv_unified = True
ctx = lib.llama_init_from_model(model, cp)
if not ctx: sys.exit("ctx init fail")
mem = lib.llama_get_memory(ctx)

state = ("Incident: the production dashboard shows a blank page for every user after login. "
         "Started 10 minutes ago, no workaround; the on-call engineer is paged.\n")
prefix_text = state + "Answer each question with a single letter.\n"
q1 = "Question: How severe is this incident?\nA) production outage for all users\nB) minor cosmetic issue\nC) single user, non-blocking\nD) feature request\nAnswer letter:"
q2 = "Question: Does this need an on-call engineer right now?\nA) yes\nB) no\nAnswer letter:"

P  = tokenize(prefix_text)
S1 = tokenize(q1)
S2 = tokenize(q2)
print(f"[2] tokeny: prefix={len(P)}  pole1={len(S1)}  pole2={len(S2)}")

t0 = time.time(); decode_tokens(P, 0, 0); t_prefill = time.time() - t0
print(f"[3] prefill (seq 0): {t_prefill*1000:.0f} ms")

lib.llama_memory_seq_cp(mem, 0, 1, 0, len(P))
lib.llama_memory_seq_cp(mem, 0, 2, 0, len(P))
print("[4] fork: seq_cp(0->1), seq_cp(0->2)  [KV + stan rekurencyjny hybrydy]")

# jeden batched decode: oba sufiksy naraz, logits na ostatnim tokenie każdej gałęzi
n1, n2 = len(S1), len(S2)
tok  = (llama_token * (n1 + n2))(*S1, *S2)
pos  = (llama_pos   * (n1 + n2))(*( [len(P) + i for i in range(n1)] + [len(P) + i for i in range(n2)] ))
nsid = (C.c_int32   * (n1 + n2))(*([1] * (n1 + n2)))
seq_arrays = [(llama_seq_id * 1)(1) for _ in range(n1)] + [(llama_seq_id * 1)(2) for _ in range(n2)]
seqp = (C.POINTER(llama_seq_id) * (n1 + n2))(*[C.cast(a, C.POINTER(llama_seq_id)) for a in seq_arrays])
lg   = (C.c_int8 * (n1 + n2))(*([0] * (n1 - 1) + [1] + [0] * (n2 - 1) + [1]))
batch = llama_batch(n1 + n2, tok, None, pos, nsid, seqp, lg)
t0 = time.time(); rc = lib.llama_decode(ctx, batch); t_branch = time.time() - t0
check(rc, "batched branch decode")
print(f"[5] batched decode dwoch galezi ({n1+n2} tok): {t_branch*1000:.0f} ms")

def probs_from_logits(ptr, cand_ids):
    if not ptr: raise RuntimeError('NULL logits pointer')
    lgts = {tid: ptr[tid] for tid in cand_ids}
    m = max(lgts.values())
    ex = {k: math.exp(v - m) for k, v in lgts.items()}
    s = sum(ex.values())
    return {k: ex[k] / s for k in ex}, lgts

letters = {ch: tokenize(ch)[0] for ch in "ABCD"}
cand = list(letters.values())

p1, _ = probs_from_logits(lib.llama_get_logits_ith(ctx, n1 - 1), cand)   # gałąź 1 (pole: severity)
p2, _ = probs_from_logits(lib.llama_get_logits_ith(ctx, n1 + n2 - 1), cand)   # gałąź 2 (pole: oncall)
inv = {v: k for k, v in letters.items()}
fmt = lambda p: {inv[k]: round(v, 4) for k, v in sorted(p.items(), key=lambda x: -x[1])}
print(f"[6] GAŁĄŹ 1 (severity)  : {fmt(p1)}")
print(f"    GAŁĄŹ 2 (oncall)    : {fmt(p2)}")

# ---------- TEST RÓWNOWAŻNOŚCI: to samo pole liczone sekwencyjnie ----------
cp2 = lib.llama_context_default_params()
cp2.n_ctx = 2048; cp2.n_batch = 512; cp2.n_ubatch = 512; cp2.n_seq_max = 1
cp2.kv_unified = True
ctx2 = lib.llama_init_from_model(model, cp2)
def decode_seq(tokens, pos0, logits_last):
    global ctx
    saved = ctx; ctx = ctx2
    try: return decode_tokens(tokens, pos0, 0, logits_last)
    finally: ctx = saved
decode_seq(P, 0, False)
decode_seq(S1, len(P), True)
pref, _ = probs_from_logits(lib.llama_get_logits_ith(ctx2, len(S1) - 1), cand)
print(f"[7] SEKWENCYJNIE (pole 1): {fmt(pref)}")
maxdiff = max(abs(pref[k] - p1[k]) for k in cand)
print(f"    -> max |roznica| fork vs sekwencyjnie: {maxdiff:.2e}  {'OK (izomorfizm)' if maxdiff < 1e-3 else '!!! ROZJAZD'}")

lib.llama_free(ctx); lib.llama_free(ctx2); lib.llama_model_free(model)
print("[8] sprzatanie OK")
