"""ctypes structs + function bindings for the pinned llama.cpp ABI

Milestone: E1a.

Transplant docs/evidence/poc-ctypes-20260917.py verbatim: the struct field order is
verified against b11026. Mandatory call order (SPEC 2.2, PoC pitfalls):
    CDLL(libggml) -> ggml_backend_load_all_from_path(rt) -> llama_backend_init()
    -> load model -> init context with kv_unified=True.

Importing this module never loads a library: `load_libraries()` is the only entry point, and
it caches what it loaded (`loaded_runtimes()`), so a test process cannot accidentally dlopen
a half-installed runtime.
"""
from __future__ import annotations

import ctypes as C  # noqa: N812
import os
import pathlib
from dataclasses import dataclass, field

from ggufone.errors import RuntimeMissingError, RuntimeSymbolsError
from ggufone.runtime.finder import library_names

llama_token = C.c_int32
llama_pos = C.c_int32
llama_seq_id = C.c_int32
RTLD_GLOBAL = getattr(C, "RTLD_GLOBAL", 0)
#: `void (*)(enum ggml_log_level level, const char * text, void * user_data)` — b11026.
LLAMA_LOG_CALLBACK = C.CFUNCTYPE(None, C.c_int, C.c_char_p, C.c_void_p)
#: Installed callbacks. `ggml_log_set(NULL, …)` is the documented reset, but a build that ignored
#: it would leave C holding a pointer to a Python object: keep every callback alive for the
#: process (see `engine.session.capture_llama_logs`).
_LIVE_LOG_CALLBACKS: list[object] = []


def register_log_callback(callback: object) -> object:
    """Keep a C-installed callback alive for the life of the process (returns it unchanged).

    `ggml_log_set(NULL, …)` is the documented reset, but a build that ignored it would leave C
    holding a pointer to a Python object; this list makes that impossible.
    """
    _LIVE_LOG_CALLBACKS.append(callback)
    return callback


def live_log_callbacks() -> tuple[object, ...]:
    """The log callbacks this process has installed (kept referenced on purpose)."""
    return tuple(_LIVE_LOG_CALLBACKS)


# --------------------------------------------------------------------- structs
# Field order is verbatim from the executed PoC; tests/test_ctypes_binding.py re-parses the
# PoC and fails if this drifts. Do not reorder or "clean up" these fields.
class llama_batch(C.Structure):
    _fields_ = [("n_tokens", C.c_int32),
                ("token", C.POINTER(llama_token)),
                ("embd", C.POINTER(C.c_float)),
                ("pos", C.POINTER(llama_pos)),
                ("n_seq_id", C.POINTER(C.c_int32)),
                ("seq_id", C.POINTER(C.POINTER(llama_seq_id))),
                ("logits", C.POINTER(C.c_int8))]


class llama_chat_message(C.Structure):
    """`struct llama_chat_message { const char * role; const char * content; }` (b11026)."""

    _fields_ = [("role", C.c_char_p), ("content", C.c_char_p)]


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


MANDATORY_ORDER = (
    "CDLL(libggml.so, RTLD_GLOBAL)",
    "ggml_backend_load_all_from_path(<runtime_dir>)",
    "llama_backend_init()",
    "llama_model_load_from_file()",
    "llama_init_from_model() with kv_unified=True",
)

_LOADED: dict[str, Runtime] = {}


@dataclass
class Runtime:
    """A loaded bundle: the two CDLLs plus the bound function signatures."""

    directory: pathlib.Path
    ggml: C.CDLL
    llama: C.CDLL
    system: str | None = None
    bindings: dict[str, object] = field(default_factory=dict)

    def free(self) -> None:
        try:
            if "llama_backend_free" in self.bindings:
                self.llama.llama_backend_free()
        except OSError:  # pragma: no cover - defensive
            pass


def loaded_runtimes() -> tuple[str, ...]:
    return tuple(_LOADED)


def load_libraries(directory: str | os.PathLike[str], *, system: str | None = None,
                   load_backends: bool = True) -> Runtime:
    """dlopen the bundle in the mandatory order and bind the C ABI."""
    directory = pathlib.Path(directory)
    if not directory.is_dir():
        raise RuntimeMissingError(
            f"E_RUNTIME_MISSING: {directory} is not a directory")
    key = str(directory.resolve())
    if key in _LOADED:
        return _LOADED[key]
    names = library_names(system)

    def _open(name: str) -> C.CDLL:
        path = directory / name
        if not path.exists():
            raise RuntimeMissingError(
                f"E_RUNTIME_MISSING: {path} is missing; the runtime bundle is incomplete "
                f"(re-run `ggufone init --force`)")
        try:
            return C.CDLL(str(path), mode=RTLD_GLOBAL)
        except OSError as exc:
            raise RuntimeMissingError(
                f"E_RUNTIME_MISSING: cannot load {path} ({exc}); the file is truncated or "
                f"not a valid shared library — re-run `ggufone init --force`") from exc

    ggml_base = directory / names["ggml_base"]
    if ggml_base.exists():
        _open(names["ggml_base"])
    ggml = _open(names["ggml"])          # ggml FIRST (PoC pitfall 1)
    llama = _open(names["llama"])
    if load_backends:
        _load_backends(ggml, directory)
    llama.llama_backend_init()
    runtime = Runtime(directory=directory, ggml=ggml, llama=llama, system=system)
    _bind(runtime)
    _LOADED[key] = runtime
    return runtime


def _load_backends(ggml: C.CDLL, directory: pathlib.Path) -> None:
    fn = getattr(ggml, "ggml_backend_load_all_from_path", None)
    if fn is None:  # pragma: no cover - only on a broken bundle
        return
    fn.argtypes = [C.c_char_p]
    fn.restype = None
    fn(str(directory).encode())


def _bind(runtime: Runtime) -> None:
    llama, ggml = runtime.llama, runtime.ggml
    sigs: dict[str, tuple[list[object], object]] = {
        "llama_backend_init": ([], None),
        "llama_backend_free": ([], None),
        "llama_model_default_params": ([], llama_model_params),
        "llama_context_default_params": ([], llama_context_params),
        "llama_model_load_from_file": ([C.c_char_p, llama_model_params], C.c_void_p),
        "llama_model_free": ([C.c_void_p], None),
        "llama_init_from_model": ([C.c_void_p, llama_context_params], C.c_void_p),
        "llama_free": ([C.c_void_p], None),
        "llama_model_get_vocab": ([C.c_void_p], C.c_void_p),
        "llama_vocab_n_tokens": ([C.c_void_p], C.c_int32),
        "llama_model_n_layer": ([C.c_void_p], C.c_int32),
        "llama_model_n_embd": ([C.c_void_p], C.c_int32),
        "llama_n_ctx": ([C.c_void_p], C.c_uint32),
        "llama_n_seq_max": ([C.c_void_p], C.c_uint32),
        "llama_model_meta_val_str": ([C.c_void_p, C.c_char_p, C.c_char_p, C.c_size_t],
                                     C.c_int32),
        "llama_model_chat_template": ([C.c_void_p, C.c_char_p], C.c_char_p),
        "llama_get_memory": ([C.c_void_p], C.c_void_p),
        "llama_memory_seq_cp": ([C.c_void_p, llama_seq_id, llama_seq_id, llama_pos,
                                llama_pos], None),
        "llama_memory_seq_rm": ([C.c_void_p, llama_seq_id, llama_pos, llama_pos], C.c_bool),
        "llama_memory_seq_keep": ([C.c_void_p, llama_seq_id], None),
        "llama_state_seq_get_size": ([C.c_void_p, llama_seq_id], C.c_size_t),
        # Header @ b11026 (include/llama.h:897/905): these two take the SEQUENCE'S TOKENS, not a
        # raw buffer. Getting this wrong is silent — the file is written with garbage and the
        # loader rejects it much later ("token count in sequence state file exceeded capacity").
        "llama_state_seq_save_file": ([C.c_void_p, C.c_char_p, llama_seq_id,
                                       C.POINTER(llama_token), C.c_size_t], C.c_size_t),
        "llama_state_seq_load_file": ([C.c_void_p, C.c_char_p, llama_seq_id,
                                       C.POINTER(llama_token), C.c_size_t,
                                       C.POINTER(C.c_size_t)], C.c_size_t),
        "llama_batch_init": ([C.c_int32, C.c_int32, C.c_int32], llama_batch),
        "llama_batch_free": ([llama_batch], None),
        "llama_batch_get_one": ([C.POINTER(llama_token), C.c_int32], llama_batch),
        "llama_decode": ([C.c_void_p, llama_batch], C.c_int32),
        "llama_get_logits_ith": ([C.c_void_p, C.c_int32], C.POINTER(C.c_float)),
        "llama_tokenize": ([C.c_void_p, C.c_char_p, C.c_int32, C.POINTER(llama_token),
                            C.c_int32, C.c_bool, C.c_bool], C.c_int32),
        "llama_token_to_piece": ([C.c_void_p, llama_token, C.c_char_p, C.c_int32, C.c_int32,
                                  C.c_bool], C.c_int32),
        # Header @ b11026 (include/llama.h:1222/1230): the chat-template entry point takes a
        # `const llama_chat_message *` and (tmpl, chat, n_msg, add_ass, buf, length) — six
        # arguments, in that order. E1c renders through it (chain step 2).
        "llama_chat_apply_template": ([C.c_char_p, C.POINTER(llama_chat_message), C.c_size_t,
                                       C.c_bool, C.c_char_p, C.c_int32], C.c_int32),
        "llama_chat_builtin_templates": ([C.POINTER(C.c_char_p), C.c_size_t], C.c_int32),
        "llama_synchronize": ([C.c_void_p], None),
    }
    missing: list[str] = []
    for name, (argtypes, restype) in sigs.items():
        fn = getattr(llama, name, None)
        if fn is None:
            missing.append(name)
            continue
        fn.argtypes = argtypes
        fn.restype = restype
        runtime.bindings[name] = fn
    if missing:
        raise RuntimeSymbolsError(
            f"E_RUNTIME_SYMBOLS: libllama.so is missing {len(missing)} required symbol(s): "
            f"{', '.join(sorted(missing))}; the installed runtime is not the pinned b11026 "
            f"bundle (re-run `ggufone init --force`)")
    for name in ("ggml_backend_load_all", "ggml_backend_load_all_from_path"):
        fn = getattr(ggml, name, None)
        if fn is None:
            raise RuntimeSymbolsError(
                f"E_RUNTIME_SYMBOLS: libggml.so is missing {name} — the backend loader must "
                f"run before any model load (PoC pitfall 1)")
    _bind_optional(runtime)


def _bind_optional(runtime: Runtime) -> None:
    """Signatures that are knowledge-but-not-requirements (E1c FIX: the backend log handler).

    `llama_log_set(callback, user_data)` lets the engine read the backend's own diagnostics — the
    only place an allocation failure is reported (card t_8cb0a05e). A bundle without it still loads
    models; it just cannot be classified from its log, so it is bound when present and never
    demanded.

    `llama_log_get` is deliberately NOT bound: at b11026 it is a tail jump to
    `ggml_log_get(callback *, void **)` — two OUT parameters — and calling it like the older
    no-argument getter segfaults the process (measured on the pinned CPU bundle: `ggufone ask`
    exited 139 with faulthandler pointing at the call). `capture_llama_logs` resets the handler
    with a NULL callback instead, which is the documented reset in both ABI generations.
    """
    for name, (argtypes, restype) in {
        "llama_log_set": ([LLAMA_LOG_CALLBACK, C.c_void_p], None),
    }.items():
        fn = getattr(runtime.llama, name, None)
        if fn is None:
            continue
        fn.argtypes = argtypes
        fn.restype = restype
        runtime.bindings[name] = fn


def token_piece(runtime: Runtime, vocab: C.c_void_p, token: int, *, special: bool = True) -> str:
    """Decode one token id back to text (special tokens rendered as their literal form)."""
    size = 256
    for _ in range(4):
        buf = C.create_string_buffer(size)
        written = runtime.llama.llama_token_to_piece(vocab, token, buf, size, 0, special)
        if written >= 0:
            return buf.raw[:written].decode("utf-8", errors="replace")
        size = max(size * 4, -written + 1)
    return ""


def tokenize(runtime: Runtime, vocab: C.c_void_p, text: str, *, add_special: bool = False,
             max_tokens: int | None = None) -> list[int]:
    """Tokenize with the model vocab (over-allocating like the PoC: byte length + 64)."""
    raw = text.encode()
    cap = max_tokens or (len(raw) + 64)
    arr = (llama_token * cap)()
    n = runtime.llama.llama_tokenize(vocab, raw, len(raw), arr, cap, add_special, True)
    if n < 0:
        raise RuntimeSymbolsError(f"E_RUNTIME_SYMBOLS: llama_tokenize returned {n}")
    return list(arr[:n])
