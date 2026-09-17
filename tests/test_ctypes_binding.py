"""ctypes binding: struct layouts are transplanted verbatim from the executed PoC.

The PoC (docs/evidence/poc-ctypes-20260917.py) is the reference implementation; its struct
field order was verified against llama.h @ b11026 on this box. If our structs ever drift from
it, the ABI is wrong and the engine would read garbage — so the test parses the PoC itself.
"""
from __future__ import annotations

import ast
import pathlib
import platform

import pytest

from ggufone.errors import GgufoneError
from ggufone.runtime import ctypes_binding, finder

ROOT = pathlib.Path(__file__).resolve().parents[1]
POC = ROOT / "docs" / "evidence" / "poc-ctypes-20260917.py"


def poc_structs() -> dict[str, list[str]]:
    tree = ast.parse(POC.read_text())
    out: dict[str, list[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name.startswith("llama_"):
            names = []
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) \
                        and getattr(stmt.targets[0], "id", "") == "_fields_":
                    for elt in stmt.value.elts:  # type: ignore[attr-defined]
                        names.append(elt.elts[0].value)
            out[node.name] = names
    return out


def test_structs_match_the_poc_verbatim() -> None:
    want = poc_structs()
    assert set(want) == {"llama_batch", "llama_model_params", "llama_context_params"}
    ours = {
        "llama_batch": [f[0] for f in ctypes_binding.llama_batch._fields_],
        "llama_model_params": [f[0] for f in ctypes_binding.llama_model_params._fields_],
        "llama_context_params": [f[0] for f in ctypes_binding.llama_context_params._fields_],
    }
    assert ours == want


def test_context_params_expose_the_mandatory_kv_unified_flag() -> None:
    names = [f[0] for f in ctypes_binding.llama_context_params._fields_]
    assert "kv_unified" in names          # PoC pitfall 2: seq_cp asserts without it
    assert "n_seq_max" in names and "n_rs_seq" in names
    model_names = [f[0] for f in ctypes_binding.llama_model_params._fields_]
    assert "n_gpu_layers" in model_names and "tensor_buft_overrides" in model_names


def test_importing_the_module_loads_no_library() -> None:
    """Importing must never dlopen a bundle — checked in a fresh interpreter.

    In-process this can only be asserted before anything else loads the runtime; the E1b model
    tests (which legitimately call `load_libraries`) run in the same session, so the invariant
    that matters — "import is inert" — is pinned in a child process.
    """
    import os
    import subprocess
    import sys
    root = pathlib.Path(__file__).resolve().parents[1]
    environment = {**os.environ, "PYTHONPATH": str(root / "src")}
    code = ("import ggufone.runtime.ctypes_binding as b;"
            "assert b.loaded_runtimes() == (), b.loaded_runtimes();"
            "assert not hasattr(b, 'llama_decode_enabled');"
            "print('inert')")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env=environment, check=False, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "inert"


def test_mandatory_call_order_is_documented_and_pinned() -> None:
    order = " | ".join(ctypes_binding.MANDATORY_ORDER).replace(" ", "")
    assert order.index("ggml_backend_load_all_from_path") < order.index("llama_backend_init")
    assert "kv_unified=True" in order


def test_load_libraries_reports_missing_files(tmp_path: pathlib.Path) -> None:
    with pytest.raises(GgufoneError) as exc:
        ctypes_binding.load_libraries(tmp_path)
    assert exc.value.code == "E_RUNTIME_MISSING"
    assert "libllama" in str(exc.value) or "libggml" in str(exc.value)


def test_load_libraries_surfaces_a_dlopen_failure(tmp_path: pathlib.Path) -> None:
    names = finder.library_names("linux")
    for key in ("ggml_base", "ggml", "llama"):
        (tmp_path / names[key]).write_text("this is not an ELF shared object\n")
    with pytest.raises(GgufoneError) as exc:
        ctypes_binding.load_libraries(tmp_path, system="linux")
    assert exc.value.code in ("E_RUNTIME_MISSING", "E_RUNTIME_SYMBOLS")
    assert str(tmp_path) in str(exc.value)


@pytest.mark.parametrize(("system", "suffix"), [
    ("linux", "libllama.so"),
    ("darwin", "libllama.dylib"),
    ("windows", "llama.dll"),
])
def test_library_names_per_platform(system: str, suffix: str) -> None:
    assert finder.library_names(system)["llama"] == suffix


def test_library_names_default_to_the_running_platform() -> None:
    got = finder.library_names()
    if platform.system().lower() == "linux":
        assert got["llama"] == "libllama.so"
    assert set(got) == {"llama", "ggml", "ggml_base"}


# ------------------------------------------------------------------ live ABI pin (needs runtime)
def _runtime_dir() -> pathlib.Path:
    import os
    env = os.environ.get("GGUFONE_RUNTIME_DIR")
    if env and (pathlib.Path(env) / "libllama.so").exists():
        return pathlib.Path(env)
    for base in (pathlib.Path.home() / ".hermes" / "runtime",
                 pathlib.Path.home() / ".local" / "share" / "ggufone" / "runtime"):
        for candidate in sorted(base.glob("*/")):
            if (candidate / "libllama.so").exists():
                return candidate
    pytest.skip("no llama.cpp runtime on this box (set GGUFONE_RUNTIME_DIR)")


@pytest.mark.model
def test_state_seq_file_signatures_match_the_header() -> None:
    """include/llama.h @ b11026:897/905 — the state-file pair takes the sequence's TOKENS.

    The scaffold's first guess (a raw `void * dst/size` buffer pair) is silently wrong: the save
    writes a file the loader then rejects ("token count in sequence state file exceeded
    capacity"), which is exactly how E1b's A-E1b-8 caught it. This pin needs a bundle on disk
    because ctypes only exposes `argtypes` after `load_libraries()`.
    """
    import ctypes as C
    before = dict(ctypes_binding._LOADED)          # keep `loaded_runtimes()` honest for the
    try:                                           # no-eager-load test that follows
        runtime = ctypes_binding.load_libraries(_runtime_dir())
        save = runtime.bindings["llama_state_seq_save_file"]
        load = runtime.bindings["llama_state_seq_load_file"]
        assert list(save.argtypes) == [C.c_void_p, C.c_char_p, ctypes_binding.llama_seq_id,
                                       C.POINTER(ctypes_binding.llama_token), C.c_size_t]
        assert list(load.argtypes) == [C.c_void_p, C.c_char_p, ctypes_binding.llama_seq_id,
                                       C.POINTER(ctypes_binding.llama_token), C.c_size_t,
                                       C.POINTER(C.c_size_t)]
        assert save.restype is C.c_size_t and load.restype is C.c_size_t
    finally:
        ctypes_binding._LOADED.clear()
        ctypes_binding._LOADED.update(before)
