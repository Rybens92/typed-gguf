"""Context lifecycle: load the model, prefill once, fork per question, waves, state save/load.

Milestone: E1b. SPEC 2.2 (mandatory call order, ctx params) + SPEC 2.3 steps 2-10.

This is the only module that calls libllama for decisions. `ModelHandle` owns the model and
its vocabulary (`tokenize`, `n_vocab`); `ModelSession` owns the context (`prefill`, `fork`,
`release`, `decode`). Two objects because the context must be sized from the *tokenized* prompt
(SPEC 2.2: `n_ctx = prefix + longest question + margin`), which needs the vocabulary first.

Pitfalls this module encodes (all PoC-verified, `docs/evidence/poc-ctypes-20260917.py`):
  1. `ggml_backend_load_all_from_path(<rt>)` must run before any model load — done inside
     `ctypes_binding.load_libraries()`;
  2. `kv_unified = True` is mandatory for `llama_memory_seq_cp` (without it the cross-stream
     copy trips `GGML_ASSERT(is_full)` and kills the process);
  3. `llama_get_logits_ith(ctx, i)` is indexed by the token's position **within the batch**
     (not by the output order), which is why rows are read by batch index.

No sampler is ever created and no token is generated: `decode()` is the only entry point, it
takes a fully materialised batch, and it never feeds its own output back in (A-E1b-9).
"""
from __future__ import annotations

import contextlib
import ctypes as C  # noqa: N812
import dataclasses
import hashlib
import json
import os
import pathlib
import struct
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from ggufone.engine.decide import Batch, ContextPlan, PrefillInfo, SessionMeta
from ggufone.errors import (
    DecodeFailedError,
    ModelArchUnsupportedError,
    PrefillFailedError,
    RuntimeMissingError,
    StateLoadFailedError,
)
from ggufone.registry import gguf, store
from ggufone.runtime import capability, ctypes_binding, finder, fit

GGML_TYPE_IDS = {"auto": 1, "f16": 1, "q8_0": 8, "q4_0": 2}   # ggml_type: F16=1, Q4_0=2, Q8_0=8
LLAMA_FLASH_ATTN_TYPE_AUTO = 0
LLAMA_FLASH_ATTN_TYPE_ENABLED = 1
DEFAULT_N_BATCH = 512
STATE_SUFFIX = ".bin"
META_SUFFIX = ".meta.json"
STATE_META_SCHEMA = "ggufone.state/v1"
#: `void (*)(enum ggml_log_level, const char * text, void * user_data)` — b11026.
LLAMA_LOG_CALLBACK = C.CFUNCTYPE(None, C.c_int, C.c_char_p, C.c_void_p)


@contextlib.contextmanager
def capture_llama_logs(runtime: ctypes_binding.Runtime) -> Iterator[list[str]]:
    """Collect libllama/ggml's own diagnostics while the block runs.

    The backends report an allocation failure only through this handler (`GGML_LOG_ERROR`), so a
    load that returns NULL cannot be classified from its return value alone. A bundle without the
    symbol is not an error — the capture simply stays empty and the caller falls back to a CPU
    retry (the swap itself is `ctypes_binding`'s optional `llama_log_set` binding).

    Two traps this function is shaped around, both measured on the pinned b11026 CPU bundle:

    * **`llama_log_get` is NOT a no-argument getter in this build.** It is a tail jump to
      `ggml_log_get(callback *, void **)`, i.e. it *writes through two out-parameters*; calling it
      the way older releases allowed (`llama_log_get()`) segfaults the process
      (`uv run ggufone ask …` exited 139, faulthandler pointed straight at the call). Production
      therefore never reads the previous handler back: it restores ggml's default with a NULL
      callback, which is the documented reset in both ABI generations.
    * the callback must outlive its installation. A future build that ignores the NULL reset would
      otherwise leave C calling a freed Python object, so every installed callback is kept in
      :data:`_LIVE_LOG_CALLBACKS` for the life of the process.
    """
    lines: list[str] = []
    llama = runtime.llama
    setter = getattr(llama, "llama_log_set", None)
    if setter is None:  # pragma: no cover - b11026 always has it
        yield lines
        return

    def collect(_level: int, text: bytes | None, _data: object) -> None:
        lines.append((text or b"").decode("utf-8", "replace").rstrip("\n"))

    callback = ctypes_binding.register_log_callback(
        ctypes_binding.LLAMA_LOG_CALLBACK(collect))
    setter(callback, None)
    try:
        yield lines
    finally:
        setter(ctypes_binding.LLAMA_LOG_CALLBACK(0), None)   # NULL -> ggml's default handler


def _safe_state_name(state_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_.@" else "_" for ch in state_id)


def _token_digest(tokens: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(int(token).to_bytes(4, "little", signed=True))
    return digest.hexdigest()


class ModelHandle:
    """A loaded GGUF model + its vocabulary (no context yet).

    `placement` records HOW the model was placed (`--no-fit`, the fit plan's offload, or a
    degraded retry) so `response.engine` can say it outright instead of leaving the reader to
    interpret `n_gpu_layers: 0` (card t_8cb0a05e, requirement 3).
    """

    def __init__(self, runtime: ctypes_binding.Runtime, model: C.c_void_p, path: str,
                 *, arch: str | None, load_ms: float, n_gpu_layers: int = 0,
                 fit_plan: Any | None = None,
                 placement: Placement | None = None) -> None:
        self.runtime = runtime
        self.model = model
        self.path = path
        self.arch = arch
        self.load_ms = load_ms
        self.n_gpu_layers = int(n_gpu_layers)
        self.fit_plan = fit_plan
        self.placement = placement or Placement(note="", n_gpu_layers=self.n_gpu_layers,
                                                kv_type="auto")
        llama = runtime.llama
        self.vocab = llama.llama_model_get_vocab(model)
        self.n_vocab = int(llama.llama_vocab_n_tokens(self.vocab))
        self.n_layer = int(llama.llama_model_n_layer(model))
        if not self.vocab or self.n_vocab <= 0:
            raise RuntimeMissingError(
                f"E_RUNTIME_SYMBOLS: the loaded model reports {self.n_vocab} vocabulary tokens; "
                f"the bundle and the model do not match")

    @property
    def warnings(self) -> tuple[str, ...]:
        return self.placement.warnings

    def tokenize(self, text: str, *, add_special: bool = False) -> list[int]:
        return ctypes_binding.tokenize(self.runtime, self.vocab, text, add_special=add_special)

    def close(self) -> None:
        if self.model:
            self.runtime.llama.llama_model_free(self.model)
            self.model = None

    def __enter__(self) -> ModelHandle:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class Placement:
    """How the model ended up placed, and why — the `engine.placement` surface.

    `note` is the sentence a human reads; `attempts` is the raw ladder walk (one line per load
    attempt, with the classification of its failure) so an OOM report can be audited without
    re-running the box.
    """

    note: str
    n_gpu_layers: int
    kv_type: str = "auto"
    degraded: bool = False
    attempts: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"note": self.note, "n_gpu_layers": self.n_gpu_layers, "kv_type": self.kv_type,
                "degraded": self.degraded, "attempts": list(self.attempts),
                "warnings": list(self.warnings)}


def _placement_note(plan: Any | None, degraded: bool, fit_disabled: bool) -> str:
    if plan is None:
        return ("fit disabled: CPU only (--no-fit sets n_gpu_layers=0)" if fit_disabled
                else "no fit plan: CPU only (n_gpu_layers=0)")
    if degraded:
        return (f"degraded after a backend allocation failure: {plan.n_gpu_layers} layer(s) "
                f"offloaded, kv_type={plan.kv_type}")
    if plan.n_gpu_layers < 0:
        # `-1` (llama.cpp's "every layer", what the benchmark's GPU default passes) is not "nothing
        # offloaded": report the request, not a CPU outcome this function cannot know (a CPU-only
        # bundle and a fully offloaded Vulkan load both land here).
        return (f"all layers requested: n_gpu_layers={plan.n_gpu_layers} "
                f"(kv_type={plan.kv_type})")
    if plan.n_gpu_layers == 0:
        return f"CPU only: the fit plan offloads nothing (kv_type={plan.kv_type})"
    return f"fit plan: {plan.n_gpu_layers} layer(s) offloaded, kv_type={plan.kv_type}"


def default_free_device_bytes() -> int | None:
    """How much device memory the driver says is free right now (None when it cannot answer)."""
    from ggufone.registry import recommend

    memory = recommend.device_memory()
    return memory.free_bytes if memory.total_bytes > 0 else None


def _load_model(llama: Any, model_path: pathlib.Path, n_gpu_layers: int) -> Any:
    """One load attempt with the placement's layer count (`llama_model_load_from_file`)."""
    params = llama.llama_model_default_params()
    params.n_gpu_layers = int(n_gpu_layers)
    return llama.llama_model_load_from_file(str(model_path).encode(), params)


def open_model(path: str | os.PathLike[str], *, runtime_dir: str | os.PathLike[str] | None = None,
               home: pathlib.Path | None = None, system: str | None = None,
               fit_plan: Any | None = None, fit_disabled: bool = False,
               degrade: bool = True,
               free_probe: Callable[[], int | None] | None = None) -> ModelHandle:
    """Load a GGUF model through the pinned runtime, after the arch pre-flight (SPEC 2.2/A11).

    `fit_plan` (E1c) contributes the placement: `n_gpu_layers` comes from the plan
    (`llama_model_params.n_gpu_layers`); without a plan — or with `--no-fit` — the model is
    placed on the CPU and the placement note says so. Anything that *names a placement* is
    accepted, not only a full `fit.FitPlan`: `fit.coerce_plan` normalizes it once, right here, so
    a minimal object (the benchmark's `Placement`, `n_gpu_layers` and nothing else) cannot reach
    the ladder as an `AttributeError` (card t_31b3943a).

    **Allocation failures degrade, they do not kill the run** (card t_8cb0a05e). The backend's own
    log is captured through `llama_log_set`, classified (`fit.classify_load_failure`) and, when it
    names an allocation failure, the load is retried down `fit.downgrade_steps` — fewer layers,
    then a smaller kv_type, then CPU-only. A failure that is *not* an allocation failure gets one
    CPU-only retry (a log can be silent about the cause); if that fails too the original
    classification is raised, with both attempts named.
    """
    model_path = pathlib.Path(path)
    if not model_path.exists():
        raise RuntimeMissingError(f"E_MODEL_NOT_FOUND: {model_path} does not exist")
    rt_dir = pathlib.Path(runtime_dir) if runtime_dir else finder.find_runtime(home=home,
                                                                              system=system)
    if rt_dir is None:
        raise RuntimeMissingError(
            f"E_RUNTIME_MISSING: no llama.cpp runtime installed under "
            f"{(home or store.data_home()) / 'runtime'}; run `ggufone init` (no compiler "
            f"needed) or set GGUFONE_RUNTIME_DIR")
    arch = None
    try:
        arch = gguf.arch_of(gguf.parse_gguf_metadata(model_path)["kv"])
    except Exception as exc:  # noqa: BLE001 - a corrupt header must not hide the runtime check
        raise RuntimeMissingError(
            f"E_GGUF_CORRUPT: cannot read {model_path} header ({exc})") from exc
    if arch:
        capability.require_arch(rt_dir, arch, lock=None)
    runtime = ctypes_binding.load_libraries(rt_dir, system=system)
    llama = runtime.llama
    plan = fit.coerce_plan(fit_plan) if fit_plan is not None else None
    facts = _model_facts_for_ladder(model_path, plan)
    queue: list[Any | None] = [plan]
    # At LOAD time only the layer count matters (the KV cache does not exist yet): walk fewer
    # layers down to CPU-only, and leave the kv_type rungs to the context init, which is where
    # a KV cache is actually allocated. A plan that pinned no rung (`auto`) starts at the top of
    # the KV ladder, so the layer rungs are still built for it.
    walk = [step for step in fit.degrade_ladder(plan, facts)
            if plan is not None and facts is not None
            and step.kv_type == fit.kv_start(plan.kv_type)] \
        if (degrade and plan is not None and facts is not None) else []
    attempts: list[str] = []
    oom_seen = False
    walk_started = False
    cpu_retry_used = False
    index = 0
    while index < len(queue):
        candidate = queue[index]
        index += 1
        n_gpu_layers = int(getattr(candidate, "n_gpu_layers", 0) or 0)
        started = time.perf_counter()
        with capture_llama_logs(runtime) as captured:
            model = _load_model(llama, model_path, n_gpu_layers)
        load_ms = (time.perf_counter() - started) * 1000.0
        if model:
            degraded = index > 1
            warnings: list[str] = []
            if oom_seen:
                warnings.append("W_BACKEND_OOM")
            if degraded:
                warnings.append("W_FIT_DOWNGRADE")
            placement = Placement(
                note=_placement_note(candidate, degraded=degraded, fit_disabled=fit_disabled),
                n_gpu_layers=n_gpu_layers,
                kv_type=str(getattr(candidate, "kv_type", "auto") or "auto"),
                degraded=degraded, attempts=tuple(attempts), warnings=tuple(warnings))
            return ModelHandle(runtime, model, str(model_path), arch=arch, load_ms=load_ms,
                               n_gpu_layers=n_gpu_layers, fit_plan=candidate,
                               placement=placement)
        text = "\n".join(captured)
        kind = fit.classify_load_failure(text)
        oom_seen = oom_seen or kind == "oom"
        attempts.append(f"n_gpu_layers={n_gpu_layers} -> {kind}")
        if kind == "oom":
            if walk and not walk_started:
                queue.extend(walk)          # walk fewer layers, ending at CPU-only
                walk_started = True
                continue
            if index < len(queue):
                continue                    # the walk is still unrolling
            needed = fit.allocation_bytes_from_log(text)
            if needed is None and candidate is not None and facts is not None:
                needed = fit.plan_device_bytes(candidate, facts)
            raise fit.backend_oom_error(candidate, free_bytes=_free_bytes(free_probe),
                                        needed_bytes=needed, log_tail=text, attempts=attempts)
        if not cpu_retry_used and not walk_started and plan is not None and facts is not None \
                and n_gpu_layers != 0:
            cpu_retry_used = True           # a silent failure still deserves one CPU attempt
            queue.append(_cpu_only(plan, facts))
            continue
        raise _load_failure(model_path, arch, attempts, text)
    raise _load_failure(model_path, arch, attempts, "")   # pragma: no cover - queue never empty


def _model_facts_for_ladder(model_path: pathlib.Path, fit_plan: Any | None) -> Any | None:
    """Header-only model facts for the ladder (no sha256: hashing a 4 GiB file to retry it is
    pointless)."""
    try:
        return fit.ModelFacts.read(model_path, want_sha256=False,
                                   sha256=getattr(fit_plan, "model_sha256", None) or None)
    except Exception:  # noqa: BLE001 - a ladder is an improvement, never a new failure mode
        return None


def _cpu_only(plan: Any, facts: Any) -> Any:
    steps = fit.degrade_ladder(plan, facts)
    for step in steps:
        if int(step.n_gpu_layers) == 0:
            return step
    return steps[-1]  # pragma: no cover - the ladder always ends at CPU-only


def _free_bytes(free_probe: Callable[[], int | None] | None) -> int | None:
    try:
        return int(free_probe()) if free_probe is not None else default_free_device_bytes()
    except Exception:  # noqa: BLE001 - a diagnostic must never mask the real failure
        return None


def _load_failure(path: pathlib.Path, arch: str | None, attempts: list[str],
                  log_tail: str) -> ModelArchUnsupportedError:
    detail = "; ".join(attempts) or "one attempt"
    tail = " ".join(line.strip() for line in log_tail.splitlines() if line.strip())[-400:]
    return ModelArchUnsupportedError(
        f"E_MODEL_ARCH_UNSUPPORTED: llama.cpp could not load {path} (arch {arch or 'unknown'}); "
        f"the pinned runtime must support the architecture (run `ggufone doctor`). "
        f"placements tried: {detail}"
        + (f"; backend log tail: {tail}" if tail else ""))


class ModelSession:
    """A llama.cpp context plus the fork/wave bookkeeping the engine needs."""

    def __init__(self, handle: ModelHandle, plan: ContextPlan, *,
                 backend: str = "cpu", decode_spy: Callable[[Batch], None] | None = None,
                 states_home: pathlib.Path | None = None, degrade: bool = True) -> None:
        self.handle = handle
        self.plan = plan
        self.backend = backend
        self.decode_spy = decode_spy
        self.decode_calls = 0
        self.loaded_states: list[str] = []
        self.states_home = states_home
        self.kv_type = plan.kv_type           # what the request (or plan) asked for — HEAD contract
        self.kv_type_used = ""               # the rung the context was really created with
        self.extra_warnings: list[str] = []
        llama = handle.runtime.llama
        self.ctx, failure, kv_used = _init_context(llama, handle, plan, degrade=degrade,
                                                  warnings=self.extra_warnings)
        if self.ctx:
            self.kv_type_used = kv_used
        if not self.ctx:
            raise failure or RuntimeMissingError(
                f"E_RUNTIME_MISSING: llama_init_from_model failed (n_ctx={plan.n_ctx}, "
                f"n_seq_max={plan.n_seq_max}); the runtime refused these context parameters")
        self.memory = llama.llama_get_memory(self.ctx)
        self._runtime_name = _runtime_name(handle)

    # ---- surface
    @property
    def meta(self) -> SessionMeta:
        placement = getattr(self.handle, "placement", None)
        if placement is not None and self.kv_type_used and self.kv_type_used != placement.kv_type:
            # the context resolved `auto` (or degraded) to a concrete rung: report the one that
            # was really used, next to the request's own value in `engine.kv_type`
            placement = dataclasses.replace(placement, kv_type=self.kv_type_used)
        carried = tuple(getattr(placement, "warnings", ()) or ())
        return SessionMeta(runtime=self._runtime_name, backend=self.backend,
                           n_ctx=int(self.handle.runtime.llama.llama_n_ctx(self.ctx)),
                           n_seq_max=int(self.handle.runtime.llama.llama_n_seq_max(self.ctx)),
                           kv_unified=True, threads=int(self.plan.threads),
                           n_vocab=self.handle.n_vocab, model_path=self.handle.path,
                           model_alias=None, load_ms=self.handle.load_ms,
                           kv_type=self.kv_type,
                           n_gpu_layers=int(getattr(self.handle, "n_gpu_layers", 0) or 0),
                           placement=placement,
                           placement_warnings=tuple(dict.fromkeys(carried
                                                                  + tuple(self.extra_warnings))))

    def tokenize(self, text: str) -> list[int]:
        return self.handle.tokenize(text)

    def close(self) -> None:
        if self.ctx:
            self.handle.runtime.llama.llama_free(self.ctx)
            self.ctx = None

    def __enter__(self) -> ModelSession:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- mechanics
    def prefill(self, tokens: Sequence[int], *, state_id: str | None = None,
                state_cache: bool = True, save_state: bool = False) -> PrefillInfo:
        """Decode the shared prefix once on seq 0, or reload a saved prefix state (SPEC 2.3)."""
        if not tokens:
            raise PrefillFailedError("E_PREFILL_FAILED: the prefix tokenized to zero tokens")
        path = self._state_path(state_id) if state_id else None
        if state_cache and path is not None and path.exists() \
                and self._load_state(path, tokens):
            self.loaded_states.append(state_id or path.name)
            return PrefillInfo(prefill_tokens=0, prefill_ms=0.0, prefill_reused=True,
                               state_id=state_id, state_path=str(path))
        started = time.perf_counter()
        self.decode(Batch(tokens=tuple(tokens), seq_ids=(0,) * len(tokens),
                          positions=tuple(range(len(tokens))),
                          logits=tuple(False for _ in tokens)))
        prefill_ms = (time.perf_counter() - started) * 1000.0
        if save_state and path is not None:
            self._save_state(path, tokens)
        return PrefillInfo(prefill_tokens=len(tokens), prefill_ms=prefill_ms,
                           prefill_reused=False, state_id=state_id,
                           state_path=str(path) if path else None)

    def fork(self, src: int, dst: int, upto: int) -> None:
        """`llama_memory_seq_cp` — KV cells *and* recurrent state (hybrid models) move together."""
        self.handle.runtime.llama.llama_memory_seq_cp(self.memory, src, dst, 0, int(upto))

    def release(self, seq: int) -> None:
        """Drop every cell of one sequence so a seq id can be reused by the next wave."""
        if seq == 0:
            return
        self.handle.runtime.llama.llama_memory_seq_rm(self.memory, seq, -1, -1)

    def decode(self, batch: Batch) -> list[list[float]]:
        """One `llama_decode`; returns the full-vocab row at every `logits=1` position."""
        if self.decode_spy is not None:
            self.decode_spy(batch)
        n = batch.n_tokens
        tokens = (ctypes_binding.llama_token * n)(*batch.tokens)
        positions = (ctypes_binding.llama_pos * n)(*batch.positions)
        n_seq_id = (C.c_int32 * n)(*([1] * n))
        seq_arrays = [(ctypes_binding.llama_seq_id * 1)(seq) for seq in batch.seq_ids]
        seq_pointers = (C.POINTER(ctypes_binding.llama_seq_id) * n)(
            *[C.cast(array, C.POINTER(ctypes_binding.llama_seq_id)) for array in seq_arrays])
        logits = (C.c_int8 * n)(*[1 if flag else 0 for flag in batch.logits])
        llama_batch = ctypes_binding.llama_batch(n, tokens, None, positions, n_seq_id,
                                                 seq_pointers, logits)
        self.decode_calls += 1
        rc = self.handle.runtime.llama.llama_decode(self.ctx, llama_batch)
        if rc != 0:
            raise DecodeFailedError(f"E_DECODE_FAILED: llama_decode returned {rc}")
        self.handle.runtime.llama.llama_synchronize(self.ctx)
        rows: list[list[float]] = []
        n_vocab = self.handle.n_vocab
        for index in batch.logits_indices():
            pointer = self.handle.runtime.llama.llama_get_logits_ith(self.ctx, index)
            if not pointer:
                raise DecodeFailedError(
                    f"E_DECODE_FAILED: no logits at batch position {index} (logits=1 was set "
                    f"there, but the context returned a null row)")
            rows.append(list(struct.unpack(f"<{n_vocab}f", C.string_at(pointer, 4 * n_vocab))))
        return rows

    # ---- saved prefix states (SPEC 2.3.10)
    def _state_path(self, state_id: str) -> pathlib.Path:
        directory = self.states_home or store.states_dir()
        return directory / f"{_safe_state_name(state_id)}{STATE_SUFFIX}"

    def _state_meta_path(self, path: pathlib.Path) -> pathlib.Path:
        return path.with_name(path.name + META_SUFFIX)

    def _load_state(self, path: pathlib.Path, expected_tokens: Sequence[int]) -> bool:
        """Restore a saved prefix state into seq 0 after checking our own integrity metadata.

        `llama_state_seq_load_file(ctx, filepath, dest_seq_id, tokens_out, n_token_capacity,
        n_token_count_out)` both restores the state and hands back the tokens the file carries
        (include/llama.h @ b11026:905). Two guards run BEFORE that call, because libllama
        *aborts the process* on a state file whose declared token count or payload is wrong
        (measured: a truncated file killed the interpreter with SIGABRT): the sidecar metadata
        must match the file byte-for-byte, and the file must declare this prefix's token count.
        A rejected entry is removed, so the caller re-prefills instead of crashing.
        """
        meta_path = self._state_meta_path(path)
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise self._bad_state(
                path, f"no usable metadata sidecar ({exc.__class__.__name__})") from exc
        size = path.stat().st_size
        if meta.get("bytes") != size:
            raise self._bad_state(
                path, f"the file is {size} bytes but the entry recorded {meta.get('bytes')} "
                      f"(truncated or overwritten)")
        if meta.get("tokens") != len(expected_tokens) or meta.get("sha256_tokens") != \
                _token_digest(expected_tokens):
            raise self._bad_state(path, "the entry belongs to a different prefix")
        with open(path, "rb") as handle:
            header = handle.read(12)
        # layout (llama-context.cpp @ b11026:3280): [u32 magic][u32 version][u32 n_tokens][tokens]
        declared = int.from_bytes(header[8:12], "little") if len(header) == 12 else -1
        if declared != len(expected_tokens):
            raise self._bad_state(
                path, f"the file declares {declared} tokens, this prefix has "
                      f"{len(expected_tokens)}")

        capacity = max(len(expected_tokens), 1)
        tokens_out = (ctypes_binding.llama_token * capacity)()
        count_out = C.c_size_t(0)
        read = self.handle.runtime.llama.llama_state_seq_load_file(
            self.ctx, str(path).encode(), 0, tokens_out, capacity, C.byref(count_out))
        if read <= 0:
            raise self._bad_state(path, f"llama_state_seq_load_file returned {read} bytes")
        got = list(tokens_out[:int(count_out.value)])
        if got != list(expected_tokens):
            raise self._bad_state(
                path, f"the file holds {len(got)} tokens that do not match this prefix "
                      f"({len(expected_tokens)} tokens)")
        return True

    def _bad_state(self, path: pathlib.Path, detail: str) -> StateLoadFailedError:
        """Invalidate the cache entry (A-E1b-8: the next call must re-prefill, not crash)."""
        for target in (path, self._state_meta_path(path)):
            with contextlib.suppress(OSError):
                target.unlink()
        return StateLoadFailedError(
            f"E_STATE_LOAD_FAILED: {path} is not a usable prefix state ({detail}); the cache "
            f"entry was removed — retry and the prefix will be decoded again")

    def _save_state(self, path: pathlib.Path, tokens: Sequence[int]) -> None:
        """Persist seq 0's state for `tokens` (the C call writes the file itself).

        `llama_state_seq_save_file(ctx, filepath, seq_id, tokens, n_tokens)` stores the tokens
        next to the KV/recurrent state, which is what lets `_load_state` prove the file belongs
        to this prefix. The temp-file + rename keeps a crashed write from poisoning the cache,
        and the sidecar records the exact byte size so a truncated file is detectable without
        asking libllama (which aborts instead of returning an error).
        """
        llama = self.handle.runtime.llama
        path.parent.mkdir(parents=True, exist_ok=True)
        token_array = (ctypes_binding.llama_token * len(tokens))(*tokens)
        tmp = path.with_name(path.name + ".tmp")
        tmp.unlink(missing_ok=True)
        written = int(llama.llama_state_seq_save_file(self.ctx, str(tmp).encode(), 0,
                                                      token_array, len(tokens)))
        if written <= 0:
            tmp.unlink(missing_ok=True)
            raise PrefillFailedError(
                f"E_PREFILL_FAILED: llama_state_seq_save_file wrote {written} bytes to {tmp}; "
                f"the prefix state was not persisted")
        os.replace(tmp, path)
        self._state_meta_path(path).write_text(json.dumps({
            "schema": STATE_META_SCHEMA,
            "tokens": len(tokens),
            "sha256_tokens": _token_digest(tokens),
            "bytes": path.stat().st_size,
            "model_path": self.handle.path,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, indent=1), encoding="utf-8")


def _context_params(llama: Any, plan: ContextPlan, kv_type: str) -> Any:
    """`llama_context_params` for one context attempt (SPEC 2.2 + the kv_type under test)."""
    params = llama.llama_context_default_params()
    params.n_ctx = int(plan.n_ctx)
    params.n_batch = max(DEFAULT_N_BATCH, int(plan.n_ctx))
    params.n_ubatch = DEFAULT_N_BATCH
    params.n_seq_max = int(plan.n_seq_max)
    params.n_threads = int(plan.threads)
    params.n_threads_batch = int(plan.threads)
    kv_type_id = GGML_TYPE_IDS.get(kv_type, GGML_TYPE_IDS["f16"])
    params.type_k = kv_type_id
    params.type_v = kv_type_id
    params.kv_unified = True                     # pitfall 2: mandatory for seq_cp
    params.no_perf = False
    if kv_type_id != GGML_TYPE_IDS["f16"]:
        # a quantized V cache requires flash attention in llama.cpp
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED
    return params


def _init_context(llama: Any, handle: ModelHandle, plan: ContextPlan, *,
                  degrade: bool, warnings: list[str]) -> tuple[Any | None, Any | None, str]:
    """Create the context, degrading the KV type on an allocation failure (requirement 3).

    The KV cache is allocated here, so this is the one place where a smaller `kv_type` actually
    helps: f16 -> q8_0 -> q4_0, each step recorded as `W_KV_TYPE_DOWNGRADE` + `W_FIT_DOWNGRADE`.
    Returns `(ctx, failure, kv_type_used)`; `failure` is the typed error to raise when no rung
    worked, `None` when the failure was not memory-related (the caller's own message then fits).
    """
    ladder = _kv_ladder(plan.kv_type, degrade=degrade)
    start = ladder[0] if ladder else plan.kv_type
    last: tuple[str, str, int] | None = None
    for position, rung in enumerate(ladder):
        with capture_llama_logs(handle.runtime) as captured:
            ctx = llama.llama_init_from_model(handle.model, _context_params(llama, plan, rung))
        if ctx:
            if rung != start:
                # the request's own value (`auto` included) resolved to `start`; only a rung
                # BELOW that is a downgrade worth warning about
                warnings.append("W_KV_TYPE_DOWNGRADE")
                warnings.append("W_FIT_DOWNGRADE")
            return ctx, None, rung
        text = "\n".join(captured)
        kind = fit.classify_load_failure(text)
        last = (kind, text, position)
        if kind != "oom":                          # not memory: a smaller KV cache cannot help
            break
    if last is not None:
        kind, text, position = last
        tail = " ".join(line.strip() for line in text.splitlines() if line.strip())[-300:]
        if kind == "oom":
            return None, fit.backend_oom_error(
                handle.fit_plan, free_bytes=None,
                needed_bytes=fit.allocation_bytes_from_log(text), log_tail=text,
                attempts=[f"ctx kv_type={rung} -> oom" for rung in ladder[:position + 1]]), ""
        return None, RuntimeMissingError(
            f"E_RUNTIME_MISSING: llama_init_from_model failed (n_ctx={plan.n_ctx}, "
            f"n_seq_max={plan.n_seq_max}, kv_type={plan.kv_type}); the runtime refused these "
            f"context parameters ({kind})"
            + (f"; backend log tail: {tail}" if tail else "")), ""
    return None, None, ""


def _kv_ladder(kv_type: str, *, degrade: bool) -> list[str]:
    """The KV types to try, best first.

    `auto` (the default of every request) and any unknown value start at f16 — which is what
    `GGML_TYPE_IDS.get(kv_type, f16)` resolves them to. An empty ladder would mean "create no
    context at all", and that is how a `--no-fit` run with the default kv_type failed once: the
    context init loop never ran (card t_8cb0a05e follow-up).
    """
    start = kv_type if kv_type in fit.KV_DOWNGRADE_ORDER else fit.KV_DOWNGRADE_ORDER[0]
    index = fit.KV_DOWNGRADE_ORDER.index(start)
    return list(fit.KV_DOWNGRADE_ORDER[index:] if degrade else (start,))


def _runtime_name(handle: ModelHandle) -> str:
    """`llama.cpp b11026` — the pinned tag when the build number is readable."""
    build = capability.build_number(_runtime_dir_of(handle))
    return f"llama.cpp b{build}" if build else "llama.cpp"


def _runtime_dir_of(handle: ModelHandle) -> pathlib.Path:
    return pathlib.Path(handle.runtime.directory)


def runtime_backend(home: pathlib.Path | None = None) -> str:
    """The backend `init` proved working on this host (report-only; defaults to cpu)."""
    record = finder.runtime_record(home) or {}
    return str(record.get("backend_working") or record.get("backend_requested") or "cpu")
