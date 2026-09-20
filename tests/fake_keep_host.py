"""The warm host, in a child process, without a model: the real `keep.host.Server` + a fake engine.

`tests/test_keep_client.py` spawns *this* file (via the client's `spawn` seam) so the client's
half — spawn, reuse, swap, stale cleanup, inline fallback, `keep status`/`keep stop` — is exercised
against a real child process, a real unix socket and the real ledger. Only the engine is fake, the
same trade `tests/fake_engine.py` makes for the decision tests.

The mode comes from the environment (`TYPED_GGUF_KEEP_FAKE`), never from `typed_gguf/`: nothing in
the product knows this file exists.

    echo    (default)  answer every request with the payload's own questions echoed back
    slow               answer after `TYPED_GGUF_KEEP_FAKE_SLEEP` seconds (default 0.5)
    hang               never answer the first request (the client is killed mid-request)
    crash              die inside the decision (`os._exit(9)`, the ICD-SIGSEGV shape)
    error              raise a typed engine error (`E_PREFILL_FAILED`, exit code 3)
"""
from __future__ import annotations

import os
import sys
import time

from typed_gguf.errors import PrefillFailedError
from typed_gguf.keep import host as host_module

MODE = os.environ.get("TYPED_GGUF_KEEP_FAKE", "echo")
SLEEP = float(os.environ.get("TYPED_GGUF_KEEP_FAKE_SLEEP", "0.5"))


class FakeHandle:
    """The one thing the server does with a handle: close it (which frees the device, for real)."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _answers(payload: dict) -> dict:
    questions = payload.get("questions") or {}
    answers = {}
    for qid, body in questions.items():
        kind = body.get("type", "noul")
        if kind == "choice":
            answers[qid] = {"type": kind, "choice": sorted(body.get("criteria") or {})[0],
                            "probabilities": {sorted(body.get("criteria") or {})[0]: 1.0}}
        elif kind == "score":
            answers[qid] = {"type": kind, "score": 0.5, "probabilities": {"0.5": 1.0}}
        else:
            answers[qid] = {"type": "noul", "noul": 0.5, "probabilities": {"yes": 1.0}}
    return answers


def make_loaded(spec: host_module.HostSpec) -> host_module.Loaded:
    if MODE == "slowstart":
        # the model takes minutes to load: the *client* must give up by name, not hang forever
        time.sleep(float(os.environ.get("TYPED_GGUF_KEEP_FAKE_SLEEP", "30")))
    handle = FakeHandle()

    def decide(payload: dict) -> dict:
        if MODE == "error":
            raise PrefillFailedError(
                "E_PREFILL_FAILED: the fake engine refused this prefix (tests/fake_keep_host.py)")
        if MODE == "crash":
            os._exit(9)                     # what the NVIDIA ICD does to a session on the way out
        if MODE == "hang":
            time.sleep(float(os.environ.get("TYPED_GGUF_KEEP_FAKE_SLEEP", "30")))
        elif MODE == "slow":
            time.sleep(SLEEP)
        answers = _answers(payload)
        return {
            "model": spec.model,
            "engine": {"runtime": "llama.cpp b11026 (fake)", "backend": "cpu",
                       "backend_source": "explicit", "devices": ["CPU"], "device_buffers": {},
                       "effective_backend": "cpu", "n_ctx": 4096, "kv_type": "auto",
                       "n_gpu_layers": 0, "placement": {"note": "fake placement"}},
            "answers": answers,
            "usage": {"input_tokens": 4, "output_tokens": len(answers), "questions": len(answers),
                      "prefill_tokens": 4, "waves": 1, "forks": 0, "decode_steps": len(answers)},
            # the host paid this load once; a warm decision reports 0 (`engine.keep` carries the
            # real number) — exactly what the CLI's own warm path does
            "timings": {"model_load_ms": 0.0, "prefill_ms": 0.5, "questions_ms": 1.0,
                        "total_ms": 1.5},
            "warnings": [],
        }

    return host_module.Loaded(handle=handle, decide=decide, model=spec.model,
                              model_path=spec.model_path,
                              placement={"note": "fake placement", "n_gpu_layers": 0},
                              devices={"devices": ["CPU"], "compute_buffers": {"CPU": 1},
                                       "effective_backend": "cpu"},
                              model_load_ms=12.5)


def main(argv: list[str]) -> int:
    path = argv[argv.index("--spec") + 1]
    spec = host_module.read_spec(path)
    server = host_module.Server(spec, load=lambda: make_loaded(spec))
    return server.serve()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
