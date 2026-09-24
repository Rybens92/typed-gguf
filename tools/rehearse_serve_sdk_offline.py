"""Offline rehearsal of `tools/host_gate_serve.sh`'s SDK step (card `t_f5d8b6c7`).

The host gate's acceptance is a real `typesafe-sdk==0.7.1` client against the real 4B on the real
box. This script rehearses *everything except the model and the GPU*: it serves the in-process app
on loopback with the same real decision path the CLI uses (the real engine over
`tests/fake_engine.py`'s deterministic session — `tests/test_serve.py`'s trade) and then runs the
gate's own client driver (`tools/host_gate_serve_client.py`) against it inside the pinned SDK venv.

What it proves: the SDK's own request bytes reach the server, the server's response validates
against the SDK's pydantic models, the typed answers survive `model_dump()`, and the driver's
checks pass on a body the *real* mapping produced.

What only the host can prove: that the numbers come from the 4B through the resident keep host.

    python3 tools/rehearse_serve_sdk_offline.py --sdk-python /path/to/sdk-venv/bin/python

Exit 0 when the driver exits 0 and its verdict is printed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import socket
import subprocess
import sys
import threading

REPO = pathlib.Path(__file__).resolve().parents[1]
ALIAS = "jev-latest"
RESOLVED = "spark-x2.5-4b-q8_0"


def _load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_home(root: pathlib.Path) -> pathlib.Path:
    """A data home with one alias whose file exists — what the served route resolves."""
    from typed_gguf.registry import store

    home = root / "home"
    (home / "models").mkdir(parents=True, exist_ok=True)
    weights = home / "models" / f"{RESOLVED}.gguf"
    weights.write_bytes(b"GGUF" + bytes(64))               # the file's existence is what is read
    entry = store.Entry(alias=RESOLVED, path=str(weights), arch="qwen35", quant="Q8_0",
                        size=weights.stat().st_size)
    store.save_registry(store.Registry(aliases={RESOLVED: entry}, current=RESOLVED),
                        store.registry_path(home))
    return home


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sdk-python", default=sys.executable,
                        help="a python that has the pinned typesafe-sdk installed")
    parser.add_argument("--root", default=None, help="a scratch dir (default: a temp dir)")
    parser.add_argument("--keep", action="store_true", help="do not delete the scratch dir")
    opts = parser.parse_args(argv)

    import tempfile

    from typed_gguf.api import http as serve

    # `tests/test_serve.py` imports its siblings as `tests.<module>`; do the same here
    for entry in (str(REPO), str(REPO / "src")):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    test_serve = _load_module("rehearsal_test_serve", REPO / "tests" / "test_serve.py")
    root = pathlib.Path(opts.root) if opts.root else pathlib.Path(
        tempfile.mkdtemp(prefix="rehearse-serve-"))
    home = build_home(root)

    # the same decide callable shape the CLI wires: one payload in, one native body out
    def decide(payload: dict) -> dict:
        return test_serve.mixed_engine_body(payload)

    app = serve.App(decide=decide, home=home, keep_alive=600.0, default_format="native",
                    log=lambda line: print(f"  server: {line}", flush=True))
    port = free_port()
    server = serve.make_server(app, "127.0.0.1", port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"rehearsal server: http://127.0.0.1:{port} (home={home})")

    try:
        argv = [opts.sdk_python, str(REPO / "tools" / "host_gate_serve_client.py"),
                "--base-url", f"http://127.0.0.1:{port}", "--out", str(root / "body.json")]
        import os

        env = {**os.environ, "TYPESAFE_BASE_URL": f"http://127.0.0.1:{port}",
               "TYPESAFE_API_KEY": "rehearsal-not-a-secret"}
        completed = subprocess.run(argv, cwd=REPO, env=env, capture_output=True, text=True)
        print(completed.stdout)
        if completed.stderr.strip():
            print("--- driver stderr ---\n" + completed.stderr, file=sys.stderr)
        body = json.loads((root / "body.json").read_text()) if (root / "body.json").exists() else {}
        if body:
            print("served model: " + repr(body.get("model")))
            print("answer types: " + json.dumps({qid: answer.get("type")
                                                 for qid, answer in body["answers"].items()}))
        print(f"\ndriver exit: {completed.returncode} (scratch dir: {root})")
        return completed.returncode
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        if not opts.keep:
            import shutil

            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
