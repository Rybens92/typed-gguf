"""Launcher: run the typed-gguf CLI from the checkout's src/ without any install."""
import sys

REPO_SRC = "/workspace/ggufone/src"
if REPO_SRC not in sys.path:
    sys.path.insert(0, REPO_SRC)

from typed_gguf.cli import run  # noqa: E402

if __name__ == "__main__":
    sys.exit(run())
