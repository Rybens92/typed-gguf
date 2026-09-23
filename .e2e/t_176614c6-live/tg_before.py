"""Launcher: run the typed-gguf CLI from the *pristine* HEAD worktree (card t_176614c6 receipts)."""
import sys

REPO_SRC = "/work/t176614c6-before/src"
if REPO_SRC not in sys.path:
    sys.path.insert(0, REPO_SRC)

from typed_gguf.cli import run  # noqa: E402

if __name__ == "__main__":
    sys.exit(run())
