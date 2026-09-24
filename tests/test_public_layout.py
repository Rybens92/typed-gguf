"""Public-layout gate (card `t_f2636df1`): the repository root carries no dev-run clutter.

The owner's layout decision (thread `state/groupchat/typed-gguf-e1.md`, 2026-09-24): a root
dot-folder is clutter — only `.github`-class entries (tooling the *repository* itself needs) stay
at the root, and every receipt a card produces lives in a project subfolder. That rule is only a
rule if a green suite can enforce it, so this file fails the build the moment a new dot-entry
appears at the root: a fresh `.e7x/` receipt dir, a stray `.mutmut-cache`, a data home someone
pointed at the checkout.

Two shapes are pinned, because the gate has to hold on a machine that has *used* the project as
well as on a fresh clone:

* the live tree (`ROOT`) — every dot-entry there is one of the allowlisted tooling names;
* a bare checkout built on disk (only `.git/` + `.github/`) — an allowlisted name that is simply
  *absent* is not a failure: the allowlist says what may exist, never what must.
"""
from __future__ import annotations

import fnmatch
import pathlib

#: tooling the repository itself carries or a run creates: allowed as a directory
ALLOWED_DOT_DIRS = frozenset({
    ".git",             # the checkout
    ".github",          # CI, workflows, issue templates
    ".venv",            # the project virtualenv (uv)
    ".pytest_cache",    # pytest's own cache
    ".ruff_cache",      # the linter's own cache
    ".mypy_cache",      # the type checker's own cache
    ".hermes",          # a session/agent home pointed at the checkout
})
#: allowed as a file (a dot-*file* named like one of the dirs above is still clutter)
ALLOWED_DOT_FILES = frozenset({
    ".gitignore",
    ".coverage",
})
#: `mutmut` writes its cache as `.mutmut-cache` (sqlite) or `.mutmut-cache.<something>`
ALLOWED_DOT_FILE_GLOBS = (".mutmut-cache*",)
#: A linked worktree carries `.git` as a *file* (`gitdir: <checkout>/.git/worktrees/<id>`), so
#: `.git` is the one name allowed in either shape — a reviewer running the suite from a worktree
#: (and a `git worktree add` checkout in CI) must not read as a layout violation.
DOT_NAME_ALLOWED_EITHER_WAY = frozenset({".git"})

ROOT = pathlib.Path(__file__).resolve().parents[1]


def is_allowed_dot_entry(name: str, *, is_dir: bool) -> bool:
    """May this dot-name sit in the repository root, in this shape?"""
    if name in DOT_NAME_ALLOWED_EITHER_WAY:
        return True
    if is_dir:
        return name in ALLOWED_DOT_DIRS
    return name in ALLOWED_DOT_FILES or any(
        fnmatch.fnmatch(name, pattern) for pattern in ALLOWED_DOT_FILE_GLOBS)


def offenders(root: pathlib.Path = ROOT) -> list[str]:
    """Every dot-entry of `root` the allowlist does not name, as `name` (+ `/` for a directory)."""
    found: list[str] = []
    for entry in sorted(root.iterdir()):
        if not entry.name.startswith("."):
            continue
        if is_allowed_dot_entry(entry.name, is_dir=entry.is_dir()):
            continue
        found.append(entry.name + "/" if entry.is_dir() else entry.name)
    return found


def test_the_repository_root_carries_only_allowed_dot_entries() -> None:
    """The gate itself: a new root dot-folder is the failure this card exists to prevent."""
    found = offenders()
    assert found == [], (
        "the repository root carries dot-entries the layout allowlist does not name — a receipt "
        "dir belongs under `docs/evidence/<name>/` (QA notes under `docs/qa/`), never at the "
        "root:\n  " + "\n  ".join(found))


def test_a_bare_checkout_passes_though_most_allowed_names_are_absent(tmp_path) -> None:
    """A fresh clone has `.git/` and `.github/` and nothing else: absent is not a failure."""
    (tmp_path / ".git").mkdir()
    (tmp_path / ".github").mkdir()
    (tmp_path / ".gitignore").write_text("", encoding="utf-8")
    (tmp_path / "README.md").write_text("", encoding="utf-8")
    assert offenders(tmp_path) == []


def test_a_new_root_dot_entry_fails_and_tooling_names_do_not(tmp_path) -> None:
    """The other half: the gate *finds* clutter. A green gate above is not an empty walk."""
    (tmp_path / ".git").mkdir()
    (tmp_path / ".e7x").mkdir()
    (tmp_path / ".t99zz").mkdir()
    (tmp_path / ".gauntlet").mkdir()
    assert offenders(tmp_path) == [".e7x/", ".gauntlet/", ".t99zz/"]
    # the shapes a real run creates stay allowed, and a dot-file is judged as a file
    (tmp_path / ".mutmut-cache").write_text("", encoding="utf-8")
    (tmp_path / ".mutmut-cache.20260924").write_text("", encoding="utf-8")
    (tmp_path / ".coverage").write_text("", encoding="utf-8")
    (tmp_path / ".pytest_cache").mkdir()
    (tmp_path / ".hermes").mkdir()
    assert offenders(tmp_path) == [".e7x/", ".gauntlet/", ".t99zz/"]
    # …and a *directory* spelled like an allowed file is not allowed (the allowlist is typed)
    (tmp_path / ".pytest_cache").rmdir()
    (tmp_path / ".pytest_cache").mkdir()
    (tmp_path / ".coverage").unlink()
    (tmp_path / ".coverage").mkdir()
    assert offenders(tmp_path) == [".coverage/", ".e7x/", ".gauntlet/", ".t99zz/"]
