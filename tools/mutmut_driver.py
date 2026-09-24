"""Drive mutmut with the container workarounds applied in-process.

Two container facts, both fatal before a single mutant is generated:

* `mutmut` copies the source tree with `shutil.copy2`, which carries extended attributes; this
  container cannot re-apply `security.selinux` on newly written files, so the copy dies with
  `PermissionError: [Errno 13] Permission denied: 'mutants/...'` (the same failure E1a documented).
* the committed evidence trees (`docs/evidence/`) hold venvs whose `bin/python*` and
  `lib*.so.0.*` are dangling symlinks; `shutil.copytree` follows them and dies with
  `shutil.Error: No such file or directory: docs/evidence/.../venv/bin/python`.

Neither affects the mutation *result* — xattrs are not read by anything here, and a source tree
wants links copied as links — so both are disabled for the process that drives mutmut.

Usage::

    uv run --extra dev --with mutmut python tools/mutmut_driver.py run --max-children 3
"""
from __future__ import annotations

import shutil
import sys


def _no_xattr(*args: object, **kwargs: object) -> None:
    return None


shutil._copyxattr = _no_xattr  # type: ignore[attr-defined]

_original_copytree = shutil.copytree


def _symlink_safe_copytree(src, dst, symlinks=False, ignore=None, copy_function=shutil.copy2,
                           ignore_dangling_symlinks=False, dirs_exist_ok=False):
    """Copy links as links.

    `also_copy` carries `docs`, and the committed evidence trees hold venvs whose `bin/python*`
    are symlinks into a prefix that no longer exists here. Following them aborts the sweep before
    a single mutant is generated (`shutil.Error: No such file or directory: docs/evidence/.../venv/
    bin/python`); copying the link itself is what a source tree wants anyway.

    The signature has to match `shutil.copytree`'s exactly: its own recursion calls it positionally.
    """
    return _original_copytree(src, dst, symlinks=True, ignore=ignore,
                              copy_function=copy_function, ignore_dangling_symlinks=True,
                              dirs_exist_ok=dirs_exist_ok)


shutil.copytree = _symlink_safe_copytree  # type: ignore[assignment]

from mutmut.__main__ import cli  # noqa: E402

if __name__ == "__main__":
    sys.argv = ["mutmut", *sys.argv[1:]]
    cli()
