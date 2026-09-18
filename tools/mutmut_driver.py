"""Drive mutmut with the container workaround applied in-process.

`mutmut` copies the source tree with `shutil.copy2`, which carries extended attributes; this
container cannot re-apply `security.selinux` on newly written files, so the copy dies with
`PermissionError: [Errno 13] Permission denied: 'mutants/...'` before a single mutant is
generated (the same failure E1a documented). Nothing about the mutation *result* depends on
xattrs, so the xattr copy step is disabled for the process that drives mutmut.

Usage::

    uv run --extra dev --with mutmut python tools/mutmut_driver.py run --max-children 3
"""
from __future__ import annotations

import shutil
import sys


def _no_xattr(*args: object, **kwargs: object) -> None:
    return None


shutil._copyxattr = _no_xattr  # type: ignore[attr-defined]

from mutmut.__main__ import cli  # noqa: E402

if __name__ == "__main__":
    sys.argv = ["mutmut", *sys.argv[1:]]
    cli()
