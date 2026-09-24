#!/usr/bin/env python3
"""Which `ggufone` package does this interpreter import? (t7c9 gate plumbing check)."""
import ggufone

print("ggufone:", ggufone.__file__)
try:
    import pytest
    print("pytest:", pytest.__version__)
except Exception as error:  # noqa: BLE001
    print("pytest: missing:", error)
