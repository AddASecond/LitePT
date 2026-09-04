#!/usr/bin/env python3
"""Run OCC unit tests."""
from __future__ import annotations

import unittest
from pathlib import Path

_IMPL = Path(__file__).resolve().parent / "_impl"


def main() -> int:
    suite = unittest.defaultTestLoader.discover(str(_IMPL), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
