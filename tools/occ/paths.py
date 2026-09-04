"""LitePT repo + tools/occ import paths."""
from __future__ import annotations

import sys
from pathlib import Path

OCC = Path(__file__).resolve().parent
ROOT = OCC.parents[1]  # LitePT/


def ensure_import_path(*, tools: bool = False, repo: bool = False) -> Path:
    """Put tools/occ (and optionally tools/, repo root) on sys.path. Returns ROOT."""
    for path in (OCC, *((ROOT / "tools",) if tools else ()), *((ROOT,) if repo else ())):
        s = str(path)
        if s not in sys.path:
            sys.path.insert(0, s)
    return ROOT
