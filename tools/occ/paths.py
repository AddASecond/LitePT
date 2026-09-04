"""LitePT repo paths + shared Mongo/rawdata constants (no CUDA, no OCC logic)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

OCC = Path(__file__).resolve().parent
ROOT = OCC.parents[1]

RAW_ROOTS = tuple(Path(f"/data/rawdata{s}") for s in ("", "-1", "-2", "-3", "-4"))
DEFAULT_ASSET_ROOT = Path("/data/rawdata-4/occupancy")
DEFAULT_URI = os.environ.get(
    "ROBOTRUCK_MONGO_URI",
    "mongodb://krk030-mongodb:27017/?authSource=perception_experiment",
)


def ensure_import_path(*, tools: bool = False, repo: bool = False) -> Path:
    for path in (OCC, *((ROOT / "tools",) if tools else ()), *((ROOT,) if repo else ())):
        s = str(path)
        if s not in sys.path:
            sys.path.insert(0, s)
    return ROOT
