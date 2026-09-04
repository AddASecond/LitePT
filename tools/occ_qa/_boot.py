"""Bootstrap occ_qa (no LitePT occ production imports; no occ_viewer dependency)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

QA = Path(__file__).resolve().parent
ROOT = QA.parents[1]
RAW_ROOTS = tuple(Path(f"/data/rawdata{s}") for s in ("", "-1", "-2", "-3", "-4"))
DEFAULT_URI = os.environ.get(
    "ROBOTRUCK_MONGO_URI",
    "mongodb://krk030-mongodb:27017/?authSource=perception_experiment",
)

if str(QA) not in sys.path:
    sys.path.insert(0, str(QA))
