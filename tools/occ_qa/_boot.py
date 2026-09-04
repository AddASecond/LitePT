"""Bootstrap C lane (occ_qa). Do not put tools/occ on sys.path."""
from __future__ import annotations

import sys
from pathlib import Path

QA = Path(__file__).resolve().parent
VIEWER = QA.parent / "occ_viewer"
for path in (QA, VIEWER):
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)

from env_paths import ROOT, RAW_ROOTS, DEFAULT_URI, ensure_c_path  # noqa: E402

ensure_c_path(QA)
