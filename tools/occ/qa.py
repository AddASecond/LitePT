#!/usr/bin/env python3
"""OCC QA / debug CLI — offline filter review and triage viz.

Subcommands:
  scan     pose/PC badcase scan (or --retier-only)
  triage   BEV+camera triage images for REJECT / HIGH+WARN
  layer    legacy layer_metrics scan (feeds global_pc for scan)
  project  single-frame projection sanity check
  video    export MP4 from an OCC scene package

Delivery reject gate remains tools/occ/quality_gate.py inside produce/export.
Examples:
  python tools/occ/qa.py scan --all --workers 6
  python tools/occ/qa.py scan --retier-only
  python tools/occ/qa.py triage --from-metrics --rank-by score
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

_IMPL = Path(__file__).resolve().parent / "_impl"
_CMDS = {
    "scan": _IMPL / "pose_badcase.py",
    "triage": _IMPL / "triage.py",
    "layer": _IMPL / "layer_scan.py",
    "project": _IMPL / "validate_projection.py",
    "video": _IMPL / "scene_video.py",
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help", "help"}:
        print(__doc__.strip(), file=sys.stderr)
        print("\ncommands:", ", ".join(sorted(_CMDS)), file=sys.stderr)
        return 0
    cmd = sys.argv[1]
    if cmd not in _CMDS:
        print(f"unknown command: {cmd}", file=sys.stderr)
        print("commands:", ", ".join(sorted(_CMDS)), file=sys.stderr)
        return 2
    target = _CMDS[cmd]
    sys.argv = [str(target), *sys.argv[2:]]
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
