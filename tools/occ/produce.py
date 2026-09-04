#!/usr/bin/env python3
"""OCC delivery CLI — one entry for production runners.

Subcommands:
  export   export scene OCC package (quality gate ON by default)
  pipeline materialize → export → store one clip
  batch    lidar14 tagged batch → pipeline → GSS publish
  inproc   HAMI-safe multi-clip export+store in one process
  warmup   precompute pred caches
  store    content-addressed store + mongo ingest

Examples:
  python tools/occ/produce.py export --clip ... --backup-root ...
  python tools/occ/produce.py pipeline --raw-frame-collection ... --clip-id ...
  python tools/occ/produce.py batch --stride 5 --write
  source .cuda_env.sh && python tools/occ/produce.py inproc --clips-json ...
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

_IMPL = Path(__file__).resolve().parent / "_impl"
_ROOT_SCRIPTS = {
    "export": _IMPL / "export_scene.py",
    "store": _IMPL / "store.py",
    "pipeline": _IMPL / "pipeline.py",
    "batch": _IMPL / "batch.py",
    "inproc": _IMPL / "inproc.py",
    "warmup": _IMPL / "warmup.py",
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help", "help"}:
        print(__doc__.strip(), file=sys.stderr)
        print("\ncommands:", ", ".join(sorted(_ROOT_SCRIPTS)), file=sys.stderr)
        return 0
    cmd = sys.argv[1]
    if cmd not in _ROOT_SCRIPTS:
        print(f"unknown command: {cmd}", file=sys.stderr)
        print("commands:", ", ".join(sorted(_ROOT_SCRIPTS)), file=sys.stderr)
        return 2
    target = _ROOT_SCRIPTS[cmd]
    sys.argv = [str(target), *sys.argv[2:]]
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
