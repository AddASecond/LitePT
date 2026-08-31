#!/usr/bin/env python3
"""Evaluate the production geometry gate on every cached Robotruck clip."""
from __future__ import annotations

import argparse
import importlib.util
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_gate():
    spec = importlib.util.spec_from_file_location(
        "robotruck_quality_gate_scan", ROOT / "tools/robotruck_quality_gate.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def scan_one(args: tuple[str, float]) -> dict:
    path_text, skip_initial_seconds = args
    clip_dir = Path(path_text)
    timestamps = sorted(
        frame.parent.name
        for frame in (clip_dir / "frames").glob("*/frame.json")
        if (frame.parent / "lidar_merge.bin").is_file()
    )
    total_frames = len(timestamps)
    if timestamps and skip_initial_seconds > 0:
        cutoff = int(timestamps[0]) + int(skip_initial_seconds * 1e9)
        timestamps = [ts for ts in timestamps if int(ts) >= cutoff]
    try:
        result = load_gate().assess_clip_geometry(clip_dir, timestamps)
        return {
            "scene": clip_dir.name,
            "clip_id": clip_dir.name.removeprefix("batch_s5_"),
            "stride": 1,
            "total_frames": total_frames,
            "evaluated_frames": len(timestamps),
            "allow_occ": bool(result["allow_occ"]),
            "reasons": result["reasons"],
            "warnings": result["warnings"],
            "geometry_quality": result,
        }
    except Exception as error:
        return {
            "scene": clip_dir.name,
            "clip_id": clip_dir.name.removeprefix("batch_s5_"),
            "stride": 1,
            "total_frames": total_frames,
            "evaluated_frames": len(timestamps),
            "allow_occ": False,
            "reasons": [f"scan_error:{type(error).__name__}:{error}"],
            "warnings": [],
        }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-root", type=Path, default=ROOT / "exp/robotruck/raw_volume_cache")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-initial-seconds", type=float, default=1.0)
    args = ap.parse_args()
    clips = sorted(
        path for path in args.cache_root.glob("batch_s5_*")
        if (path / "frames").is_dir()
    )
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(scan_one, (str(path), args.skip_initial_seconds)): path
            for path in clips
        }
        for number, future in enumerate(as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            state = "KEEP" if row["allow_occ"] else "REJECT"
            print(f"[{number}/{len(clips)}] {state} {row['scene']}", flush=True)
    rows.sort(key=lambda row: row["scene"])
    rejected = [row for row in rows if not row["allow_occ"]]
    report = {
        "schema_version": "robotruck_occ_quality_scan/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cache_root": str(args.cache_root.resolve()),
        "stride": 1,
        "skip_initial_seconds": args.skip_initial_seconds,
        "total_clips": len(rows),
        "saved_clips": len(rows) - len(rejected),
        "filtered_clips": len(rejected),
        "scan_errors": sum(
            any(reason.startswith("scan_error:") for reason in row["reasons"])
            for row in rejected
        ),
        "rejected": rejected,
        "clips": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: report[k] for k in (
        "total_clips", "saved_clips", "filtered_clips", "scan_errors"
    )}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
