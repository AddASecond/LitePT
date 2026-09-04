#!/usr/bin/env python3
"""Infer: precompute preds for a materialized clip (agg-stride ∪ export-stride frames)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cuda_env import setup_cuda_env
from paths import ROOT, ensure_import_path

ensure_import_path(tools=True)
# fake mode must not touch CUDA; infer warms up via cuda_env
_will_infer = "--mode" not in sys.argv or "--mode" in sys.argv and (
    len(sys.argv) <= sys.argv.index("--mode") + 1
    or sys.argv[sys.argv.index("--mode") + 1] == "infer"
)
setup_cuda_env(warmup=_will_infer)

import numpy as np
import infer_robotruck_mongo_frame as h


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip-dir", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--agg-stride", type=int, default=2)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=8)
    ap.add_argument("--config-file", default="configs/waymo/semseg-litept-small-v1m1.py")
    ap.add_argument(
        "--weight",
        default="checkpoints/waymo-semseg-litept-small-v1m1/model/model_best.pth",
    )
    ap.add_argument("--grid-size", type=float, default=0.05)
    ap.add_argument(
        "--mode", choices=["infer", "fake"], default="infer",
        help="fake: label everything class 7 (no CUDA)",
    )
    args = ap.parse_args(argv)

    clip = Path(args.clip_dir)
    idx = json.loads((clip / "frames_index.json").read_text())
    tss = [str(e["timestamp"]) for e in idx if e.get("has_lidar")]
    need = list(dict.fromkeys(
        tss[::max(1, args.agg_stride)] + tss[::max(1, args.stride)][: args.max_frames]
    ))

    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    todo = []
    for ts in need:
        binp = clip / "frames" / ts / "lidar_merge.bin"
        pp = pred_dir / f"{ts}_pred.npy"
        if not binp.is_file():
            continue
        n = binp.stat().st_size // (4 * len(h.LIDAR_COLS))
        if pp.is_file():
            try:
                if np.load(pp).shape[0] == n:
                    continue
            except Exception:
                pass
        todo.append((ts, n))
    print(f"pred_warmup: need={len(need)} todo={len(todo)}", flush=True)
    if args.mode == "fake":
        for ts, n in todo:
            np.save(pred_dir / f"{ts}_pred.npy", np.full(n, 7, np.int32))
        print(f"  fake labels written: {len(todo)}", flush=True)
        return 0
    if not todo:
        return 0

    model, _ = h.load_segmentor(ROOT / args.config_file, ROOT / args.weight, "cuda")
    for i, (ts, n) in enumerate(todo):
        pts = h.load_lidar_bin(
            clip / "frames" / ts / "lidar_merge.bin", num_cols=len(h.LIDAR_COLS)
        )
        coord = pts[:, :3].astype(np.float32)
        strength = np.tanh(pts[:, 3].reshape(-1, 1) / 255.0).astype(np.float32)
        pred = h.infer_frame(model, coord, strength, "cuda", args.grid_size)
        np.save(pred_dir / f"{ts}_pred.npy", pred.astype(np.int32))
        print(f"  [{i+1}/{len(todo)}] {ts} N={len(coord)} pred={pred.shape[0]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
