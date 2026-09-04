#!/usr/bin/env python3
"""为 materialized clip 批量预生成 pred（单 CUDA 进程，避免 HAMI 冲突段错误）。

只计算 exporter 实际需要的帧并集:
  static agg: frames_index[::agg-stride]
  导出帧:     frames_index[::stride][:max-frames]
已有且形状匹配的 pred 跳过。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_OCC = Path(__file__).resolve().parent
for path in (_OCC, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cuda_env import setup_cuda_env

setup_cuda_env(warmup=False)

_will_infer = "--mode infer" in " ".join(sys.argv) or "--mode" not in " ".join(sys.argv)
if _will_infer and os.environ.get("LITEPT_SKIP_CUDA_WARMUP") != "1":
    try:
        import torch as _torch
        _ = _torch.cuda.is_available()
    except Exception:
        pass

import numpy as np
import infer_robotruck_mongo_frame as h


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-dir", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--agg-stride", type=int, default=2)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=8)
    ap.add_argument("--config-file", default="configs/waymo/semseg-litept-small-v1m1.py")
    ap.add_argument("--weight",
                    default="checkpoints/waymo-semseg-litept-small-v1m1/model/model_best.pth")
    ap.add_argument("--grid-size", type=float, default=0.05)
    ap.add_argument("--mode", choices=["infer", "fake"], default="infer",
                    help="fake: 全部标成静态类7（零CUDA，用于坏case几何查验）")
    args = ap.parse_args()

    clip = Path(args.clip_dir)
    idx = json.loads((clip / "frames_index.json").read_text())
    tss = [str(e["timestamp"]) for e in idx if e.get("has_lidar")]
    agg_ts = tss[::max(1, args.agg_stride)]
    exp_ts = tss[::max(1, args.stride)][: args.max_frames]
    need = list(dict.fromkeys(agg_ts + exp_ts))

    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    todo = []
    for ts in need:
        binp = clip / "frames" / ts / "lidar_merge.bin"
        pp = pred_dir / f"{ts}_pred.npy"
        if not binp.is_file():
            continue
        n = (binp.stat().st_size // (4 * len(h.LIDAR_COLS)))
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

    device = "cuda"
    model, _ = h.load_segmentor(ROOT / args.config_file, ROOT / args.weight, device)
    for i, (ts, n) in enumerate(todo):
        pts = h.load_lidar_bin(clip / "frames" / ts / "lidar_merge.bin",
                               num_cols=len(h.LIDAR_COLS))
        coord = pts[:, :3].astype(np.float32)
        strength = np.tanh(pts[:, 3].reshape(-1, 1) / 255.0).astype(np.float32)
        pred = h.infer_frame(model, coord, strength, device, args.grid_size)
        np.save(pred_dir / f"{ts}_pred.npy", pred.astype(np.int32))
        print(f"  [{i+1}/{len(todo)}] {ts} N={len(coord)} pred={pred.shape[0]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
