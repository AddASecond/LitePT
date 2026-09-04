#!/usr/bin/env python3
"""A lane: precompute preds for a materialized clip (agg-stride ∪ export-stride).

Contract: writes {pred_dir}/{ts}_pred.npy (int32, N = lidar points).
See tools/occ/CONTRACTS.md. Production (B) must call this via CLI only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))


def _setup_cuda_env() -> None:
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ.pop("_CUDA_COMPAT_PATH", None)
    os.environ.pop("Path", None)
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if "/usr/lib/x86_64-linux-gnu" not in ld.split(":"):
        head = "/usr/lib/x86_64-linux-gnu"
        cudalib = "/usr/local/cuda/targets/x86_64-linux/lib"
        os.environ["LD_LIBRARY_PATH"] = f"{head}:{cudalib}" + (f":{ld}" if ld else "")
    os.environ["HAMI_DISABLE_WARN"] = "1"
    os.environ["CUDA_MODULE_LOADING"] = "EAGER"
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6;8.9;9.0+PTX"
    try:
        import torch

        torch.cuda.is_available()
    except Exception:
        pass


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
        "--mode",
        choices=["infer", "fake"],
        default="infer",
        help="fake: label everything class 7 (no CUDA)",
    )
    args = ap.parse_args(argv)

    if args.mode == "infer":
        _setup_cuda_env()

    import numpy as np
    import infer_robotruck_mongo_frame as h

    clip = Path(args.clip_dir)
    idx = json.loads((clip / "frames_index.json").read_text())
    tss = [str(e["timestamp"]) for e in idx if e.get("has_lidar")]
    need = list(
        dict.fromkeys(
            tss[:: max(1, args.agg_stride)] + tss[:: max(1, args.stride)][: args.max_frames]
        )
    )

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
