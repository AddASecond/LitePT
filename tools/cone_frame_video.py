#!/usr/bin/env python3
"""锥桶 case 逐帧视频：BEV + 7 路相机投影（与已确认的 PNG 预览布局一致）。

完全复用 cone_frame_preview.py（其投影又复用 render_robotruck_clip_video.py 管线）。

用法:
  .venv_smoke/bin/python tools/cone_frame_video.py [--stride 5] [--max-frames 0] [--fps 10]
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BACKUP = ROOT / "exp/robotruck/raw_volume_cache"
OUT = ROOT / "exp/robotruck/cone_videos_v2"

GOOD_CLIPS = [
    "batch_s5_11c2aa2d-2618-45d8-ab28-5cf1529eca84",
    "batch_s5_6bc60101-e684-438b-9983-b74c1cc5fe2b",
    "batch_s5_f1683bae-a2a6-496a-b9a7-9cc2ccfbc3e0",
    "batch_s5_72ebcd30-693f-417e-855d-ae66447d2020",
    "batch_s5_9d17f68b-3594-4412-b39e-deb7e7250de9",
    "batch_s5_701ff460-2b29-4433-b492-39e7ae2310b6",
    "batch_s5_c5d9f4b4-e410-4241-8e46-f5653db90f5b",
]


def load_preview_module():
    path = ROOT / "tools" / "cone_frame_preview.py"
    spec = importlib.util.spec_from_file_location("cfp_video", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfp = load_preview_module()
rrv = cfp.rrv


def render_frame(clip_dir: Path, ts: str, pose_samples, frame_idx: int) -> np.ndarray | None:
    """单帧 → 3200x900 BGR（BEV + 7 相机，2x4 网格）。无 pred 返回 None。"""
    fr = clip_dir / "frames" / ts
    pred_path = clip_dir / "preds" / f"{ts}_pred.npy"
    if not (fr / "lidar_merge.bin").is_file() or not pred_path.is_file():
        return None
    arr = np.fromfile(fr / "lidar_merge.bin", dtype=np.float32).reshape(-1, 7)
    xyz = arr[:, :3]
    pred = np.load(pred_path).astype(np.int32).reshape(-1)
    cone_mask = pred == cfp.CONE_ID
    cone_idx = np.where(cone_mask)[0]
    clusters = cfp.cone_clusters_for_frame(clip_dir.name, ts, xyz[cone_mask])

    meta = rrv.json.loads((fr / "frame.json").read_text())
    sensors = meta["dependency"]["sensors"]
    clusters = cfp.filter_clusters_by_rgb(
        clusters, sensors, pose_samples, int(ts), fr,
    )
    clustered_mask, orphan_mask, centroids = cfp.masks_from_clusters(
        len(xyz), cone_idx, clusters,
    )
    highlight_mask = cfp.sparse_cluster_points(xyz, cone_idx, clusters)

    panels = [cfp.bev_panel(xyz, clustered_mask, orphan_mask, ts, clip_dir.name)]
    for cam_name in rrv.CAM_ORDER:
        img_path = fr / f"{cam_name}.jpg"
        if not img_path.is_file() or cam_name not in sensors:
            continue
        img_bgr = cv2.cvtColor(np.array(rrv.Image.open(img_path).convert("RGB")), cv2.COLOR_RGB2BGR)
        panel = cfp.cam_panel(
            img_bgr, xyz, pred, highlight_mask, orphan_mask, centroids, sensors[cam_name],
            pose_samples, int(ts), cam_name, frame_idx,
        )
        panels.append(panel)  # BGR

    while len(panels) % 4:
        panels.append(np.zeros_like(panels[0]))
    rows = [np.hstack(panels[i:i + 4]) for i in range(0, len(panels), 4)]
    return np.vstack(rows)  # BGR


def run_clip(clip: str, stride: int, max_frames: int, fps: float) -> tuple[int, str, int]:
    clip_dir = BACKUP / clip
    all_ts = rrv.list_clip_frames(clip_dir)
    sel = all_ts[::stride]
    if max_frames > 0:
        sel = sel[:max_frames]
    pose_samples = rrv.build_clip_pose_samples(clip_dir, all_ts)

    OUT.mkdir(parents=True, exist_ok=True)
    out_mp4 = OUT / f"cone_v2__{clip[9:17]}.mp4"
    writer = None
    n_written = n_skipped = 0
    for i, ts in enumerate(sel):
        try:
            frame = render_frame(clip_dir, ts, pose_samples, i)
        except Exception as exc:
            print(f"  [{clip[9:17]}] frame {ts} error: {exc}", flush=True)
            n_skipped += 1
            continue
        if frame is None:
            n_skipped += 1
            continue
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (w, h))
            if not writer.isOpened():
                return 3, str(out_mp4), 0
        writer.write(frame)
        n_written += 1
        if n_written % 10 == 0:
            print(f"  [{clip[9:17]}] {n_written}/{len(sel)} frames", flush=True)
    if writer is not None:
        writer.release()
    return (0 if n_written else 2), str(out_mp4), n_written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--fps", type=float, default=10.0)
    args = ap.parse_args()

    ok = 0
    for clip in GOOD_CLIPS:
        print(f"[{clip[9:17]}] start", flush=True)
        rc, out, n = run_clip(clip, args.stride, args.max_frames, args.fps)
        print(f"[{clip[9:17]}] rc={rc} frames={n} -> {out}", flush=True)
        ok += rc == 0
    print(f"DONE ok={ok}/{len(GOOD_CLIPS)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
