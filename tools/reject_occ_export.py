#!/usr/bin/env python3
"""把 REJECT clips 导出为 OCC viewer 场景：选择性 materialize(≤30帧) → export_robotruck_occ_scene。

用法:
  .venv_smoke/bin/python tools/reject_occ_export.py            # 全部 23 条
  .venv_smoke/bin/python tools/reject_occ_export.py --limit 3  # 先试 3 条
之后:
  .venv_smoke/bin/python tools/occ_viewer/serve.py --scenes-root exp/robotruck/occ_scenes --host 0.0.0.0 --port 8899
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

# HAMI 惩罚期防段错误：本驱动及子进程全部零 CUDA（fake 标签 + --device cpu）
os.environ.setdefault("LITEPT_SKIP_CUDA_WARMUP", "1")

LIST = ROOT / "exp/robotruck/pose_badcase/final_badcase_list.txt"
CACHE = ROOT / "exp/robotruck/raw_volume_cache"
SCENES = ROOT / "exp/robotruck/occ_scenes"
LOG = ROOT / "exp/robotruck/reject_occ_export.log"
PY = ROOT / ".venv_smoke/bin/python"

DB = "perception_experiment"
FRAMES_COL = "raw_data_frames_lidar14_0813"

_spec = importlib.util.spec_from_file_location(
    "occ_pipe", ROOT / "tools/run_robotruck_occ_mongo_pipeline.py")
occ_pipe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(occ_pipe)  # 副作用: _setup_cuda_env()
INGEST, STORE = occ_pipe.INGEST, occ_pipe.STORE


def reject_clips() -> list[tuple[str, str]]:
    rows = []
    for line in LIST.read_text().splitlines():
        cols = line.split("\t")
        if cols and cols[0] == "REJECT":
            rows.append((cols[1], cols[2]))
    return rows


def selective_materialize(db, clip_id: str, every: int, max_frames: int) -> int:
    cache = CACHE / f"batch_s5_{clip_id}"
    frames = list(db[FRAMES_COL].find({"clip_id": clip_id}).sort("timestamp", 1))
    if not frames:
        raise RuntimeError("no frames in mongo")
    sel = frames[::every][:max_frames]
    index = []
    for raw in sel:
        ts = str(raw["timestamp"])
        fdir = cache / "frames" / ts
        md5 = INGEST.raw_lidar_md5(raw) or raw.get("md5")
        if not (fdir / "lidar_merge.bin").is_file():
            if not md5:
                continue
            occ_pipe.link_or_convert_lidar(STORE.resolve_raw(md5, "lidar"),
                                           fdir / "lidar_merge.bin")
        ncams = 0
        sensors = ((raw.get("dependency") or {}).get("sensors") or {})
        for name, sensor in sensors.items():
            if not name.startswith("camera") or not isinstance(sensor, dict):
                continue
            cmd5 = sensor.get("md5")
            if not isinstance(cmd5, str) or not INGEST.MD5_RE.fullmatch(cmd5):
                continue
            tgt = fdir / f"{name}.jpg"
            if not (tgt.exists() or tgt.is_symlink()):
                try:
                    tgt.symlink_to(STORE.resolve_raw(cmd5, "camera"))
                except Exception:
                    continue
            ncams += 1
        if not (fdir / "frame.json").is_file():
            occ_pipe.json_write(fdir / "frame.json", raw)
        index.append({"timestamp": raw["timestamp"], "md5": md5,
                      "has_lidar": True, "n_cameras": ncams})
    occ_pipe.json_write(cache / "frames_index.json", index)
    return len(index)


def fake_labels(clip_dirname: str, agg_stride: int = 2, stride: int = 4,
                max_frames: int = 8) -> int:
    """全部标成静态类7（零CUDA）。静态聚合=全点按pose聚合，鬼影最直观。"""
    import numpy as np
    clip = CACHE / clip_dirname
    idx = json.loads((clip / "frames_index.json").read_text())
    tss = [str(e["timestamp"]) for e in idx if e.get("has_lidar")]
    need = list(dict.fromkeys(tss[::max(1, agg_stride)] + tss[::max(1, stride)][:max_frames]))
    pred_dir = clip / "preds"
    if pred_dir.is_symlink() or (pred_dir.exists() and not pred_dir.is_dir()):
        pred_dir.unlink()
    pred_dir.mkdir(parents=True, exist_ok=True)
    n_done = 0
    for ts in need:
        binp = clip / "frames" / ts / "lidar_merge.bin"
        if not binp.is_file():
            continue
        n = binp.stat().st_size // (4 * 7)
        pp = pred_dir / f"{ts}_pred.npy"
        if pp.is_file():
            try:
                if np.load(pp).shape[0] == n:
                    continue
            except Exception:
                pass
        np.save(pp, np.full(n, 7, np.int32))
        n_done += 1
    return n_done


def export_clip(clip_dirname: str) -> int:
    # 清掉旧 static_agg 缓存，强制按本次 frames_index(30帧) 重建聚合层
    for root in (ROOT / "exp/robotruck/clip_video", CACHE):
        stale = root / clip_dirname / "static_agg"
        if stale.is_dir():
            import shutil
            shutil.rmtree(stale, ignore_errors=True)
            print(f"    cleared stale static_agg: {stale}", flush=True)
    # 阶段1: 内联写假标签（零CUDA零子进程）
    n = fake_labels(clip_dirname)
    print(f"    fake labels written: {n}", flush=True)
    # 阶段2: exporter 纯 reuse，无推理（--device cpu 不碰 CUDA）
    cmd = [str(PY), str(ROOT / "tools/export_robotruck_occ_scene.py"),
           "--clip", clip_dirname,
           "--backup-root", "exp/robotruck/raw_volume_cache",
           "--out-dir", "exp/robotruck/occ_scenes",
           "--stride", "4", "--max-frames", "8",
           "--agg-stride", "2", "--static-voxel", "0.2", "--occ-voxel", "0.2",
           "--reuse-pred", "--export-points",
           "--no-geometry-quality-gate", "--device", "cpu",
           "--pred-dir", str(CACHE / clip_dirname / "preds")]
    import os as _os
    import time
    env = dict(_os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    rc = -1
    for attempt in range(3):
        rc = subprocess.run(cmd, cwd=str(ROOT), env=env).returncode
        if rc == 0:
            break
        print(f"    exporter attempt {attempt+1} rc={rc}, retrying in 10s", flush=True)
        time.sleep(10)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--mat-frames", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="", help="逗号分隔的 clip uuid 前缀，只跑这些")
    args = ap.parse_args()

    clips = reject_clips()
    if args.only:
        keys = [k.strip() for k in args.only.split(",") if k.strip()]
        clips = [c for c in clips if any(c[0].startswith(k) for k in keys)]
    if args.limit:
        clips = clips[:args.limit]
    LOG.write_text(f"export {len(clips)} REJECT scenes @ {SCENES}\n")
    ok, fail = [], []
    for i, (cid, reason) in enumerate(clips):
        dirname = f"batch_s5_{cid}"
        print(f"[{i+1}/{len(clips)}] {cid[:13]} {reason}", flush=True)
        try:
            n = selective_materialize(_get_db(), cid, args.every, args.mat_frames)
            print(f"    materialized {n} frames", flush=True)
            rc = export_clip(dirname)
            if rc == 0 and (SCENES / dirname / "index.json").is_file():
                ok.append(dirname)
                print(f"    EXPORT OK -> {SCENES / dirname}", flush=True)
            else:
                fail.append((dirname, f"rc={rc}"))
                print(f"    EXPORT FAIL rc={rc}", flush=True)
        except Exception as exc:
            fail.append((dirname, str(exc)[:200]))
            print(f"    FAIL: {exc}", flush=True)
            traceback.print_exc()
        with open(LOG, "a") as f:
            f.write(f"{dirname}\t{'OK' if dirname in ok else 'FAIL'}\n")
    print(f"\nDONE ok={len(ok)} fail={len(fail)}")
    for d, e in fail:
        print(f"  FAIL {d}: {e}")
    return 0


_DB = None


def _get_db():
    global _DB
    if _DB is None:
        from pymongo import MongoClient
        c = MongoClient(INGEST.DEFAULT_URI, serverSelectionTimeoutMS=10000)
        _DB = c[DB]
    return _DB


if __name__ == "__main__":
    raise SystemExit(main())
