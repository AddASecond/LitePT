#!/usr/bin/env python3
"""剩余 REJECT clip 的本地盘导出（绕开 NFS 抖动）。"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
os.environ.setdefault("LITEPT_SKIP_CUDA_WARMUP", "1")

CACHE = ROOT / "exp/robotruck/raw_volume_cache"
SCENES = ROOT / "exp/robotruck/occ_scenes"
WORK = Path("/tmp/reject_work")
PY = ROOT / ".venv_smoke/bin/python"

CLIPS = [
    "5b751fd8-5603-416e-a7cd-b7fc9722c029",
    "1c27295c-417b-485c-a051-bc26b36e5f91",
    "1daa64ef-fab6-430b-93e5-2c568e08c91c",
    "c47ca55d-5883-4bc6-a258-c3b1cb41c550",
    "dd6a1553-7db1-4cc1-b7c3-76feb6d2ab9f",
    "a8a2fa4e-ff6b-4190-94a3-dca6376b5b6d",
    "0038952f-2da1-4eb1-8b40-83b728debfaa",
]


def export_one(uuid: str) -> str:
    dirname = f"batch_s5_{uuid}"
    local = WORK / dirname
    if local.exists():
        shutil.rmtree(local, ignore_errors=True)
    local.parent.mkdir(parents=True, exist_ok=True)
    src = CACHE / dirname
    print(f"  copy {dirname} -> local ...", flush=True)
    shutil.copytree(src, local, symlinks=True)
    import numpy as np
    idxj = json.loads((local / "frames_index.json").read_text())
    pred_dir = local / "preds"
    pred_dir.mkdir(parents=True, exist_ok=True)
    n_fake = 0
    for e in idxj:
        if not e.get("has_lidar"):
            continue
        ts = str(e["timestamp"])
        binp = local / "frames" / ts / "lidar_merge.bin"
        pp = pred_dir / f"{ts}_pred.npy"
        if not binp.is_file():
            continue
        n = binp.stat().st_size // 28
        if pp.is_file():
            try:
                if np.load(pp).shape[0] == n:
                    continue
            except Exception:
                pass
        np.save(pp, np.full(n, 7, np.int32))
        n_fake += 1
    print(f"  fake labels: {n_fake} written (all frames)", flush=True)
    for stale in (local / "static_agg", ROOT / "exp/robotruck/clip_video" / dirname / "static_agg"):
        if stale.is_dir():
            shutil.rmtree(stale, ignore_errors=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    cmd = [str(PY), str(ROOT / "tools/export_robotruck_occ_scene.py"),
           "--clip", dirname,
           "--backup-root", str(WORK),
           "--out-dir", str(WORK / "scenes"),
           "--stride", "4", "--max-frames", "8",
           "--agg-stride", "2", "--static-voxel", "0.2", "--occ-voxel", "0.2",
           "--reuse-pred", "--export-points",
           "--no-geometry-quality-gate", "--device", "cpu",
           "--pred-dir", str(local / "preds")]
    rc = -1
    for attempt in range(3):
        rc = subprocess.run(cmd, cwd=str(ROOT), env=env).returncode
        if rc == 0 and (WORK / "scenes" / dirname / "index.json").is_file():
            break
        print(f"  attempt {attempt+1} rc={rc}, retry in 20s", flush=True)
        time.sleep(20)
    if rc != 0:
        return f"FAIL rc={rc}"
    dst = SCENES / dirname
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    shutil.move(str(WORK / "scenes" / dirname), str(dst))
    if (dst / "index.json").is_file():
        return "OK"
    return "FAIL move"


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "scenes").mkdir(parents=True, exist_ok=True)
    results = {}
    for i, uuid in enumerate(CLIPS):
        try:
            r = export_one(uuid)
        except Exception as exc:
            r = f"FAIL {type(exc).__name__}: {exc}"[:200]
        results[uuid[:8]] = r
        print(f"[{i+1}/{len(CLIPS)}] {uuid[:13]} -> {r}", flush=True)
    print("\nSUMMARY:", json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
