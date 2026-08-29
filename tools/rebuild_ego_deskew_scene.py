#!/usr/bin/env python3
"""Rebuild viewer points from nodeskew PCD timestamps and ego poses."""
from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import export_robotruck_occ_scene as exporter
from validate_deskew_reference import read_pcd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("occ_store", ROOT / "tools/store_robotruck_occ_gridfs.py")
store = importlib.util.module_from_spec(spec); spec.loader.exec_module(store)

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--clip",required=True)
    ap.add_argument("--cache-root",default="exp/robotruck/raw_volume_cache")
    ap.add_argument("--scene-root",default="exp/robotruck/occ_scenes")
    args=ap.parse_args()
    cache=Path(args.cache_root)/args.clip; scene=Path(args.scene_root)/args.clip
    index=json.loads((scene/"index.json").read_text())
    rows=[]
    for p in (cache/"frames").glob("*/frame.json"):
        d=json.loads(p.read_text()); e=(d.get("dependency") or {}).get("ego_pose")
        if e and e.get("pose"):
            s=e["header"]["stamp"]; t=int(s["sec"])*1_000_000_000+int(s["nanosec"])
            q=e["pose"]["orientation"]; x=e["pose"]["position"]
            rows.append((t,[x[k] for k in "xyz"],[q[k] for k in ("x","y","z","w")]))
    rows.sort(); tt=np.array([r[0] for r in rows],np.float64)
    pp=np.array([r[1] for r in rows],np.float64); qq=np.array([r[2] for r in rows],np.float64)
    for i in range(1,len(qq)):
        if np.dot(qq[i-1],qq[i])<0: qq[i]*=-1
    def poses(qt):
        qt=np.asarray(qt,np.float64)
        p=np.column_stack([np.interp(qt,tt,pp[:,j]) for j in range(3)])
        q=np.column_stack([np.interp(qt,tt,qq[:,j]) for j in range(4)])
        q/=np.linalg.norm(q,axis=1,keepdims=True); return p,q
    allx=[]; alll=[]; alli=[]
    for e in index["frames"]:
        ts=str(e.get("timestamp") or e["frame_id"]); fd=cache/"frames"/ts
        doc=json.loads((fd/"frame.json").read_text()); sensors=doc["dependency"]["sensors"]
        nd=sensors["lidar_merge_nodeskew"]; pcd=read_pcd(store.resolve_raw(nd["md5"],"lidar"))
        xyz=np.column_stack([pcd[k] for k in ("x","y","z")]).astype(np.float64)
        pos,q=poses(pcd["timestamp"]); world=Rotation.from_quat(q).apply(xyz)+pos
        ref=int((sensors.get("lidar_merge_deskew") or {}).get("timestamp") or ts)
        rp,rq=poses([ref]); corrected=Rotation.from_quat(rq[0]).inv().apply(world-rp[0]).astype(np.float32)
        out=scene/"frames"/ts; lab=np.fromfile(out/"frame_sensor_points_labels.u8.bin",np.uint8)
        lid=pcd["lidar_id"].astype(np.uint8)
        if len(lab)!=len(corrected): raise ValueError(f"{ts}: label/point mismatch")
        corrected.tofile(out/"frame_sensor_points_xyz.f32.bin"); lid.tofile(out/"frame_sensor_points_lidar_id.u8.bin")
        meta=json.loads((out/"meta.json").read_text()); a=meta["assets"]["frame_sensor_points"]
        a.update({"source":"lidar_merge_nodeskew","deskew_method":"ego_pose_per_point_timestamp","deskew_reference_timestamp":ref,"filtering":"none"})
        (out/"meta.json").write_text(json.dumps(meta,indent=2))
        pose=doc["dependency"]["ego_pose"]["pose"]; xm=exporter.sag.transform_points(corrected,exporter.sag.ego_pose_to_T_map_vehicle(pose))
        allx.append(xm); alll.append(lab); alli.append(lid)
    x=np.concatenate(allx); l=np.concatenate(alll); ids=np.concatenate(alli); d=scene/"point_aggregate"; d.mkdir(exist_ok=True)
    x.tofile(d/"xyz_map.f32.bin"); l.tofile(d/"labels.u8.bin"); ids.tofile(d/"lidar_id.u8.bin")
    index["point_aggregate"].update({"source":"lidar_merge_nodeskew","deskew_method":"ego_pose_per_point_timestamp","n":len(x)})
    (scene/"index.json").write_text(json.dumps(index,indent=2))
    print(json.dumps({"clip":args.clip,"frames":len(index["frames"]),"points":len(x),"deskew_method":"ego_pose_per_point_timestamp"}))
    return 0
if __name__=="__main__": raise SystemExit(main())
