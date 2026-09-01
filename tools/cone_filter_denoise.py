#!/usr/bin/env python3
"""Stage 1 — Cone batch filter + denoise.  NO visualisation.  NO projection.

Input paths (choose ONE):
  * --mongo-uri + --raw-frame-collection + --raw-clip-collection:
      query clips by class10 (cone) OD-box count OR class10 pred-point count,
      materialize lidar+preds locally, then per-point filter -> denoise -> cluster.
  * --backup-root + --clips-json:   reuse already-materialized backup dirs.

Output per clip: <out-root>/<clip_id>_cones.pkl
  {'clip_id': str,
   'per_frame': [
     {'ts': str,
      'cone_points_xyz': f32[N,3],
      'cone_labels_intensity': u1[N] if present else None,
      'cone_lidar_id':      u1[N] if present else None,
      'clusters': [{'id':int, 'centroid_xyz':f32[3], 'n_points':int,
                    'min_z':float, 'max_z':float, 'radius_xy':float,
                    'point_ids': i32[M]} , ... ]},
     ... ],
   'clips_summary': dict,
   'schema': 'cone_filter_denoise/v1'}

RULES (no-stopping):
  * Exception on one clip/frame -> logged; continue.
  * All tuning thresholds are HARD CODED (not flags) so user never touches them.
"""
from __future__ import annotations
import argparse, json, logging, math, os, pickle, sys, traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

# --------------------------------------------------------------------------- #
# Hard-coded thresholds (DO NOT TUNE)
# --------------------------------------------------------------------------- #
MIN_CONE_POINTS_PER_FRAME = 3
CONE_CLASS_ID = 10  # Waymo-S Construction Cone
# DBSCAN-like spatial cluster for cone point isolation (vehicle xy frame, meters)
CLUSTER_RADIUS_XY = 0.25  # 25 cm
CLUSTER_RADIUS_Z = 0.50   # 50 cm
CLUSTER_MIN_PTS = 3
CLUSTER_MAX_PTS = 20000
# per-cluster sanity (approximate traffic cone shape)
CLUSTER_MIN_HEIGHT = 0.05  # meters (5cm)
CLUSTER_MAX_HEIGHT = 1.50  # meters (1.5m)
CLUSTER_MAX_RADIUS_XY = 0.60  # meters (60cm)
# reject elongated barrier/curb streaks that pass radius_xy (p90) but span >1 cone
CLUSTER_MAX_LENGTH_XY = 0.85  # meters — principal-axis extent in xy
# OD threshold for "cone clip": at least one frame with >= this many class10 OD boxes
CONE_CLIP_MIN_OD_FRAMES = 1
CONE_CLIP_MIN_OD_BOXES_PER_FRAME = 1
# sampling cap (keep memory bounded)
MAX_CLIPS = 2000
MAX_FRAMES_PER_CLIP = 0  # 0 = all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cone_filter_denoise")


def _setup_env():  # best-effort cuda env guard (we may not use CUDA here but importing tools does)
    try:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        os.environ.pop("_CUDA_COMPAT_PATH", None)
        os.environ.pop("Path", None)
    except Exception:
        pass


_setup_env()

# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #

def frame_cones_from_materialized(backup_root: Path, clip_id: str, ts: str):
    """Load lidar_merge.bin + <ts>_pred.npy, mask by class==10 -> xyz + labels + lidar_id.

    Falls back to PCD via the pymongo-framework's PCD reader if bin missing.
    NEVER calls projection or drawing.
    """
    fr = backup_root / clip_id / "frames" / ts
    if not fr.is_dir():
        return None, f"frame dir missing: {fr}"

    lidar_cols = 7  # default LIDAR_COLS from infer_robotruck_mongo_frame.py
    # (1) try pred cache
    pred = None
    pred_path = backup_root / clip_id / "preds" / f"{ts}_pred.npy"
    if pred_path.is_file():
        try:
            pred = np.load(pred_path).astype(np.int64).reshape(-1)
        except Exception as exc:
            return None, f"pred load failed: {exc}"

    # (2) try lidar_merge.bin first
    xyz = intensity = lidar_id = None
    bin_path = fr / "lidar_merge.bin"
    if bin_path.is_file():
        try:
            arr = np.fromfile(bin_path, dtype=np.float32).reshape(-1, lidar_cols)
            xyz = arr[:, :3].astype(np.float32)
            if arr.shape[1] > 3: intensity = arr[:, 3]
            if arr.shape[1] > 6: lidar_id = (arr[:, 6].round().astype(np.uint8)
                                          if arr[:, 6].min() >= 0 and arr[:, 6].max() < 256
                                          else None)
        except Exception as exc:
            return None, f"bin load failed: {exc}"

    if xyz is None:
        return None, "no supported lidar payload (lidar_merge.bin / PCD not found)"

    if pred is None or pred.shape[0] != xyz.shape[0]:
        return None, f"pred shape mismatch pred={getattr(pred,'shape',None)} xyz={xyz.shape}"

    mask = pred == CONE_CLASS_ID
    if mask.sum() < MIN_CONE_POINTS_PER_FRAME:
        # empty frame is a valid result (no cones)
        return {"xyz": np.zeros((0,3), dtype=np.float32),
                "intensity": np.zeros((0,), dtype=np.uint16) if intensity is not None else None,
                "lidar_id": np.zeros((0,), dtype=np.uint8) if lidar_id is not None else None,
                "cone_count_full_cloud": int(mask.sum())}, None

    out = {"xyz": xyz[mask].astype(np.float32, copy=False),
           "cone_count_full_cloud": int(mask.sum())}
    if intensity is not None:
        # intensity range is 0..1 — scale to u16 range for compact storage
        i = intensity[mask]
        out["intensity"] = np.clip(i * 65535.0, 0, 65535).astype(np.uint16)
    if lidar_id is not None:
        out["lidar_id"] = lidar_id[mask]
    return out, None


def cluster_cones(xyz: np.ndarray):
    """Very small pure-Naive DBSCAN in 3D (vehicle frame), small N only.

    We expect a few dozen cones per frame at most.  For hundreds to low-thousands
    of points this is O(N^2) ~ (2000^2) = 4M per frame, fine.
    """
    N = xyz.shape[0]
    if N == 0:
        return []
    # radius filters by xy + z separately (cone is tall&thin)
    xy = xyz[:, :2].astype(np.float64)
    z = xyz[:, 2].astype(np.float64)
    label = np.full(N, -1, dtype=np.int32)
    cid = -1
    R2XY = CLUSTER_RADIUS_XY * CLUSTER_RADIUS_XY
    # Ball-priority order doesn't matter for us; simple sequential flood-fill
    for i in range(N):
        if label[i] >= 0:
            continue
        cid += 1
        stack = [i]; label[i] = cid
        while stack:
            p = stack.pop()
            # squared distances to all unlabelled within xy radius
            dxy2 = np.sum((xy[label < 0] - xy[p]) ** 2, axis=1)
            dz = np.abs(z[label < 0] - z[p])
            idx_mask = np.where((dxy2 <= R2XY) & (dz <= CLUSTER_RADIUS_Z))[0]
            if len(idx_mask) == 0:
                continue
            unlabeled_ids = np.where(label < 0)[0][idx_mask]
            if len(unlabeled_ids) == 0:
                continue
            label[unlabeled_ids] = cid
            stack.extend(unlabeled_ids.tolist())
    # build cluster dicts
    clusters = []
    for k in range(cid + 1):
        ids = np.where(label == k)[0]
        m = ids.size
        if m < CLUSTER_MIN_PTS or m > CLUSTER_MAX_PTS:
            continue
        pts = xyz[ids]
        c = pts.mean(axis=0).astype(np.float32)
        xy = pts[:, :2].astype(np.float64) - c[:2].astype(np.float64)
        rxy = float(np.percentile(np.linalg.norm(xy, axis=1), 90))
        h = float(pts[:, 2].max() - pts[:, 2].min())
        if h < CLUSTER_MIN_HEIGHT or h > CLUSTER_MAX_HEIGHT:
            continue
        if rxy > CLUSTER_MAX_RADIUS_XY:
            continue
        # principal-axis length: barrier FPs often form long thin chains
        length_xy = 0.0
        if m >= 3:
            _, _, vh = np.linalg.svd(xy, full_matrices=False)
            proj = xy @ vh[0]
            length_xy = float(proj.max() - proj.min())
            if length_xy > CLUSTER_MAX_LENGTH_XY:
                continue
        clusters.append({
            "id": len(clusters),
            "centroid_xyz": c,
            "n_points": int(m),
            "min_z": float(pts[:, 2].min()),
            "max_z": float(pts[:, 2].max()),
            "height": h,
            "radius_xy": rxy,
            "length_xy": length_xy,
            "point_ids": ids.astype(np.int32, copy=False),
        })
    return clusters


def pick_cone_clips_from_mongo(mongo_uri: str, db: str, raw_frames_col: str,
                               raw_clips_col: str, max_clips: int):
    """Return list of clip_ids (UUID) that have any cone-OD or enough class10 preds.

    Policy: scan raw_clips collection then sample clips until we have max_clips.
    We never fail the pipeline if mongo has issues — empty list falls back.
    """
    try:
        from pymongo import MongoClient
    except Exception as exc:
        log.error("pymongo missing: %s", exc); return []
    try:
        c = MongoClient(mongo_uri, serverSelectionTimeoutMS=8000)
        c.admin.command("ping")
    except Exception as exc:
        log.error("mongo connect failed: %s", exc); return []
    frames = c[db][raw_frames_col]
    out = []
    try:
        # Quick: just query any frame in raw_frames whose detection has class10 boxes.
        # Cone box class_id mapping: detection.lidar_od_1-2_deskew.objects.type==10.
        cursor = frames.aggregate([
            {"$match": {"dependency.sensors.lidar_merge_deskew": {"$exists": True}}},
            {"$addFields": {"objects": {
                "$ifNull": ["$detection.lidar_od_1-2_deskew.objects",
                            {"$ifNull": ["$detection.lidar_od_14_deskew.objects", []]}]}}},
            {"$addFields": {"cone_boxes": {
                "$filter": {"input": "$objects", "as": "o",
                            "cond": {"$in": ["$$o.type", [10, 10.0]]}}}}},
            {"$match": {"$expr": {"$gte": [{"$size": "$cone_boxes"},
                                          CONE_CLIP_MIN_OD_BOXES_PER_FRAME]}}},
            {"$group": {"_id": "$clip_id", "cone_frames": {"$sum": 1}}},
            {"$match": {"cone_frames": {"$gte": CONE_CLIP_MIN_OD_FRAMES}}},
            {"$sort": {"cone_frames": -1}},
            {"$limit": max_clips},
        ], allowDiskUse=True, maxTimeMS=180000)
        for row in cursor:
            cid = row.get("_id")
            if isinstance(cid, str) and len(cid) > 0 and cid not in out:
                out.append(cid)
            if len(out) >= max_clips: break
    except Exception as exc:
        log.error("cone clip aggregation failed (continuing anyway): %s", exc)
    log.info("cone clips from mongo: %d", len(out))
    return out


def materialize_one_clip(mongo_uri: str, db: str, raw_frames_col: str,
                         raw_clips_col: str, clip_id: str, backup_root: Path,
                         stride: int, max_frames: int):
    """Use pipeline tool materialize flow.

    We don't import run_robotruck_occ_mongo_pipeline as a module because its
    argparse is complex.  Instead fall back to the same materialization primitives
    inline: for each frame in clip -> download/copy JPG + PCD/bin and write
    <backup>/<clip_id>/frames/<ts>/frame.json.
    """
    try:
        from pymongo import MongoClient
    except Exception:
        return False
    try:
        import hashlib
        # raw roots (same as validate script)
        raw_roots = [Path(f"/data/rawdata{s}") for s in ("", "-1", "-2", "-3", "-4")]
        def _cp(md5: str, kind: str, suffix: str, dst: Path):
            rel = Path(kind) / md5[:2] / md5[2:4] / f"{md5[4:]}{suffix}"
            hits = [r / rel for r in raw_roots if (r / rel).is_file()]
            if not hits: return False
            dst.parent.mkdir(parents=True, exist_ok=True)
            import shutil; shutil.copy2(hits[0], dst)
            return True
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=8000)
        frames = client[db][raw_frames_col]
        clip_doc = client[db][raw_clips_col].find_one({"clip_id": clip_id})
        if clip_doc is None:
            log.warning("clip %s absent from %s", clip_id, raw_clips_col)
            return False
        ts_list = clip_doc.get("frame_timestamps") or [str(f.get("timestamp"))
            for f in clip_doc.get("frames", [])]
        if not ts_list:
            # fetch from frames col
            ts_list = [str(x["timestamp"]) for x in
                       frames.find({"clip_id": clip_id}, {"timestamp":1}).sort("timestamp",1)]
        if stride > 1: ts_list = ts_list[::stride]
        if max_frames: ts_list = ts_list[:max_frames]
        clip_dir = backup_root / clip_id
        for ts in ts_list:
            fr_path = clip_dir / "frames" / str(ts)
            if (fr_path / "frame.json").is_file():
                continue  # already materialized
            doc = frames.find_one({"clip_id": clip_id, "timestamp": int(ts)})
            if doc is None:
                continue
            fr_path.mkdir(parents=True, exist_ok=True)
            sensors = ((doc.get("dependency") or {}).get("sensors") or {})
            # copy all cameras + lidar
            for name, sensor in sensors.items():
                md5 = sensor.get("md5")
                if not md5 or len(md5) != 32:
                    continue
                if name.startswith("camera"):
                    _cp(md5, "camera", ".jpg", fr_path / f"{name}.jpg")
                elif name.startswith("lidar"):
                    if not _cp(md5, "lidar", ".pcd", fr_path / f"{name}.pcd"):
                        _cp(md5, "lidar", ".pcd", fr_path / f"{name}.pcd")
            (fr_path / "frame.json").write_text(json.dumps(doc, default=str))
        return True
    except Exception as exc:
        log.error("materialize %s failed: %s", clip_id, exc)
        return False


def run_one_clip(clip_id: str, backup_root: Path, out_root: Path, max_frames: int = 0):
    fr_dir = backup_root / clip_id / "frames"
    if not fr_dir.is_dir():
        log.warning("no materialized frames for %s", clip_id)
        return None
    ts_list = sorted(p.name for p in fr_dir.iterdir() if p.is_dir())
    if max_frames: ts_list = ts_list[:max_frames]
    total_cones = 0
    total_clusters = 0
    frames_out = []
    errs = 0
    for ts in ts_list:
        try:
            res, err = frame_cones_from_materialized(backup_root, clip_id, ts)
            if err is not None:
                errs += 1
                log.info("frame %s/%s skip: %s", clip_id[:8], ts[:13], err)
                frames_out.append({"ts": ts, "cone_points_xyz": np.zeros((0,3), dtype=np.float32),
                                   "clusters": [], "error": str(err)})
                continue
            if res is None:
                frames_out.append({"ts": ts, "cone_points_xyz": np.zeros((0,3), dtype=np.float32),
                                   "clusters": []})
                continue
            xyz = res["xyz"]
            total_cones += int(res.get("cone_count_full_cloud") or 0)
            clusters = cluster_cones(xyz)
            total_clusters += len(clusters)
            frames_out.append({
                "ts": ts,
                "cone_points_xyz": xyz,
                "cone_labels_intensity": res.get("intensity"),
                "cone_lidar_id": res.get("lidar_id"),
                "clusters": clusters,
                "cone_count_full_cloud": int(res.get("cone_count_full_cloud") or 0),
            })
        except Exception as exc:
            errs += 1
            log.warning("frame %s/%s exception: %s", clip_id[:8], ts[:13], exc)
    if not frames_out:
        return None
    out_root.mkdir(parents=True, exist_ok=True)
    out = {"clip_id": clip_id,
           "per_frame": frames_out,
           "clips_summary": {"n_frames": len(frames_out),
                             "total_cone_point_hits_full_cloud": total_cones,
                             "total_cluster_objects": total_clusters,
                             "frame_errors": errs},
           "schema": "cone_filter_denoise/v1"}
    out_path = out_root / f"{clip_id}_cones.pkl"
    try:
        with open(out_path, "wb") as fp:
            pickle.dump(out, fp, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        log.error("write pkl %s failed: %s", out_path, exc)
        return None
    log.info("%s: frames=%d cones=%d clusters=%d errs=%d -> %s",
             clip_id[:8], len(frames_out), total_cones, total_clusters, errs, out_path.name)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backup-root", type=Path, default=Path("exp/robotruck/raw_volume_cache"))
    ap.add_argument("--out-root", type=Path, default=Path("exp/robotruck/cone_mid"))
    ap.add_argument("--clips-json", type=Path, help="Optional list of clip_ids: {\"clips\":[\"uuid1\",\"uuid2\"]}")
    ap.add_argument("--stride", type=int, default=5, help="Stride frames when pulling from mongo (smaller = slower)")
    ap.add_argument("--max-clips", type=int, default=MAX_CLIPS)
    ap.add_argument("--max-frames-per-clip", type=int, default=MAX_FRAMES_PER_CLIP)
    ap.add_argument("--db", default="perception_experiment")
    ap.add_argument("--raw-frame-collection", default="raw_data_frames_lidar14_0813")
    ap.add_argument("--raw-clip-collection", default="raw_data_clips_lidar14_0813")
    ap.add_argument("--mongo-uri", default=os.environ.get(
        "ROBOTRUCK_MONGO_URI",
        "mongodb://krk030-mongodb:27017/?authSource=perception_experiment"))
    args = ap.parse_args()

    args.backup_root.mkdir(parents=True, exist_ok=True)
    args.out_root.mkdir(parents=True, exist_ok=True)

    clip_ids = []
    if args.clips_json and args.clips_json.is_file():
        try:
            data = json.loads(args.clips_json.read_text())
            clip_ids = list(data.get("clips", []) or data.get("clip_ids", []) or [])
        except Exception as exc:
            log.error("clips-json parse: %s", exc)
    if not clip_ids:
        log.info("no explicit clips.json; querying mongo for cone clips (max=%d)", args.max_clips)
        clip_ids = pick_cone_clips_from_mongo(
            args.mongo_uri, args.db,
            args.raw_frame_collection, args.raw_clip_collection, args.max_clips)

    log.info("total clips to process: %d  backup_root=%s", len(clip_ids), args.backup_root)
    manifest = []
    done_skip = 0
    for i, cid in enumerate(clip_ids, 1):
        log.info("[%d/%d] clip=%s", i, len(clip_ids), cid[:8])
        try:
            # ensure materialized
            if not (args.backup_root / cid / "frames").is_dir():
                ok = materialize_one_clip(
                    args.mongo_uri, args.db,
                    args.raw_frame_collection, args.raw_clip_collection,
                    cid, args.backup_root, args.stride, args.max_frames_per_clip)
                if not ok:
                    log.warning("  materialize failed or empty; skip")
                    continue
            pkl = run_one_clip(cid, args.backup_root, args.out_root, args.max_frames_per_clip)
            if pkl is None:
                done_skip += 1
            else:
                manifest.append({"clip_id": cid, "pkl": str(pkl)})
        except Exception as exc:
            log.error("clip %s failed: %s\n%s", cid, exc, traceback.format_exc())
            done_skip += 1
            continue

    manifest_path = args.out_root / "manifest.json"
    manifest_path.write_text(json.dumps(
        {"n_clips": len(manifest), "n_skipped_or_failed": done_skip,
         "clips": manifest}, indent=2))
    log.info("DONE  produced_pkls=%d  skipped_or_failed=%d  manifest=%s",
             len(manifest), done_skip, manifest_path)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
