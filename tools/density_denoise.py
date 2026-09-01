#!/usr/bin/env python3
"""Density-based denoising for cone point clouds.

Pipeline (per frame, vehicle frame xyz):
  1. SOR  — Statistical Outlier Removal: mean distance to k-NN, reject points
            above (mu + alpha * sigma).
  2. DBSCAN — proper Ester et al. density clustering (cKDTree accelerated):
            core points (>= min_pts within eps), border points join via any
            core neighbour, everything else is noise and dropped.
  3. Cone geometry sanity — per-cluster height / xy-radius / principal-axis
            length checks (same thresholds as cone_filter_denoise).

CLI:
  # re-denoise existing stage-1 pkls -> <out-root>/<clip>_cones_denoised.pkl
  .venv_smoke/bin/python tools/density_denoise.py --pkl exp/robotruck/cone_mid/batch_s5_11c2aa2d-2618-45d8-ab28-5cf1529eca84_cones.pkl

  # single cloud npy (N,3) in, (N,) bool kept-mask npy + stats out
  .venv_smoke/bin/python tools/density_denoise.py --npy cloud.npy --npy-out kept.npy

  # synthetic self-test
  .venv_smoke/bin/python tools/density_denoise.py --selftest
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

# --------------------------------------------------------------------------- #
# Hard-coded defaults (CLI flags can override)
# --------------------------------------------------------------------------- #
SOR_K = 16                # neighbours used for SOR mean-distance (excl. self)
SOR_ALPHA = 1.0           # threshold = mu + alpha * sigma
SOR_MIN_POINTS = 20       # skip SOR for tiny clouds (kNN unreliable)

DBSCAN_EPS = 0.30         # metres, 3D neighbourhood radius
DBSCAN_MIN_PTS = 4        # core-point threshold (incl. self)

CLUSTER_MIN_SIZE = 3      # drop clusters smaller than this
CLUSTER_MIN_HEIGHT = 0.05 # cone shape sanity (same as stage-1)
CLUSTER_MAX_HEIGHT = 1.50
CLUSTER_MAX_RADIUS_XY = 0.60
CLUSTER_MAX_LENGTH_XY = 0.85


# --------------------------------------------------------------------------- #
# Step 1 — Statistical Outlier Removal
# --------------------------------------------------------------------------- #
def sor_mask(xyz: np.ndarray, k: int = SOR_K, alpha: float = SOR_ALPHA) -> np.ndarray:
    """True = keep.  Reject points whose mean kNN distance > mu + alpha*sigma."""
    n = xyz.shape[0]
    if n < max(SOR_MIN_POINTS, k + 1):
        return np.ones(n, dtype=bool)
    tree = cKDTree(xyz)
    # query k+1 because the nearest neighbour is the point itself
    d, _ = tree.query(xyz, k=k + 1, workers=-1)
    mean_d = d[:, 1:].mean(axis=1)          # exclude self
    mu, sigma = mean_d.mean(), mean_d.std()
    return mean_d <= (mu + alpha * sigma)


# --------------------------------------------------------------------------- #
# Step 2 — DBSCAN (union-find on core/border edges)
# --------------------------------------------------------------------------- #
def dbscan_labels(xyz: np.ndarray, eps: float = DBSCAN_EPS,
                  min_pts: int = DBSCAN_MIN_PTS) -> np.ndarray:
    """Return labels: cluster id >= 0, noise = -1."""
    n = xyz.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    tree = cKDTree(xyz)
    pairs = tree.query_pairs(r=eps, output_type="ndarray")
    counts = np.bincount(
        np.concatenate([pairs[:, 0], pairs[:, 1], np.arange(n)]),
        minlength=n,
    )  # neighbour counts including self
    core = counts >= min_pts

    # union all edges that touch at least one core point
    parent = np.arange(n, dtype=np.int64)

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:            # path compression
            parent[x], x = root, parent[x]
        return root

    touch_core = core[pairs[:, 0]] | core[pairs[:, 1]]
    for i, j in pairs[touch_core]:
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[rj] = ri

    labels = np.full(n, -1, dtype=np.int32)
    roots = {}
    for i in range(n):
        if not core[i] and counts[i] == 1:  # only self in range -> noise
            continue
        r = find(i)
        if r not in roots:
            roots[r] = len(roots)
        labels[i] = roots[r]
    return labels


# --------------------------------------------------------------------------- #
# Step 3 — cone geometry sanity per cluster
# --------------------------------------------------------------------------- #
def cluster_sanity(xyz: np.ndarray, labels: np.ndarray,
                   min_size: int = CLUSTER_MIN_SIZE) -> tuple[np.ndarray, list[dict]]:
    """Return (kept_mask, cluster_records).  Drops clusters failing cone shape."""
    kept = np.zeros(xyz.shape[0], dtype=bool)
    records = []
    for cid in range(int(labels.max()) + 1 if labels.size else 0):
        ids = np.where(labels == cid)[0]
        if ids.size < min_size:
            continue
        pts = xyz[ids]
        c = pts.mean(axis=0)
        xy = pts[:, :2] - c[:2]
        rxy = float(np.percentile(np.linalg.norm(xy, axis=1), 90))
        height = float(pts[:, 2].max() - pts[:, 2].min())
        length_xy = 0.0
        if ids.size >= 3:
            _, _, vh = np.linalg.svd(xy, full_matrices=False)
            proj = xy @ vh[0]
            length_xy = float(proj.max() - proj.min())
        ok = (CLUSTER_MIN_HEIGHT <= height <= CLUSTER_MAX_HEIGHT
              and rxy <= CLUSTER_MAX_RADIUS_XY
              and length_xy <= CLUSTER_MAX_LENGTH_XY)
        if ok:
            kept[ids] = True
            records.append({
                "cluster_id": cid, "n_points": int(ids.size),
                "centroid_xyz": c.astype(np.float32), "height": height,
                "radius_xy": rxy, "length_xy": length_xy,
                "point_ids": ids.astype(np.int32, copy=False),
            })
    return kept, records


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #
def density_denoise(xyz: np.ndarray, sor_k: int = SOR_K, sor_alpha: float = SOR_ALPHA,
                    eps: float = DBSCAN_EPS, min_pts: int = DBSCAN_MIN_PTS,
                    min_size: int = CLUSTER_MIN_SIZE) -> tuple[np.ndarray, dict]:
    """Return (kept_mask, info).  kept_mask[i] True = point survives denoising."""
    info: dict = {"n_in": int(xyz.shape[0])}
    if xyz.shape[0] == 0:
        info.update({"n_after_sor": 0, "n_clusters": 0, "n_kept": 0, "n_noise": 0})
        return np.zeros(0, dtype=bool), info

    m1 = sor_mask(xyz, sor_k, sor_alpha)
    info["n_after_sor"] = int(m1.sum())
    info["sor_removed"] = info["n_in"] - info["n_after_sor"]

    labels = dbscan_labels(xyz[m1], eps, min_pts)
    kept_sub, records = cluster_sanity(xyz[m1], labels, min_size)

    kept = np.zeros(xyz.shape[0], dtype=bool)
    kept[np.where(m1)[0][kept_sub]] = True
    info.update({
        "n_clusters": len(records),
        "n_kept": int(kept.sum()),
        "n_noise": info["n_in"] - int(kept.sum()),
        "clusters": [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                      for k, v in r.items() if k != "point_ids"} for r in records],
    })
    return kept, info


# --------------------------------------------------------------------------- #
# Self-test: synthetic cones + uniform noise
# --------------------------------------------------------------------------- #
def selftest() -> int:
    rng = np.random.default_rng(0)
    clouds, expect_cones = [], []
    for _ in range(5):                       # 5 synthetic cones
        base = rng.uniform([-30, -30, -1.0], [30, 30, 0.0])
        pts = np.column_stack([
            base[0] + rng.normal(0, 0.06, 30),
            base[1] + rng.normal(0, 0.06, 30),
            base[2] + np.abs(rng.normal(0, 0.18, 30)) * 0.9 + 0.05,
        ])
        clouds.append(pts)
        expect_cones.append(pts.mean(axis=0))
    noise = rng.uniform([-40, -40, -2.0], [40, 40, 2.0], size=(200, 3))
    xyz = np.vstack(clouds + [noise]).astype(np.float32)

    kept, info = density_denoise(xyz)
    # every true cone centroid must have kept points nearby
    hits = 0
    for c in expect_cones:
        d = np.linalg.norm(xyz[kept] - c, axis=1)
        hits += bool((d < 0.5).any())
    noise_kept = int(kept.sum() - sum(r["n_points"] for r in
                                      _iter_records(kept, xyz)))
    print(json.dumps(info, indent=2))
    print(f"cones_recovered={hits}/5  noise_kept~={noise_kept}")
    ok = hits == 5 and info["n_noise"] >= 150
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _iter_records(kept: np.ndarray, xyz: np.ndarray):
    kept_idx = np.where(kept)[0]
    _, records = cluster_sanity(xyz[kept_idx], dbscan_labels(xyz[kept_idx]))
    yield from records


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def run_pkl(pkl_path: Path, out_root: Path | None) -> Path | None:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    tot_in = tot_kept = 0
    for fr in data.get("per_frame", []):
        xyz = fr.get("cone_points_xyz")
        if xyz is None or len(xyz) == 0:
            fr["denoise"] = {"n_in": 0, "n_kept": 0}
            continue
        kept, info = density_denoise(xyz.astype(np.float32))
        fr["cone_points_xyz_denoised"] = xyz[kept].astype(np.float32)
        for key in ("cone_labels_intensity", "cone_lidar_id"):
            if fr.get(key) is not None and len(fr[key]) == len(kept):
                fr[f"{key}_denoised"] = fr[key][kept]
        fr["denoise"] = {k: v for k, v in info.items() if k != "clusters"}
        fr["denoise_clusters"] = info.get("clusters", [])
        tot_in += info["n_in"]; tot_kept += info["n_kept"]
    data["schema"] = "cone_filter_denoise/v2_density"
    data["denoise_summary"] = {"n_in": tot_in, "n_kept": tot_kept,
                               "keep_ratio": round(tot_kept / max(tot_in, 1), 4)}
    out_root = out_root or pkl_path.parent
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / pkl_path.name.replace("_cones.pkl", "_cones_denoised.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"{pkl_path.name}: in={tot_in} kept={tot_kept} "
          f"ratio={data['denoise_summary']['keep_ratio']} -> {out_path}")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pkl", type=Path, nargs="*", help="stage-1 cone pkls to denoise")
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--npy", type=Path, help="single (N,3) cloud")
    ap.add_argument("--npy-out", type=Path, default=None)
    ap.add_argument("--sor-k", type=int, default=SOR_K)
    ap.add_argument("--sor-alpha", type=float, default=SOR_ALPHA)
    ap.add_argument("--eps", type=float, default=DBSCAN_EPS)
    ap.add_argument("--min-pts", type=int, default=DBSCAN_MIN_PTS)
    ap.add_argument("--min-size", type=int, default=CLUSTER_MIN_SIZE)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    if args.npy:
        xyz = np.load(args.npy).astype(np.float32).reshape(-1, 3)
        kept, info = density_denoise(xyz, args.sor_k, args.sor_alpha,
                                     args.eps, args.min_pts, args.min_size)
        if args.npy_out:
            np.save(args.npy_out, kept)
        print(json.dumps(info, indent=2))
        return 0

    if args.pkl:
        for p in args.pkl:
            run_pkl(p, args.out_root)
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
