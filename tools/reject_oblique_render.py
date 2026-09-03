#!/usr/bin/env python3
"""REJECT clips — 高分辨率斜视聚合图，供人工一眼确认 pose/点云分层。

相对旧版增强：
  - 完整 bag_name + clip_id（不再截断）
  - 中文文字解释「为什么 REJECT、图上该看什么」
  - 锚点优先用 v2 anomaly_centers（否则退回最大 |d2| 帧）
  - 输出 PNG + index.md 对照表

用法:
  .venv_smoke/bin/python tools/reject_oblique_render.py
  .venv_smoke/bin/python tools/reject_oblique_render.py --limit 2
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import textwrap
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

LIST = ROOT / "exp/robotruck/pose_badcase/final_badcase_list.json"
METRICS = ROOT / "exp/robotruck/pose_badcase/v2_metrics.json"
CACHE = ROOT / "exp/robotruck/raw_volume_cache"
OUTDIR = ROOT / "exp/robotruck/reject_oblique_hires"

MONGO_URI = os.environ.get(
    "ROBOTRUCK_MONGO_URI",
    "mongodb://krk030-mongodb:27017/?authSource=perception_experiment",
)
DB = "perception_experiment"
CLIPS_COL = "raw_data_clips_lidar14_0813"

N_AGG = 30
RADIUS_M = 60.0
ELEV_DEG = 45.0
CAM_DIST_M = 95.0
FOV_DEG = 70.0
IMG_W, IMG_H = 3200, 1800
PTS_PER_FRAME = 25000
POINT_RADIUS = 1
PANEL_H = 340  # top text panel


def _load_layer_scan():
    spec = importlib.util.spec_from_file_location("layer_scan", ROOT / "tools/layer_scan.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _waymo_colors_bgr() -> np.ndarray:
    spec = importlib.util.spec_from_file_location("viz", ROOT / "visualize.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rgb = (mod.WAYMO_COLORS * 255.0).astype(np.uint8)
    return rgb[:, ::-1].copy()


def load_reject_rows() -> list[dict]:
    rows = json.loads(LIST.read_text())
    return [r for r in rows if r.get("tier") == "REJECT"]


def load_metrics_by_id() -> dict[str, dict]:
    if not METRICS.is_file():
        return {}
    return {m["clip_id"]: m for m in json.loads(METRICS.read_text())}


def fetch_bag_names(clip_ids: list[str]) -> dict[str, str]:
    from pymongo import MongoClient

    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
    out = {}
    for doc in client[DB][CLIPS_COL].find(
        {"clip_id": {"$in": clip_ids}},
        {"clip_id": 1, "bag_name": 1, "clip_name": 1},
    ):
        out[doc["clip_id"]] = {
            "bag_name": doc.get("bag_name") or "",
            "clip_name": doc.get("clip_name") or "",
        }
    return out


def explain_flags(flags: list[str], row: dict) -> list[str]:
    """Human-readable explanation (ASCII for OpenCV overlay; Chinese also in index.md)."""
    lines = []
    if any(f.startswith("GATE:high_confidence_pose_drift") for f in flags):
        lines.append(
            "Why REJECT: kinematics look vehicle-feasible, but along-track pose-shift "
            "search consistently prefers non-zero displacement (systematic pose drift)."
        )
        lines.append(
            "Look for: parallel double-images / lateral ghosting of guardrail, curb, poles "
            f"(gate shift ~ {_fmt(row.get('med_shift') or 0.7)} m)."
        )
    elif any(f.startswith("GATE:layering_with_pose_inconsistency") for f in flags):
        lines.append(
            "Why REJECT: lidar height-layering AND cross-frame pose misalignment both fire."
        )
        lines.append(
            "Look for: road/guardrail appearing as stacked thin sheets that never coincide."
        )
    elif any(f.startswith("PC_EXTREME") for f in flags):
        lines.append(
            "Why REJECT: at an infeasible/jump window, cross-frame cloud alignment error is extreme "
            f"(anom_pc={_fmt(row.get('max_anom_pc'))} m, ratio={_fmt(row.get('pc_ratio'))})."
        )
        lines.append(
            "Look for: same physical structure copied to multiple places near the anchor (ghosting)."
        )
    elif any(f.startswith("PC_CONFIRMED") for f in flags):
        lines.append(
            "Why REJECT: anomaly-window PC mismatch is high AND global uniform-window pc_p20 "
            f"is also elevated (gpc={_fmt(row.get('global_pc_p20'))} m) — not a single-frame spike."
        )
        lines.append(
            "Look for: longitudinal/vertical ghosting along road edges across the aggregate."
        )
    else:
        lines.append(f"Why REJECT: {','.join(flags)}")
        lines.append(
            "Look for: static structure ghosting/layering; a clean clip should show one crisp layer."
        )

    lines.append(
        f"Kinematics: feasible={row.get('feasible')}  n_infeas={row.get('n_infeas')}  "
        f"max|d2|={_fmt(row.get('max_d2'))}m  max_accel={_fmt(row.get('max_accel'))} m/s^2"
    )
    return lines


def explain_flags_zh(flags: list[str], row: dict) -> list[str]:
    """Chinese copy for markdown index."""
    lines = []
    if any(f.startswith("GATE:high_confidence_pose_drift") for f in flags):
        lines.append(
            "判定：轨迹运动学看似可达，但沿行驶方向 pose 平移搜索多段一致偏向非零位移 → 系统性漂移。"
        )
        lines.append(
            f"图上应看：护栏/路缘/杆状物是否平行双影或横向错层（shift≈{_fmt(row.get('med_shift') or 0.7)}m）。"
        )
    elif any(f.startswith("GATE:layering_with_pose_inconsistency") for f in flags):
        lines.append("判定：多雷达高度分层与跨帧 pose 对齐失败同时成立。")
        lines.append("图上应看：路面/护栏是否呈上下多层薄片，静态结构无法重合。")
    elif any(f.startswith("PC_EXTREME") for f in flags):
        lines.append(
            f"判定：在轨迹不可达/跳变窗口，跨帧点云对齐误差极大"
            f"（anom_pc={_fmt(row.get('max_anom_pc'))}m，ratio={_fmt(row.get('pc_ratio'))}）。"
        )
        lines.append("图上应看：锚点附近同一物理结构被复制到多个位置（明显鬼影）。")
    elif any(f.startswith("PC_CONFIRMED") for f in flags):
        lines.append(
            f"判定：诡异窗口点云错位抬升，且全局均匀窗 pc_p20 也抬升"
            f"（gpc={_fmt(row.get('global_pc_p20'))}m）→ 非单帧尖刺误报。"
        )
        lines.append("图上应看：道路边缘/护栏纵向或垂向重影，整段聚合发糊。")
    else:
        lines.append(f"判定：{','.join(flags)}")
        lines.append("图上应看：静态结构是否重影/分层。")
    return lines


def _fmt(v) -> str:
    if v is None:
        return "NA"
    try:
        return f"{float(v):.3g}"
    except Exception:
        return str(v)


def worst_jump_index(frames: list) -> tuple[int, float, float]:
    valid_i = [i for i, f in enumerate(frames) if f[1] is not None]
    if len(valid_i) < 3:
        mid = valid_i[len(valid_i) // 2] if valid_i else 0
        return mid, 0.0, 0.0
    P = np.array([frames[i][1] for i in valid_i], np.float64)
    T = np.array([frames[i][0] for i in valid_i], np.int64)
    dt = np.diff(T).astype(np.float64) / 1e9
    dt = np.where(dt <= 0, np.nan, dt)
    d2 = P[2:] - 2 * P[1:-1] + P[:-2]
    d2_norm = np.linalg.norm(d2, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        accel = d2_norm / (dt[1:] * dt[:-1])
    accel = np.where(np.isfinite(accel), accel, -1.0)
    k = int(np.argmax(d2_norm))
    wi_valid = k + 1
    return int(valid_i[wi_valid]), float(d2_norm[k]), float(accel[k])


def pick_anchor(frames: list, metrics: dict | None) -> tuple[int, float, float, str]:
    """Prefer first v2 anomaly center (valid-local → full index), else worst |d2|."""
    wi_d2, d2_m, accel = worst_jump_index(frames)
    if metrics and metrics.get("anomaly_centers"):
        valid_i = [i for i, f in enumerate(frames) if f[1] is not None]
        c0 = int(metrics["anomaly_centers"][0])
        if 0 <= c0 < len(valid_i):
            return int(valid_i[c0]), d2_m, accel, "anomaly_center"
    return wi_d2, d2_m, accel, "worst_d2"


def select_neighbor_indices(n_frames: int, worst_i: int, n_agg: int = N_AGG) -> list[int]:
    half = n_agg // 2
    lo = max(0, worst_i - half)
    hi = min(n_frames, lo + n_agg)
    lo = max(0, hi - n_agg)
    return list(range(lo, hi))


def heading_at(frames: list, worst_i: int) -> float:
    pts = []
    for j in range(max(0, worst_i - 3), min(len(frames), worst_i + 4)):
        if frames[j][1] is not None:
            pts.append(frames[j][1][:2])
    if len(pts) < 2:
        return 0.0
    d = np.asarray(pts[-1]) - np.asarray(pts[0])
    if np.linalg.norm(d) < 1e-3:
        return 0.0
    return float(math.atan2(d[1], d[0]))


def load_pred_labels(clip_id: str, ts: str, n_pts: int) -> np.ndarray | None:
    pred = CACHE / f"batch_s5_{clip_id}" / "preds" / f"{ts}_pred.npy"
    if not pred.is_file():
        return None
    try:
        lab = np.load(pred).astype(np.int32).reshape(-1)
    except Exception:
        return None
    if lab.shape[0] != n_pts:
        return None
    return lab


def aggregate_window(
    ls,
    clip_id: str,
    frames: list,
    idxs: list[int],
    center_xy: np.ndarray,
    radius: float,
    color_mode: str,
    pts_per_frame: int,
) -> tuple[np.ndarray, np.ndarray]:
    waymo_bgr = _waymo_colors_bgr() if color_mode == "semantic" else None
    chunks_xyz, chunks_col = [], []
    for i in idxs:
        ts, t, R, md5 = frames[i]
        if t is None or R is None or md5 is None:
            continue
        try:
            xyz = ls.load_cloud(clip_id, str(ts), md5)
        except Exception:
            continue
        if len(xyz) == 0:
            continue
        labels = load_pred_labels(clip_id, str(ts), len(xyz)) if color_mode == "semantic" else None
        if len(xyz) > pts_per_frame:
            step = max(1, len(xyz) // pts_per_frame)
            sel = np.arange(0, len(xyz), step)[:pts_per_frame]
            xyz = xyz[sel]
            if labels is not None:
                labels = labels[sel]
        xyz_map = xyz @ R.T + t
        dxy = xyz_map[:, :2] - center_xy.reshape(1, 2)
        keep = np.sum(dxy * dxy, axis=1) <= radius * radius
        xyz_map = xyz_map[keep]
        if len(xyz_map) == 0:
            continue
        if color_mode == "semantic" and labels is not None:
            lab = labels[keep]
            col = np.full((len(lab), 3), 40, np.uint8)
            ok = (lab >= 0) & (lab < len(waymo_bgr))
            col[ok] = waymo_bgr[lab[ok]]
        else:
            col = np.zeros((len(xyz_map), 3), np.uint8)
        chunks_xyz.append(xyz_map.astype(np.float64))
        chunks_col.append(col)
    if not chunks_xyz:
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
    xyz_all = np.concatenate(chunks_xyz)
    col_all = np.concatenate(chunks_col)
    if color_mode != "semantic":
        z = xyz_all[:, 2]
        z0 = float(np.percentile(z, 5))
        z1 = float(np.percentile(z, 95))
        if z1 <= z0 + 1e-3:
            z1 = z0 + 1.0
        tnorm = np.clip((z - z0) / (z1 - z0), 0.0, 1.0)
        gray = (tnorm * 255).astype(np.uint8).reshape(-1, 1)
        col_all = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO).reshape(-1, 3)
    return xyz_all, col_all


def perspective_project(
    xyz: np.ndarray,
    eye: np.ndarray,
    lookat: np.ndarray,
    world_up: np.ndarray,
    fov_v_deg: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = lookat - eye
    forward = forward / max(1e-9, np.linalg.norm(forward))
    right = np.cross(forward, world_up)
    rn = np.linalg.norm(right)
    if rn < 1e-9:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / rn
    up = np.cross(right, forward)
    up = up / max(1e-9, np.linalg.norm(up))
    rel = xyz - eye.reshape(1, 3)
    x = rel @ right
    y = rel @ up
    z = rel @ forward
    f = 1.0 / math.tan(math.radians(fov_v_deg) * 0.5)
    aspect = width / float(height)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = (x / z) * (f / aspect)
        v = (y / z) * f
    u = (u * 0.5 + 0.5) * width
    v = (-v * 0.5 + 0.5) * height
    return u, v, z


def render_image(
    xyz: np.ndarray,
    colors_bgr: np.ndarray,
    center: np.ndarray,
    heading: float,
    elev_deg: float,
    cam_dist: float,
    fov_deg: float,
    width: int,
    height: int,
) -> np.ndarray:
    fwd = np.array([math.cos(heading), math.sin(heading), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    e = math.radians(elev_deg)
    eye = center - fwd * (cam_dist * math.cos(e)) + up * (cam_dist * math.sin(e))
    lookat = center.copy()
    img = np.full((height, width, 3), 18, np.uint8)
    if len(xyz) == 0:
        return img
    u, v, z = perspective_project(xyz, eye, lookat, up, fov_deg, width, height)
    ok = (z > 1.0) & np.isfinite(u) & np.isfinite(v)
    ok &= (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not ok.any():
        return img
    u_i = np.clip(np.rint(u[ok]).astype(np.int32), 0, width - 1)
    v_i = np.clip(np.rint(v[ok]).astype(np.int32), 0, height - 1)
    z_ok = z[ok]
    col = colors_bgr[ok]
    order = np.argsort(-z_ok)
    zbuf = np.full((height, width), np.inf, np.float64)
    for ui, vi, zz, c in zip(u_i[order], v_i[order], z_ok[order], col[order]):
        if zz >= zbuf[vi, ui]:
            continue
        zbuf[vi, ui] = zz
        if POINT_RADIUS <= 0:
            img[vi, ui] = c
        else:
            cv2.circle(img, (int(ui), int(vi)), POINT_RADIUS, (int(c[0]), int(c[1]), int(c[2])), -1)
    return img


def draw_text_panel(width: int, lines: list[str], panel_h: int = PANEL_H) -> np.ndarray:
    """Dark panel with wrapped white/yellow text (full IDs readable)."""
    panel = np.full((panel_h, width, 3), 24, np.uint8)
    y = 28
    for i, line in enumerate(lines):
        # wrap long lines to ~140 chars at 0.7 scale on 3200px
        chunks = textwrap.wrap(line, width=120) or [""]
        color = (0, 220, 255) if i < 2 else (230, 230, 230)
        for chunk in chunks:
            cv2.putText(
                panel, chunk, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 3, cv2.LINE_AA
            )
            cv2.putText(
                panel, chunk, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 1, cv2.LINE_AA
            )
            y += 28
            if y > panel_h - 12:
                return panel
    return panel


def render_clip(
    ls,
    row: dict,
    meta: dict,
    metrics: dict | None,
    *,
    elev: float,
    radius: float,
    n_agg: int,
    color_mode: str,
    out_path: Path,
) -> dict:
    clip_id = row["clip_id"]
    bag_name = meta.get("bag_name") or "UNKNOWN_BAG"
    flags = row.get("flags") or []
    reason = ",".join(flags)

    recs = ls.fetch_records([clip_id])
    frames = recs[0]["frames"]
    if len(frames) < 5:
        return {"status": "no_frames", "n_frames": len(frames)}

    wi, d2_m, accel, anchor_src = pick_anchor(frames, metrics)
    if frames[wi][1] is None:
        return {"status": "worst_has_no_pose", "worst_i": wi}

    idxs = select_neighbor_indices(len(frames), wi, n_agg=n_agg)
    idxs = [i for i in idxs if frames[i][1] is not None and frames[i][3] is not None]
    if len(idxs) < 3:
        return {"status": "too_few_neighbors", "worst_i": wi, "n_sel": len(idxs)}

    center = np.asarray(frames[wi][1], dtype=np.float64)
    head = heading_at(frames, wi)
    xyz, col = aggregate_window(
        ls, clip_id, frames, idxs, center[:2], radius, color_mode, PTS_PER_FRAME,
    )
    view = render_image(
        xyz, col, center, head, elev, CAM_DIST_M, FOV_DEG, IMG_W, IMG_H - PANEL_H,
    )

    explain = explain_flags(flags, {**row, **(metrics or {})})
    panel_lines = [
        f"bag_name: {bag_name}",
        f"clip_id:  {clip_id}",
        f"REJECT  flags: {reason}",
        *explain,
        (
            f"view: anchor=f{wi}/{len(frames)-1} ({anchor_src})  |d2|={d2_m:.2f}m  "
            f"accel={accel:.1f}  agg={len(idxs)}frames  ±{radius:.0f}m  elev={elev:.0f}°  "
            f"pts={len(xyz):,}  heading={math.degrees(head):.1f}°  "
            f"perspective oblique along-road"
        ),
    ]
    panel = draw_text_panel(IMG_W, panel_lines, PANEL_H)
    img = np.vstack([panel, view])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return {
        "status": "ok",
        "bag_name": bag_name,
        "clip_id": clip_id,
        "worst_i": wi,
        "anchor_src": anchor_src,
        "d2_m": d2_m,
        "accel": accel,
        "n_frames_clip": len(frames),
        "n_agg": len(idxs),
        "n_points": int(len(xyz)),
        "heading_deg": math.degrees(head),
        "flags": flags,
        "explain": explain,
        "out": str(out_path),
    }


def write_index_md(summary: list[dict], out_dir: Path) -> None:
    lines = [
        "# REJECT clip 斜视聚合可视化",
        "",
        "每个 PNG：上方完整 `bag_name` / `clip_id` + 英文说明；下方以问题锚点为中心、邻近约 30 帧聚合的斜视点云。",
        "",
        "**如何确认有问题**：护栏/路缘/杆状物出现双影，或路面呈多层薄片、静态结构无法重合。"
        "干净场景应只有一层清晰结构。",
        "",
    ]
    for i, s in enumerate(summary, 1):
        cid = s.get("clip_id") or ""
        bag = s.get("bag_name") or ""
        flags = ",".join(s.get("flags") or [])
        lines.append(f"## {i}. `{bag}`")
        lines.append("")
        lines.append(f"- **bag_name**: `{bag}`")
        lines.append(f"- **clip_id**: `{cid}`")
        lines.append(f"- **flags**: `{flags}`")
        if s.get("status") != "ok":
            lines.append(f"- **status**: `{s.get('status')}`")
            lines.append("")
            continue
        rel = Path(s["out"]).name
        lines.append(f"- **image**: [{rel}](./{rel})")
        lines.append(
            f"- **anchor**: f{s.get('worst_i')} ({s.get('anchor_src')}), "
            f"|d2|={s.get('d2_m'):.2f}m, agg={s.get('n_agg')} frames"
        )
        lines.append("")
        lines.append("**说明**")
        for e in s.get("explain_zh") or s.get("explain") or []:
            lines.append(f"- {e}")
        lines.append("")
    (out_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--elev", type=float, default=ELEV_DEG)
    ap.add_argument("--radius", type=float, default=RADIUS_M)
    ap.add_argument("--n-agg", type=int, default=N_AGG)
    ap.add_argument("--color", choices=("height", "semantic"), default="height")
    ap.add_argument("--out-dir", type=Path, default=OUTDIR)
    ap.add_argument("--clips", nargs="*", default=[])
    args = ap.parse_args()

    # clear old reject_*.png from previous looser list to avoid confusion
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for old in args.out_dir.glob("reject_*.png"):
        old.unlink()

    ls = _load_layer_scan()
    rows = load_reject_rows()
    metrics_map = load_metrics_by_id()
    if args.clips:
        want = set(args.clips)
        rows = [r for r in rows if r["clip_id"] in want or r["clip_id"][:8] in want]
    if args.limit > 0:
        rows = rows[: args.limit]

    bag_map = fetch_bag_names([r["clip_id"] for r in rows])
    print(f"{len(rows)} REJECT → {args.out_dir}", flush=True)

    summary = []
    for i, row in enumerate(rows):
        cid = row["clip_id"]
        meta = bag_map.get(cid) or {}
        bag = meta.get("bag_name") or "UNKNOWN_BAG"
        out = args.out_dir / f"reject_{i+1:02d}_{cid}_fPENDING.png"
        try:
            info = render_clip(
                ls, row, meta, metrics_map.get(cid),
                elev=args.elev, radius=args.radius, n_agg=args.n_agg,
                color_mode=args.color, out_path=out,
            )
        except Exception as exc:
            info = {
                "status": f"ERR {type(exc).__name__}: {exc}"[:200],
                "clip_id": cid,
                "bag_name": bag,
                "flags": row.get("flags"),
            }
        if info.get("status") == "ok":
            final = args.out_dir / f"reject_{i+1:02d}_{cid}_f{info['worst_i']}.png"
            if out != final and out.is_file():
                out.replace(final)
                info["out"] = str(final)
            msg = (
                f"{bag}  f{info['worst_i']} |d2|={info['d2_m']:.2f}m "
                f"pts={info['n_points']:,}"
            )
        else:
            if out.is_file():
                out.unlink()
            msg = str(info.get("status"))
        info["clip_id"] = cid
        info["bag_name"] = bag
        info["flags"] = row.get("flags")
        if "explain" not in info:
            info["explain"] = explain_flags(row.get("flags") or [], row)
        info["explain_zh"] = explain_flags_zh(
            row.get("flags") or [], {**row, **(metrics_map.get(cid) or {})}
        )
        summary.append(info)
        print(f"[{i+1}/{len(rows)}] {cid}  {msg}", flush=True)

    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_index_md(summary, args.out_dir)
    n_ok = sum(1 for s in summary if s.get("status") == "ok")
    print(f"DONE {n_ok}/{len(rows)}  index={args.out_dir / 'index.md'}", flush=True)
    return 0 if n_ok == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
