#!/usr/bin/env python3
"""Stage 3 — Projection diagnostics. READ ONLY. NEVER TUNES.

Checklist (strictly bounded — NO calib changes):
  [A] Timestamp sync
      A1.  lidar timestamp      vs  frame (lidar_dir) ts          Δ_ns
      A2.  lidar timestamp      vs  ego_pose.header.stamp        Δ_ns
      A3.  lidar timestamp      vs  each camera.timestamp        Δ_ns
      A4.  Inter-frame Δ_lidar monotonicity + any large gap.
  [B] Ego-pose sanity (vehicle-frame)
      B1.  quaternion |q| ≈ 1 (± tol)
      B2.  per-axis position / angle first-difference: no spikes >> mean+Nσ
      B3.  RPY in plausible range: |roll| < 0.2 rad, |pitch| < 0.2 rad,
           |yaw jump between consecutive| < 0.1 rad (unless turning)
  [C] Extrinsic sanity (static)
      C1.  cam_extrinsic T_v_c shape (4,4), det(R)=±1, R^T R ≈ I
      C2.  camera principal point cx ∈ [0.2·w , 0.8·w], cy ∈ [0.2·h , 0.8·h]
      C3.  K shape = (3,3), K[2,2]==1, K[0,1]==0, K[1,0]==0
      C4.  K intrinsic vs calibration_image width/height ratio (fx≈fy within 10%)
  [D] (NOT DONE) intrinsic distortion offsets — we leave K/distortion alone.

Outputs:
  stdout logger              — per-clip aggregate stats
  <out-root>/<clip>_diag.log — per-frame detail (raw deltas, spikes)
  <out-root>/PROJECTION_ISSUES.md    — ONE file with all clips' flagged issues
                                       (human readable, for post-12:00 analysis)
Rules:
  Any frame/clip failure → log & continue.  NEVER sys.exit inside analysis.
  Never write anywhere except under --out-root.
  Never call cv2.projectPoints or draw.  We only do numeric checks.
"""
from __future__ import annotations
import argparse, bisect, json, logging, math, os, sys, traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

# --------------------------------------------------------------------------- #
# Hard diagnostic thresholds (DO NOT TUNE — document the flag instead!)
# --------------------------------------------------------------------------- #
# timestamps
TS_LIDAR_EGO_MAX_NS    = 20_000_000  # 20 ms between lidar ts vs ego header stamp
TS_LIDAR_CAM_MAX_NS    = 40_000_000  # 40 ms between lidar ts vs camera ts
TS_INTERFRAME_GAP_NS   = 500_000     # >0.5ms jump between lidar ticks (unusual for 10/20 Hz)
# pose
POSE_Q_NORM_MAX_DELTA  = 1e-3        # ||q|| ± tol
POSE_ROLL_MAX_ABS      = 0.25        # radians  (~14°)
POSE_PITCH_MAX_ABS     = 0.25
POSE_JUMP_SIGMAS       = 6           # spike detector
POSE_MIN_N             = 10          # need >=10 samples to run jump detector
# calibration static
CAM_CX_FRAC_MIN        = 0.20
CAM_CX_FRAC_MAX        = 0.80
CAM_FX_FY_RATIO_MIN    = 0.90
CAM_FX_FY_RATIO_MAX    = 1.10
CAM_RORTHO_MAX         = 1e-4        # ||R^T R - I||_inf


ISSUES_FILE_NAME = "PROJECTION_ISSUES.md"

def setup_loggers(out_root: Path):
    out_root.mkdir(parents=True, exist_ok=True)
    lgr = logging.getLogger("proj_diag")
    lgr.setLevel(logging.DEBUG)
    for h in list(lgr.handlers): lgr.removeHandler(h)
    sh = logging.StreamHandler(sys.stdout); sh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    sh.setFormatter(fmt); lgr.addHandler(sh)
    return lgr


# ---------- Pure helpers -----------------------------------------------------

def quat_to_rpy(q_xyzw: np.ndarray) -> tuple[float, float, float]:
    x, y, z, w = float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]), float(q_xyzw[3])
    # roll (x-axis), pitch (y-axis), yaw (z-axis)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    if sinp >= 1: pitch = math.pi / 2
    elif sinp <= -1: pitch = -math.pi / 2
    else: pitch = math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def r3(r, c): return max(abs(r[i][c]) for i in range(3))


def pose_rpy_xyz(pose_dict: dict):
    p = pose_dict
    q = np.array([p["orientation"][k] for k in ("x", "y", "z", "w")], np.float64)
    t = np.array([p["position"][k] for k in ("x", "y", "z")], np.float64)
    r, py, yaw = quat_to_rpy(q)
    return q, t, r, py, yaw


def pose_stamp_ns(ego: dict, fallback_ns: int) -> int:
    stamp = ((ego or {}).get("header") or {}).get("stamp") or {}
    if "sec" in stamp:
        return int(stamp["sec"]) * 1_000_000_000 + int(stamp.get("nanosec", 0))
    return int(fallback_ns)


# ---------- Per-clip diagnostic loop ----------------------------------------

class ClipReport:
    def __init__(self, clip_id: str):
        self.clip_id = clip_id
        self.issues: list[dict] = []
        self.stats: dict = {}

    def flag(self, severity: str, category: str, short: str, frame: str | None = None,
             detail: str | None = None):
        self.issues.append({
            "severity": severity,   # info / warn / error
            "category": category,   # TS_A1 / TS_A2 / ... / B1 / C1 ...
            "frame": frame,
            "short": short,
            "detail": detail or "",
        })

    def __len__(self):
        return len(self.issues)


def diag_clip(clip_id: str, backup_root: Path, out_root: Path,
              lgr: logging.Logger) -> ClipReport:
    rpt = ClipReport(clip_id)
    clip_dir = backup_root / clip_id
    fr_dir = clip_dir / "frames"
    if not fr_dir.is_dir():
        rpt.flag("error", "GEN", f"clip dir missing: {clip_dir}")
        return rpt

    # frames (sorted by name which == ns timestamp)
    ts_list = sorted(p.name for p in fr_dir.iterdir() if p.is_dir())
    if not ts_list:
        rpt.flag("error", "GEN", "no frame dirs")
        return rpt
    rpt.stats["n_frames"] = len(ts_list)

    # detail log per clip
    clip_log_path = out_root / f"{clip_id}_diag.log"
    detail_handler = logging.FileHandler(clip_log_path, mode="w")
    detail_handler.setLevel(logging.DEBUG)
    dlog = logging.getLogger(f"proj_diag_{clip_id}")
    dlog.setLevel(logging.DEBUG)
    for h in list(dlog.handlers): dlog.removeHandler(h)
    dlog.addHandler(detail_handler)

    pose_samples = []         # (ns, pose_dict)
    lidar_dts = []            # inter-lidar dt (ns)
    lidar_ego_delta = []      # |lidar_ts - ego_stamp|
    per_cam_delta: dict[str, list[int]] = defaultdict(list)
    per_cam_present: set[str] = set()
    cams_with_static_issues: set[str] = set()

    # A4. inter-frame lidar monotonicity + gap
    last_lidar: int | None = None
    # pose arrays for B2/B3
    pose_t_arr = []
    pose_roll_arr = []; pose_pitch_arr = []; pose_yaw_arr = []
    pose_stamps = []
    n_missing_ego = 0

    for ts in ts_list:
        meta_path = fr_dir / ts / "frame.json"
        if not meta_path.is_file():
            rpt.flag("warn", "GEN", f"frame.json missing", ts)
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception as exc:
            rpt.flag("error", "GEN", f"frame.json parse fail {exc}", ts)
            continue

        dep = meta.get("dependency") or {}
        sensors = dep.get("sensors") or {}

        # -------- [A1] lidar ts vs folder ts
        lidar_names = [k for k in sensors.keys() if k.startswith("lidar")]
        frame_lidar_ts = int(ts)
        this_lidar_stamps = []
        for k in lidar_names:
            s = sensors[k] or {}
            st = s.get("timestamp")
            if isinstance(st, (int, float)) and st:
                st = int(st)
                this_lidar_stamps.append(st)
                d = abs(st - frame_lidar_ts)
                if d > 1_000:  # only record >1us to avoid noise
                    dlog.debug("A1 frame=%s sensor=%s Δ_lidar-frame_ns=%d", ts, k, d)
        lidar_ts = this_lidar_stamps[0] if this_lidar_stamps else frame_lidar_ts
        if last_lidar is not None:
            dts = lidar_ts - last_lidar
            lidar_dts.append(dts)
            if dts < 0:
                rpt.flag("warn", "TS_A4", f"lidar ts non-monotonic dt={dts}ns", ts)
            elif dts > TS_INTERFRAME_GAP_NS:
                rpt.flag("warn", "TS_A4",
                         f"lidar inter-frame gap large dt={dts/1e3:.1f}ms", ts)
        last_lidar = lidar_ts

        # -------- [A2] lidar ts vs ego_pose.header.stamp
        ego = dep.get("ego_pose") or {}
        pose = ego.get("pose")
        if not pose:
            n_missing_ego += 1
            rpt.flag("warn", "TS_A2", "ego_pose.pose missing (skip interp)", ts)
            dlog.debug("A2 frame=%s pose_missing", ts)
        else:
            ego_stamp_ns = pose_stamp_ns(ego, frame_lidar_ts)
            d_ego = abs(lidar_ts - ego_stamp_ns)
            lidar_ego_delta.append(d_ego)
            if d_ego > TS_LIDAR_EGO_MAX_NS:
                rpt.flag("warn", "TS_A2",
                         f"|lidar - ego.stamp|={d_ego/1e3:.1f}ms (>{TS_LIDAR_EGO_MAX_NS/1e6:.0f}ms)",
                         ts)
            pose_samples.append((ego_stamp_ns, pose))
            q, t, r, py, yaw = pose_rpy_xyz(pose)
            pose_stamps.append(ego_stamp_ns)
            pose_t_arr.append(t)
            pose_roll_arr.append(r); pose_pitch_arr.append(py); pose_yaw_arr.append(yaw)

            # -------- [B1] quaternion norm
            qn = float(np.linalg.norm(q))
            if abs(qn - 1.0) > POSE_Q_NORM_MAX_DELTA:
                rpt.flag("warn", "POSE_B1", f"|q|={qn:.5f} |1-q|={abs(qn-1):.2e}", ts)

        # -------- [A3] lidar ts vs each camera ts + [C] static cam calib
        cam_names = [k for k in sensors.keys() if k.startswith("camera")]
        cams_with_static_checked = set()
        for k in cam_names:
            per_cam_present.add(k)
            s = sensors[k] or {}
            cts = s.get("timestamp")
            if isinstance(cts, (int, float)) and cts:
                dc = abs(int(cts) - lidar_ts)
                per_cam_delta[k].append(dc)
                if dc > TS_LIDAR_CAM_MAX_NS:
                    rpt.flag("warn", "TS_A3",
                             f"{k} |cam - lidar|={dc/1e3:.1f}ms (>{TS_LIDAR_CAM_MAX_NS/1e6:.0f}ms)",
                             ts)
            # Static calibration checks (only once per camera, first frame with calib)
            if k not in cams_with_static_checked:
                cams_with_static_checked.add(k)
                camdoc = s.get("calibration") or s
                intr = camdoc.get("intrinsic")
                extr = camdoc.get("extrinsic")
                if intr and extr:
                    try:
                        K = np.asarray(intr["intrinsic"], np.float64).reshape(3, 3)
                        dist = np.asarray(intr["distortion"], np.float64).reshape(-1)
                        w = int(intr.get("width", 0)); h = int(intr.get("height", 0))
                        T = np.asarray(extr["transformation"], np.float64).reshape(4, 4)
                        R = T[:3, :3]
                        # C3 K shape
                        if abs(K[2, 2] - 1.0) > 1e-6 or abs(K[0, 1]) > 1e-6 or abs(K[1, 0]) > 1e-6:
                            rpt.flag("warn", "CAL_C3",
                                     f"{k} K shape unexpected K[2,2]={K[2,2]:.3f} skew={K[0,1]:.3f},{K[1,0]:.3f}",
                                     ts)
                        # C2 cx,cy
                        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
                        if w > 0 and not (CAM_CX_FRAC_MIN * w <= cx <= CAM_CX_FRAC_MAX * w):
                            rpt.flag("warn", "CAL_C2",
                                     f"{k} cx={cx:.1f} outside [{CAM_CX_FRAC_MIN*w:.0f},{CAM_CX_FRAC_MAX*w:.0f}] w={w}",
                                     ts)
                        if h > 0 and not (CAM_CX_FRAC_MIN * h <= cy <= CAM_CX_FRAC_MAX * h):
                            rpt.flag("warn", "CAL_C2",
                                     f"{k} cy={cy:.1f} outside [{CAM_CX_FRAC_MIN*h:.0f},{CAM_CX_FRAC_MAX*h:.0f}] h={h}",
                                     ts)
                        # C4 fx≈fy
                        if min(abs(fx), abs(fy)) > 1e-6:
                            ratio = fx / fy
                            if not (CAM_FX_FY_RATIO_MIN <= ratio <= CAM_FX_FY_RATIO_MAX):
                                rpt.flag("warn", "CAL_C4",
                                         f"{k} fx/fy={ratio:.3f} outside [{CAM_FX_FY_RATIO_MIN},{CAM_FX_FY_RATIO_MAX}]",
                                         ts)
                        # C1 det(R) ±1 + R^T R ≈ I
                        detR = float(np.linalg.det(R))
                        if not 0.999 <= abs(detR) <= 1.001:
                            rpt.flag("warn", "CAL_C1", f"{k} det(R)={detR:.5f}", ts)
                        orth = R.T @ R - np.eye(3)
                        if np.abs(orth).max() > CAM_RORTHO_MAX:
                            rpt.flag("warn", "CAL_C1",
                                     f"{k} R^T R != I max_abs={np.abs(orth).max():.2e}",
                                     ts)
                    except Exception as exc:
                        rpt.flag("warn", "CAL_GEN", f"{k} calib check exception {exc}", ts)

    # ---- aggregate summary logs
    rpt.stats["n_missing_ego"] = n_missing_ego
    rpt.stats["cams_seen"] = sorted(per_cam_present)
    if lidar_ego_delta:
        a = np.array(lidar_ego_delta)
        rpt.stats["lidar_ego_delta_ns"] = {
            "p50": int(np.percentile(a, 50)), "p95": int(np.percentile(a, 95)),
            "max": int(a.max())}
    for k, arr in per_cam_delta.items():
        a = np.array(arr)
        rpt.stats[f"cam_{k}_delta_ns"] = {
            "p50": int(np.percentile(a, 50)), "p95": int(np.percentile(a, 95)),
            "max": int(a.max())}
    if lidar_dts:
        a = np.array(lidar_dts, np.float64)
        rpt.stats["inter_lidar_dt_ns"] = {
            "mean_ns": float(a.mean()), "min": int(a.min()), "max": int(a.max())}

    # ---- [B2/B3] first-diff jump detection after loop
    if len(pose_samples) >= POSE_MIN_N:
        t_arr = np.array(pose_t_arr, np.float64)      # N,3
        roll = np.array(pose_roll_arr); pitch = np.array(pose_pitch_arr); yaw = np.array(pose_yaw_arr)
        droll = np.diff(roll); dpitch = np.diff(pitch)
        # yaw jumps need to be unwrapped
        dyaw = np.diff(np.unwrap(yaw))
        dxyz = np.linalg.norm(np.diff(t_arr, axis=1 if t_arr.ndim==2 else 0), axis=-1)
        if t_arr.ndim != 2:
            dxyz = np.zeros_like(droll)  # guard
        for name, d in (("roll", droll), ("pitch", dpitch), ("yaw", dyaw), ("trans_m", dxyz)):
            m = float(np.mean(d)); s = float(np.std(d))
            thr = max(1e-8, m + POSE_JUMP_SIGMAS * s)
            for i, v in enumerate(d.tolist()):
                if abs(v) > thr:
                    rpt.flag("warn", "POSE_B2",
                             f"Δ{name} jump={abs(v):.4f}  μ±{POSE_JUMP_SIGMAS}σ=({thr:.4f})",
                             frame=ts_list[i+1],
                             detail=f"μ={m:.2e} σ={s:.2e} i={i}")
        # B3 static plausible ranges
        for i, ts in enumerate(ts_list):
            if abs(roll[i]) > POSE_ROLL_MAX_ABS:
                rpt.flag("warn", "POSE_B3",
                         f"|roll|={abs(roll[i]):.3f} rad > {POSE_ROLL_MAX_ABS}", ts)
            if abs(pitch[i]) > POSE_PITCH_MAX_ABS:
                rpt.flag("warn", "POSE_B3",
                         f"|pitch|={abs(pitch[i]):.3f} rad > {POSE_PITCH_MAX_ABS}", ts)
    else:
        rpt.flag("info", "POSE_B2", f"pose samples ({len(pose_samples)}) < {POSE_MIN_N} skip jump test")

    severity_counts = Counter(i["severity"] for i in rpt.issues)
    lgr.info("clip=%8s frames=%d  issues[err/warn/info]=[%d/%d/%d]  detail=%s",
             clip_id[:8], len(ts_list),
             severity_counts.get("error",0), severity_counts.get("warn",0),
             severity_counts.get("info",0), clip_log_path.name)
    # flush detail
    detail_handler.close()
    dlog.removeHandler(detail_handler)
    return rpt


# ---------- Render issues markdown -------------------------------------------

def issues_categories_short():
    return {
        "TS_A1": "lidar stamp vs folder ts",
        "TS_A2": "lidar ↔ ego_pose stamp",
        "TS_A3": "lidar ↔ camera stamp",
        "TS_A4": "inter-frame lidar monotonicity / gap",
        "POSE_B1": "quaternion unit-norm",
        "POSE_B2": "pose 1st-diff jumps (6σ outlier)",
        "POSE_B3": "static roll/pitch plausibility",
        "CAL_C1": "camera extrinsic orthogonality + det(R)",
        "CAL_C2": "principal point cx,cy ∈ middle of image",
        "CAL_C3": "K[2,2]=1 and zero-skew shape",
        "CAL_C4": "fx/fy ratio (≈1 within 10%)",
        "CAL_GEN": "camera calibration read exception",
        "GEN":     "general (file I/O etc.)",
    }


def write_issues_markdown(all_reports: list[ClipReport], out_root: Path):
    out_root.mkdir(parents=True, exist_ok=True)
    p = out_root / ISSUES_FILE_NAME
    lines = []
    lines.append("# Projection Issues Report (READ-ONLY, NO FIXES APPLIED)\n")
    lines.append(f"- clips scanned: {len(all_reports)}")
    total_err = sum(1 for r in all_reports for i in r.issues if i["severity"] == "error")
    total_warn = sum(1 for r in all_reports for i in r.issues if i["severity"] == "warn")
    total_info = sum(1 for r in all_reports for i in r.issues if i["severity"] == "info")
    lines.append(f"- issue counts: error={total_err}  warn={total_warn}  info={total_info}\n")
    lines.append("## Legend (categories checked)\n")
    for k, v in issues_categories_short().items():
        lines.append(f"- **{k}** — {v}")
    lines.append("")
    lines.append("> ## Scope / out-of-scope\n")
    lines.append("> - ✅ Timestamp sync (lidar ↔ ego ↔ cams)\n")
    lines.append("> - ✅ Ego-pose continuity (q-norm, roll/pitch bounds, 6σ outlier jumps)\n")
    lines.append("> - ✅ Static camera calib sanity (K shape, R orth, cx/cy range, fx/fy ratio)\n")
    lines.append("> - ❌ Intrinsic distortion (k1..k5) — NEVER touched per task instructions\n")
    lines.append("> - ❌ Extrinsic refinement (camera-vehicle mount offsets) — NEVER touched\n")
    lines.append("> - ❌ cv2.projectPoints model choice (pinhole vs fisheye) — NEVER touched\n")
    lines.append("")
    # Summary per clip
    lines.append("## Clip-by-clip summary\n")
    lines.append("| clip (prefix) | frames | errors | warns | infos | delta_lidar↔ego_p95 (ms) | worst cam↔lidar_p95 (ms) |\n")
    lines.append("|---|---|---|---|---|---|---|\n")
    for r in all_reports:
        c = Counter(i["severity"] for i in r.issues)
        p95_ego = (r.stats.get("lidar_ego_delta_ns") or {}).get("p95", 0)
        cam_p95s = [(k, (r.stats.get(f"cam_{k}_delta_ns") or {}).get("p95", 0))
                    for k in r.stats.get("cams_seen", [])]
        worst_cam = max(cam_p95s, key=lambda x: x[1]) if cam_p95s else ("-", 0)
        lines.append(f"| `{r.clip_id[:12]}`… | {r.stats.get('n_frames',0)} | "
                     f"{c.get('error',0)} | {c.get('warn',0)} | {c.get('info',0)} | "
                     f"{p95_ego/1e6:.2f} | {worst_cam[0]} {worst_cam[1]/1e6:.2f} |\n")
    lines.append("")
    # Per-clip issue lists (warn+)
    lines.append("## Per-clip flagged issues (warn and above)\n")
    any_issue = False
    for r in all_reports:
        flagged = [i for i in r.issues if i["severity"] in ("warn", "error")]
        if not flagged: continue
        any_issue = True
        lines.append(f"### `{r.clip_id[:36]}`\n")
        lines.append("| sev | cat | frame | short | detail |\n")
        lines.append("|---|---|---|---|---|\n")
        for i in flagged[:500]:  # cap per clip
            lines.append(f"| {i['severity']} | {i['category']} | "
                         f"`{(i.get('frame') or '')[:16]}`… | "
                         f"{i['short'].replace('|','/')[:120]} | "
                         f"{(i.get('detail') or '').replace('|','/')[:120]} |\n")
        if len(flagged) > 500:
            lines.append(f"| … | … | … | (truncated; total flagged={len(flagged)} for this clip) | … |\n")
        lines.append("")
    if not any_issue:
        lines.append("_No warn/error-level issues flagged on any clip._\n")
    p.write_text("\n".join(lines))
    return p


# ---------- main ------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backup-root", type=Path, default=Path("exp/robotruck/raw_volume_cache"))
    ap.add_argument("--manifest", type=Path, default=Path("exp/robotruck/cone_mid/manifest.json"),
                    help="cone_filter_denoise manifest (optional).  If missing, scan all clips in backup.")
    ap.add_argument("--clips-json", type=Path)
    ap.add_argument("--out-root", type=Path, default=Path("exp/robotruck/proj_diag"))
    ap.add_argument("--max-clips", type=int, default=0, help="0 = unlimited")
    args = ap.parse_args()

    lgr = setup_loggers(args.out_root)
    clip_ids: list[str] = []
    if args.manifest.is_file():
        try:
            clip_ids = [row["clip_id"] for row in (json.loads(args.manifest.read_text())
                                                   .get("clips", []))]
        except Exception as exc:
            lgr.warning("manifest read failed: %s", exc)
    if not clip_ids and args.clips_json and args.clips_json.is_file():
        try:
            clip_ids = list((json.loads(args.clips_json.read_text())
                             .get("clips") or []))
        except Exception as exc:
            lgr.warning("clips-json read failed: %s", exc)
    if not clip_ids and args.backup_root.is_dir():
        clip_ids = sorted(p.name for p in args.backup_root.iterdir() if p.is_dir())
    if args.max_clips:
        clip_ids = clip_ids[:args.max_clips]
    lgr.info("total clips for diagnostics: %d", len(clip_ids))

    reports: list[ClipReport] = []
    for i, cid in enumerate(clip_ids, 1):
        lgr.info("[%d/%d] diag %s", i, len(clip_ids), cid[:8])
        try:
            reports.append(diag_clip(cid, args.backup_root, args.out_root, lgr))
        except Exception as exc:
            lgr.error("diag %s crashed: %s\n%s", cid[:8], exc, traceback.format_exc())
            r = ClipReport(cid)
            r.flag("error", "GEN", f"diagnostic crashed: {exc}")
            reports.append(r)

    md_path = write_issues_markdown(reports, args.out_root)
    lgr.info("DONE  reports=%d  issues MD -> %s", len(reports), md_path)
    return 0


if __name__ == "__main__":
    try: sys.exit(main())
    except KeyboardInterrupt: sys.exit(130)
