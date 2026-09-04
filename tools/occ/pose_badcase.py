#!/usr/bin/env python3
"""Offline pose/PC layering QA (not delivery SoT — that is quality_gate.py).

Five tunables: T_MED / T_GPC / T_MAX / T_NINF / T_DRIFT_REL → REJECT/HIGH/WARN/CLEAN.
Viz: tools/occ/triage.py. Out: exp/robotruck/pose_badcase/
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from paths import ROOT, ensure_import_path

ensure_import_path()
from layer_scan import (  # noqa: E402
    CLIPS_COL,
    DB,
    MONGO_URI,
    SUBSAMPLE,
    fetch_records,
    load_cloud,
)

OUT = ROOT / "exp/robotruck/pose_badcase"

# ---- 运动学测量上限（固定，不是业务旋钮）----
SPEED_MAX = 40.0      # m/s ≈ 144 km/h
VRATE_MAX = 3.0       # m/s 垂直
D2_MAX = 1.0          # m 三帧空间二阶差分
ACCEL_MAX = 20.0      # m/s² ≈ 2g

PAIR_GAP = 10
N_ANOMALY = 4
N_REF = 3
N_SHIFT_PAIRS = 5
SHIFT_GRID = np.linspace(-1.2, 1.2, 13)

# ---- 仅 5 个可调 REJECT 阈值 ----
T_MED = 0.45          # 多窗口中位 PC 误差 (m)：持续分层
T_GPC = 0.31          # 全局 PC 误差 (m)：整段发虚
T_MAX = 1.19          # 局部最差 PC 误差 (m)：尖刺够狠
T_NINF = 12           # 不可达帧数：轨迹够乱
T_DRIFT_REL = 0.06    # pose-shift 相对改善：系统性漂移

# HIGH 线（非 REJECT 旋钮）
HIGH_PC = 0.70

# 人工标签（回归）
LABEL_BAD = {
    "0038952f-2da1-4eb1-8b40-83b728debfaa",
    "3c4ddc70-dd06-4c5b-9c01-5fd55e29a246",
    "0c85fa31-dd09-4adc-a2d9-ab76725e438e",
    "c47ca55d-5883-4bc6-a258-c3b1cb41c550",
    "9542b395-5657-4b74-8f43-dd86eabdcdfb",
    "d3067307-7370-4ca6-8580-92c0ef404547",
    "f40552ec-3190-4aad-a2f9-c4ac5dfc479b",
    "0a1f71ee-7c7a-426d-838d-4a51b305da76",
    "7ffb9577-e3d3-4432-9d79-9fe39ae325f4",
    "0882cb83-550b-4c54-8b91-cf2061e07c4a",
    "ede2c75f-9f99-4301-9c14-93727e71e66a",
    "59037e92-5639-4a40-b058-7151eb387be1",
    "d3f27db5-040d-4941-8811-cee6cf45d27c",
}
LABEL_SUSPECT = {
    "32534087-9356-49ef-a008-b398080bf370",
    "7c8416b9-297d-4861-8085-573b55cc046c",
    "635a9056-6e91-4df2-86b8-4cfa222b900a",
    "65d420de-8ae4-44aa-a0a5-1bf435500822",
    "2ec9449e-1de6-465c-b904-ca837e6b48c5",
    "98809b49-1ba2-4b0e-92d8-b4e9ad75d2b4",
    "6cff709a-fb5e-4659-8dae-ead00f58b8f5",
    "5549bc4e-b1c6-41d1-ad05-fc58e01fa758",
    "c9a4788c-4c44-4d5d-bc82-6871ec4ea6ff",
}
# 人工确认：不像 badcase（多为弱 DRIFT 误杀）
LABEL_FP_OK = {
    "f201f1fb-04ed-414f-bb9e-b76e56c00248",
    "f5d99f16-f107-4f53-9d7a-43ac73601488",
    "b0924e25-bf76-4138-83fb-dab2262d15ad",
    "9fa61ccf-fdad-4691-bfaf-cf0eac17cdf8",
    "3832f01d-dced-4562-bc9c-b17f274655e3",
}
LABEL_GOOD = {
    "0742da0f-a84d-497c-9286-269b862a7930",
    "0bf9c403-97fc-4100-b1b8-9a32f351951a",
    "10087156-0de5-40eb-9938-cecdeeb764e3",
    "095aef75-e726-42a1-ab57-853fad15f8f3",
    "185c87c8-b002-4790-ac8c-d8065c38d06e",
    "28b5b1ea-37b6-47b6-a176-215d5b2127e9",
    "370d5e67-e742-4854-b654-40ed6f802f4d",
    "b7807e70-fb23-4e36-99e9-9ef0ded1c1a9",
}
LABEL_FP_BRIDGE = {
    "8e347b59-be4b-4279-956f-b015a16859a9",
    "051104df-252e-4303-80e3-c304bdedc6f0",
    "cc1c3784-e412-4921-b7a4-582f91b4c1ea",
    "1daa64ef-fab6-430b-93e5-2c568e08c91c",
}

KNOWN = LABEL_BAD


def _kinematics(fr: list) -> dict:
    valid = [(i, f) for i, f in enumerate(fr) if f[1] is not None]
    if len(valid) < 5:
        return {"ok": False, "reason": "too_short"}
    P = np.array([f[1] for _, f in valid], dtype=np.float64)
    T = np.array([f[0] for _, f in valid], dtype=np.int64)
    dt = np.diff(T).astype(np.float64) / 1e9
    dt = np.where(dt <= 0, np.nan, dt)
    dp = np.diff(P, axis=0)
    with np.errstate(invalid="ignore"):
        speed = np.linalg.norm(dp, axis=1) / dt
        vrate = np.abs(dp[:, 2]) / dt
        d2 = P[2:] - 2 * P[1:-1] + P[:-2]
        d2n = np.linalg.norm(d2, axis=1)
        accel = d2n / (dt[1:] * dt[:-1])
    n = len(valid)
    infeas = np.zeros(n, dtype=bool)
    for i in range(len(speed)):
        if (np.isfinite(speed[i]) and speed[i] > SPEED_MAX) or (
            np.isfinite(vrate[i]) and vrate[i] > VRATE_MAX
        ):
            infeas[i] = infeas[i + 1] = True
    for i in range(len(d2n)):
        if d2n[i] > D2_MAX or (np.isfinite(accel[i]) and accel[i] > ACCEL_MAX):
            infeas[i + 1] = True
    return {
        "ok": True,
        "valid": valid,
        "P": P,
        "speed": speed,
        "vrate": vrate,
        "d2n": d2n,
        "accel": accel,
        "infeas": infeas,
        "feasible": not bool(infeas.any()),
        "n_infeas": int(infeas.sum()),
        "max_speed": float(np.nanmax(speed)) if len(speed) else None,
        "max_vrate": float(np.nanmax(vrate)) if len(vrate) else None,
        "max_d2": float(np.nanmax(d2n)) if len(d2n) else None,
        "max_accel": float(np.nanmax(accel)) if len(accel) else None,
    }


def _pick_centers(kin: dict, k: int = N_ANOMALY) -> list[int]:
    n = len(kin["valid"])
    score = np.zeros(n, dtype=np.float64)
    score[1:-1] = kin["d2n"]
    score = score + kin["infeas"].astype(np.float64) * 10.0
    picked: list[int] = []
    for i in np.argsort(-score):
        i = int(i)
        if any(abs(i - p) < 12 for p in picked):
            continue
        picked.append(i)
        if len(picked) >= k:
            break
    if kin["feasible"]:
        for frac in (0.2, 0.4, 0.6, 0.8):
            c = int(frac * (n - 1))
            if all(abs(c - p) >= 12 for p in picked):
                picked.append(c)
    return picked


def _ref_centers(kin: dict, anom: list[int]) -> list[int]:
    n = len(kin["valid"])
    infeas = kin["infeas"]
    cands = [int(f * (n - 1)) for f in (0.2, 0.4, 0.6, 0.8)]
    out = []
    for c in cands:
        if any(abs(c - p) < 15 for p in anom):
            continue
        if infeas.any() and infeas[max(0, c - 5): min(n, c + 6)].any():
            continue
        out.append(c)
        if len(out) >= N_REF:
            break
    return out or cands[:N_REF]


def _pc_pair_metric(pa: np.ndarray, pb: np.ndarray) -> float:
    """Same as layer_scan.pc_pair_metric but workers=1 (safe under ProcessPool)."""
    if len(pa) > SUBSAMPLE:
        pa = pa[:: len(pa) // SUBSAMPLE]
    if len(pb) > SUBSAMPLE:
        pb = pb[:: len(pb) // SUBSAMPLE]
    tree = cKDTree(pa)
    d, _ = tree.query(pb, k=1, workers=1)
    return float(np.percentile(d, 20))


def _pair_cloud(cid: str, valid: list, a: int, b: int, cloud_cache: dict | None = None):
    fa, fb = valid[a][1], valid[b][1]
    if fa[2] is None or fb[2] is None or not fa[3] or not fb[3]:
        return None

    def _get(ts: str, md5: str):
        if cloud_cache is None:
            return load_cloud(cid, ts, md5)
        key = (ts, md5)
        hit = cloud_cache.get(key)
        if hit is None:
            hit = load_cloud(cid, ts, md5)
            cloud_cache[key] = hit
        return hit

    try:
        A = _get(str(fa[0]), fa[3])
        B = _get(str(fb[0]), fb[3])
    except Exception:
        return None
    pa = A @ fa[2].T + fa[1]
    pb = B @ fb[2].T + fb[1]
    return pa, pb, float(_pc_pair_metric(pa, pb)), fa[1], fb[1]


def _pose_shift(pa: np.ndarray, pb: np.ndarray, direction: np.ndarray):
    step_a = max(1, len(pa) // 6000)
    step_b = max(1, len(pb) // 6000)
    tree = cKDTree(pa[::step_a])
    pb0 = pb[::step_b]
    losses = []
    for s in SHIFT_GRID:
        # workers=1: ProcessPool 内禁止再开线程风暴
        d = tree.query(pb0 + s * direction, workers=1)[0]
        d = d[d < 1.2]
        if len(d) < 80:
            losses.append(np.inf)
            continue
        k = max(80, int(0.7 * len(d)))
        losses.append(float(np.median(np.partition(d, k - 1)[:k])))
    best = int(np.argmin(losses))
    mid = len(SHIFT_GRID) // 2
    if not np.isfinite(losses[best]) or not np.isfinite(losses[mid]):
        return None
    rel = max(0.0, (losses[mid] - losses[best]) / max(1e-9, losses[mid]))
    return float(SHIFT_GRID[best]), float(rel), float(losses[mid])


def _pc_at_centers(
    cid: str,
    valid: list,
    centers: list[int],
    gap: int = PAIR_GAP,
    cloud_cache: dict | None = None,
) -> list[float]:
    n = len(valid)
    vals = []
    for c in centers:
        a = max(0, c - gap)
        b = min(n - 1, c + gap)
        if b - a < gap:
            continue
        row = _pair_cloud(cid, valid, a, b, cloud_cache=cloud_cache)
        if row is not None:
            vals.append(row[2])
    return vals


def _uniform_shift_search(
    cid: str,
    valid: list,
    gap: int = PAIR_GAP,
    n_pairs: int = N_SHIFT_PAIRS,
    cloud_cache: dict | None = None,
):
    n = len(valid)
    if n <= gap + 2:
        return None
    starts = np.linspace(0, n - gap - 1, num=min(n_pairs, max(1, n - gap)), dtype=int)
    pcs, shifts, rels, signs = [], [], [], []
    for a in starts:
        b = int(a) + gap
        row = _pair_cloud(cid, valid, int(a), b, cloud_cache=cloud_cache)
        if row is None:
            continue
        pa, pb, pc, ta, tb = row
        pcs.append(pc)
        direction = tb - ta
        nn = float(np.linalg.norm(direction))
        if nn < 1e-6:
            continue
        hit = _pose_shift(pa, pb, direction / nn)
        if hit is None:
            continue
        sh, rel, _ = hit
        shifts.append(abs(sh))
        rels.append(rel)
        signs.append(sh)
    if not shifts:
        return None
    sigs = [s for s in signs if abs(s) >= 0.2]
    if sigs:
        cons = max(sum(s > 0 for s in sigs), sum(s < 0 for s in sigs)) / len(sigs)
    else:
        cons = 0.0
    return {
        "n_pairs": len(shifts),
        "med_pc": float(np.median(pcs)) if pcs else None,
        "max_pc": float(np.max(pcs)) if pcs else None,
        "med_shift": float(np.median(shifts)),
        "med_rel": float(np.median(rels)),
        "sign_consistency": float(cons),
        "n_significant": len(sigs),
    }


def decide(m: dict, _gate_reasons: list[str] | None = None) -> tuple[str, list[str]]:
    """简洁判定：5 个可调阈值，三条 REJECT 路径，无旧 GATE。

    REJECT:
      DRIFT  — 大平移搜索一致改善 + 局部 PC 证据（抑制弱假阳）
      SOFT   — 弱尖刺 + 中位持续差 + 全局发虚（max 须 < 1m）
      HARD   — 局部够尖 + 中位够高（或尖局部+低 gpc）+ 轨迹乱
    """
    max_a = float(m.get("max_anom_pc") or 0.0)
    med_a = float(m.get("med_anom_pc") or 0.0)
    ratio = float(m.get("pc_ratio") or 0.0)
    gpc = float(m.get("global_pc_p20") or 0.0)
    ninf = int(m.get("n_infeas") or 0)
    shift = float(m.get("med_shift") or 0.0)
    rel = float(m.get("med_rel") or 0.0)
    cons = float(m.get("sign_consistency") or 0.0)
    n_sig = int(m.get("n_significant") or 0)
    feas = m.get("feasible")

    # 1) 系统性 pose 漂移（0038952f / d3f27db5）
    # shift 要够大；max_a 要有可见错位；gpc 不能整段发虚（排除 b092 类）
    if (
        shift >= 1.2
        and cons >= 0.8
        and n_sig >= 4
        and rel >= T_DRIFT_REL
        and max_a >= 0.33
        and gpc <= 0.28
    ):
        return "REJECT", [f"DRIFT:shift{shift:.2f}/rel{rel:.3f}"]

    # 后续 PC 路径要求轨迹不可达
    if feas is not False:
        if max_a >= HIGH_PC:
            return "HIGH", [f"PC_BORDERLINE:{max_a:.3f}"]
        if shift >= 0.5 and cons >= 0.8 and n_sig >= 4:
            return "HIGH", [f"POSE_SHIFT_WEAK:{shift:.2f}m"]
        return "CLEAN", []

    # 天桥式单尖刺：ratio 虚高且不可达帧不够多 → 不走 HARD/SOFT
    spike_ok = (ratio < 15.0) or (ninf >= 30)

    # 2) 弱尖刺 + 全局发虚（7ffb9577）；max≥1 留给 HARD，避免 c9a 类误杀
    if (
        spike_ok
        and med_a >= T_MED
        and gpc >= T_GPC
        and 0.45 <= max_a < 1.0
        and ratio >= 1.05
        and ninf <= 8
    ):
        return "REJECT", [f"SOFT:med{med_a:.3f}/gpc{gpc:.3f}"]

    # 3) 强局部错位 + 轨迹乱
    # med≥0.55 覆盖多数 BAD；窄分支保留 0a1f71ee，排除 5549 类疑似
    hard_med = med_a >= 0.55 or (
        max_a >= 1.4 and med_a >= 0.38 and ratio >= 4.0 and gpc <= 0.20
    )
    if (
        spike_ok
        and max_a >= T_MAX
        and hard_med
        and ninf >= T_NINF
        and ratio >= 2.0
    ):
        return "REJECT", [f"HARD:max{max_a:.3f}/med{med_a:.3f}/n{ninf}"]

    if max_a >= HIGH_PC:
        return "HIGH", [f"PC_BORDERLINE:{max_a:.3f}"]
    if shift >= 0.5 and cons >= 0.8 and n_sig >= 4:
        return "HIGH", [f"POSE_SHIFT_WEAK:{shift:.2f}m"]
    return "WARN", [f"INFEASIBLE_ONLY:n={ninf}"]


def reason_cat(flags: list[str] | None) -> str:
    if not flags:
        return "NONE"
    return flags[0].split(":", 1)[0]


def suspicion_score(m: dict, tier: str | None = None) -> float:
    """可疑度：越高越该先看。"""
    tier = tier or m.get("tier") or "CLEAN"
    max_a = float(m.get("max_anom_pc") or 0.0)
    med_a = float(m.get("med_anom_pc") or 0.0)
    gpc = float(m.get("global_pc_p20") or 0.0)
    ninf = float(m.get("n_infeas") or 0.0)
    shift = float(m.get("med_shift") or 0.0)
    rel = float(m.get("med_rel") or 0.0)
    boost = {"REJECT": 1000.0, "HIGH": 100.0, "WARN": 10.0}.get(tier, 0.0)
    layer = med_a * (1.0 + max_a) * (1.0 + 3.0 * gpc) * (1.0 + ninf / 25.0)
    drift = shift * (1.0 + 20.0 * rel)
    return float(boost + max(layer, drift))


def load_global_pc() -> dict[str, float]:
    path = OUT / "layer_metrics.json"
    if not path.is_file():
        return {}
    out = {}
    for row in json.loads(path.read_text()):
        if row.get("pc_p20") is not None:
            out[row["clip_id"]] = float(row["pc_p20"])
    return out


def regress_labels(metrics: list[dict]) -> int:
    """回归人工标签；返回失败数。"""
    by_id = {m["clip_id"]: m for m in metrics}
    fails = 0
    print("\n===== LABEL REGRESSION =====")
    for lab, ids, expect in (
        ("BAD", LABEL_BAD, "REJECT"),
        ("SUSPECT", LABEL_SUSPECT, "NOT_REJECT"),
        ("GOOD", LABEL_GOOD, "NOT_REJECT"),
        ("FP_BRIDGE", LABEL_FP_BRIDGE, "NOT_REJECT"),
        ("FP_OK", LABEL_FP_OK, "NOT_REJECT"),
    ):
        for cid in sorted(ids):
            m = by_id.get(cid)
            if m is None:
                print(f"FAIL {lab:9} MISSING {cid}")
                fails += 1
                continue
            tier, flags = decide(m)
            ok = (tier == "REJECT") if expect == "REJECT" else (tier != "REJECT")
            if not ok:
                fails += 1
            print(
                f"{'OK' if ok else 'FAIL':4} {lab:9} expect={expect:10} got={tier:6} "
                f"{cid[:13]} {flags}"
            )
    print(f"regress fails: {fails}")
    return fails


def scan_clip(rec: dict) -> dict:
    cid = rec["clip_id"]
    fr = rec["frames"]
    m: dict = {"clip_id": cid, "n_frames": len(fr)}
    try:
        kin = _kinematics(fr)
        if not kin.get("ok"):
            m.update({"feasible": None, "flags_raw": ["TOO_SHORT"], "tier": "CLEAN"})
            return m
        m.update({
            "feasible": kin["feasible"],
            "n_infeas": kin["n_infeas"],
            "max_speed": kin["max_speed"],
            "max_vrate": kin["max_vrate"],
            "max_d2": kin["max_d2"],
            "max_accel": kin["max_accel"],
        })
        valid = kin["valid"]
        anom = _pick_centers(kin)
        refs = _ref_centers(kin, anom)
        m["anomaly_centers"] = anom
        m["ref_centers"] = refs

        cloud_cache: dict = {}
        anom_pcs = _pc_at_centers(cid, valid, anom, cloud_cache=cloud_cache)
        ref_pcs = _pc_at_centers(cid, valid, refs, cloud_cache=cloud_cache)
        m["max_anom_pc"] = round(float(np.max(anom_pcs)), 3) if anom_pcs else None
        m["med_anom_pc"] = round(float(np.median(anom_pcs)), 3) if anom_pcs else None
        m["med_ref_pc"] = round(float(np.median(ref_pcs)), 3) if ref_pcs else None
        if m["max_anom_pc"] is not None and m["med_ref_pc"] and m["med_ref_pc"] > 1e-6:
            m["pc_ratio"] = round(m["max_anom_pc"] / m["med_ref_pc"], 3)
        else:
            m["pc_ratio"] = None

        need_shift = bool(kin["feasible"]) or (
            m["max_anom_pc"] is None or m["max_anom_pc"] < HIGH_PC
        )
        if need_shift:
            sh = _uniform_shift_search(cid, valid, cloud_cache=cloud_cache)
            if sh:
                m.update({
                    "med_shift": round(sh["med_shift"], 3),
                    "med_rel": round(sh["med_rel"], 4),
                    "sign_consistency": round(sh["sign_consistency"], 3),
                    "n_significant": sh["n_significant"],
                    "shift_max_pc": round(sh["max_pc"], 3) if sh["max_pc"] is not None else None,
                })
        return m
    except Exception as exc:
        m.update({
            "feasible": None,
            "error": f"{type(exc).__name__}:{exc}"[:200],
            "tier": "CLEAN",
            "flags": [f"ERROR:{type(exc).__name__}"],
        })
        return m


def write_outputs(metrics: list[dict]) -> tuple[list[dict], int]:
    global_pc = load_global_pc()
    for m in metrics:
        if m.get("clip_id") in global_pc:
            m["global_pc_p20"] = global_pc[m["clip_id"]]

    fails = regress_labels(metrics)

    final = []
    for m in metrics:
        if m.get("error"):
            m["tier"] = "CLEAN"
            m["flags"] = m.get("flags") or ["ERROR"]
            continue
        tier, flags = decide(m)
        m["tier"] = tier
        m["flags"] = flags
        m["reason_cat"] = reason_cat(flags)
        m["suspicion_score"] = suspicion_score(m, tier)
        if tier == "CLEAN":
            continue
        final.append({
            "clip_id": m["clip_id"],
            "tier": tier,
            "flags": flags,
            "reason_cat": m["reason_cat"],
            "suspicion_score": m["suspicion_score"],
            "feasible": m.get("feasible"),
            "n_infeas": m.get("n_infeas"),
            "max_accel": m.get("max_accel"),
            "max_vrate": m.get("max_vrate"),
            "max_d2": m.get("max_d2"),
            "max_anom_pc": m.get("max_anom_pc"),
            "med_anom_pc": m.get("med_anom_pc"),
            "med_ref_pc": m.get("med_ref_pc"),
            "pc_ratio": m.get("pc_ratio"),
            "global_pc_p20": m.get("global_pc_p20"),
            "med_shift": m.get("med_shift"),
            "med_rel": m.get("med_rel"),
            "sign_consistency": m.get("sign_consistency"),
            "anomaly_centers": m.get("anomaly_centers"),
        })

    order = {"REJECT": 0, "HIGH": 1, "WARN": 2}
    final.sort(
        key=lambda x: (
            order[x["tier"]],
            -(x.get("suspicion_score") or 0),
            -(x.get("max_anom_pc") or 0),
            -(x.get("n_infeas") or 0),
        )
    )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "v2_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1))
    (OUT / "final_badcase_list.json").write_text(json.dumps(final, ensure_ascii=False, indent=1))
    with open(OUT / "final_badcase_list.txt", "w") as f:
        for x in final:
            f.write(
                f"{x['tier']}\t{x['clip_id']}\t{x.get('reason_cat')}\t"
                f"score={x.get('suspicion_score'):.3f}\t{','.join(x['flags'])}\t"
                f"feasible={x.get('feasible')}\tn_infeas={x.get('n_infeas')}\t"
                f"accel={x.get('max_accel')}\tvrate={x.get('max_vrate')}\td2={x.get('max_d2')}\t"
                f"anom_pc={x.get('max_anom_pc')}\tmed_anom={x.get('med_anom_pc')}\t"
                f"ref_pc={x.get('med_ref_pc')}\tratio={x.get('pc_ratio')}\t"
                f"gpc={x.get('global_pc_p20')}\tshift={x.get('med_shift')}\t"
                f"cons={x.get('sign_consistency')}\n"
            )
    rej = [x for x in final if x["tier"] == "REJECT"]
    (OUT / "reject_list_v2.txt").write_text("\n".join(x["clip_id"] for x in rej) + "\n")

    suspect = [x for x in final if x["tier"] in ("HIGH", "WARN")]
    (OUT / "suspect_ranked.json").write_text(
        json.dumps(suspect, ensure_ascii=False, indent=1)
    )
    with open(OUT / "suspect_ranked.txt", "w") as f:
        for i, x in enumerate(suspect, 1):
            f.write(
                f"{i:04d}\t{x['tier']}\t{x.get('reason_cat')}\t"
                f"{x.get('suspicion_score'):.3f}\t{x['clip_id']}\t"
                f"{','.join(x['flags'])}\tanom={x.get('max_anom_pc')}\t"
                f"med={x.get('med_anom_pc')}\tgpc={x.get('global_pc_p20')}\t"
                f"ninf={x.get('n_infeas')}\n"
            )

    print("\n===== SUMMARY =====")
    print(
        "tiers:", dict(Counter(x["tier"] for x in final)),
        "cats:", dict(Counter(x.get("reason_cat") for x in final)),
        "clean:", sum(1 for m in metrics if m.get("tier") == "CLEAN"),
        "errors:", sum(1 for m in metrics if m.get("error")),
        "total:", len(metrics),
    )
    print(f"params: T_MED={T_MED} T_GPC={T_GPC} T_MAX={T_MAX} T_NINF={T_NINF} T_DRIFT_REL={T_DRIFT_REL}")
    print(f"REJECT ({len(rej)}):")
    for x in rej:
        print(
            f"  {x['clip_id']}  score={x.get('suspicion_score'):.1f}  "
            f"{','.join(x['flags'])}  "
            f"feas={x.get('feasible')} anom_pc={x.get('max_anom_pc')} gpc={x.get('global_pc_p20')}"
        )
    print(f"suspect HIGH+WARN: {len(suspect)} → {OUT / 'suspect_ranked.txt'}")
    print(f"\nwrote {OUT / 'final_badcase_list.txt'}")
    return final, fails


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips", nargs="*", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=40)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--retier-only",
        action="store_true",
        help="reuse v2_metrics.json, only re-run decide + label regression",
    )
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    if args.retier_only:
        metrics = json.loads((OUT / "v2_metrics.json").read_text())
        print(f"retier-only: {len(metrics)} metrics", flush=True)
        _, fails = write_outputs(metrics)
        return 1 if fails else 0

    from pymongo import MongoClient

    c = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
    if args.all:
        clip_ids = [d["clip_id"] for d in c[DB][CLIPS_COL].find({}, {"clip_id": 1}) if d.get("clip_id")]
    else:
        clip_ids = list(args.clips)
    if args.limit:
        clip_ids = clip_ids[: args.limit]
    print(f"clips: {len(clip_ids)} workers={args.workers} batch={args.batch_size}", flush=True)

    print("fetching poses…", flush=True)
    recs = fetch_records(clip_ids)
    print(f"fetched {len(recs)}; scanning…", flush=True)

    checkpoint = OUT / "v2_metrics.partial.json"
    metrics: list[dict] = []
    if checkpoint.is_file() and args.resume:
        try:
            metrics = json.loads(checkpoint.read_text())
            done = {m["clip_id"] for m in metrics}
            recs = [r for r in recs if r["clip_id"] not in done]
            print(f"resume: keep {len(done)}, remain {len(recs)}", flush=True)
        except Exception:
            metrics = []

    batch_size = max(10, args.batch_size)
    workers = max(1, args.workers)
    n_batches = max(1, (len(recs) + batch_size - 1) // batch_size) if recs else 0
    for batch_i in range(0, len(recs), batch_size):
        batch = recs[batch_i: batch_i + batch_size]
        print(f"  batch {batch_i // batch_size + 1}/{n_batches} size={len(batch)}", flush=True)
        if workers <= 1:
            batch_out = [scan_clip(r) for r in batch]
        else:
            with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=8) as ex:
                batch_out = list(ex.map(scan_clip, batch, chunksize=1))
        metrics.extend(batch_out)
        checkpoint.write_text(json.dumps(metrics, ensure_ascii=False))
        print(f"  checkpoint {len(metrics)} clips", flush=True)

    if checkpoint.is_file():
        checkpoint.unlink()

    _, fails = write_outputs(metrics)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
