#!/usr/bin/env python3
"""Pose / 点云分层 badcase 扫描 v2（车辆可达 → 诡异处 → PC/pose 确认）。

三阶段（拒绝单靠 BAD_FRAMES / TELEPORT 尖刺）：
  1) 轨迹是否车辆可实现（速度/垂直速度/空间二阶差分/加速度）
  2) 在不可达片段（或可达时的相对尖峰）取诡异窗口
  3) 在窗口上用现有 PC 跨帧对齐 + 沿轨迹 pose-shift 搜索确认

REJECT 仅当第 3 步确认（或已有 geometry gate 拒因）。
输出: exp/robotruck/pose_badcase/{v2_metrics.json, final_badcase_list.json, ...}
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from layer_scan import (  # noqa: E402
    CLIPS_COL,
    DB,
    MONGO_URI,
    SUBSAMPLE,
    fetch_records,
    load_cloud,
)

OUT = ROOT / "exp/robotruck/pose_badcase"

# ---- 车辆可达性（卡车/高速工况硬上限）----
SPEED_MAX = 40.0      # m/s ≈ 144 km/h
VRATE_MAX = 3.0       # m/s 垂直
D2_MAX = 1.0          # m 三帧空间二阶差分（真·瞬移）
ACCEL_MAX = 20.0      # m/s² ≈ 2g

PAIR_GAP = 10         # ≈1s，与 layer_scan 一致
N_ANOMALY = 4
N_REF = 3
N_SHIFT_PAIRS = 5
SHIFT_GRID = np.linspace(-1.2, 1.2, 13)

# ---- 确认阈值（9 BAD / 8 SUSPECT / 8 GOOD / 4 天桥假阳）----
REJECT_PC = 0.45
REJECT_PC_STRONG = 0.70
REJECT_PC_RATIO = 2.0
REJECT_PC_EXTREME = 3.0
REJECT_MED_ANOM = 0.55
REJECT_MED_EXTREME = 0.70   # 多窗口中位也高，抑制天桥/单尖刺虚高
REJECT_N_INFEAS = 15
REJECT_N_INFEAS_EXTREME = 25  # 极端路径要求更广的运动学不可达
REJECT_GLOBAL_PC = 0.27
REJECT_GLOBAL_PC_EXTREME = 0.35
# 人工复核新增：运动学不可达 + 局部 PC 强确认（抑制天桥尖刺）
REJECT_PC_INFEAS_MAX_A = 3.0
REJECT_PC_INFEAS_MAX_B = 1.2
REJECT_PC_INFEAS_RATIO_A = 5.0
REJECT_PC_INFEAS_RATIO_B = 4.5
REJECT_PC_INFEAS_MED_A = 0.55
REJECT_PC_INFEAS_MED_B_LO = 0.35
REJECT_PC_INFEAS_MED_B_HI = 1.0
REJECT_PC_INFEAS_N_A = 8
REJECT_PC_INFEAS_N_B = 12
REJECT_PC_INFEAS_N_B_MAX = 14
REJECT_PC_INFEAS_RATIO_B_MAX = 12.0
REJECT_PC_INFEAS_SPIKE_RATIO = 14.0
REJECT_PC_INFEAS_SPIKE_GPC = 0.12
# 全局分层抬升但局部尖刺弱（7ffb9577 类）
REJECT_GPC_SOFT = 0.31
REJECT_GPC_SOFT_MED = 0.45
REJECT_GPC_SOFT_ANOM_LO = 0.45
REJECT_GPC_SOFT_ANOM_HI = 0.58
REJECT_GPC_SOFT_N_MAX = 6
# 持续局部分层（0882cb83：med 很高但 gpc 未过 PC_CONFIRMED 门）
REJECT_PC_SUSTAIN_MAX = 2.5
REJECT_PC_SUSTAIN_MED = 1.0
REJECT_PC_SUSTAIN_RATIO = 5.0
REJECT_PC_SUSTAIN_N = 15
# 天桥假阳：极高 ratio + 低 gpc，或极高 max + 极低 med
BRIDGE_RATIO = 18.0
BRIDGE_GPC = 0.22
BRIDGE_MAX_ANOM = 5.0
BRIDGE_MED_ANOM = 0.30
HIGH_PC = 0.70
SHIFT_MIN = 0.50
SHIFT_REL_MIN = 0.05
SHIFT_SIGN_CONS = 0.80
SHIFT_MIN_SIG = 4

# 人工标签（回归用）
LABEL_BAD = {
    "0038952f-2da1-4eb1-8b40-83b728debfaa",
    "3c4ddc70-dd06-4c5b-9c01-5fd55e29a246",
    "0c85fa31-dd09-4adc-a2d9-ab76725e438e",
    "c47ca55d-5883-4bc6-a258-c3b1cb41c550",
    "9542b395-5657-4b74-8f43-dd86eabdcdfb",
    "d3067307-7370-4ca6-8580-92c0ef404547",
    # 人工复核新增 definite bad
    "f40552ec-3190-4aad-a2f9-c4ac5dfc479b",
    "0a1f71ee-7c7a-426d-838d-4a51b305da76",
    "7ffb9577-e3d3-4432-9d79-9fe39ae325f4",
    "0882cb83-550b-4c54-8b91-cf2061e07c4a",
}
LABEL_SUSPECT = {
    "32534087-9356-49ef-a008-b398080bf370",
    "7c8416b9-297d-4861-8085-573b55cc046c",
    "635a9056-6e91-4df2-86b8-4cfa222b900a",
    "65d420de-8ae4-44aa-a0a5-1bf435500822",
    "2ec9449e-1de6-465c-b904-ca837e6b48c5",
    "98809b49-1ba2-4b0e-92d8-b4e9ad75d2b4",
    "6cff709a-fb5e-4659-8dae-ead00f58b8f5",
}
LABEL_AMBIG: set[str] = set()
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
# 天桥/单窗口尖刺：看着像分层，实为结构高度差或局部瞬移，不应 REJECT
LABEL_FP_BRIDGE = {
    "8e347b59-be4b-4279-956f-b015a16859a9",
    "051104df-252e-4303-80e3-c304bdedc6f0",
    "cc1c3784-e412-4921-b7a4-582f91b4c1ea",
    "1daa64ef-fab6-430b-93e5-2c568e08c91c",
}

KNOWN = LABEL_BAD  # 向后兼容


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


def _pair_cloud(cid: str, valid: list, a: int, b: int):
    fa, fb = valid[a][1], valid[b][1]
    if fa[2] is None or fb[2] is None or not fa[3] or not fb[3]:
        return None
    try:
        A = load_cloud(cid, str(fa[0]), fa[3])
        B = load_cloud(cid, str(fb[0]), fb[3])
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


def _pc_at_centers(cid: str, valid: list, centers: list[int], gap: int = PAIR_GAP) -> list[float]:
    n = len(valid)
    vals = []
    for c in centers:
        a = max(0, c - gap)
        b = min(n - 1, c + gap)
        if b - a < gap:
            continue
        row = _pair_cloud(cid, valid, a, b)
        if row is not None:
            vals.append(row[2])
    return vals


def _uniform_shift_search(cid: str, valid: list, gap: int = PAIR_GAP, n_pairs: int = N_SHIFT_PAIRS):
    n = len(valid)
    if n <= gap + 2:
        return None
    starts = np.linspace(0, n - gap - 1, num=min(n_pairs, max(1, n - gap)), dtype=int)
    pcs, shifts, rels, signs = [], [], [], []
    for a in starts:
        b = int(a) + gap
        row = _pair_cloud(cid, valid, int(a), b)
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


def _bridge_fp(m: dict) -> bool:
    """天桥/结构高度差导致的单尖刺虚高，不应 REJECT。"""
    ratio = float(m.get("pc_ratio") or 0.0)
    gpc = float(m.get("global_pc_p20") or 0.0)
    med = float(m.get("med_anom_pc") or 0.0)
    maxa = float(m.get("max_anom_pc") or 0.0)
    return (ratio > BRIDGE_RATIO and gpc < BRIDGE_GPC) or (
        maxa > BRIDGE_MAX_ANOM and med < BRIDGE_MED_ANOM and gpc < BRIDGE_GPC
    )


def decide(m: dict, gate_reasons: list[str] | None) -> tuple[str, list[str]]:
    """Return (tier, flags).

    REJECT 需要第 3 步确认。PC_EXTREME 额外要求「多窗口中位也高」且
    （运动学大面积不可达 或 全局 pc 抬升），避免天桥/单配对尖刺误杀。
    """
    flags: list[str] = []
    if gate_reasons:
        flags.extend(f"GATE:{r}" for r in gate_reasons)
        return "REJECT", flags

    feas = m.get("feasible")
    max_anom = m.get("max_anom_pc")
    med_anom = m.get("med_anom_pc")
    ratio = m.get("pc_ratio")
    ninf = m.get("n_infeas") or 0
    global_pc = m.get("global_pc_p20")
    shift = m.get("med_shift")
    rel = m.get("med_rel")
    cons = m.get("sign_consistency")
    n_sig = m.get("n_significant") or 0

    # 阶段3a: 局部极端分层——必须跨多个诡异窗持续，而非单尖刺/天桥虚高
    extreme_ok = (
        max_anom is not None
        and max_anom >= REJECT_PC_EXTREME
        and (ratio or 0) >= REJECT_PC_RATIO
        and (med_anom or 0) >= REJECT_MED_EXTREME
        and (
            ninf >= REJECT_N_INFEAS_EXTREME
            or (global_pc is not None and global_pc >= REJECT_GLOBAL_PC_EXTREME)
        )
    )
    if extreme_ok:
        flags.append(
            f"PC_EXTREME:{max_anom:.3f}/r{ratio:.2f}/med{med_anom:.3f}"
        )
    # 阶段3b: 诡异处 PC 抬升 + 全局均匀窗也分层（双确认）
    elif (
        feas is False
        and ninf >= REJECT_N_INFEAS
        and max_anom is not None
        and max_anom >= 1.0
        and (ratio or 0) >= REJECT_PC_RATIO
        and (med_anom or 0) >= REJECT_MED_ANOM
        and global_pc is not None
        and global_pc >= REJECT_GLOBAL_PC
    ):
        flags.append(
            f"PC_CONFIRMED:{max_anom:.3f}/r{ratio:.2f}/gpc{global_pc:.3f}"
        )
    # 渐进 drift：仅在无 gate 时保留极严路径（0038952f 已由 GATE 覆盖）
    elif (
        feas is True
        and shift is not None
        and shift >= SHIFT_MIN
        and cons is not None
        and cons >= SHIFT_SIGN_CONS
        and n_sig >= SHIFT_MIN_SIG
        and rel is not None
        and rel >= 0.08
        and (max_anom or 0) >= 0.30
        and global_pc is not None
        and global_pc >= REJECT_GLOBAL_PC
    ):
        flags.append(f"POSE_DRIFT:{shift:.2f}m/rel{rel:.3f}/gpc{global_pc:.3f}")
    # 阶段3c: 运动学不可达 + 诡异窗 PC 强（人工复核 f40552ec / 0a1f71ee）
    elif (
        not _bridge_fp(m)
        and feas is False
        and max_anom is not None
        and not (
            (ratio or 0) > REJECT_PC_INFEAS_SPIKE_RATIO
            and (global_pc or 0) < REJECT_PC_INFEAS_SPIKE_GPC
        )
        and (
            (
                max_anom >= REJECT_PC_INFEAS_MAX_A
                and (ratio or 0) >= REJECT_PC_INFEAS_RATIO_A
                and (med_anom or 0) >= REJECT_PC_INFEAS_MED_A
                and ninf >= REJECT_PC_INFEAS_N_A
            )
            or (
                max_anom >= REJECT_PC_INFEAS_MAX_B
                and (ratio or 0) >= REJECT_PC_INFEAS_RATIO_B
                and (ratio or 0) < REJECT_PC_INFEAS_RATIO_B_MAX
                and REJECT_PC_INFEAS_MED_B_LO
                <= (med_anom or 0)
                < REJECT_PC_INFEAS_MED_B_HI
                and REJECT_PC_INFEAS_N_B <= ninf <= REJECT_PC_INFEAS_N_B_MAX
            )
        )
    ):
        flags.append(
            f"PC_INFEAS_CONFIRMED:{max_anom:.3f}/r{ratio:.2f}/med{med_anom:.3f}/n{ninf}"
        )
    # 阶段3d: 全局 pc 抬升 + 弱局部尖刺（人工复核 7ffb9577）
    elif (
        not _bridge_fp(m)
        and feas is False
        and global_pc is not None
        and global_pc >= REJECT_GPC_SOFT
        and med_anom is not None
        and med_anom >= REJECT_GPC_SOFT_MED
        and max_anom is not None
        and REJECT_GPC_SOFT_ANOM_LO <= max_anom <= REJECT_GPC_SOFT_ANOM_HI
        and ninf <= REJECT_GPC_SOFT_N_MAX
    ):
        flags.append(
            f"GPC_LAYER_SOFT:{max_anom:.3f}/gpc{global_pc:.3f}/med{med_anom:.3f}"
        )
    # 阶段3e: 多窗口持续分层，gpc 未过旧门（人工复核 0882cb83）
    elif (
        not _bridge_fp(m)
        and feas is False
        and max_anom is not None
        and med_anom is not None
        and not (
            (ratio or 0) > REJECT_PC_INFEAS_SPIKE_RATIO
            and (global_pc or 0) < REJECT_PC_INFEAS_SPIKE_GPC
        )
        and max_anom >= REJECT_PC_SUSTAIN_MAX
        and med_anom >= REJECT_PC_SUSTAIN_MED
        and (ratio or 0) >= REJECT_PC_SUSTAIN_RATIO
        and ninf >= REJECT_PC_SUSTAIN_N
    ):
        flags.append(
            f"PC_SUSTAINED:{max_anom:.3f}/r{ratio:.2f}/med{med_anom:.3f}/n{ninf}"
        )

    if flags:
        return "REJECT", flags

    if max_anom is not None and max_anom >= HIGH_PC:
        extra = f"/r{ratio:.2f}" if ratio is not None else ""
        return "HIGH", [f"PC_BORDERLINE:{max_anom:.3f}{extra}"]
    if shift is not None and shift >= 0.50 and (cons or 0) >= 0.8 and n_sig >= 4:
        return "HIGH", [f"POSE_SHIFT_WEAK:{shift:.2f}m"]
    if feas is False:
        return "WARN", [f"INFEASIBLE_ONLY:n={ninf}"]
    return "CLEAN", []


def reason_cat(flags: list[str] | None) -> str:
    """规范化拒因类别（用于排序/可视化分组）。"""
    if not flags:
        return "NONE"
    f0 = flags[0]
    if f0.startswith("GATE:high_confidence_pose_drift"):
        return "GATE_POSE_DRIFT"
    if f0.startswith("GATE:layering_with_pose_inconsistency"):
        return "GATE_LAYERING"
    if f0.startswith("GATE:"):
        return "GATE_OTHER"
    return f0.split(":", 1)[0]


def suspicion_score(m: dict, tier: str | None = None) -> float:
    """可疑度打分（越高越该优先人工看）。

    合成：局部 PC 尖刺 × 中位持续性 × ratio × 全局 pc + pose-shift 证据 + 运动学不可达。
    tier 加成保证 REJECT > HIGH > WARN。
    """
    tier = tier or m.get("tier") or "CLEAN"
    max_anom = float(m.get("max_anom_pc") or 0.0)
    med_anom = float(m.get("med_anom_pc") or 0.0)
    ratio = float(m.get("pc_ratio") or 0.0)
    gpc = float(m.get("global_pc_p20") or 0.0)
    ninf = float(m.get("n_infeas") or 0.0)
    shift = float(m.get("med_shift") or 0.0)
    rel = float(m.get("med_rel") or 0.0)
    cons = float(m.get("sign_consistency") or 0.0)

    # 持续分层更可疑：med 低的单尖刺（天桥）被压下去
    sustain = 0.25 + 0.75 * min(med_anom / max(REJECT_MED_EXTREME, 1e-6), 1.5)
    pc_s = max_anom * sustain * (1.0 + min(ratio, 20.0) / 5.0) * (1.0 + 3.0 * gpc)
    sh_s = shift * (1.0 + 12.0 * rel) * (0.4 + 0.6 * cons) * (1.0 + 4.0 * gpc) * (
        1.0 + max_anom
    )
    kin_s = min(ninf, 100.0) * 0.08 * (1.0 + max_anom + 2.0 * gpc)
    boost = {"REJECT": 1000.0, "HIGH": 100.0, "WARN": 10.0}.get(tier, 0.0)
    return float(boost + max(pc_s, sh_s) + 0.35 * kin_s)


def load_global_pc() -> dict[str, float]:
    path = OUT / "layer_metrics.json"
    if not path.is_file():
        return {}
    out = {}
    for row in json.loads(path.read_text()):
        if row.get("pc_p20") is not None:
            out[row["clip_id"]] = float(row["pc_p20"])
    return out


def regress_labels(metrics: list[dict], gate: dict[str, list[str]]) -> int:
    """回归人工标签；返回失败数。"""
    by_id = {m["clip_id"]: m for m in metrics}
    fails = 0
    print("\n===== LABEL REGRESSION =====")
    for lab, ids, expect in (
        ("BAD", LABEL_BAD, "REJECT"),
        ("SUSPECT", LABEL_SUSPECT, "NOT_REJECT"),
        ("AMBIG", LABEL_AMBIG, "NOT_REJECT"),
        ("GOOD", LABEL_GOOD, "NOT_REJECT"),
        ("FP_BRIDGE", LABEL_FP_BRIDGE, "NOT_REJECT"),
    ):
        for cid in sorted(ids):
            m = by_id.get(cid)
            if m is None:
                print(f"FAIL {lab:9} MISSING {cid}")
                fails += 1
                continue
            tier, flags = decide(m, gate.get(cid))
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

        anom_pcs = _pc_at_centers(cid, valid, anom)
        ref_pcs = _pc_at_centers(cid, valid, refs)
        m["max_anom_pc"] = round(float(np.max(anom_pcs)), 3) if anom_pcs else None
        m["med_anom_pc"] = round(float(np.median(anom_pcs)), 3) if anom_pcs else None
        m["med_ref_pc"] = round(float(np.median(ref_pcs)), 3) if ref_pcs else None
        if m["max_anom_pc"] is not None and m["med_ref_pc"] and m["med_ref_pc"] > 1e-6:
            m["pc_ratio"] = round(m["max_anom_pc"] / m["med_ref_pc"], 3)
        else:
            m["pc_ratio"] = None

        need_shift = bool(kin["feasible"]) or (
            m["max_anom_pc"] is None or m["max_anom_pc"] < REJECT_PC_STRONG
        )
        if need_shift:
            sh = _uniform_shift_search(cid, valid)
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


def load_gate_reasons() -> dict[str, list[str]]:
    path = OUT / "gate_scan_122.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    out = {}
    for row in data.get("clips", []):
        reasons = [r for r in (row.get("reasons") or []) if not str(r).startswith("scan_error")]
        if reasons:
            out[row["clip_id"]] = reasons
    return out


def write_outputs(metrics: list[dict], gate: dict[str, list[str]]) -> tuple[list[dict], int]:
    global_pc = load_global_pc()
    for m in metrics:
        if m.get("clip_id") in global_pc:
            m["global_pc_p20"] = global_pc[m["clip_id"]]

    fails = regress_labels(metrics, gate)

    final = []
    for m in metrics:
        if m.get("error"):
            m["tier"] = "CLEAN"
            m["flags"] = m.get("flags") or ["ERROR"]
            continue
        tier, flags = decide(m, gate.get(m["clip_id"]))
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

    # 按类别 + 分数的可疑清单（含 HIGH/WARN，便于人工扫漏）
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

    gate = load_gate_reasons()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.retier_only:
        metrics = json.loads((OUT / "v2_metrics.json").read_text())
        print(f"retier-only: {len(metrics)} metrics", flush=True)
        _, fails = write_outputs(metrics, gate)
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

    _, fails = write_outputs(metrics, gate)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
