#!/usr/bin/env python3
"""[DEPRECATED] 旧四路合并逻辑；最终清单请用 tools/pose_badcase_v2.py。

v2 流程：车辆可达性 → 诡异窗口 → PC/pose-shift 确认后才 REJECT。
本脚本仅保留便于对照旧结果。

旧规则（过松，会把大量运动学尖刺但无鬼影的 clip 标成 REJECT）:
REJECT: gate 拒掉 ∨ 坏帧占比≥5% ∨ 点云跨帧错位 pc_p20≥0.45m
HIGH  : 坏帧占比 2-5% ∨ pc_p20 0.35-0.45m
WARN  : 有 pose 异常 flag 但坏帧占比 <2%（单帧尖刺，肉眼不可见）
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "exp/robotruck/pose_badcase"

REJECT_RATIO = 0.05
HIGH_RATIO = 0.02
REJECT_PC = 0.45
HIGH_PC = 0.35


def main() -> int:
    gate = json.load(open(OUT / "gate_scan_122.json"))
    layer = {m["clip_id"]: m for m in json.load(open(OUT / "layer_metrics.json"))}
    ratio = {cid: v.get("ratio") for cid, v in
             json.load(open(OUT / "badframe_ratio.json")).items()}

    rows = {}
    for row in gate["clips"]:
        rows[row["clip_id"]] = {
            "clip_id": row["clip_id"], "gate_reasons": row["reasons"],
            "pose_shift": row["geometry_quality"].get("pose", {}).get("median_abs_shift"),
            "gate_layer": row["geometry_quality"].get("layer", {}).get("median_score")}
    for cid, m in layer.items():
        r = rows.setdefault(cid, {"clip_id": cid, "gate_reasons": [], "pose_shift": None,
                                  "gate_layer": None})
        r.update({"pose_flags": m.get("flags", []), "max_accel": m.get("max_accel"),
                  "max_vrate": m.get("max_vrate"), "pc_p20": m.get("pc_p20")})
    for cid, r in rows.items():
        r["bad_ratio"] = ratio.get(cid)

    final = []
    for cid, r in rows.items():
        br = r.get("bad_ratio")
        br = -1.0 if br is None else br
        pc = r.get("pc_p20") or 0
        if r["gate_reasons"]:
            tier, why = "REJECT", [f"GATE:{x}" for x in r["gate_reasons"]]
        elif br >= REJECT_RATIO:
            tier, why = "REJECT", [f"BAD_FRAMES:{r.get('bad_ratio'):.1%}"]
        elif pc >= REJECT_PC:
            tier, why = "REJECT", [f"PC_LAYER:{pc}"]
        elif br >= HIGH_RATIO:
            tier, why = "HIGH", [f"BAD_FRAMES:{r.get('bad_ratio'):.1%}"]
        elif pc >= HIGH_PC:
            tier, why = "HIGH", [f"PC_LAYER:{pc}"]
        elif r.get("pose_flags"):
            tier, why = "WARN", r["pose_flags"]
        else:
            continue
        final.append({"clip_id": cid, "tier": tier, "flags": why,
                      "pose_flags": r.get("pose_flags", []), "bad_ratio": r.get("bad_ratio"),
                      "max_accel": r.get("max_accel"), "max_vrate": r.get("max_vrate"),
                      "pc_p20": r.get("pc_p20"), "pose_shift": r.get("pose_shift"),
                      "gate_layer": r.get("gate_layer")})
    order = {"REJECT": 0, "HIGH": 1, "WARN": 2}
    final.sort(key=lambda x: (order[x["tier"]], -(x.get("bad_ratio") or 0)))
    (OUT / "final_badcase_list.json").write_text(json.dumps(final, ensure_ascii=False, indent=1))
    with open(OUT / "final_badcase_list.txt", "w") as f:
        for x in final:
            f.write(f"{x['tier']}\t{x['clip_id']}\t{','.join(x['flags'])}\t"
                    f"pose_flags={','.join(x['pose_flags'])}\tbad_ratio={x['bad_ratio']}\t"
                    f"accel={x['max_accel']}\tvrate={x['max_vrate']}\tpc_p20={x['pc_p20']}\t"
                    f"shift={x['pose_shift']}\n")
    print("tiers:", dict(Counter(x["tier"] for x in final)), "total:", len(final), "/ 909")
    print("\nREJECT (%d):" % sum(1 for x in final if x["tier"] == "REJECT"))
    for x in final:
        if x["tier"] == "REJECT":
            print(f"  {x['clip_id']}  {','.join(x['flags'])}  "
                  f"nflags={','.join(x['pose_flags'])}  accel={x['max_accel']}  pc={x['pc_p20']}")
    print("\n3 target clips:")
    for t in ["0038952f-2da1-4eb1-8b40-83b728debfaa",
              "3c4ddc70-dd06-4c5b-9c01-5fd55e29a246",
              "0c85fa31-dd09-4adc-a2d9-ab76725e438e",
              "6bc60101-e684-438b-9983-b74c1cc5fe2b",
              "701ff460-2b29-4433-b492-39e7ae2310b6"]:
        hit = [x for x in final if x["clip_id"] == t]
        print(" ", t[:13], "->", (hit[0]["tier"], hit[0]["flags"]) if hit else "CLEAN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
