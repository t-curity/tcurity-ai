#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

# 실행 방식(모듈/스크립트) 상관없이 import 되게
import sys
THIS = Path(__file__).resolve()
ROOT = THIS.parents[2]  # tcurity-ai
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.phase_b.feature_extractor import extract_features_from_drag, DEFAULT_GAP_MS  # noqa


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def fmt_f(x: float) -> str:
    return f"{x:.4f}"


def load_json_first(path: Path) -> dict | None:
    raw = path.read_text(encoding="utf-8", errors="replace").lstrip()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    dec = json.JSONDecoder()
    try:
        obj, _ = dec.raw_decode(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


# --------- geometry: quadratic bezier ----------
def quad_bezier(p0, p1, p2, u: float) -> Tuple[float, float]:
    x = (1 - u) * (1 - u) * p0[0] + 2 * (1 - u) * u * p1[0] + u * u * p2[0]
    y = (1 - u) * (1 - u) * p0[1] + 2 * (1 - u) * u * p1[1] + u * u * p2[1]
    return x, y


def gen_segment_points(
    start: Tuple[float, float],
    end: Tuple[float, float],
    n_points: int,
    base_dt: float,
    dt_sigma: float,
    jitter: float,
    curve: float,
    micro_pause_p: float,
    pause_min: int,
    pause_max: int,
    start_t: int,
) -> List[Dict[str, object]]:
    sx, sy = start
    ex, ey = end

    # control point: midpoint + perpendicular offset
    mx, my = (sx + ex) / 2, (sy + ey) / 2
    dx, dy = (ex - sx), (ey - sy)
    nx, ny = (-dy, dx)
    norm = math.hypot(nx, ny) or 1.0
    nx, ny = nx / norm, ny / norm
    sign = random.choice([-1.0, 1.0])
    cx, cy = (mx + sign * nx * curve, my + sign * ny * curve)

    t = start_t
    pts: List[Dict[str, object]] = []

    for i in range(n_points):
        u = i / (n_points - 1) if n_points > 1 else 1.0
        x, y = quad_bezier((sx, sy), (cx, cy), (ex, ey), u)

        # mid에서 더 흔들리게 (사람 느낌)
        scale = 0.4 + 0.6 * (1.0 - abs(0.5 - u) * 2)
        x += random.gauss(0, jitter * scale)
        y += random.gauss(0, jitter * scale)

        dt = base_dt + random.gauss(0, dt_sigma)
        dt = max(8.0, min(40.0, dt))

        if random.random() < micro_pause_p:
            dt += random.randint(pause_min, pause_max)

        t += int(round(dt))
        pts.append({"x": fmt_f(clamp01(x)), "y": fmt_f(clamp01(y)), "t": int(t)})

    return pts


def gen_human_matched_points(hfeat: Dict[str, float]) -> List[Dict[str, object]]:
    """
    사람 feature를 타겟으로 세그먼트 수/idle gap/dt 분포를 맞춘 points 생성
    """
    drag_count = int(max(1, round(hfeat.get("drag_count", 2.0))))
    idle_mean = float(hfeat.get("idle_time_mean", 0.0))
    idle_std = float(hfeat.get("idle_time_std", 0.0))
    avg_dt = float(hfeat.get("avg_dt", 16.0))
    dt_std = float(hfeat.get("dt_std", 3.0))

    # 사람 분포를 그대로 쓰되, 봇이 “완전히 동일”하면 안 되니까 살짝만 노이즈
    base_dt = max(10.0, min(22.0, avg_dt + random.gauss(0, 1.0)))
    dt_sigma = max(1.5, min(8.0, dt_std * 0.6 + 1.5))

    # path_efficiency를 사람과 겹치게 만들기: jitter/curve 과하지 않게
    jitter = 0.0025 + random.random() * 0.0015          # 0.0025~0.0040
    curve = 0.035 + random.random() * 0.030             # 0.035~0.065

    # micro pause는 너무 많으면 바로 들킴 → 낮게
    micro_pause_p = 0.01 + random.random() * 0.02       # 1~3%
    pause_min, pause_max = 40, 120

    # 세그먼트별 점수
    seg_points = []
    total_pts = random.randint(70, 120)
    # seg마다 적당히 분배
    splits = [0] + sorted(random.sample(range(10, total_pts - 10), k=max(0, drag_count - 1))) + [total_pts]
    counts = [splits[i + 1] - splits[i] for i in range(drag_count)]
    counts = [max(12, c) for c in counts]  # 너무 짧으면 특징이 비어짐

    # start/end는 랜덤하지만 화면 안에서 자연스럽게
    sx, sy = random.random(), random.random()
    start_t = 0
    pts_all: List[Dict[str, object]] = []
    cur = (sx, sy)

    for si in range(drag_count):
        # end는 근처로 조금 이동(너무 멀면 이상)
        ex = clamp01(cur[0] + random.gauss(0.12, 0.08))
        ey = clamp01(cur[1] + random.gauss(0.10, 0.08))
        end = (ex, ey)

        seg = gen_segment_points(
            start=cur,
            end=end,
            n_points=counts[si],
            base_dt=base_dt,
            dt_sigma=dt_sigma,
            jitter=jitter,
            curve=curve,
            micro_pause_p=micro_pause_p,
            pause_min=pause_min,
            pause_max=pause_max,
            start_t=start_t,
        )
        pts_all.extend(seg)

        # idle gap 삽입: 사람의 idle_mean/std를 목표로
        if si < drag_count - 1:
            if idle_mean <= 0:
                gap = DEFAULT_GAP_MS + random.randint(10, 80)
            else:
                gap = int(max(DEFAULT_GAP_MS + 10, random.gauss(idle_mean, max(20.0, idle_std))))
            # 다음 세그 시작 시간을 점프
            start_t = int(pts_all[-1]["t"]) + gap
        cur = end

    return pts_all


def build_bot_like(human: dict, points: List[Dict[str, object]]) -> dict:
    meta = human.get("metadata", {}) or {}
    res = meta.get("res") or {"w": 0, "h": 0}
    return {
        "label": "bot",
        "timestamp": iso_now(),
        "target": human.get("target", "unknown"),
        "correct_count": int(human.get("correct_count", 4)),
        "is_perfect": bool(human.get("is_perfect", True)),
        "points": points,
        "metadata": {
            "ua": meta.get("ua", ""),
            "res": res,
            "sim": "bot_human_matched_v1",
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human-dir", type=str, default="data/phase_b/human")
    ap.add_argument("--out-dir", type=str, default="data/phase_b/bot")
    ap.add_argument("--per-human", type=int, default=1)
    ap.add_argument("--max-out", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    human_dir = (ROOT / args.human_dir).resolve()
    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)

    files = sorted(human_dir.rglob("*.json"))
    if not files:
        raise SystemExit(f"No human json under {human_dir}")

    created = 0
    for fp in files:
        human = load_json_first(fp)
        if not human:
            continue

        hfeat = extract_features_from_drag(human)
        if not hfeat:
            continue

        for k in range(args.per_human):
            pts = gen_human_matched_points(hfeat)
            bot = build_bot_like(human, pts)

            out_name = f"{fp.stem}_bot_m_{k+1:02d}.json"
            (out_dir / out_name).write_text(json.dumps(bot, ensure_ascii=False, indent=2), encoding="utf-8")
            created += 1
            if args.max_out and created >= args.max_out:
                break
        if args.max_out and created >= args.max_out:
            break

    print(f"[OK] created={created} -> {out_dir}")


if __name__ == "__main__":
    main()
