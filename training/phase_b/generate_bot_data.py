# python training/phase_b/generate_bot_data.py \
#  --per-human 1 --max-out 3000 --overshoot --multiseg
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


# ==========================================================
# Time / Format
# ==========================================================
def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def fmt_f(x: float) -> str:
    return f"{x:.4f}"


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


# ==========================================================
# Robust JSON loader (handles "Extra data", NDJSON, junk tail)
# ==========================================================
def load_json_objects(text: str) -> List[dict]:
    text = text.lstrip()
    if not text:
        return []
    try:
        obj = json.loads(text)
        return [obj] if isinstance(obj, dict) else []
    except json.JSONDecodeError:
        pass

    dec = json.JSONDecoder()
    objs: List[dict] = []
    idx = 0
    n = len(text)

    while idx < n:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        try:
            obj, next_idx = dec.raw_decode(text, idx)
            if isinstance(obj, dict):
                objs.append(obj)
            idx = next_idx
        except json.JSONDecodeError:
            break

    return objs


def load_first_json(path: Path) -> dict | None:
    raw = path.read_text(encoding="utf-8", errors="replace")
    objs = load_json_objects(raw)
    return objs[0] if objs else None


# ==========================================================
# Human-ish bot points generator
# - curved path (quadratic bezier)
# - jitter
# - variable dt + micro pauses
# - optional overshoot correction
# - optional multi-segment time gaps (to mimic multiple drags)
# ==========================================================
def generate_humanish_points(
    start: Tuple[float, float],
    end: Tuple[float, float],
    start_t: int,
    n_points: int,
    base_dt: int,
    jitter: float,
    curve: float,
    micro_pause_p: float,
    overshoot: bool,
) -> List[Dict[str, object]]:
    sx, sy = start
    ex, ey = end

    # control point for quadratic bezier: midpoint + perpendicular offset
    mx, my = (sx + ex) / 2, (sy + ey) / 2
    dx, dy = (ex - sx), (ey - sy)
    nx, ny = (-dy, dx)
    norm = math.hypot(nx, ny) or 1.0
    nx, ny = nx / norm, ny / norm
    sign = random.choice([-1.0, 1.0])
    cx, cy = (mx + sign * nx * curve, my + sign * ny * curve)

    t_cur = start_t
    pts: List[Dict[str, object]] = []

    for i in range(max(2, n_points)):
        u = i / (n_points - 1) if n_points > 1 else 1.0

        # quadratic bezier
        x = (1 - u) * (1 - u) * sx + 2 * (1 - u) * u * cx + u * u * ex
        y = (1 - u) * (1 - u) * sy + 2 * (1 - u) * u * cy + u * u * ey

        # jitter (slightly larger mid-way)
        scale = 0.4 + 0.6 * (1.0 - abs(0.5 - u) * 2)
        x += random.gauss(0, jitter * scale)
        y += random.gauss(0, jitter * scale)

        # variable dt
        dt = base_dt + int(random.gauss(0, 6))
        dt = max(8, min(55, dt))

        # micro pause
        if random.random() < micro_pause_p:
            dt += random.randint(60, 220)

        t_cur += dt
        pts.append({"x": fmt_f(clamp01(x)), "y": fmt_f(clamp01(y)), "t": int(t_cur)})

    # overshoot + correction (very human-ish)
    if overshoot and n_points >= 30:
        ox = ex + random.choice([-1, 1]) * random.uniform(0.002, 0.012)
        oy = ey + random.choice([-1, 1]) * random.uniform(0.002, 0.012)
        for _ in range(random.randint(2, 4)):
            t_cur += random.randint(12, 30)
            pts.append({"x": fmt_f(clamp01(ox)), "y": fmt_f(clamp01(oy)), "t": int(t_cur)})
        for _ in range(random.randint(2, 4)):
            t_cur += random.randint(12, 30)
            pts.append({"x": fmt_f(clamp01(ex)), "y": fmt_f(clamp01(ey)), "t": int(t_cur)})

    return pts


def add_multisegment_time_gaps(points: List[Dict[str, object]], segments: int) -> List[Dict[str, object]]:
    """
    points는 연속이지만, 시간에 큰 gap을 넣어서 extractor가 segment로 나누게 함.
    """
    if segments <= 1 or len(points) < 40:
        return points

    segments = min(segments, 5)
    cut_count = segments - 1
    cuts = sorted(random.sample(range(10, len(points) - 10), k=cut_count))

    out: List[Dict[str, object]] = []
    gap_added = 0
    last = 0
    for cut in cuts + [len(points)]:
        chunk = points[last:cut]
        if not chunk:
            continue
        for p in chunk:
            out.append({"x": p["x"], "y": p["y"], "t": int(p["t"]) + gap_added})
        # gap must exceed your DEFAULT_GAP_MS (e.g. 250ms)
        gap_added += random.randint(280, 1100)
        last = cut
    return out


def pick_start_end_from_human(sample: dict) -> Tuple[Tuple[float, float], Tuple[float, float], int]:
    pts = sample.get("points", [])
    if not pts or len(pts) < 2:
        sx, sy = random.random(), random.random()
        ex, ey = random.random(), random.random()
        return (sx, sy), (ex, ey), 0

    s = pts[0]
    e = pts[-1]
    start = (safe_float(s.get("x", 0.1), 0.1), safe_float(s.get("y", 0.1), 0.1))
    end = (safe_float(e.get("x", 0.9), 0.9), safe_float(e.get("y", 0.9), 0.9))
    start_t = safe_int(s.get("t", 0), 0)
    return start, end, start_t


def build_bot_sample_like(human_sample: dict, points: List[Dict[str, object]]) -> dict:
    meta = human_sample.get("metadata", {}) or {}
    res = (meta.get("res") or {"w": 0, "h": 0})

    # 결과값(correct_count/is_perfect)은 운영에서 입력피처로 쓰지 않는 걸 권장하지만,
    # 스키마 맞추려고 일단 human 값 복사 (원하면 랜덤으로 약간 흔들리게 바꿔도 됨)
    return {
        "label": "bot",
        "timestamp": iso_now(),
        "target": human_sample.get("target", "unknown"),
        "correct_count": safe_int(human_sample.get("correct_count", 4), 4),
        "is_perfect": bool(human_sample.get("is_perfect", True)),
        "points": points,
        "metadata": {
            "ua": meta.get("ua", ""),
            "vendor": meta.get("vendor", ""),
            "res": res,
            "sim": "bot_humanish_v1",
        },
    }


def main():
    ap = argparse.ArgumentParser(description="Phase B human-ish bot data generator")
    ap.add_argument("--human-dir", type=str, default="data/phase_b/human")
    ap.add_argument("--out-dir", type=str, default="data/phase_b/bot")
    ap.add_argument("--seed", type=int, default=42)

    # how many bots to create
    ap.add_argument("--per-human", type=int, default=1, help="human 1개당 human-ish bot 몇 개 만들지")
    ap.add_argument("--max-out", type=int, default=3000, help="최대 생성 파일 수 (0이면 제한 없음)")

    # behavior knobs
    ap.add_argument("--min-points", type=int, default=55)
    ap.add_argument("--max-points", type=int, default=110)
    ap.add_argument("--base-dt", type=int, default=16)
    ap.add_argument("--jitter", type=float, default=0.004)
    ap.add_argument("--curve", type=float, default=0.07)
    ap.add_argument("--micro-pause-p", type=float, default=0.06)
    ap.add_argument("--overshoot", action="store_true", help="끝에서 살짝 지나쳤다 되돌리기")
    ap.add_argument("--multiseg", action="store_true", help="여러 드래그처럼 시간 gap 넣기")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]  # tcurity-ai
    human_dir = (root / args.human_dir).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)

    human_files = sorted(human_dir.rglob("*.json"))
    if not human_files:
        raise SystemExit(f"No human json files under: {human_dir}")

    created = 0
    skipped = 0

    for hf in human_files:
        human = load_first_json(hf)
        if human is None:
            skipped += 1
            continue

        start, end, start_t = pick_start_end_from_human(human)

        for k in range(args.per_human):
            n_points = random.randint(args.min_points, args.max_points)
            pts = generate_humanish_points(
                start=start,
                end=end,
                start_t=start_t,
                n_points=n_points,
                base_dt=args.base_dt,
                jitter=args.jitter,
                curve=args.curve,
                micro_pause_p=args.micro_pause_p,
                overshoot=args.overshoot,
            )
            if args.multiseg:
                pts = add_multisegment_time_gaps(pts, segments=random.randint(2, 4))

            bot = build_bot_sample_like(human, pts)

            out_name = f"{hf.stem}_bot_h_{k+1:02d}.json"
            (out_dir / out_name).write_text(json.dumps(bot, ensure_ascii=False, indent=2), encoding="utf-8")
            created += 1

            if args.max_out and created >= args.max_out:
                break
        if args.max_out and created >= args.max_out:
            break

    print(f"[OK] created={created}, skipped_bad_json={skipped}")
    print(f"[OK] bot_dir={out_dir}")


if __name__ == "__main__":
    main()
