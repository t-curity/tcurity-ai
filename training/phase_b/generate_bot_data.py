#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[generate_bot_data.py] - 사람 데이터 복사 + 미세 변형

가장 정교한 봇: 실제 사람 points를 그대로 가져와서 아주 미세하게만 변형
- 좌표에 미세 노이즈
- 시간에 미세 노이즈
- 일부 점 추가/삭제

이론적으로 거의 구분 불가능해야 함

사용법:
  python -m training.phase_b.generate_bot_data --per-human 1 --max-out 3000
"""

from __future__ import annotations

import argparse
import json
import random
import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import sys
THIS = Path(__file__).resolve()
ROOT = THIS.parents[2]  # tcurity-ai
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.phase_b.feature_extractor import extract_features_from_drag


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


def perturb_points(original_points: List[Dict], intensity: str = "micro") -> List[Dict]:
    """
    사람 points를 미세하게 변형
    
    intensity:
    - "micro": 거의 티 안 나게 (0.1~0.5% 변형)
    - "light": 살짝 (0.5~1% 변형)
    - "medium": 좀 더 (1~2% 변형)
    """
    if not original_points:
        return []
    
    # intensity에 따른 노이즈 크기
    noise_config = {
        "micro": {"xy": 0.002, "t_ratio": 0.02, "skip_p": 0.02, "dup_p": 0.02},
        "light": {"xy": 0.004, "t_ratio": 0.04, "skip_p": 0.03, "dup_p": 0.03},
        "medium": {"xy": 0.008, "t_ratio": 0.06, "skip_p": 0.05, "dup_p": 0.05},
    }
    cfg = noise_config.get(intensity, noise_config["micro"])
    
    new_points = []
    prev_t = 0
    
    for i, pt in enumerate(original_points):
        # 가끔 점 스킵 (2~5%)
        if i > 0 and i < len(original_points) - 1 and random.random() < cfg["skip_p"]:
            continue
        
        # 좌표 추출
        x = float(pt.get("x", 0))
        y = float(pt.get("y", 0))
        t = int(pt.get("t", 0))
        
        # 미세한 좌표 노이즈
        x += random.gauss(0, cfg["xy"])
        y += random.gauss(0, cfg["xy"])
        x = clamp01(x)
        y = clamp01(y)
        
        # 미세한 시간 노이즈
        if i > 0:
            dt = t - int(original_points[i-1].get("t", 0))
            dt_noise = random.gauss(0, dt * cfg["t_ratio"])
            t = prev_t + max(5, int(dt + dt_noise))
        
        new_points.append({
            "x": fmt_f(x),
            "y": fmt_f(y),
            "t": int(t)
        })
        prev_t = t
        
        # 가끔 점 복제 (약간 다른 위치에) - 2~5%
        if random.random() < cfg["dup_p"]:
            dup_x = x + random.gauss(0, cfg["xy"] * 0.5)
            dup_y = y + random.gauss(0, cfg["xy"] * 0.5)
            dup_t = t + random.randint(3, 8)
            new_points.append({
                "x": fmt_f(clamp01(dup_x)),
                "y": fmt_f(clamp01(dup_y)),
                "t": int(dup_t)
            })
            prev_t = dup_t
    
    return new_points


def transform_human_to_bot(human: dict, intensity: str = "micro") -> dict:
    """
    사람 데이터를 봇 데이터로 변환
    points만 미세하게 변형
    """
    # 원본 points 가져오기
    original_points = human.get("points", [])
    
    # behavior 구조인 경우
    if not original_points and "behavior" in human:
        behavior = human.get("behavior", {})
        if isinstance(behavior, dict):
            original_points = behavior.get("points", [])
    
    if not original_points:
        return None
    
    # points 변형
    new_points = perturb_points(original_points, intensity)
    
    if not new_points:
        return None
    
    # 봇 데이터 생성
    meta = human.get("metadata", {}) or {}
    res = meta.get("res") or {"w": 0, "h": 0}
    
    return {
        "label": "bot",
        "timestamp": iso_now(),
        "target": human.get("target", "unknown"),
        "correct_count": int(human.get("correct_count", 4)),
        "is_perfect": bool(human.get("is_perfect", True)),
        "points": new_points,
        "metadata": {
            "ua": meta.get("ua", ""),
            "res": res,
            "sim": f"bot_clone_{intensity}",
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human-dir", type=str, default="data/phase_b/human")
    ap.add_argument("--out-dir", type=str, default="data/phase_b/bot")
    ap.add_argument("--per-human", type=int, default=1)
    ap.add_argument("--max-out", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--intensity", type=str, default="micro", 
                    choices=["micro", "light", "medium"],
                    help="변형 강도: micro(거의 동일), light(살짝), medium(좀 더)")
    args = ap.parse_args()

    human_dir = (ROOT / args.human_dir).resolve()
    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)

    # human + human_pred 둘 다 사용
    human_dirs = [human_dir]
    human_pred_dir = human_dir.parent / "human_pred"
    if human_pred_dir.exists():
        human_dirs.append(human_pred_dir)
    
    files = []
    for d in human_dirs:
        files.extend(sorted(d.rglob("*.json")))
    
    if not files:
        raise SystemExit(f"No human json found")

    print(f"[INFO] Human files: {len(files)}")
    print(f"[INFO] Output dir: {out_dir}")
    print(f"[INFO] Intensity: {args.intensity}")
    print(f"[INFO] Per human: {args.per_human}, Max: {args.max_out}")
    print()

    created = 0
    feature_diffs = []  # feature 차이 추적
    
    for fp in files:
        human = load_json_first(fp)
        if not human:
            continue

        # 원본 feature
        human_feat = extract_features_from_drag(human)
        if not human_feat:
            continue

        for k in range(args.per_human):
            # 다양한 intensity 섞기
            if args.per_human > 1:
                intensities = ["micro", "micro", "light"]  # micro 비중 높게
                intensity = random.choice(intensities)
            else:
                intensity = args.intensity
            
            bot = transform_human_to_bot(human, intensity)
            if not bot:
                continue
            
            # 생성된 봇 feature
            bot_feat = extract_features_from_drag(bot)
            
            # feature 차이 계산
            if bot_feat and human_feat:
                diff = {
                    "path_efficiency": abs(bot_feat.get("path_efficiency", 0) - human_feat.get("path_efficiency", 0)),
                    "avg_dt": abs(bot_feat.get("avg_dt", 0) - human_feat.get("avg_dt", 0)),
                    "speed_std": abs(bot_feat.get("speed_std", 0) - human_feat.get("speed_std", 0)),
                }
                feature_diffs.append(diff)

            out_name = f"{fp.stem}_bot_{k+1:02d}.json"
            (out_dir / out_name).write_text(json.dumps(bot, ensure_ascii=False, indent=2), encoding="utf-8")
            created += 1
            
            if created % 500 == 0:
                print(f"  ... created {created}")
                # 평균 차이 출력
                if feature_diffs:
                    avg_diff = {
                        key: sum(d[key] for d in feature_diffs[-500:]) / min(500, len(feature_diffs))
                        for key in ["path_efficiency", "avg_dt", "speed_std"]
                    }
                    print(f"      avg diff: path_eff={avg_diff['path_efficiency']:.4f}, "
                          f"avg_dt={avg_diff['avg_dt']:.2f}, speed_std={avg_diff['speed_std']:.5f}")
            
            if args.max_out and created >= args.max_out:
                break
        if args.max_out and created >= args.max_out:
            break

    # 최종 통계
    print(f"\n[OK] Created {created} v4 bot samples -> {out_dir}")
    
    if feature_diffs:
        print("\n[Feature Difference Summary]")
        for key in ["path_efficiency", "avg_dt", "speed_std"]:
            values = [d[key] for d in feature_diffs]
            avg = sum(values) / len(values)
            print(f"  {key}: avg_diff={avg:.5f}")


if __name__ == "__main__":
    main()