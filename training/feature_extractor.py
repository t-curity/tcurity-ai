#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
feature_extractor_v2.py

드래그 궤적에서 20개 핵심 특징을 추출하는 모듈 (최적화 버전)

변경사항:
- 기존 36개 → 20개로 압축
- 중요도 낮은 피처 제거
- 봇 탐지에 효과적인 시간 패턴 피처 4개 추가

사용법:
    from feature_extractor_v2 import extract_features
    features = extract_features(points)  # points: [{"x": ..., "y": ..., "t": ...}, ...]
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import stats
from scipy.signal import find_peaks


Point = Dict[str, float]

# 20개 피처 이름
FEATURE_NAMES = [
    # 1. 궤적 기본 (2개)
    'y_range', 'total_time',
    
    # 2. 속도 (4개)
    'mean_speed', 'std_speed', 'max_speed', 'iqr_speed',
    
    # 3. 가속도 (1개)
    'mean_acc',
    
    # 4. Jerk (3개) - 움직임 자연스러움
    'mean_abs_jerk', 'std_jerk', 'max_abs_jerk',
    
    # 5. 방향 (1개)
    'std_angle_change',
    
    # 6. 시간 간격 (3개) - ⭐ 봇 구분 핵심
    'mean_dt', 'max_dt', 'cv_time',
    
    # 7. 피크/초기속도 (2개)
    'num_peaks', 'initial_speed',
    
    # 8. 새로 추가된 봇 탐지 피처 (4개) - ⭐ NEW
    'speed_entropy',       # 속도 분포 엔트로피 (봇은 낮음)
    'dt_entropy',          # 시간간격 엔트로피 (봇은 낮음)
    'end_deceleration',    # 끝부분 감속률 (사람은 끝에서 느려짐)
    'start_acceleration',  # 시작부분 가속률 (사람은 시작에서 빨라짐)
]


def sanitize_points(
    points: List[Dict[str, Any]],
    *,
    sort_by_t: bool = True,
    merge_same_t: bool = True,
    same_t_eps: float = 0.0,
) -> Tuple[List[Point], Dict[str, int]]:
    """
    Timestamp 정리:
    - 유효하지 않은 데이터 제거
    - t 기준 정렬
    - 동일 timestamp 병합 (dt=0 방지)
    """
    cleaned: List[Point] = []
    dropped = 0

    for p in points:
        try:
            x = float(p["x"])
            y = float(p["y"])
            t = float(p["t"])
        except Exception:
            dropped += 1
            continue

        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(t)):
            dropped += 1
            continue

        cleaned.append({"x": x, "y": y, "t": t})

    if len(cleaned) < 2:
        return cleaned, {"dropped": dropped, "reordered": 0, "merged": 0}

    reordered = 0
    if sort_by_t:
        for i in range(1, len(cleaned)):
            if cleaned[i]["t"] < cleaned[i - 1]["t"]:
                reordered = 1
                break
        cleaned.sort(key=lambda p: p["t"])

    merged = 0
    if merge_same_t:
        dedup: List[Point] = [cleaned[0]]
        for p in cleaned[1:]:
            if abs(p["t"] - dedup[-1]["t"]) <= same_t_eps:
                dedup[-1] = p
                merged += 1
            else:
                dedup.append(p)
        cleaned = dedup

    return cleaned, {"dropped": dropped, "reordered": reordered, "merged": merged}


def _calc_entropy(values: np.ndarray, bins: int = 10) -> float:
    """히스토그램 기반 엔트로피 계산"""
    if len(values) < 2:
        return 0.0
    
    # 값의 범위가 너무 작으면 0 반환
    if np.max(values) - np.min(values) < 1e-10:
        return 0.0
    
    hist, _ = np.histogram(values, bins=bins, density=True)
    hist = hist[hist > 0]  # 0인 bin 제거
    
    if len(hist) == 0:
        return 0.0
    
    # 정규화
    hist = hist / hist.sum()
    
    # 엔트로피 계산
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    return float(entropy)


def extract_features(
    points: List[Dict[str, Any]],
    *,
    min_points: int = 10,
    sanitize_time: bool = True,
    same_t_eps: float = 0.0,
    # 하위 호환성을 위해 무시되는 파라미터들
    line: Any = None,
    normalize_to_line: bool = False,
    normalize_xy: bool = False,
    img_w: float = 400.0,
    img_h: float = 200.0,
) -> Optional[np.ndarray]:
    """
    드래그 궤적에서 20개 핵심 특징 추출 (최적화 버전)
    
    Args:
        points: [{"x": float, "y": float, "t": float}, ...]
        min_points: 최소 포인트 수 (기본 10)
        sanitize_time: timestamp 정리 여부 (기본 True)
    
    Returns:
        np.ndarray (20,) or None (포인트 부족 시)
    """
    if sanitize_time:
        pts, _ = sanitize_points(points, sort_by_t=True, merge_same_t=True, same_t_eps=same_t_eps)
    else:
        pts = []
        for p in points:
            try:
                pts.append({"x": float(p["x"]), "y": float(p["y"]), "t": float(p["t"])})
            except Exception:
                continue

    if len(pts) < min_points:
        return None

    arr = np.array([[p["x"], p["y"], p["t"]] for p in pts], dtype=float)
    
    t = arr[:, 2] - arr[0, 2]
    t[t == 0] = 1e-6
    x = arr[:, 0]
    y = arr[:, 1]

    features: List[float] = []

    # ========================================
    # 1. 궤적 기본 (2개)
    # ========================================
    features.append(float(np.max(y) - np.min(y)))       # y_range
    features.append(float(t[-1]))                       # total_time

    # ========================================
    # 2. 속도 (4개)
    # ========================================
    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(t)
    dt[dt == 0] = 1e-6

    dist = np.sqrt(dx**2 + dy**2)
    speed = dist / dt

    features.append(float(np.mean(speed)))              # mean_speed
    features.append(float(np.std(speed)))               # std_speed
    features.append(float(np.max(speed)))               # max_speed
    features.append(float(np.percentile(speed, 75) - np.percentile(speed, 25)))  # iqr_speed

    # ========================================
    # 3. 가속도 (1개)
    # ========================================
    acc = np.diff(speed)
    features.append(float(np.mean(acc)))                # mean_acc

    # ========================================
    # 4. Jerk (3개) - 움직임 자연스러움
    # ========================================
    if len(acc) > 1:
        jerk = np.diff(acc)
        features.append(float(np.mean(np.abs(jerk))))   # mean_abs_jerk
        features.append(float(np.std(jerk)))            # std_jerk
        features.append(float(np.max(np.abs(jerk))))    # max_abs_jerk
    else:
        features.extend([0.0, 0.0, 0.0])

    # ========================================
    # 5. 방향 (1개)
    # ========================================
    angles = np.arctan2(dy, dx)
    angle_changes = np.diff(angles) if len(angles) > 1 else np.array([])
    
    if len(angle_changes) > 0:
        angle_changes = (angle_changes + np.pi) % (2 * np.pi) - np.pi
        features.append(float(np.std(angle_changes)))   # std_angle_change
    else:
        features.append(0.0)

    # ========================================
    # 6. 시간 간격 (3개) - ⭐ 봇 구분 핵심
    # ========================================
    features.append(float(np.mean(dt)))                 # mean_dt
    features.append(float(np.max(dt)))                  # max_dt
    cv_time = float(np.std(dt) / (np.mean(dt) + 1e-6))
    features.append(cv_time)                            # cv_time

    # ========================================
    # 7. 피크/초기속도 (2개)
    # ========================================
    if len(speed) > 5:
        peaks, _ = find_peaks(speed, distance=3)
        features.append(float(len(peaks)))              # num_peaks
    else:
        features.append(0.0)

    if len(speed) >= 5:
        features.append(float(np.mean(speed[:5])))      # initial_speed
    else:
        features.append(float(np.mean(speed)))

    # ========================================
    # 8. 새로 추가된 봇 탐지 피처 (4개) - ⭐ NEW
    # ========================================
    
    # speed_entropy: 속도 분포 엔트로피 (봇은 균일해서 낮음, 사람은 다양해서 높음)
    features.append(_calc_entropy(speed, bins=10))      # speed_entropy
    
    # dt_entropy: 시간간격 엔트로피 (봇은 균일해서 낮음, 사람은 불규칙해서 높음)
    features.append(_calc_entropy(dt, bins=10))         # dt_entropy
    
    # end_deceleration: 끝부분 감속률 (사람은 끝에서 느려짐, 봇은 일정)
    n = len(speed)
    if n >= 10:
        mid_speed = np.mean(speed[n//3:2*n//3])
        end_speed = np.mean(speed[-5:])
        if mid_speed > 1e-10:
            end_decel = (mid_speed - end_speed) / mid_speed
        else:
            end_decel = 0.0
    else:
        end_decel = 0.0
    features.append(float(end_decel))                   # end_deceleration
    
    # start_acceleration: 시작부분 가속률 (사람은 시작에서 빨라짐, 봇은 일정)
    if n >= 10:
        start_speed = np.mean(speed[:5])
        mid_speed = np.mean(speed[n//3:2*n//3])
        if mid_speed > 1e-10:
            start_accel = (mid_speed - start_speed) / mid_speed
        else:
            start_accel = 0.0
    else:
        start_accel = 0.0
    features.append(float(start_accel))                 # start_acceleration

    return np.array(features, dtype=float)


# 하위 호환성
extract_features_v2 = extract_features


# 테스트용
if __name__ == "__main__":
    print(f"피처 개수: {len(FEATURE_NAMES)}")
    print("\n피처 목록:")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"  [{i:2d}] {name}")
