#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
feature_extractor_v3_combined.py

드래그 궤적에서 40개 특징을 추출하는 모듈 (통합 버전)

변경사항:
- v1의 36개 피처 유지 (검증된 성능)
- v2의 새로운 봇 탐지 피처 4개 추가
- 총 40개 피처

사용법:
    from feature_extractor import extract_features
    features = extract_features(points)  # points: [{"x": ..., "y": ..., "t": ...}, ...]
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import stats
from scipy.signal import find_peaks


Point = Dict[str, float]

# 40개 특징 이름 (디버깅/분석용)
FEATURE_NAMES = [
    # === v1 피처 (36개) ===
    # 1. 기본 궤적 통계 (4개)
    'num_points', 'x_range', 'y_range', 'total_time',
    # 2. 속도 특징 (8개)
    'mean_speed', 'std_speed', 'max_speed', 'min_speed', 
    'median_speed', 'iqr_speed', 'skew_speed', 'kurt_speed',
    # 3. 가속도 특징 (3개)
    'mean_acc', 'std_acc', 'max_abs_acc',
    # 4. Jerk 특징 (3개)
    'mean_abs_jerk', 'std_jerk', 'max_abs_jerk',
    # 5. 방향 특징 (4개)
    'mean_abs_angle_change', 'std_angle_change', 'max_abs_angle_change', 'sharp_turns',
    # 6. 시간 간격 특징 (5개)
    'mean_dt', 'std_dt', 'max_dt', 'min_dt', 'cv_time',
    # 7. 멈춤/정지 특징 (2개)
    'pauses', 'pause_ratio',
    # 8. 미세 움직임 (1개)
    'micro_movement_ratio',
    # 9. 궤적 매끄러움 (2개)
    'mean_smoothness', 'std_smoothness',
    # 10. 피크 특징 (1개)
    'num_peaks',
    # 11. 초기 반응 (1개)
    'initial_speed',
    # 12. 궤적 선형성 (2개)
    'mean_deviation_from_line', 'max_deviation_from_line',
    
    # === v2 새 피처 (4개) - 봇 탐지 강화 ===
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
    드래그 궤적에서 40개 특징 추출 (v1 36개 + v2 새 4개)
    
    Args:
        points: [{"x": float, "y": float, "t": float}, ...]
        min_points: 최소 포인트 수 (기본 10)
        sanitize_time: timestamp 정리 여부 (기본 True)
    
    Returns:
        np.ndarray (40,) or None (포인트 부족 시)
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
    # 1. 기본 궤적 통계 (4개)
    # ========================================
    features.append(float(len(pts)))                    # num_points
    features.append(float(np.max(x) - np.min(x)))       # x_range
    features.append(float(np.max(y) - np.min(y)))       # y_range
    features.append(float(t[-1]))                       # total_time

    # ========================================
    # 2. 속도 특징 (8개)
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
    features.append(float(np.min(speed)))               # min_speed
    features.append(float(np.median(speed)))            # median_speed
    features.append(float(np.percentile(speed, 75) - np.percentile(speed, 25)))  # iqr_speed
    features.append(float(stats.skew(speed)))           # skew_speed
    features.append(float(stats.kurtosis(speed)))       # kurt_speed

    # ========================================
    # 3. 가속도 특징 (3개)
    # ========================================
    acc = np.diff(speed)
    features.append(float(np.mean(acc)))                # mean_acc
    features.append(float(np.std(acc)))                 # std_acc
    features.append(float(np.max(np.abs(acc))))         # max_abs_acc

    # ========================================
    # 4. Jerk 특징 (3개)
    # ========================================
    if len(acc) > 1:
        jerk = np.diff(acc)
        features.append(float(np.mean(np.abs(jerk))))   # mean_abs_jerk
        features.append(float(np.std(jerk)))            # std_jerk
        features.append(float(np.max(np.abs(jerk))))    # max_abs_jerk
    else:
        features.extend([0.0, 0.0, 0.0])

    # ========================================
    # 5. 방향 특징 (4개)
    # ========================================
    angles = np.arctan2(dy, dx)
    angle_changes = np.diff(angles) if len(angles) > 1 else np.array([])
    
    # 각도 변화를 -π ~ π 범위로 정규화
    if len(angle_changes) > 0:
        angle_changes = (angle_changes + np.pi) % (2 * np.pi) - np.pi
        features.append(float(np.mean(np.abs(angle_changes))))  # mean_abs_angle_change
        features.append(float(np.std(angle_changes)))           # std_angle_change
        features.append(float(np.max(np.abs(angle_changes))))   # max_abs_angle_change
        # 급격한 방향 전환 횟수 (>90도)
        sharp_turns = np.sum(np.abs(angle_changes) > np.pi / 2)
        features.append(float(sharp_turns))                     # sharp_turns
    else:
        features.extend([0.0, 0.0, 0.0, 0.0])

    # ========================================
    # 6. 시간 간격 특징 (5개)
    # ========================================
    features.append(float(np.mean(dt)))                 # mean_dt
    features.append(float(np.std(dt)))                  # std_dt
    features.append(float(np.max(dt)))                  # max_dt
    features.append(float(np.min(dt)))                  # min_dt
    cv_time = float(np.std(dt) / (np.mean(dt) + 1e-6))
    features.append(cv_time)                            # cv_time

    # ========================================
    # 7. 멈춤/정지 특징 (2개)
    # ========================================
    pause_threshold = np.percentile(speed, 10)
    pauses = np.sum(speed < pause_threshold)
    features.append(float(pauses))                      # pauses
    features.append(float(pauses / len(speed)))         # pause_ratio

    # ========================================
    # 8. 미세 움직임 (1개)
    # ========================================
    small_movements = np.sum(dist < 2.0)  # 2픽셀 미만
    features.append(float(small_movements / len(dist))) # micro_movement_ratio

    # ========================================
    # 9. 궤적 매끄러움 (2개)
    # ========================================
    if len(x) >= 3:
        smoothness_scores = []
        for i in range(len(x) - 2):
            v1 = np.array([x[i+1] - x[i], y[i+1] - y[i]])
            v2 = np.array([x[i+2] - x[i+1], y[i+2] - y[i+1]])
            
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 > 1e-12 and norm2 > 1e-12:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                smoothness_scores.append(float(cos_angle))
        
        if smoothness_scores:
            features.append(float(np.mean(smoothness_scores)))  # mean_smoothness
            features.append(float(np.std(smoothness_scores)))   # std_smoothness
        else:
            features.extend([0.0, 0.0])
    else:
        features.extend([0.0, 0.0])

    # ========================================
    # 10. 피크 특징 (1개)
    # ========================================
    if len(speed) > 5:
        peaks, _ = find_peaks(speed, distance=3)
        features.append(float(len(peaks)))              # num_peaks
    else:
        features.append(0.0)

    # ========================================
    # 11. 초기 반응 (1개)
    # ========================================
    if len(speed) >= 5:
        features.append(float(np.mean(speed[:5])))      # initial_speed
    else:
        features.append(float(np.mean(speed)))

    # ========================================
    # 12. 궤적 선형성 (2개)
    # ========================================
    if len(x) > 2:
        start = np.array([x[0], y[0]])
        end = np.array([x[-1], y[-1]])
        line_vec = end - start
        line_length = np.linalg.norm(line_vec)
        
        if line_length > 1e-12:
            deviations = []
            for i in range(len(x)):
                point = np.array([x[i], y[i]])
                point_vec = point - start
                projection = np.dot(point_vec, line_vec) / (line_length ** 2) * line_vec
                deviation = np.linalg.norm(point_vec - projection)
                deviations.append(deviation)
            
            features.append(float(np.mean(deviations)))     # mean_deviation_from_line
            features.append(float(np.max(deviations)))      # max_deviation_from_line
        else:
            features.extend([0.0, 0.0])
    else:
        features.extend([0.0, 0.0])

    # ========================================
    # 13. 새로 추가된 봇 탐지 피처 (4개) - v2에서 가져옴
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
extract_features_v3 = extract_features


# 테스트용
if __name__ == "__main__":
    print(f"피처 개수: {len(FEATURE_NAMES)}")
    print("\n피처 목록:")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"  [{i:2d}] {name}")
    
    # 간단한 테스트
    test_points = [
        {"x": 0, "y": 0, "t": 0},
        {"x": 10, "y": 5, "t": 16},
        {"x": 25, "y": 8, "t": 33},
        {"x": 45, "y": 6, "t": 50},
        {"x": 70, "y": 4, "t": 66},
        {"x": 100, "y": 3, "t": 83},
        {"x": 135, "y": 5, "t": 100},
        {"x": 175, "y": 7, "t": 116},
        {"x": 220, "y": 4, "t": 133},
        {"x": 270, "y": 2, "t": 150},
        {"x": 320, "y": 0, "t": 166},
    ]
    
    feat = extract_features(test_points)
    if feat is not None:
        print(f"\n추출된 피처 ({len(feat)}개):")
        for i, (name, val) in enumerate(zip(FEATURE_NAMES, feat)):
            print(f"  [{i:2d}] {name:30s} = {val:.6f}")
    else:
        print("\n피처 추출 실패 (포인트 부족)")