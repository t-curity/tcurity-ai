#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
기존 14개 feature + 새로운 12개 feature = 총 26개

새로 추가된 feature:
- acceleration_mean: 평균 가속도
- acceleration_std: 가속도 표준편차
- jerk_mean: 평균 jerk (가속도의 변화율)
- jerk_std: jerk 표준편차
- curvature_mean: 평균 곡률
- curvature_std: 곡률 표준편차
- angular_velocity_mean: 평균 각속도
- angular_velocity_std: 각속도 표준편차
- velocity_autocorr: 속도 자기상관 (연속성)
- straightness: 직진성 (straight_distance / path_length)
- max_deviation: 최대 이탈 거리
- pause_ratio: 멈춤 비율 (dt > threshold인 구간)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Any

DEFAULT_GAP_MS = 300  # idle time 기준


# ============================================================
# 기존 14개 Feature (FEATURES_BEHAVIOR_14)
# ============================================================
FEATURES_BEHAVIOR_14 = [
    "drag_duration",
    "avg_dt",
    "dt_std",
    "path_length",
    "straight_distance",
    "path_efficiency",
    "avg_speed",
    "speed_std",
    "direction_changes",
    "micro_pause_count",
    "drag_count",
    "session_duration",
    "idle_time_mean",
    "idle_time_std",
]

# ============================================================
# 새로운 12개 Feature
# ============================================================
FEATURES_ADVANCED_12 = [
    "acceleration_mean",
    "acceleration_std",
    "jerk_mean",
    "jerk_std",
    "curvature_mean",
    "curvature_std",
    "angular_velocity_mean",
    "angular_velocity_std",
    "velocity_autocorr",
    "straightness",
    "max_deviation",
    "pause_ratio",
]

# 전체 26개 Feature
FEATURES_ALL_26 = FEATURES_BEHAVIOR_14 + FEATURES_ADVANCED_12

# 결과 포함 버전 (28개)
FEATURES_WITH_RESULT_16 = FEATURES_BEHAVIOR_14 + ["correct_count", "is_perfect"]
FEATURES_WITH_RESULT_28 = FEATURES_ALL_26 + ["correct_count", "is_perfect"]


def _safe_std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_val = sum(values) / len(values)
    variance = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(max(0, variance))


def _safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _distance(p1: Dict, p2: Dict) -> float:
    x1, y1 = float(p1.get("x", 0)), float(p1.get("y", 0))
    x2, y2 = float(p2.get("x", 0)), float(p2.get("y", 0))
    return math.hypot(x2 - x1, y2 - y1)


def _angle(p1: Dict, p2: Dict) -> float:
    """두 점 사이의 각도 (라디안)"""
    x1, y1 = float(p1.get("x", 0)), float(p1.get("y", 0))
    x2, y2 = float(p2.get("x", 0)), float(p2.get("y", 0))
    return math.atan2(y2 - y1, x2 - x1)


def _angle_diff(a1: float, a2: float) -> float:
    """두 각도 차이 (-pi ~ pi)"""
    diff = a2 - a1
    while diff > math.pi:
        diff -= 2 * math.pi
    while diff < -math.pi:
        diff += 2 * math.pi
    return diff


def _point_to_line_distance(point: Dict, line_start: Dict, line_end: Dict) -> float:
    """점에서 직선까지의 거리"""
    px, py = float(point.get("x", 0)), float(point.get("y", 0))
    x1, y1 = float(line_start.get("x", 0)), float(line_start.get("y", 0))
    x2, y2 = float(line_end.get("x", 0)), float(line_end.get("y", 0))
    
    # 직선의 길이
    line_len = math.hypot(x2 - x1, y2 - y1)
    if line_len < 1e-9:
        return math.hypot(px - x1, py - y1)
    
    # 점에서 직선까지 수직 거리
    dist = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1) / line_len
    return dist


def _autocorrelation(values: List[float], lag: int = 1) -> float:
    """자기상관 계산"""
    if len(values) <= lag:
        return 0.0
    
    mean_val = _safe_mean(values)
    n = len(values)
    
    numerator = sum((values[i] - mean_val) * (values[i + lag] - mean_val) 
                    for i in range(n - lag))
    denominator = sum((v - mean_val) ** 2 for v in values)
    
    if denominator < 1e-9:
        return 0.0
    
    return numerator / denominator


def extract_features(sample: dict, include_advanced: bool = True) -> Optional[Dict[str, float]]:
    """
    고급 feature 포함 추출
    
    Args:
        sample: JSON 데이터
        include_advanced: True면 26개, False면 14개 feature
    
    Returns:
        feature dictionary
    """
    # points 추출
    points = sample.get("points", [])
    if not points and "behavior" in sample:
        behavior = sample.get("behavior", {})
        if isinstance(behavior, dict):
            points = behavior.get("points", [])
    
    if not points or len(points) < 3:
        return None
    
    # 시간 정렬
    try:
        points = sorted(points, key=lambda p: int(p.get("t", 0)))
    except:
        return None
    
    # ============================================================
    # 기본 계산
    # ============================================================
    n = len(points)
    t_start = int(points[0].get("t", 0))
    t_end = int(points[-1].get("t", 0))
    
    session_duration = max(1, t_end - t_start)
    
    # dt, distance, speed 계산
    dts = []
    distances = []
    speeds = []
    angles = []
    
    for i in range(1, n):
        dt = int(points[i].get("t", 0)) - int(points[i-1].get("t", 0))
        dt = max(1, dt)
        dts.append(dt)
        
        d = _distance(points[i-1], points[i])
        distances.append(d)
        
        speed = d / (dt / 1000.0)  # pixels per second
        speeds.append(speed)
        
        angle = _angle(points[i-1], points[i])
        angles.append(angle)
    
    # path length, straight distance
    path_length = sum(distances) if distances else 0.0
    straight_distance = _distance(points[0], points[-1])
    path_efficiency = straight_distance / max(path_length, 1e-9)
    path_efficiency = min(1.0, path_efficiency)
    
    # direction changes
    direction_changes = 0
    for i in range(1, len(angles)):
        if abs(_angle_diff(angles[i-1], angles[i])) > math.pi / 4:
            direction_changes += 1
    
    # micro pause count
    micro_pause_count = sum(1 for dt in dts if dt > 50)
    
    # segment 분리 (idle time 기준)
    segments = []
    current_seg = [points[0]]
    idle_times = []
    
    for i in range(1, n):
        dt = int(points[i].get("t", 0)) - int(points[i-1].get("t", 0))
        if dt > DEFAULT_GAP_MS:
            if len(current_seg) > 1:
                segments.append(current_seg)
            idle_times.append(dt)
            current_seg = [points[i]]
        else:
            current_seg.append(points[i])
    
    if len(current_seg) > 1:
        segments.append(current_seg)
    
    drag_count = max(1, len(segments))
    
    # idle time stats
    idle_time_mean = _safe_mean(idle_times) if idle_times else 0.0
    idle_time_std = _safe_std(idle_times) if idle_times else 0.0
    
    # drag duration (idle 제외)
    drag_duration = session_duration - sum(idle_times)
    drag_duration = max(1, drag_duration)
    
    # ============================================================
    # 기본 14개 Feature
    # ============================================================
    feat = {
        "drag_duration": float(drag_duration),
        "avg_dt": _safe_mean(dts),
        "dt_std": _safe_std(dts),
        "path_length": path_length,
        "straight_distance": straight_distance,
        "path_efficiency": path_efficiency,
        "avg_speed": _safe_mean(speeds),
        "speed_std": _safe_std(speeds),
        "direction_changes": float(direction_changes),
        "micro_pause_count": float(micro_pause_count),
        "drag_count": float(drag_count),
        "session_duration": float(session_duration),
        "idle_time_mean": idle_time_mean,
        "idle_time_std": idle_time_std,
    }
    
    if not include_advanced:
        return feat
    
    # ============================================================
    # 고급 12개 Feature
    # ============================================================
    
    # 1. Acceleration (가속도)
    accelerations = []
    for i in range(1, len(speeds)):
        dt = dts[i] / 1000.0  # seconds
        if dt > 0:
            acc = (speeds[i] - speeds[i-1]) / dt
            accelerations.append(acc)
    
    feat["acceleration_mean"] = _safe_mean(accelerations)
    feat["acceleration_std"] = _safe_std(accelerations)
    
    # 2. Jerk (가속도의 변화율)
    jerks = []
    for i in range(1, len(accelerations)):
        dt = dts[i+1] / 1000.0 if i+1 < len(dts) else 0.016
        if dt > 0:
            jerk = (accelerations[i] - accelerations[i-1]) / dt
            jerks.append(jerk)
    
    feat["jerk_mean"] = _safe_mean(jerks)
    feat["jerk_std"] = _safe_std(jerks)
    
    # 3. Curvature (곡률)
    curvatures = []
    for i in range(1, len(angles)):
        # 곡률 = 각도 변화 / 이동 거리
        angle_diff = abs(_angle_diff(angles[i-1], angles[i]))
        dist = distances[i] if i < len(distances) else 0.001
        if dist > 1e-6:
            curvature = angle_diff / dist
            curvatures.append(curvature)
    
    feat["curvature_mean"] = _safe_mean(curvatures)
    feat["curvature_std"] = _safe_std(curvatures)
    
    # 4. Angular Velocity (각속도)
    angular_velocities = []
    for i in range(1, len(angles)):
        dt = dts[i] / 1000.0 if i < len(dts) else 0.016
        if dt > 0:
            angle_diff = _angle_diff(angles[i-1], angles[i])
            angular_vel = angle_diff / dt
            angular_velocities.append(angular_vel)
    
    feat["angular_velocity_mean"] = _safe_mean(angular_velocities)
    feat["angular_velocity_std"] = _safe_std(angular_velocities)
    
    # 5. Velocity Autocorrelation (속도 연속성)
    feat["velocity_autocorr"] = _autocorrelation(speeds, lag=1)
    
    # 6. Straightness (직진성)
    feat["straightness"] = path_efficiency  # 이미 계산됨
    
    # 7. Max Deviation (최대 이탈 거리)
    max_deviation = 0.0
    if len(points) >= 2:
        for p in points:
            dev = _point_to_line_distance(p, points[0], points[-1])
            max_deviation = max(max_deviation, dev)
    
    feat["max_deviation"] = max_deviation
    
    # 8. Pause Ratio (멈춤 비율)
    pause_threshold = 40  # ms
    pause_count = sum(1 for dt in dts if dt > pause_threshold)
    feat["pause_ratio"] = pause_count / max(1, len(dts))
    
    return feat


def extract_features_from_drag(sample: dict, include_advanced: bool = True) -> Optional[Dict[str, float]]:
    return extract_features(sample, include_advanced)


# ============================================================
# 테스트
# ============================================================
if __name__ == "__main__":
    # 테스트 데이터
    test_sample = {
        "points": [
            {"x": 0.1, "y": 0.1, "t": 0},
            {"x": 0.15, "y": 0.12, "t": 20},
            {"x": 0.2, "y": 0.15, "t": 45},
            {"x": 0.28, "y": 0.18, "t": 70},
            {"x": 0.35, "y": 0.2, "t": 100},
            {"x": 0.4, "y": 0.22, "t": 150},
            {"x": 0.5, "y": 0.25, "t": 200},
        ]
    }
    
    feat = extract_features(test_sample, include_advanced=True)
    
    print("=" * 50)
    print("Feature Extractor V2 Test")
    print("=" * 50)
    
    if feat:
        print(f"\nTotal features: {len(feat)}")
        print("\n[Basic 14 Features]")
        for key in FEATURES_BEHAVIOR_14:
            print(f"  {key}: {feat.get(key, 0):.4f}")
        
        print("\n[Advanced 12 Features]")
        for key in FEATURES_ADVANCED_12:
            print(f"  {key}: {feat.get(key, 0):.4f}")
    else:
        print("Failed to extract features")