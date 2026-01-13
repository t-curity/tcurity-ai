# tcurity-ai/training/phase_b/feature_extractor.py
from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import Dict, List, Tuple, Any


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
DEFAULT_GAP_MS = 250          # dt가 이 이상이면 다른 드래그로 간주
DEFAULT_JUMP_DIST = 0.25      # 정규화 좌표에서 이 이상 점프하면 다른 드래그로 간주

DIR_CHANGE_DEG = 45.0         # 방향 변화 카운트 임계값 (deg)
MICRO_PAUSE_FACTOR = 2.0      # avg_dt의 몇 배 이상이면 micro pause로 간주


# ------------------------------------------------------------
# Feature name presets (train에서 그대로 쓰기 좋게)
# ------------------------------------------------------------
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

FEATURES_WITH_RESULT_16 = FEATURES_BEHAVIOR_14 + ["correct_count", "is_perfect"]


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _f(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _i(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle_deg(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """angle at point b between segment ba and bc"""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag = math.hypot(*ba) * math.hypot(*bc)
    if mag == 0:
        return 0.0

    cosv = max(-1.0, min(1.0, dot / mag))
    return math.degrees(math.acos(cosv))


def _segment_points(
    points: List[dict],
    gap_ms: int = DEFAULT_GAP_MS,
    jump_dist: float = DEFAULT_JUMP_DIST,
) -> List[List[dict]]:
    """
    points를 드래그(세그먼트) 단위로 분리.
    - dt가 너무 크거나, 좌표 점프가 크면 새로운 세그먼트 시작
    """
    if not points or len(points) < 2:
        return []

    segs: List[List[dict]] = []
    cur: List[dict] = [points[0]]

    for p in points[1:]:
        prev = cur[-1]

        px, py = _f(prev.get("x")), _f(prev.get("y"))
        pt = _i(prev.get("t"))
        x, y = _f(p.get("x")), _f(p.get("y"))
        t = _i(p.get("t"))

        dt = t - pt
        jump = _dist((px, py), (x, y))

        if dt >= gap_ms or jump >= jump_dist:
            segs.append(cur)
            cur = [p]
        else:
            cur.append(p)

    if cur:
        segs.append(cur)

    # 길이 2 미만 세그먼트 제거
    segs = [s for s in segs if len(s) >= 2]
    return segs


def _segment_stats(seg: List[dict]) -> Dict[str, Any]:
    """세그먼트 1개(드래그 1회)에 대한 통계"""
    coords = [(_f(p.get("x")), _f(p.get("y"))) for p in seg]
    times = [_i(p.get("t")) for p in seg]

    dts = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    steps = [_dist(coords[i], coords[i + 1]) for i in range(len(coords) - 1)]

    duration = times[-1] - times[0]
    path_len = sum(steps)
    straight = _dist(coords[0], coords[-1])
    eff = (straight / path_len) if path_len > 0 else 0.0

    speeds = [steps[i] / dts[i] for i in range(len(steps)) if dts[i] > 0]

    # 방향 변화
    dir_changes = 0
    for i in range(1, len(coords) - 1):
        if _angle_deg(coords[i - 1], coords[i], coords[i + 1]) > DIR_CHANGE_DEG:
            dir_changes += 1

    return {
        "duration": float(duration),
        "path_length": float(path_len),
        "straight_distance": float(straight),
        "eff": float(eff),
        "dir_changes": float(dir_changes),
        "dts": dts,
        "speeds": speeds,
    }


def extract_features_from_drag(sample: dict) -> Dict[str, float]:
    """
    collector json(sample)에서 points를 읽고 특징 추출
    반환 키들은 train_random_forest.py에서 그대로 사용 가능
    """
    points = sample.get("points") or []
    if not isinstance(points, list) or len(points) < 2:
        return {}

    # 정답 관련 (MVP에서는 모델 입력에서 제외 권장)
    correct_count = _f(sample.get("correct_count", 0))
    is_perfect = 1.0 if bool(sample.get("is_perfect", False)) else 0.0

    # 세그먼트 분리
    segs = _segment_points(points, DEFAULT_GAP_MS, DEFAULT_JUMP_DIST)
    if not segs:
        segs = [points]  # fallback

    seg_stats = [_segment_stats(s) for s in segs]

    # 전체 dt/speed 풀 (세그먼트 내부만)
    all_dts: List[int] = []
    all_speeds: List[float] = []
    for st in seg_stats:
        all_dts.extend(st["dts"])
        all_speeds.extend(st["speeds"])

    # ---- time features (active drag)
    drag_duration = sum(st["duration"] for st in seg_stats)

    avg_dt = mean(all_dts) if all_dts else 0.0
    dt_std = pstdev(all_dts) if len(all_dts) >= 2 else 0.0

    # ---- geometry features
    path_length = sum(st["path_length"] for st in seg_stats)
    straight_distance = sum(st["straight_distance"] for st in seg_stats)
    path_efficiency = (straight_distance / path_length) if path_length > 0 else 0.0

    # ---- speed features
    avg_speed = (path_length / drag_duration) if drag_duration > 0 else 0.0
    speed_std = pstdev(all_speeds) if len(all_speeds) >= 2 else 0.0

    # ---- direction
    direction_changes = sum(st["dir_changes"] for st in seg_stats)

    # ---- micro pause count (세그먼트 내부 dt 기반)
    micro_pause_count = 0
    if all_dts and avg_dt > 0:
        thr = avg_dt * MICRO_PAUSE_FACTOR
        micro_pause_count = sum(1 for d in all_dts if d >= thr)

    # ---- session features (including idle gaps between segments)
    # session_duration: first point t ~ last point t
    t0 = _i(points[0].get("t"))
    t1 = _i(points[-1].get("t"))
    session_duration = float(max(0, t1 - t0))

    # idle gaps between segments: (next_seg_start - prev_seg_end) - 0 (>=0)
    idle_gaps: List[int] = []
    for a, b in zip(segs[:-1], segs[1:]):
        ta = _i(a[-1].get("t"))
        tb = _i(b[0].get("t"))
        idle = tb - ta
        if idle >= 0:
            idle_gaps.append(idle)

    idle_mean = mean(idle_gaps) if idle_gaps else 0.0
    idle_std = pstdev(idle_gaps) if len(idle_gaps) >= 2 else 0.0

    return {
        # 행동+세션 (MVP 기본 입력으로 추천)
        "drag_duration": float(drag_duration),
        "avg_dt": float(avg_dt),
        "dt_std": float(dt_std),
        "path_length": float(path_length),
        "straight_distance": float(straight_distance),
        "path_efficiency": float(path_efficiency),
        "avg_speed": float(avg_speed),
        "speed_std": float(speed_std),
        "direction_changes": float(direction_changes),
        "micro_pause_count": float(micro_pause_count),
        "drag_count": float(len(segs)),
        "session_duration": float(session_duration),
        "idle_time_mean": float(idle_mean),
        "idle_time_std": float(idle_std),

        # 결과 기반(비권장: 모델 입력에서 보통 제외)
        "correct_count": float(correct_count),
        "is_perfect": float(is_perfect),
    }
