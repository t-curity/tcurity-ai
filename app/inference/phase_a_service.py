from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import joblib
import os
import json
import uuid
import random
from datetime import datetime


DEFAULT_MODEL_DIR = Path(os.getenv("MODEL_A_DIR", "/models/phase_a"))
PHASE_A_DATA_DIR = Path(os.getenv("PHASE_A_DATA_DIR", "/data/drag_trainset"))
PHASE_A_SAVE_ENABLED = os.getenv("PHASE_A_SAVE_ENABLED", "0").strip() == "1"
PHASE_A_SAVE_RATIO = float(os.getenv("PHASE_A_SAVE_RATIO", "1.0")) 


def _import_extract_features():
    import sys
    project_root = Path(os.getenv("PROJECT_ROOT", "/app"))
    if project_root.exists() and str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    candidates = [
        "feature_extractor",
        "training.feature_extractor",
        "app.training.feature_extractor",
    ]
    last_err: Optional[Exception] = None
    for mod in candidates:
        try:
            m = __import__(mod, fromlist=["extract_features"])
            return getattr(m, "extract_features")
        except Exception as e:
            last_err = e
    raise ImportError("extract_features import failed") from last_err


extract_features = _import_extract_features()


def remove_pause_gaps(
    points: List[Dict[str, float]], 
    pause_threshold_ms: float = 200.0, 
    move_threshold: float = 0.005
) -> List[Dict[str, float]]:
    """
    멈춤 구간을 제거하고 시간을 재조정.
    멈춤 조건: 시간 gap >= pause_threshold_ms AND 위치 변화 < move_threshold
    """
    if len(points) < 2:
        return points
    
    result = [points[0].copy()]
    result[0]["t"] = 0.0
    
    accumulated_time = 0.0
    
    for i in range(1, len(points)):
        prev_t = float(points[i - 1].get("t", 0))
        curr_t = float(points[i].get("t", 0))
        dt = curr_t - prev_t
        
        prev_x = float(points[i - 1].get("x", 0))
        prev_y = float(points[i - 1].get("y", 0))
        curr_x = float(points[i].get("x", 0))
        curr_y = float(points[i].get("y", 0))
        distance = ((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2) ** 0.5
        
        is_pause = dt >= pause_threshold_ms and distance < move_threshold
        
        if is_pause:
            dt = 16.0
        
        accumulated_time += dt
        
        new_point = points[i].copy()
        new_point["t"] = accumulated_time
        result.append(new_point)
    
    return result


def coerce_points(payload: Dict[str, Any]) -> List[Dict[str, float]]:
    points = payload.get("points") or payload.get("trajectory") or payload.get("data")
    if not isinstance(points, list) or not points:
        raise ValueError("points list missing")
    out: List[Dict[str, float]] = []
    for p in points:
        if isinstance(p, (list, tuple)):
            if len(p) >= 3:
                out.append({"x": float(p[0]), "y": float(p[1]), "t": float(p[2])})
        elif isinstance(p, dict):
            if all(k in p for k in ("x", "y", "t")):
                out.append({"x": float(p["x"]), "y": float(p["y"]), "t": float(p["t"])})
    if len(out) < 3:
        raise ValueError("too few valid points")
    return out


def _calc_cv_time_from_points(points: List[Dict[str, float]]) -> float:
    """원본 포인트에서 cv_time(시간 간격 변동계수) 계산"""
    if len(points) < 2:
        return 0.0
    
    dts = []
    for i in range(1, len(points)):
        dt = float(points[i].get("t", 0)) - float(points[i-1].get("t", 0))
        dts.append(dt)
    
    if not dts:
        return 0.0
    
    dt_mean = np.mean(dts)
    dt_std = np.std(dts)
    
    if dt_mean <= 0:
        return 0.0
    
    return dt_std / dt_mean


class PhaseAInfer:
    """
    Phase A 추론 클래스.
    - score > threshold => 사람, else 봇
    - Rule-based 필터로 극단적 봇 패턴 추가 탐지
    """
    
    RULE_THRESHOLDS = {
        "cv_time_min": 0.02,
        "speed_entropy_min": 0.04,
        "dt_entropy_min": 0.03,
        "decel_accel_min": 0.05,
        "min_points": 10,
        "min_total_time_ms": 100,
    }
    
    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR):
        self.model_dir = Path(model_dir).expanduser().resolve()
        scaler_path = self.model_dir / "scaler_oneclass.pkl"
        model_path = self.model_dir / "model_oneclass.pkl"
        thr_path = self.model_dir / "threshold_oneclass.json"

        if not scaler_path.exists():
            raise FileNotFoundError(f"Missing scaler: {scaler_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model: {model_path}")
        if not thr_path.exists():
            raise FileNotFoundError(f"Missing threshold: {thr_path}")

        self.scaler = joblib.load(scaler_path)
        self.model = joblib.load(model_path)
        thr_obj = json.loads(thr_path.read_text(encoding="utf-8"))
        self.threshold = float(thr_obj["threshold"])
    
    def _rule_based_bot_check_raw(self, points: List[Dict[str, float]]) -> Optional[str]:
        """
        Rule-based 봇 탐지 (원본 데이터 기반).
        전처리 전 원본 포인트로 검사.
        """
        th = self.RULE_THRESHOLDS
        num_points = len(points)
        
        if num_points < th["min_points"]:
            return f"too_few_points={num_points}"
        
        total_time_ms = 0
        if num_points >= 2:
            total_time_ms = float(points[-1].get("t", 0)) - float(points[0].get("t", 0))
        
        if total_time_ms > 0 and total_time_ms < th["min_total_time_ms"]:
            return f"too_fast={total_time_ms:.0f}ms"
        
        cv_time = _calc_cv_time_from_points(points)
        if cv_time < th["cv_time_min"]:
            return f"cv_time={cv_time:.4f}"
        
        return None
    
    def _rule_based_bot_check_features(self, features: np.ndarray) -> Optional[str]:
        """
        Rule-based 봇 탐지 (feature 기반).
        전처리 후 추출된 feature로 검사.
        """
        th = self.RULE_THRESHOLDS
        
        # 40-feature 인덱스: speed_entropy=36, dt_entropy=37, end_deceleration=38, start_acceleration=39
        speed_entropy = features[36]
        dt_entropy = features[37]
        end_decel = features[38]
        start_accel = features[39]
        
        if speed_entropy < th["speed_entropy_min"]:
            return f"speed_entropy={speed_entropy:.4f}"
        
        if dt_entropy < th["dt_entropy_min"]:
            return f"dt_entropy={dt_entropy:.4f}"
        
        if abs(end_decel) < th["decel_accel_min"] and abs(start_accel) < th["decel_accel_min"]:
            return f"no_accel_decel"
        
        return None

    def infer_human_bot(
        self, 
        points: List[Dict[str, float]], 
        *, 
        min_points: int = 10,
        return_score: bool = False
    ) -> Dict[str, Any]:
        
        # Step 1: 원본 데이터로 Rule-based 체크
        rule_reason_raw = self._rule_based_bot_check_raw(points)
        if rule_reason_raw:
            result = {
                "pass": False,
                "label": "봇",
                "reason": f"rule_based:{rule_reason_raw}"
            }
            print(f"[AI] RULE-BASED BOT DETECTED (raw): {rule_reason_raw}")
            if return_score:
                result["score"] = None
                result["threshold"] = self.threshold
            return result
        
        # Step 2: 멈춤 구간 제거 전처리
        processed_points = remove_pause_gaps(points, pause_threshold_ms=200.0)
        
        # Step 3: Feature 추출
        feat = extract_features(
            processed_points,
            line=None,
            sanitize_time=True,
            normalize_to_line=False,
            min_points=min_points,
            same_t_eps=0.0,
        )

        if feat is None:
            result = {"pass": False, "label": "봇", "reason": "insufficient_points"}
            if return_score:
                result["score"] = None
                result["threshold"] = self.threshold
            return result

        # Step 4: Feature 기반 Rule-based 체크
        rule_reason_feat = self._rule_based_bot_check_features(feat)
        if rule_reason_feat:
            result = {
                "pass": False,
                "label": "봇",
                "reason": f"rule_based:{rule_reason_feat}"
            }
            print(f"[AI] RULE-BASED BOT DETECTED (feat): {rule_reason_feat}")
            if return_score:
                result["score"] = None
                result["threshold"] = self.threshold
            return result

        # Step 5: IsolationForest 판정
        X = np.asarray(feat, dtype=float).reshape(1, -1)
        Xs = self.scaler.transform(X)
        score = float(self.model.score_samples(Xs)[0])
        is_human = score > self.threshold

        result = {
            "pass": is_human,
            "label": "사람" if is_human else "봇"
        }

        print(f"[AI] score={score:.6f}, threshold={self.threshold}, is_human={is_human}")

        if return_score:
            result["score"] = round(score, 6)
            result["threshold"] = self.threshold

        return result


def save_phase_a_sample(
    *,
    raw_payload: Dict[str, Any],
    points: List[Dict[str, float]],
    infer: Dict[str, Any],
) -> Optional[Path]:
    """
    Phase A 요청 샘플을 json으로 저장.
    
    저장 로직:
    - Rule 실패 + AI 봇 → 저장 안 함 (확실한 봇)
    - 그 외 → human_pred/에 저장 (사람으로 가정하고 학습)
    """
    if not PHASE_A_SAVE_ENABLED:
        return None
    if PHASE_A_SAVE_RATIO < 1.0 and random.random() > PHASE_A_SAVE_RATIO:
        return None

    pred_label = infer.get("label")
    reason = infer.get("reason", "")
    is_rule_fail = reason.startswith("rule_based:")
    is_bot = pred_label in ("봇", "bot", "BOT")
    
    # Rule 실패 + AI 봇 → bot_pred/ 저장 (확실한 봇)
    # 그 외 → human_pred/ 저장 (사람으로 가정하고 학습)
    if is_rule_fail and is_bot:
        bucket = "bot_pred"
    else:
        bucket = "human_pred"

    now = datetime.now()
    ymd = now.strftime("%Y%m%d")
    stamp = now.strftime("%Y%m%d_%H%M%S_%f")
    uid = uuid.uuid4().hex[:10]

    out_dir = PHASE_A_DATA_DIR / bucket / ymd
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stamp}_{uid}.json"
    tmp_path = out_path.with_suffix(".json.tmp")

    record = {
        "points": points,
        "inference": {
            "pass": bool(infer.get("pass")),
            "pred_label": pred_label,
            "score": infer.get("score"),
            "threshold": infer.get("threshold"),
            "reason": reason,
        },
        "received_at": now.isoformat(timespec="seconds"),
        "line": raw_payload.get("line") or raw_payload.get("cutline") or raw_payload.get("guide_line"),
        "metadata": raw_payload.get("metadata"),
    }

    tmp_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, out_path)
    print(f"[SAVE] {bucket}/에 저장: {out_path.name}")
    return out_path