#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
app/inference/phase_b_service.py

Phase B: 행동 기반 RandomForest 추론 서비스

- 프론트가 보내는 points(좌표/시간)로부터 서버에서 feature_extractor를 통해 feature를 뽑고
- RandomForest 모델로 human_prob을 계산해서 threshold 기준으로 사람/봇 판정

지원 payload 예시:
1) {"points":[{x,y,t},...], "correct_count":4, "is_perfect":true}
2) {"behavior": {"points":[...], ...}}
3) {"features": {...}}  # (옵션) 이미 feature를 만들어 보내는 경우도 지원
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import joblib
import os, json, uuid, random
from datetime import datetime

DEFAULT_MODEL_DIR = Path(os.getenv("MODEL_B_DIR", "/models/phase_b"))
PHASE_B_DATA_DIR = Path(os.getenv("PHASE_B_DATA_DIR", "/data/phase_b"))
PHASE_B_SAVE_ENABLED = os.getenv("PHASE_B_SAVE_ENABLED", "0").strip() == "1"
PHASE_B_SAVE_RATIO = float(os.getenv("PHASE_B_SAVE_RATIO", "1.0"))

# 학습에서 사용한 (behavior-only) 기본 14개 feature fallback
FEATURE_NAMES_FALLBACK = [
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


def _import_extract_features():
    """
    training/phase_b/feature_extractor.py 의 extract_features_from_drag 동적 import
    (배포 환경에서 PYTHONPATH 문제를 피하려고 Phase A처럼 처리)
    """
    import sys

    project_root = Path(os.getenv("PROJECT_ROOT", "/app"))
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from training.phase_b.feature_extractor import extract_features_from_drag  # type: ignore

    return extract_features_from_drag


extract_features_from_drag = _import_extract_features()


# =============================================================================
# Payload/points coercion
# =============================================================================
def coerce_points(payload: Dict[str, Any]) -> List[Dict[str, float]]:
    """
    payload에서 points 배열을 찾아 float로 정규화
    (Phase A coerce_points랑 동일 계열)
    """
    points = payload.get("points") or payload.get("trajectory") or payload.get("data")
    if not isinstance(points, list) or not points:
        raise ValueError("points list missing")

    out: List[Dict[str, float]] = []
    for p in points:
        if not isinstance(p, dict):
            continue
        if not all(k in p for k in ("x", "y", "t")):
            continue
        out.append({"x": float(p["x"]), "y": float(p["y"]), "t": float(p["t"])})

    if len(out) < 3:
        raise ValueError("too few valid points")

    return out


def coerce_sample(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    extract_features_from_drag(sample)에 넣을 sample 형태로 정리
    """
    # behavior 래핑 지원
    if "behavior" in payload and isinstance(payload["behavior"], dict):
        payload = payload["behavior"]

    sample: Dict[str, Any] = {}

    # points
    sample["points"] = coerce_points(payload)

    # 결과 기반 feature를 쓰는 경우도 대비해서 같이 넣어둠(학습 feature_names에 없으면 자동 무시됨)
    if "correct_count" in payload:
        try:
            sample["correct_count"] = int(payload["correct_count"])
        except Exception:
            pass
    if "is_perfect" in payload:
        sample["is_perfect"] = bool(payload["is_perfect"])

    # 기타 메타는 extractor에서 안 쓰지만 포맷 유지용
    if "metadata" in payload:
        sample["metadata"] = payload["metadata"]

    return sample


def coerce_features(payload: Dict[str, Any]) -> Dict[str, float]:
    """
    이미 feature를 만들어 보내는 경우도 지원(옵션)
    """
    features = None
    if "features" in payload and isinstance(payload["features"], dict):
        features = payload["features"]
    elif "data" in payload and isinstance(payload["data"], dict) and isinstance(payload["data"].get("features"), dict):
        features = payload["data"]["features"]
    elif isinstance(payload, dict):
        # payload 자체가 feature dict인 경우
        features = payload

    if not isinstance(features, dict):
        raise ValueError("features missing")

    out: Dict[str, float] = {}
    for k, v in features.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            continue

    if not out:
        raise ValueError("no valid feature values")

    return out


# =============================================================================
# Inference
# =============================================================================
class PhaseBInfer:
    """
    - 서버 시작 시 1회 로드해서 재사용
    - human_prob > threshold => 사람(pass=True)
    """

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        *,
        model_file: Optional[str] = None,
    ):
        self.model_dir = Path(model_dir)

        # 1) model_dir가 pkl 파일이면 그대로 사용
        if self.model_dir.is_file() and self.model_dir.suffix == ".pkl":
            model_path = self.model_dir
        else:
            # 2) 디렉토리면 후보 파일명 탐색
            cand: List[Path] = []
            if model_file:
                cand.append(self.model_dir / model_file)

            cand += [
                self.model_dir / "phase_b_rf_matched.pkl",
                self.model_dir / "phase_b_rf.pkl",
                self.model_dir / "model_randomforest.pkl",
                self.model_dir / "phase_b_rf.pkl",
            ]

            model_path = next((p for p in cand if p.exists()), None)
            if model_path is None:
                raise FileNotFoundError(
                    f"Missing model pkl under: {self.model_dir} (tried: {[str(p) for p in cand]})"
                )

        payload = joblib.load(model_path)

        # train_random_forest.py에서 저장한 형태:
        # {"model": rf, "threshold": float, "feature_names": [...]}
        if isinstance(payload, dict) and "model" in payload:
            self.model = payload["model"]
            self.threshold = float(payload.get("threshold", 0.5))
            self.feature_names = list(payload.get("feature_names") or FEATURE_NAMES_FALLBACK)
        else:
            # 그냥 estimator만 들어있는 경우도 대비
            self.model = payload
            self.threshold = 0.5
            self.feature_names = FEATURE_NAMES_FALLBACK

        self._classes = list(getattr(self.model, "classes_", [0, 1]))

    def _vectorize(self, features: Dict[str, float]) -> np.ndarray:
        return np.array([float(features.get(n, 0.0)) for n in self.feature_names], dtype=float).reshape(1, -1)

    def _human_prob(self, proba_row: np.ndarray) -> float:
        """
        classes_에서 human=1의 index를 찾아 human_prob 리턴
        """
        if 1 in self._classes:
            idx = self._classes.index(1)
            return float(proba_row[idx])
        # fallback: binary면 마지막을 human으로 간주
        return float(proba_row[-1])

    def infer_human_bot(
        self,
        features: Dict[str, float],
        *,
        return_score: bool = False,
    ) -> Dict[str, Any]:
        X = self._vectorize(features)
        prob = self.model.predict_proba(X)[0]
        human_prob = self._human_prob(prob)

        is_human = human_prob > self.threshold
        result: Dict[str, Any] = {
            "pass": bool(is_human),
            "label": "사람" if is_human else "봇",
        }

        if return_score:
            result["score"] = round(float(human_prob), 6)      # Phase A처럼 score 키 사용
            result["threshold"] = float(self.threshold)

        return result

    def infer_from_payload(
        self,
        payload: Dict[str, Any],
        *,
        return_score: bool = False,
        return_features: bool = False,
    ) -> Dict[str, Any]:
        """
        payload가 points 기반이면 -> feature_extractor로 feature 생성 후 추론
        payload가 features 기반이면 -> 그대로 추론
        """
        # features가 명시적으로 있으면 그걸 우선
        if isinstance(payload, dict) and ("features" in payload):
            feats = coerce_features(payload)
        else:
            # points 기반으로 feature 생성
            sample = coerce_sample(payload)
            feats = extract_features_from_drag(sample)
            if not feats:
                raise ValueError("feature extraction failed (empty features)")

        out = self.infer_human_bot(feats, return_score=return_score)

        if return_features:
            out["features"] = feats

        return out


def save_phase_b_sample(
    *,
    normalized_payload: Dict[str, Any],  # coerce_phase_b_payload 결과
    infer: Dict[str, Any],               # infer_from_payload 결과(return_score/return_features 가능)
) -> Optional[Path]:
    if not PHASE_B_SAVE_ENABLED:
        return None
    if PHASE_B_SAVE_RATIO < 1.0 and random.random() > PHASE_B_SAVE_RATIO:
        return None

    now = datetime.now()
    ymd = now.strftime("%Y%m%d")
    stamp = now.strftime("%Y%m%d_%H%M%S_%f")
    uid = uuid.uuid4().hex[:10]

    label = infer.get("label")
    bucket = "human" if label == "사람" else ("bot" if label == "봇" else "unknown")

    out_dir = PHASE_B_DATA_DIR / bucket / ymd
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{stamp}_{uid}.json"
    tmp_path = out_path.with_suffix(".json.tmp")

    record = {
        # ✅ 원본(정규화된) 행동 데이터: 나중에 feature extractor 바꿔도 재가공 가능
        "behavior": normalized_payload,

        # ✅ 추론 결과 (원하면 score/features까지)
        "inference": {
            "pass": bool(infer.get("pass")),
            "pred_label": label,
            "score": infer.get("score"),
            "threshold": infer.get("threshold"),
            "features": infer.get("features"),  # return_features=True일 때만 들어옴
        },
        "received_at": now.isoformat(timespec="seconds"),
    }

    tmp_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, out_path)
    return out_path

# =============================================================================
# Singleton helper (FastAPI에서 재사용하기 좋음)
# =============================================================================
_default_instance: Optional[PhaseBInfer] = None


def get_instance(model_dir: str | Path = DEFAULT_MODEL_DIR) -> PhaseBInfer:
    global _default_instance
    if _default_instance is None:
        _default_instance = PhaseBInfer(model_dir=model_dir)
    return _default_instance


def reset_instance():
    global _default_instance
    _default_instance = None
