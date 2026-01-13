#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
app/inference/phase_b_service.py

Phase B: 행동 기반 RandomForest 추론 서비스
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import joblib
import os, json, uuid, random
from datetime import datetime

# 디버그 로거 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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
    import sys
    project_root = Path(os.getenv("PROJECT_ROOT", "/app"))
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from training.phase_b.feature_extractor import extract_features_from_drag
    return extract_features_from_drag


extract_features_from_drag = _import_extract_features()


def coerce_points(payload: Dict[str, Any]) -> List[Dict[str, float]]:
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
    if "behavior" in payload and isinstance(payload["behavior"], dict):
        payload = payload["behavior"]

    sample: Dict[str, Any] = {}
    sample["points"] = coerce_points(payload)

    if "correct_count" in payload:
        try:
            sample["correct_count"] = int(payload["correct_count"])
        except Exception:
            pass
    if "is_perfect" in payload:
        sample["is_perfect"] = bool(payload["is_perfect"])

    if "metadata" in payload:
        sample["metadata"] = payload["metadata"]

    return sample


def coerce_features(payload: Dict[str, Any]) -> Dict[str, float]:
    features = None
    if "features" in payload and isinstance(payload["features"], dict):
        features = payload["features"]
    elif "data" in payload and isinstance(payload["data"], dict) and isinstance(payload["data"].get("features"), dict):
        features = payload["data"]["features"]
    elif isinstance(payload, dict):
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


class PhaseBInfer:
    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        *,
        model_file: Optional[str] = None,
    ):
        self.model_dir = Path(model_dir)

        if self.model_dir.is_file() and self.model_dir.suffix == ".pkl":
            model_path = self.model_dir
        else:
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

        logger.info(f"[Phase B] 모델 로드: {model_path}")
        payload = joblib.load(model_path)

        if isinstance(payload, dict) and "model" in payload:
            self.model = payload["model"]
            self.threshold = float(payload.get("threshold", 0.5))
            self.feature_names = list(payload.get("feature_names") or FEATURE_NAMES_FALLBACK)
        else:
            self.model = payload
            self.threshold = 0.5
            self.feature_names = FEATURE_NAMES_FALLBACK

        self._classes = list(getattr(self.model, "classes_", [0, 1]))
        logger.info(f"[Phase B] threshold={self.threshold}, features={len(self.feature_names)}개")

    def _vectorize(self, features: Dict[str, float]) -> np.ndarray:
        return np.array([float(features.get(n, 0.0)) for n in self.feature_names], dtype=float).reshape(1, -1)

    def _human_prob(self, proba_row: np.ndarray) -> float:
        if 1 in self._classes:
            idx = self._classes.index(1)
            return float(proba_row[idx])
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
            result["score"] = round(float(human_prob), 6)
            result["threshold"] = float(self.threshold)

        # 디버그 로그
        logger.debug(f"[Phase B] human_prob={human_prob:.4f}, threshold={self.threshold}, pass={is_human}")

        return result

    def infer_from_payload(
        self,
        payload: Dict[str, Any],
        *,
        return_score: bool = False,
        return_features: bool = False,
    ) -> Dict[str, Any]:
        logger.debug(f"[Phase B] infer_from_payload 시작")
        
        if isinstance(payload, dict) and ("features" in payload):
            feats = coerce_features(payload)
            logger.debug(f"[Phase B] features 직접 사용: {len(feats)}개")
        else:
            sample = coerce_sample(payload)
            logger.debug(f"[Phase B] points에서 feature 추출: {len(sample['points'])}개 포인트")
            
            feats = extract_features_from_drag(sample)
            if not feats:
                raise ValueError("feature extraction failed (empty features)")
            
            # 주요 feature 로그
            logger.debug(f"[Phase B] 추출된 features:")
            for k, v in feats.items():
                logger.debug(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        out = self.infer_human_bot(feats, return_score=return_score)

        if return_features:
            out["features"] = feats

        logger.info(f"[Phase B] 결과: {out.get('label')}, score={out.get('score')}")
        return out


def save_phase_b_sample(
    *,
    normalized_payload: Dict[str, Any],
    infer: Dict[str, Any],
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
    bucket = "human_pred" if label == "사람" else ("bot_pred" if label == "봇" else "unknown")

    out_dir = PHASE_B_DATA_DIR / bucket / ymd
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{stamp}_{uid}.json"
    tmp_path = out_path.with_suffix(".json.tmp")

    record = {
        "behavior": normalized_payload,
        "inference": {
            "pass": bool(infer.get("pass")),
            "pred_label": label,
            "score": infer.get("score"),
            "threshold": infer.get("threshold"),
            "features": infer.get("features"),
        },
        "received_at": now.isoformat(timespec="seconds"),
    }

    tmp_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, out_path)
    
    logger.debug(f"[Phase B] 샘플 저장: {out_path}")
    return out_path


_default_instance: Optional[PhaseBInfer] = None


def get_instance(model_dir: str | Path = DEFAULT_MODEL_DIR) -> PhaseBInfer:
    global _default_instance
    if _default_instance is None:
        _default_instance = PhaseBInfer(model_dir=model_dir)
    return _default_instance


def reset_instance():
    global _default_instance
    _default_instance = None