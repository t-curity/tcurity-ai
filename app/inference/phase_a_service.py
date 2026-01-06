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


class PhaseAInfer:
    """
    - 앱 시작 시 1번만 로드해서 재사용 (서버 성능/운영 측면에서 필수)
    - score > threshold => 사람, else 봇
    """
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

    def infer_human_bot(
        self, 
        points: List[Dict[str, float]], 
        *, 
        min_points: int = 10,
        return_score: bool = False
    ) -> Dict[str, Any]:
        feat = extract_features(
            points,
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
            return result

        X = np.asarray(feat, dtype=float).reshape(1, -1)
        Xs = self.scaler.transform(X)
        score = float(self.model.score_samples(Xs)[0])
        is_human = score > self.threshold

        result = {
            "pass": is_human,
            "label": "사람" if is_human else "봇"
        }

        # 디버그 로그
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
    - points는 coerce_points로 정규화된 값을 저장(학습 재사용 목적)
    - infer는 score/threshold 포함 가능(return_score=True로 받은 결과)
    """
    if not PHASE_A_SAVE_ENABLED:
        return None
    if PHASE_A_SAVE_RATIO < 1.0 and random.random() > PHASE_A_SAVE_RATIO:
        return None

    now = datetime.now()
    ymd = now.strftime("%Y%m%d")
    stamp = now.strftime("%Y%m%d_%H%M%S_%f")
    uid = uuid.uuid4().hex[:10]

    pred_label = infer.get("label")
    if pred_label in ("사람", "human", "HUMAN"):
        bucket = "human_pred"
    elif pred_label in ("봇", "bot", "BOT"):
        bucket = "bot_pred"
    else:
        bucket = "unknown"

    out_dir = PHASE_A_DATA_DIR / bucket / ymd
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stamp}_{uid}.json"
    tmp_path = out_path.with_suffix(".json.tmp")

    record = {
        # ✅ 학습용 핵심
        "points": points,

        # ✅ 나중에 라벨링/분석에 도움(모델 점수 포함)
        "inference": {
            "pass": bool(infer.get("pass")),
            "pred_label": pred_label,
            "score": infer.get("score"),
            "threshold": infer.get("threshold"),
        },

        # ✅ 선택 메타(필요한 것만)
        "received_at": now.isoformat(timespec="seconds"),
        "line": raw_payload.get("line") or raw_payload.get("cutline") or raw_payload.get("guide_line"),
        "metadata": raw_payload.get("metadata"),
    }

    tmp_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, out_path)  # atomic swap
    return out_path