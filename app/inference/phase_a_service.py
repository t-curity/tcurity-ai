from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import joblib


DEFAULT_MODEL_DIR = Path("/home/ubuntu/tcurity-ai/models/phase_a")


def _import_extract_features():
    import sys
    project_root = Path("/home/ubuntu/tcurity-ai")
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

    def infer_human_bot(self, points: List[Dict[str, float]], *, min_points: int = 10) -> str:
        feat = extract_features(
            points,
            line=None,
            sanitize_time=True,
            normalize_to_line=False,
            min_points=min_points,
            same_t_eps=0.0,
        )

        # fail-closed
        if feat is None:
            return "봇"

        X = np.asarray(feat, dtype=float).reshape(1, -1)
        Xs = self.scaler.transform(X)
        score = float(self.model.score_samples(Xs)[0])
        return "사람" if score > self.threshold else "봇"
