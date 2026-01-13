from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from feature_extractor import extract_features_from_drag


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="phase_b_rf.pkl")
    ap.add_argument("--input", type=str, required=True)
    args = ap.parse_args()

    root = project_root()
    model_path = root / "models" / "phase_b" / args.model
    sample_path = Path(args.input).resolve()

    bundle = joblib.load(model_path)
    clf = bundle["model"]
    feature_names = bundle["feature_names"]
    threshold = float(bundle["threshold"])

    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    feats = extract_features_from_drag(sample)
    x = np.array([[float(feats[k]) for k in feature_names]], dtype=float)

    proba = clf.predict_proba(x)[0]
    human_score = float(proba[0])
    bot_score = float(proba[1])

    passed = human_score >= threshold
    out = {
        "pass": bool(passed),
        "label": "사람" if passed else "봇",
        "score": human_score,
        "threshold": threshold,
        # 디버깅/모니터링용(Phase A엔 없었지만 유용)
        "bot_score": bot_score,
    }

    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
