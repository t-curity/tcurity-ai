# python -m training.phase_b.train_random_forest --out phase_b_rf.pkl --target-human-pass 0.97
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, roc_auc_score

from training.phase_b.feature_extractor import (
    extract_features_from_drag,
    FEATURES_BEHAVIOR_14,
    FEATURES_WITH_RESULT_16,
)


# ------------------------------------------------------------
# Robust JSON loader (handles "Extra data")
# ------------------------------------------------------------
def load_json_first(path: Path) -> dict | None:
    """
    파일에 JSON이 2개 이상 붙어있거나(collector/에디터 실수),
    NDJSON처럼 여러 객체가 있는 경우에도 첫 객체만 최대한 복구해서 읽는다.
    """
    raw = path.read_text(encoding="utf-8", errors="replace").lstrip()
    if not raw:
        return None

    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    dec = json.JSONDecoder()
    try:
        obj, _ = dec.raw_decode(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def vectorize(feat: Dict[str, float], names: List[str]) -> List[float]:
    return [float(feat.get(k, 0.0)) for k in names]


def human_pass_threshold(human_scores: np.ndarray, target_pass: float) -> float:
    """
    threshold so that mean(score >= thr | human) ~= target_pass
    """
    if human_scores.size == 0:
        return 1.0
    q = max(0.0, min(1.0, 1.0 - target_pass))
    return float(np.quantile(human_scores, q))


def resolve_out_path(root: Path, out_arg: str) -> Path:
    """
    --out phase_b_rf.pkl 처럼 파일명만 들어오면 models/phase_b/ 아래로 저장.
    상대 경로가 들어오면 root 기준으로 resolve.
    절대 경로면 그대로 사용.
    """
    p = Path(out_arg)
    if p.is_absolute():
        return p
    if p.parent == Path("."):
        return (root / "models" / "phase_b" / p.name).resolve()
    return (root / p).resolve()


def load_dataset(root: Path, data_dir: Path, feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    human_dir = data_dir / "human"
    bot_dir = data_dir / "bot"

    X: List[List[float]] = []
    y: List[int] = []

    def ingest(folder: Path, label: int):
        if not folder.exists():
            return
        for fp in sorted(folder.rglob("*.json")):
            s = load_json_first(fp)
            if not s:
                continue
            feat = extract_features_from_drag(s)
            if not feat:
                continue
            X.append(vectorize(feat, feature_names))
            y.append(label)

    ingest(human_dir, 1)
    ingest(bot_dir, 0)

    if not X:
        raise RuntimeError(f"No valid samples under: {data_dir}")

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="data/phase_b")
    ap.add_argument("--out", type=str, default="phase_b_rf.pkl")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--target-human-pass", type=float, default=0.97)
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--test-size", type=float, default=0.15)

    # 기본: correct_count/is_perfect 제외
    ap.add_argument("--use-result-features", action="store_true")

    # RF params
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--max-depth", type=int, default=18)
    ap.add_argument("--min-samples-leaf", type=int, default=2)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]  # tcurity-ai
    data_dir = (root / args.data_dir).resolve()
    out_path = resolve_out_path(root, args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    feature_names = FEATURES_WITH_RESULT_16 if args.use_result_features else FEATURES_BEHAVIOR_14

    X, y = load_dataset(root, data_dir, feature_names)

    # split: train / (val+test)
    vt = args.val_size + args.test_size
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=vt, random_state=args.seed)
    (train_idx, temp_idx), = sss1.split(X, y)

    X_train, y_train = X[train_idx], y[train_idx]
    X_temp, y_temp = X[temp_idx], y[temp_idx]

    # split temp into val / test
    val_ratio_in_temp = args.val_size / vt
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - val_ratio_in_temp), random_state=args.seed + 1)
    (val_idx, test_idx), = sss2.split(X_temp, y_temp)

    X_val, y_val = X_temp[val_idx], y_temp[val_idx]
    X_test, y_test = X_temp[test_idx], y_temp[test_idx]

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.seed,
        class_weight="balanced",
        min_samples_leaf=args.min_samples_leaf,
        max_depth=args.max_depth,
        n_jobs=-1,
        max_features="sqrt",
    )
    clf.fit(X_train, y_train)

    # Calib threshold on VAL using human-pass target
    p_val = clf.predict_proba(X_val)[:, 1]  # prob(human)
    thr = human_pass_threshold(p_val[y_val == 1], args.target_human_pass)

    # Evaluate on TEST
    p_test = clf.predict_proba(X_test)[:, 1]
    y_pred = (p_test >= thr).astype(int)

    def pass_rate(scores: np.ndarray, thr_: float) -> float:
        return float(np.mean(scores >= thr_)) if scores.size else 0.0

    human_pass = pass_rate(p_test[y_test == 1], thr)
    bot_pass = pass_rate(p_test[y_test == 0], thr)

    try:
        auc = float(roc_auc_score(y_test, p_test))
    except Exception:
        auc = float("nan")

    print("\n[Dataset]")
    print(f"total={len(y)} human={int(np.sum(y==1))} bot={int(np.sum(y==0))}")
    print(f"train={len(y_train)} val={len(y_val)} test={len(y_test)}")
    print(f"features={len(feature_names)} use_result_features={args.use_result_features}")

    print(f"\n[Calib] target_human_pass={args.target_human_pass:.3f} -> threshold={thr:.6f}")
    print("\n[Test @ threshold]")
    print(f"human_pass={human_pass:.4f}")
    print(f"bot_pass  ={bot_pass:.4f} (lower is better)")
    print(f"ROC-AUC   ={auc:.4f}")

    print("\n[Feature Importance]")
    imps = clf.feature_importances_
    for name, imp in sorted(zip(feature_names, imps), key=lambda x: x[1], reverse=True):
        print(f"{name:24s}: {imp:.4f}")

    print("\n[Classification Report @ threshold]")
    print(classification_report(y_test, y_pred, target_names=["bot", "human"]))

    payload = {
        "model_type": "RandomForestClassifier",
        "model": clf,
        "threshold": float(thr),
        "feature_names": feature_names,
        "target_human_pass": float(args.target_human_pass),
        "use_result_features": bool(args.use_result_features),
    }

    # Save
    try:
        import joblib
        joblib.dump(payload, out_path)
    except Exception:
        import pickle
        out_path.write_bytes(pickle.dumps(payload))

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
