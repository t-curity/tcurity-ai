#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[train_random_forest.py] - 고급 버전

개선 사항:
1. 26개 고급 feature 사용 (기존 14개 + 새로운 12개)
2. GridSearch 하이퍼파라미터 튜닝
3. Cross-validation
4. Feature importance 시각화

사용법:
  # 기본 실행
  python -m training.phase_b.train_random_forest --target-human-pass 0.99

  # 하이퍼파라미터 튜닝 모드
  python -m training.phase_b.train_random_forest --tune --target-human-pass 0.99

  # 특정 파라미터 지정
  python -m training.phase_b.train_random_forest \
    --n-estimators 1000 \
    --max-depth 25 \
    --min-samples-leaf 1 \
    --target-human-pass 0.99
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import (
    StratifiedShuffleSplit, 
    GridSearchCV, 
    cross_val_score
)
from sklearn.metrics import (
    classification_report, 
    roc_auc_score, 
    precision_recall_curve,
    f1_score
)

import mlflow
import mlflow.sklearn

# Feature extractor v2 import
import sys
THIS = Path(__file__).resolve()
ROOT = THIS.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.phase_b.feature_extractor import (
    extract_features,
    FEATURES_BEHAVIOR_14,
    FEATURES_ADVANCED_12,
    FEATURES_ALL_26,
)


def _to_tracking_uri(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        return "file://" + str(Path("mlruns").resolve())
    if s.lower().startswith(("file:", "sqlite:", "http://", "https://", "databricks")):
        return s
    return s


def load_json_first(path: Path) -> dict | None:
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
    if human_scores.size == 0:
        return 1.0
    q = max(0.0, min(1.0, 1.0 - target_pass))
    return float(np.quantile(human_scores, q))


def resolve_out_path(root: Path, out_arg: str) -> Path:
    p = Path(out_arg)
    if p.is_absolute():
        return p
    if p.parent == Path("."):
        return (root / "models" / "phase_b" / p.name).resolve()
    return (root / p).resolve()


def load_dataset(
    root: Path, 
    data_dir: Path, 
    feature_names: List[str],
    include_advanced: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """데이터셋 로드 - v2 feature extractor 사용"""
    human_dir = data_dir / "human"
    human_pred_dir = data_dir / "human_pred"
    bot_dir = data_dir / "bot_v4"
    bot_pred_dir = data_dir / "bot_pred"

    X: List[List[float]] = []
    y: List[int] = []

    def ingest(folder: Path, label: int):
        if not folder.exists():
            return
        for fp in sorted(folder.rglob("*.json")):
            s = load_json_first(fp)
            if not s:
                continue
            
            if "behavior" in s and isinstance(s["behavior"], dict):
                s = s["behavior"]
            
            # v2 feature extractor 사용
            feat = extract_features(s, include_advanced=include_advanced)
            if not feat:
                continue
            X.append(vectorize(feat, feature_names))
            y.append(label)

    ingest(human_dir, 1)
    ingest(human_pred_dir, 1)
    ingest(bot_dir, 0)
    ingest(bot_pred_dir, 0)

    if not X:
        raise RuntimeError(f"No valid samples under: {data_dir}")

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32)


def tune_hyperparameters(X_train, y_train, seed: int = 42) -> Dict:
    """GridSearchCV로 최적 하이퍼파라미터 찾기"""
    print("\n" + "=" * 60)
    print("[Hyperparameter Tuning] GridSearchCV")
    print("=" * 60)
    
    param_grid = {
        'n_estimators': [300, 500, 800, 1000],
        'max_depth': [15, 20, 25, 30, None],
        'min_samples_leaf': [1, 2, 3, 5],
        'min_samples_split': [2, 5, 10],
        'max_features': ['sqrt', 'log2', 0.3, 0.5],
    }
    
    # 작은 그리드로 빠르게 테스트
    param_grid_small = {
        'n_estimators': [500, 800, 1000],
        'max_depth': [20, 25, 30],
        'min_samples_leaf': [1, 2, 3],
        'max_features': ['sqrt', 0.3],
    }
    
    base_clf = RandomForestClassifier(
        random_state=seed,
        class_weight="balanced",
        n_jobs=-1,
    )
    
    grid_search = GridSearchCV(
        estimator=base_clf,
        param_grid=param_grid_small,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1,
    )
    
    print("Searching...")
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest params: {grid_search.best_params_}")
    print(f"Best ROC-AUC: {grid_search.best_score_:.4f}")
    
    return grid_search.best_params_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="data/phase_b")
    ap.add_argument("--out", type=str, default="model_randomforest.pkl")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--target-human-pass", type=float, default=0.99)
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--test-size", type=float, default=0.15)

    # Feature 옵션
    ap.add_argument("--use-advanced-features", action="store_true", default=True,
                    help="고급 26개 feature 사용 (기본값: True)")
    ap.add_argument("--basic-features-only", action="store_true",
                    help="기본 14개 feature만 사용")

    # RF params
    ap.add_argument("--n-estimators", type=int, default=800)
    ap.add_argument("--max-depth", type=int, default=25)
    ap.add_argument("--min-samples-leaf", type=int, default=1)
    ap.add_argument("--min-samples-split", type=int, default=2)
    ap.add_argument("--max-features", type=str, default="sqrt")
    
    # 튜닝 모드
    ap.add_argument("--tune", action="store_true",
                    help="GridSearchCV로 하이퍼파라미터 튜닝")
    
    # MLflow 설정
    ap.add_argument("--experiment", type=str, default=None)
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--dataset-version", type=str, default=None)
    
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    data_dir = (root / args.data_dir).resolve()
    out_path = resolve_out_path(root, args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Feature 선택
    if args.basic_features_only:
        feature_names = FEATURES_BEHAVIOR_14
        include_advanced = False
    else:
        feature_names = FEATURES_ALL_26
        include_advanced = True

    # MLflow 설정
    tracking_uri = _to_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", ""))
    experiment_name = args.experiment or os.environ.get("MLFLOW_EXPERIMENT_NAME") or "captcha-phase-b"
    run_name = args.run_name or f"phaseB_rf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dataset_version = args.dataset_version or os.environ.get("DATASET_VERSION") or "dev"

    print("=" * 60)
    print("[Phase B] Random Forest Training")
    print("=" * 60)
    print(f"[data_dir]         {data_dir}")
    print(f"[out_path]         {out_path}")
    print(f"[mlflow]           {tracking_uri}")
    print(f"[experiment]       {experiment_name}")
    print(f"[features]         {len(feature_names)} ({'' if include_advanced else 'basic only'})")
    print(f"[tune mode]        {args.tune}")
    print()

    # 데이터 로드
    X, y = load_dataset(root, data_dir, feature_names, include_advanced)

    # split
    vt = args.val_size + args.test_size
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=vt, random_state=args.seed)
    (train_idx, temp_idx), = sss1.split(X, y)

    X_train, y_train = X[train_idx], y[train_idx]
    X_temp, y_temp = X[temp_idx], y[temp_idx]

    val_ratio_in_temp = args.val_size / vt
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - val_ratio_in_temp), random_state=args.seed + 1)
    (val_idx, test_idx), = sss2.split(X_temp, y_temp)

    X_val, y_val = X_temp[val_idx], y_temp[val_idx]
    X_test, y_test = X_temp[test_idx], y_temp[test_idx]

    # 하이퍼파라미터 튜닝
    if args.tune:
        best_params = tune_hyperparameters(X_train, y_train, args.seed)
        n_estimators = best_params.get('n_estimators', args.n_estimators)
        max_depth = best_params.get('max_depth', args.max_depth)
        min_samples_leaf = best_params.get('min_samples_leaf', args.min_samples_leaf)
        min_samples_split = best_params.get('min_samples_split', args.min_samples_split)
        max_features = best_params.get('max_features', args.max_features)
    else:
        n_estimators = args.n_estimators
        max_depth = args.max_depth
        min_samples_leaf = args.min_samples_leaf
        min_samples_split = args.min_samples_split
        max_features = args.max_features

    # MLflow 시작
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        # Tags
        mlflow.set_tag("phase", "B")
        mlflow.set_tag("model_name", "random_forest")
        mlflow.set_tag("dataset_version", dataset_version)
        mlflow.set_tag("feature_version", "v2_advanced" if include_advanced else "v1_basic")
        mlflow.set_tag("tuned", str(args.tune))

        # 모델 학습
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            max_features=max_features,
            random_state=args.seed,
            class_weight="balanced",
            n_jobs=-1,
        )
        
        print("\n[Training RandomForest]")
        print(f"  n_estimators: {n_estimators}")
        print(f"  max_depth: {max_depth}")
        print(f"  min_samples_leaf: {min_samples_leaf}")
        print(f"  min_samples_split: {min_samples_split}")
        print(f"  max_features: {max_features}")
        
        clf.fit(X_train, y_train)

        # Cross-validation score
        cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='roc_auc')
        print(f"\n[Cross-Validation] ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # Threshold calibration
        p_val = clf.predict_proba(X_val)[:, 1]
        thr = human_pass_threshold(p_val[y_val == 1], args.target_human_pass)

        # Test evaluation
        p_test = clf.predict_proba(X_test)[:, 1]
        y_pred = (p_test >= thr).astype(int)

        def pass_rate(scores: np.ndarray, thr_: float) -> float:
            return float(np.mean(scores >= thr_)) if scores.size else 0.0

        human_pass = pass_rate(p_test[y_test == 1], thr)
        bot_pass = pass_rate(p_test[y_test == 0], thr)
        bot_block = 1.0 - bot_pass

        try:
            auc = float(roc_auc_score(y_test, p_test))
        except Exception:
            auc = float("nan")

        f1 = f1_score(y_test, y_pred, average='weighted')

        # MLflow Params
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "min_samples_split": min_samples_split,
            "max_features": str(max_features),
            "target_human_pass": args.target_human_pass,
            "seed": args.seed,
            "num_features": len(feature_names),
            "include_advanced_features": include_advanced,
            "tuned": args.tune,
        })

        # MLflow Metrics
        mlflow.log_metrics({
            "human_pass": human_pass,
            "bot_pass": bot_pass,
            "bot_block": bot_block,
            "roc_auc": auc,
            "f1_score": f1,
            "cv_roc_auc_mean": cv_scores.mean(),
            "cv_roc_auc_std": cv_scores.std(),
            "threshold": thr,
            "total_samples": len(y),
            "human_samples": int(np.sum(y == 1)),
            "bot_samples": int(np.sum(y == 0)),
        })

        # Feature Importance
        imps = clf.feature_importances_
        for name, imp in zip(feature_names, imps):
            mlflow.log_metric(f"feat_imp_{name}", float(imp))

        # 출력
        print("\n[Dataset]")
        print(f"total={len(y)} human={int(np.sum(y==1))} bot={int(np.sum(y==0))}")
        print(f"train={len(y_train)} val={len(y_val)} test={len(y_test)}")
        print(f"features={len(feature_names)}")

        print(f"\n[Calib] target_human_pass={args.target_human_pass:.3f} -> threshold={thr:.6f}")
        print("\n[Test @ threshold]")
        print(f"human_pass={human_pass:.4f}")
        print(f"bot_pass  ={bot_pass:.4f}")
        print(f"bot_block ={bot_block:.4f}")
        print(f"ROC-AUC   ={auc:.4f}")
        print(f"F1-Score  ={f1:.4f}")

        print("\n[Feature Importance - Top 15]")
        sorted_imp = sorted(zip(feature_names, imps), key=lambda x: x[1], reverse=True)
        for name, imp in sorted_imp[:15]:
            marker = "★" if name in FEATURES_ADVANCED_12 else " "
            print(f"  {marker} {name:28s}: {imp:.4f}")

        print("\n[Classification Report]")
        print(classification_report(y_test, y_pred, target_names=["bot", "human"]))

        # 모델 저장
        payload = {
            "model_type": "RandomForestClassifier",
            "model": clf,
            "threshold": float(thr),
            "feature_names": feature_names,
            "target_human_pass": float(args.target_human_pass),
            "include_advanced_features": include_advanced,
            "hyperparameters": {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "min_samples_split": min_samples_split,
                "max_features": max_features,
            },
        }

        try:
            import joblib
            joblib.dump(payload, out_path)
        except Exception:
            import pickle
            out_path.write_bytes(pickle.dumps(payload))

        print(f"\nSaved: {out_path}")

        mlflow.sklearn.log_model(clf, artifact_path="random_forest_model")
        mlflow.log_artifact(str(out_path))

        print(f"\n[MLflow] Run logged to: {tracking_uri}")
        print(f"[MLflow] Experiment: {experiment_name}")


if __name__ == "__main__":
    main()