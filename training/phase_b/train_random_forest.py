# python -m training.phase_b.train_random_forest --out phase_b_rf.pkl --target-human-pass 0.97
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, roc_auc_score

import mlflow
import mlflow.sklearn

from training.phase_b.feature_extractor import (
    extract_features_from_drag,
    FEATURES_BEHAVIOR_14,
    FEATURES_WITH_RESULT_16,
)


# ------------------------------------------------------------
# MLflow 설정
# ------------------------------------------------------------
def _to_tracking_uri(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        return "file://" + str(Path("mlruns").resolve())
    if s.lower().startswith(("file:", "sqlite:", "http://", "https://", "databricks")):
        return s
    return s


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
    human_pred_dir = data_dir / "human_pred"
    bot_dir = data_dir / "bot"
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
            
            # human_pred 구조 지원: behavior.points 추출
            if "behavior" in s and isinstance(s["behavior"], dict):
                s = s["behavior"]
            
            feat = extract_features_from_drag(s)
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
    
    # MLflow 설정
    ap.add_argument("--experiment", type=str, default=None, help="MLflow experiment name")
    ap.add_argument("--run-name", type=str, default=None, help="MLflow run name")
    ap.add_argument("--dataset-version", type=str, default=None, help="Dataset version tag")
    
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]  # tcurity-ai
    data_dir = (root / args.data_dir).resolve()
    out_path = resolve_out_path(root, args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    feature_names = FEATURES_WITH_RESULT_16 if args.use_result_features else FEATURES_BEHAVIOR_14

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
    print(f"[dataset_version]  {dataset_version}")
    print()

    # 데이터 로드
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

    # MLflow 시작
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        # Tags
        mlflow.set_tag("phase", "B")
        mlflow.set_tag("model_name", "random_forest")
        mlflow.set_tag("dataset_version", dataset_version)
        mlflow.set_tag("use_result_features", str(args.use_result_features))

        # 모델 학습
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
        bot_block = 1.0 - bot_pass  # 봇 차단율

        try:
            auc = float(roc_auc_score(y_test, p_test))
        except Exception:
            auc = float("nan")

        # MLflow Params 기록
        mlflow.log_params({
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "target_human_pass": args.target_human_pass,
            "seed": args.seed,
            "val_size": args.val_size,
            "test_size": args.test_size,
            "num_features": len(feature_names),
            "use_result_features": args.use_result_features,
        })

        # MLflow Metrics 기록
        mlflow.log_metrics({
            "human_pass": human_pass,
            "bot_pass": bot_pass,
            "bot_block": bot_block,
            "roc_auc": auc,
            "threshold": thr,
            "total_samples": len(y),
            "human_samples": int(np.sum(y == 1)),
            "bot_samples": int(np.sum(y == 0)),
            "train_size": len(y_train),
            "val_size": len(y_val),
            "test_size": len(y_test),
        })

        # Feature Importance 기록
        imps = clf.feature_importances_
        for name, imp in zip(feature_names, imps):
            mlflow.log_metric(f"feat_imp_{name}", float(imp))

        # 콘솔 출력
        print("\n[Dataset]")
        print(f"total={len(y)} human={int(np.sum(y==1))} bot={int(np.sum(y==0))}")
        print(f"train={len(y_train)} val={len(y_val)} test={len(y_test)}")
        print(f"features={len(feature_names)} use_result_features={args.use_result_features}")

        print(f"\n[Calib] target_human_pass={args.target_human_pass:.3f} -> threshold={thr:.6f}")
        print("\n[Test @ threshold]")
        print(f"human_pass={human_pass:.4f}")
        print(f"bot_pass  ={bot_pass:.4f} (lower is better)")
        print(f"bot_block ={bot_block:.4f}")
        print(f"ROC-AUC   ={auc:.4f}")

        print("\n[Feature Importance]")
        for name, imp in sorted(zip(feature_names, imps), key=lambda x: x[1], reverse=True):
            print(f"{name:24s}: {imp:.4f}")

        print("\n[Classification Report @ threshold]")
        report = classification_report(y_test, y_pred, target_names=["bot", "human"])
        print(report)

        # 모델 저장
        payload = {
            "model_type": "RandomForestClassifier",
            "model": clf,
            "threshold": float(thr),
            "feature_names": feature_names,
            "target_human_pass": float(args.target_human_pass),
            "use_result_features": bool(args.use_result_features),
        }

        try:
            import joblib
            joblib.dump(payload, out_path)
        except Exception:
            import pickle
            out_path.write_bytes(pickle.dumps(payload))

        print(f"\nSaved: {out_path}")

        # MLflow에 모델 및 아티팩트 기록
        mlflow.sklearn.log_model(clf, artifact_path="random_forest_model")
        mlflow.log_artifact(str(out_path))

        print(f"\n[MLflow] Run logged to: {tracking_uri}")
        print(f"[MLflow] Experiment: {experiment_name}")
        print(f"[MLflow] Run name: {run_name}")


if __name__ == "__main__":
    main()