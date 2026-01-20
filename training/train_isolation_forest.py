#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_isolation_forest.py

Isolation Forest 기반 드래그 봇 탐지 모델 학습 스크립트
- MLflow 환경변수 기반
- 모델 산출물: 최신만 유지(고정 파일명 덮어쓰기) + atomic swap
- 선택적으로 진단 파일 저장(PHASE_A_SAVE_DIAGNOSTICS)

권장 환경변수:
  MLFLOW_TRACKING_URI=http://...:5000                                 # MLflow 서버 URI
  PHASE_A_EXPERIMENT_NAME=phase-a-isolation-forest                    # MLflow experiment 이름
  PHASE_A_DATA_DIR=/home/ubuntu/tcurity-ai/data/drag_trainset         # 사람 드래그 json
  PHASE_A_MODEL_DIR=/home/ubuntu/tcurity-ai/models/phase_a            # 최신모델 위치(고정)
  PHASE_A_DATASET_VERSION=v001                                        # 선택
  PHASE_A_SAVE_DIAGNOSTICS=0|1                                        # 선택(기본 0)
  PHASE_A_KEEP_BACKUPS=0|1|2|3...                                     # 선택(기본 0)
"""

import os
import json
import argparse
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import joblib
import mlflow
import mlflow.sklearn

import matplotlib
matplotlib.use("Agg")  # 서버 환경 안전
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

from feature_extractor import extract_features


# =============================================================================
# Utilities
# =============================================================================

def _to_tracking_uri(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        return "file://" + str(Path("mlruns").resolve())
    if s.lower().startswith(("file:", "sqlite:", "http://", "https://", "databricks")):
        return s
    return "file://" + Path(s).expanduser().resolve().as_posix()


def load_json_samples(folder: str, recursive: bool = True) -> list[dict]:
    folder_path = Path(folder).expanduser().resolve()
    if not folder_path.exists():
        raise FileNotFoundError(f"데이터 폴더를 찾을 수 없습니다: {folder_path}")

    pattern = "**/*.json" if recursive else "*.json"
    json_files = sorted(folder_path.glob(pattern))

    if not json_files:
        print(f"[WARN] JSON 파일이 없습니다: {folder_path}")
        return []

    print(f"[INFO] {len(json_files)}개 JSON 파일 발견. 예시:")
    for p in json_files[:5]:
        print(f"       - {p.name}")

    samples = []
    bad_files = 0

    for jf in json_files:
        try:
            obj = json.loads(jf.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            try:
                obj = json.loads(jf.read_text(encoding="utf-8-sig"))
            except Exception:
                bad_files += 1
                continue
        except Exception:
            bad_files += 1
            continue

        if isinstance(obj, dict):
            points = obj.get("points") or obj.get("trajectory") or obj.get("data")
            line = obj.get("line") or obj.get("cutline") or obj.get("guide_line")
        elif isinstance(obj, list):
            points = obj
            line = None
        else:
            continue

        if not isinstance(points, list) or not points:
            continue

        valid = all(
            isinstance(p, dict) and all(k in p for k in ("x", "y", "t"))
            for p in points[:3]
        )
        if not valid:
            continue

        samples.append({"path": str(jf), "points": points, "line": line})

    if bad_files:
        print(f"[WARN] {bad_files}개 파일이 유효하지 않아 스킵됨")

    return samples


def split_three(samples: list, calib_ratio: float, test_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(samples))
    rng.shuffle(idx)

    n = len(samples)
    n_calib = int(round(n * calib_ratio))
    n_test = int(round(n * test_ratio))

    return (
        [samples[i] for i in idx[n_calib + n_test:]],
        [samples[i] for i in idx[:n_calib]],
        [samples[i] for i in idx[n_calib:n_calib + n_test]],
    )


def threshold_for_target_pass(scores: np.ndarray, target_pass: float) -> float:
    return float(np.quantile(scores, 1.0 - np.clip(target_pass, 0.0, 1.0)))


def save_score_histogram(path: Path, train_scores, calib_scores, test_scores, bot_scores, threshold: float):
    plt.figure(figsize=(10, 6))
    if train_scores.size:
        plt.hist(train_scores, bins=60, alpha=0.6, label="human_train")
    if calib_scores.size:
        plt.hist(calib_scores, bins=60, alpha=0.6, label="human_calib")
    if test_scores.size:
        plt.hist(test_scores, bins=60, alpha=0.6, label="human_test")
    if bot_scores.size:
        plt.hist(bot_scores, bins=60, alpha=0.6, label="bot_synth")

    plt.axvline(threshold, color="red", linewidth=2, linestyle="--", label=f"threshold={threshold:.3f}")
    plt.legend()
    plt.title("Score Distribution (higher = more human-like)")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _atomic_replace(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.replace(src, dst)  # same FS에서 atomic


def _maybe_backup(dst_dir: Path, keep: int, files: list[str]) -> None:
    if keep <= 0:
        return

    backups_dir = dst_dir / "backups"
    backups_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bdir = backups_dir / stamp
    bdir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for fn in files:
        p = dst_dir / fn
        if p.exists():
            shutil.copy2(p, bdir / fn)
            copied += 1

    if copied == 0:
        # 백업할 기존 파일이 없으면 폴더 제거
        try:
            bdir.rmdir()
        except Exception:
            pass
        return

    # 오래된 백업 정리(최신 keep개만 유지)
    all_b = sorted([d for d in backups_dir.iterdir() if d.is_dir()], reverse=True)
    for d in all_b[keep:]:
        shutil.rmtree(d, ignore_errors=True)


def _log_model_compat(model):
    try:
        mlflow.sklearn.log_model(model, name="iforest_model")
    except TypeError:
        mlflow.sklearn.log_model(model, artifact_path="iforest_model")


# =============================================================================
# Synthetic Bot Generators (원본 유지)
# =============================================================================

def _sample_dt_ms(rng: np.random.Generator, base_ms: float = 16.666, jitter_ms: float = 6.0) -> float:
    dt = max(4.0, base_ms + rng.normal(0.0, jitter_ms))
    if rng.random() < 0.03:
        dt += rng.uniform(30, 200)
    return float(dt)


def gen_bot_linear(num: int, steps: int, seed: int, *,
                   u_noise: float = 0.0, v_noise: float = 0.0,
                   dt_jitter_ms: float = 0.0, dt_ms: float = 16.666) -> list:
    rng = np.random.default_rng(seed)
    bots = []
    for _ in range(num):
        u = np.linspace(0.0, 1.0, steps)
        v = np.zeros(steps)

        if u_noise > 0:
            u = u + rng.normal(0.0, u_noise, steps)
            u[0], u[-1] = 0.0, 1.0
        if v_noise > 0:
            v = v + rng.normal(0.0, v_noise, steps)

        u = np.clip(u, -0.10, 1.10)
        v = np.clip(v, -0.20, 0.20)

        if dt_jitter_ms > 0:
            dt = np.clip(rng.normal(dt_ms, dt_jitter_ms, steps - 1), dt_ms * 0.25, dt_ms * 3.0)
        else:
            dt = np.full(steps - 1, dt_ms)

        t = np.concatenate([[0.0], np.cumsum(dt)])
        bots.append([{"x": float(u[i]), "y": float(v[i]), "t": float(t[i])} for i in range(steps)])
    return bots


def gen_bot_curvy(num: int, steps: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    bots = []
    for _ in range(num):
        u, v, t = rng.uniform(-0.05, 0.05), rng.uniform(-0.02, 0.02), 0.0
        amp = rng.uniform(0.01, 0.08)
        freq = rng.uniform(0.05, 0.20)
        du = rng.uniform(0.7, 1.3) / steps
        phase = rng.uniform(0, 2 * np.pi)

        points = []
        for i in range(steps):
            u += du + rng.normal(0, 0.003)
            v = amp * np.sin(freq * i + phase) + rng.normal(0, 0.003)
            t += _sample_dt_ms(rng)
            points.append({"x": float(u), "y": float(v), "t": float(t)})
        bots.append(points)
    return bots


def gen_bot_stopgo(num: int, steps: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    bots = []
    for _ in range(num):
        u, v, t = rng.uniform(-0.05, 0.05), rng.uniform(-0.02, 0.02), 0.0
        du = rng.uniform(0.7, 1.3) / steps

        points = []
        for _ in range(steps):
            if rng.random() < 0.12:
                t += rng.uniform(80, 450)
                u += rng.normal(0, 0.001)
                v += rng.normal(0, 0.001)
            else:
                u += du + rng.normal(0, 0.004)
                v += rng.normal(0, 0.004)
                t += _sample_dt_ms(rng, jitter_ms=9.0)
            points.append({"x": float(u), "y": float(v), "t": float(t)})
        bots.append(points)
    return bots


def gen_bot_teleport(num: int, steps: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    bots = []
    for _ in range(num):
        u, v, t = rng.uniform(-0.05, 0.05), rng.uniform(-0.02, 0.02), 0.0
        du = rng.uniform(0.7, 1.3) / steps

        points = []
        for _ in range(steps):
            if rng.random() < 0.04:
                u += rng.uniform(-0.25, 0.25)
                v += rng.uniform(-0.15, 0.15)
                t += rng.uniform(5, 30)
            else:
                u += du + rng.normal(0, 0.004)
                v += rng.normal(0, 0.006)
                t += _sample_dt_ms(rng, jitter_ms=7.0)
            points.append({"x": float(u), "y": float(v), "t": float(t)})
        bots.append(points)
    return bots


def gen_bot_linefollow(num: int, steps: int, seed: int, *,
                       v_sigma: float = 0.012, u_sigma: float = 0.006,
                       backtrack_prob: float = 0.25, pause_prob: float = 0.20) -> list:
    rng = np.random.default_rng(seed)
    bots = []
    steps = max(2, steps)

    def smoothstep(x):
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    for _ in range(num):
        du = 1.0 / (steps - 1)
        inc = np.clip(rng.normal(du, du * 0.35, steps - 1), du * 0.05, du * 3.0)

        if rng.random() < backtrack_prob:
            k = rng.integers(1, max(2, steps // 15))
            idx = rng.choice(steps - 1, size=k, replace=False)
            inc[idx] *= -rng.uniform(0.15, 0.65, k)

        u = np.concatenate([[0.0], np.cumsum(inc)])
        u = (u - u.min()) / (u.max() + 1e-9)
        u += rng.normal(0.0, u_sigma, steps)
        u[0], u[-1] = 0.0, 1.0

        if rng.random() < 0.5:
            u = smoothstep((u - u.min()) / (u.max() - u.min() + 1e-9))
            u[0], u[-1] = 0.0, 1.0

        u = np.clip(u, -0.15, 1.15)

        v = rng.normal(0.0, v_sigma, steps)
        if rng.random() < 0.35:
            v += np.linspace(0.0, rng.normal(0.0, v_sigma * 0.6), steps)

        dt = np.clip(rng.normal(16.0, 6.0, steps - 1), 4.0, 45.0)
        if rng.random() < pause_prob:
            k = rng.integers(1, max(2, steps // 12))
            idx = rng.choice(steps - 1, size=k, replace=False)
            dt[idx] += rng.uniform(40.0, 180.0, k)

        t = np.concatenate([[0.0], np.cumsum(dt)])
        bots.append([{"x": float(u[i]), "y": float(v[i]), "t": float(t[i])} for i in range(steps)])

    return bots


def generate_bots(num: int, steps: int, seed: int, mode: str, **linear_kwargs) -> list[dict]:
    mode = (mode or "hard").lower()
    rng = np.random.default_rng(seed)

    def wrap(arr, mname):
        return [{"points": pts, "mode": mname} for pts in arr]

    generators = {
        "linear": lambda n, s: gen_bot_linear(n, steps, s, **linear_kwargs),
        "curvy": lambda n, s: gen_bot_curvy(n, steps, s),
        "stopgo": lambda n, s: gen_bot_stopgo(n, steps, s),
        "teleport": lambda n, s: gen_bot_teleport(n, steps, s),
        "linefollow": lambda n, s: gen_bot_linefollow(n, steps, s),
    }

    if mode in generators:
        return wrap(generators[mode](num, seed), mode)

    weights = np.array([0.10, 0.15, 0.15, 0.10, 0.50])
    modes = ["linear", "curvy", "stopgo", "teleport", "linefollow"]
    counts = rng.multinomial(num, weights / weights.sum())

    bots = []
    for i, (m, n) in enumerate(zip(modes, counts)):
        if n > 0:
            bots += wrap(generators[m](int(n), seed + i + 1), m)

    rng.shuffle(bots)
    return bots


# =============================================================================
# Feature Extraction
# =============================================================================

def extract_X(samples: list, *, min_points: int = 10) -> tuple[np.ndarray, list, int]:
    feats, paths, skipped = [], [], 0
    for s in samples:
        f = extract_features(
            s["points"],
            line=None,
            sanitize_time=True,
            normalize_to_line=False,
            min_points=min_points,
            same_t_eps=0.0,
        )
        if f is None:
            skipped += 1
            continue
        feats.append(f)
        paths.append(s["path"])

    X = np.vstack(feats) if feats else np.empty((0, 0), dtype=float)
    return X, paths, skipped


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Isolation Forest 드래그 봇 탐지 모델 학습",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--data-dir", default=None, help="학습 데이터 디렉토리 (우선순위: arg > $PHASE_A_DATA_DIR)")
    parser.add_argument("--output-root", default=None, help="모델 출력 루트(우선순위: arg > $PHASE_A_MODEL_DIR)")
    parser.add_argument("--experiment", default=None, help="MLflow experiment (우선순위: arg > $PHASE_A_EXPERIMENT_NAME)")
    parser.add_argument("--run-name", default=None, help="MLflow run 이름")
    parser.add_argument("--recursive", action="store_true", help="하위 폴더까지 스캔")

    parser.add_argument("--dataset_version", default=None, help="데이터셋 버전 태그(우선순위: arg > $PHASE_A_DATASET_VERSION)")

    parser.add_argument("--human-calib-ratio", type=float, default=0.10)
    parser.add_argument("--human-test-ratio", type=float, default=0.20)
    parser.add_argument("--target-human-pass", type=float, default=0.99)
    parser.add_argument("--min-points", type=int, default=10)

    parser.add_argument("--bot-num", type=int, default=3000)
    parser.add_argument("--bot-mode", type=str, default="hard")
    parser.add_argument("--bot-steps", type=int, default=60)

    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()

    # ---- env ----
    tracking_uri = _to_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", ""))
    experiment_name = args.experiment or os.environ.get("PHASE_A_EXPERIMENT_NAME") or "phase-a-isolation-forest"
    run_name = args.run_name or f"phaseA_iforest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    dataset_version = args.dataset_version or os.environ.get("PHASE_A_DATASET_VERSION") or "dev"

    save_diag = (os.environ.get("PHASE_A_SAVE_DIAGNOSTICS", "0").strip() == "1")
    keep_backups = int(os.environ.get("PHASE_A_KEEP_BACKUPS", "0").strip() or "0")

    script_dir = Path(__file__).resolve().parent
    data_dir = Path(
        args.data_dir
        or os.environ.get("PHASE_A_DATA_DIR")
        or (script_dir / "data" if (script_dir / "data").exists() else script_dir / "data_collected")
    ).expanduser().resolve()

    output_root = Path(
        args.output_root
        or os.environ.get("PHASE_A_MODEL_DIR")
        or (script_dir / "models" / "phase_a")
    ).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # 고정 파일명(운영 단순화)
    required_files = [
        "scaler_oneclass.pkl",
        "model_oneclass.pkl",
        "threshold_oneclass.json",
    ]
    optional_files = [
        "false_rejects_human_test.json",
        "score_hist.png",
    ]

    print("\n" + "=" * 60)
    print("Phase A - Isolation Forest Training")
    print("=" * 60)
    print(f"[data_dir]         {data_dir}")
    print(f"[model_output_root]{output_root}")
    print(f"[mlflow]           {tracking_uri}")
    print(f"[experiment]       {experiment_name}")
    print(f"[dataset_version]  {dataset_version}")
    print(f"[save_diagnostics] {save_diag}")
    print(f"[keep_backups]     {keep_backups}")
    print()

    # ---- mlflow ----
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("phase", "A")
        mlflow.set_tag("model_name", "isolation_forest")
        mlflow.set_tag("dataset_version", dataset_version)
        mlflow.set_tag("bot_mode", args.bot_mode)

        # 1) load
        human_all = load_json_samples(str(data_dir), recursive=args.recursive)
        if not human_all:
            raise RuntimeError(f"데이터가 없습니다: {data_dir}")

        human_train, human_calib, human_test = split_three(
            human_all, args.human_calib_ratio, args.human_test_ratio, args.random_state
        )

        # 2) features
        X_train, train_paths, _ = extract_X(human_train, min_points=args.min_points)
        X_calib, calib_paths, _ = extract_X(human_calib, min_points=args.min_points)
        X_test, test_paths, _ = extract_X(human_test, min_points=args.min_points)

        if X_train.size == 0:
            raise RuntimeError("유효한 학습 샘플이 없습니다")

        # 3) bots
        bot_items = generate_bots(args.bot_num, args.bot_steps, args.random_state, args.bot_mode)
        bot_samples = [{"path": f"synthetic:{it['mode']}:{i}", "points": it["points"]} for i, it in enumerate(bot_items)]
        X_bot, bot_paths, _ = extract_X(bot_samples, min_points=args.min_points)

        # 4) scale
        scaler = RobustScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_calib_s = scaler.transform(X_calib) if X_calib.size else np.empty((0, X_train.shape[1]))
        X_test_s = scaler.transform(X_test) if X_test.size else np.empty((0, X_train.shape[1]))
        X_bot_s = scaler.transform(X_bot) if X_bot.size else np.empty((0, X_train.shape[1]))

        # 5) fit
        model = IsolationForest(
            n_estimators=args.n_estimators,
            contamination="auto",
            random_state=args.random_state,
            n_jobs=-1,
            verbose=1
        )
        model.fit(X_train_s)

        # 6) scores
        s_train = model.score_samples(X_train_s)
        s_calib = model.score_samples(X_calib_s) if X_calib_s.size else np.array([])
        s_test = model.score_samples(X_test_s) if X_test_s.size else np.array([])
        s_bot = model.score_samples(X_bot_s) if X_bot_s.size else np.array([])

        # 7) threshold
        threshold = threshold_for_target_pass(s_calib if s_calib.size else s_train, args.target_human_pass)

        # 8) eval
        train_pass = float((s_train > threshold).mean()) if s_train.size else 0.0
        calib_pass = float((s_calib > threshold).mean()) if s_calib.size else 0.0
        test_pass = float((s_test > threshold).mean()) if s_test.size else 0.0
        bot_block = float((s_bot <= threshold).mean()) if s_bot.size else 0.0

        bot_block_by_mode = {}
        if s_bot.size:
            modes = np.array([p.split(":")[1] if ":" in p else "unknown" for p in bot_paths])
            for m in sorted(set(modes)):
                mask = modes == m
                if mask.any():
                    bot_block_by_mode[m] = float((s_bot[mask] <= threshold).mean())

        auc = 0.0
        if s_test.size and s_bot.size:
            y_true = np.concatenate([np.ones_like(s_test), np.zeros_like(s_bot)])
            y_score = np.concatenate([s_test, s_bot])
            auc = float(roc_auc_score(y_true, y_score))

        false_rejects = [{"path": p, "score": float(sc)} for p, sc in zip(test_paths, s_test) if sc <= threshold]

        # ---- 저장(최신만 유지) ----
        # (옵션) 백업
        _maybe_backup(output_root, keep_backups, required_files + (optional_files if save_diag else []))

        tmp_dir = output_root / f".tmp_{run_name}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # 필수 3종
        joblib.dump(scaler, tmp_dir / "scaler_oneclass.pkl")
        joblib.dump(model, tmp_dir / "model_oneclass.pkl")
        (tmp_dir / "threshold_oneclass.json").write_text(
            json.dumps({"threshold": float(threshold)}, indent=2),
            encoding="utf-8"
        )

        # 선택 진단
        if save_diag:
            (tmp_dir / "false_rejects_human_test.json").write_text(
                json.dumps(false_rejects, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            save_score_histogram(tmp_dir / "score_hist.png", s_train, s_calib, s_test, s_bot, float(threshold))

        # atomic swap: tmp -> output_root
        for fn in required_files:
            _atomic_replace(tmp_dir / fn, output_root / fn)

        if save_diag:
            for fn in optional_files:
                _atomic_replace(tmp_dir / fn, output_root / fn)

        # tmp dir cleanup
        shutil.rmtree(tmp_dir, ignore_errors=True)

        # ---- mlflow log ----
        mlflow.log_params({
            "data_dir": str(data_dir),
            "model_output_root": str(output_root),
            "dataset_version": dataset_version,
            "n_estimators": args.n_estimators,
            "random_state": args.random_state,
            "target_human_pass": args.target_human_pass,
            "bot_num": args.bot_num,
            "bot_mode": args.bot_mode,
            "bot_steps": args.bot_steps,
            "min_points": args.min_points,
            "human_calib_ratio": args.human_calib_ratio,
            "human_test_ratio": args.human_test_ratio,
            "save_diagnostics": int(save_diag),
        })

        mlflow.log_metrics({
            "threshold": float(threshold),
            "human_train_pass": train_pass,
            "human_calib_pass": calib_pass,
            "human_test_pass": test_pass,
            "bot_block_rate": bot_block,
            "auc": auc,
            "n_features": int(X_train.shape[1]),
            "n_train": int(X_train.shape[0]),
            "n_calib": int(X_calib.shape[0]) if X_calib.size else 0,
            "n_test": int(X_test.shape[0]) if X_test.size else 0,
            "n_bot": int(X_bot.shape[0]) if X_bot.size else 0,
        })

        for m, rate in bot_block_by_mode.items():
            mlflow.log_metric(f"bot_block_{m}", float(rate))

        # 운영 파일도 artifact로 남겨두면 "서버에서 파일 날려먹어도" 복구가 쉬움
        for fn in required_files:
            mlflow.log_artifact(str(output_root / fn))
        if save_diag:
            for fn in optional_files:
                mlflow.log_artifact(str(output_root / fn))

        _log_model_compat(model)

        print("\n" + "=" * 60)
        print("RESULT")
        print("=" * 60)
        print(f"Saved latest artifacts to: {output_root}")
        print(f"  - scaler_oneclass.pkl")
        print(f"  - model_oneclass.pkl")
        print(f"  - threshold_oneclass.json (threshold={threshold:.6f})")
        if save_diag:
            print("  - false_rejects_human_test.json / score_hist.png")
        print()


if __name__ == "__main__":
    main()