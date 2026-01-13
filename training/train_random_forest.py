#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_random_forest.py

Phase B: 이미지 분류 CAPTCHA - Random Forest 행동 패턴 분류기 학습

3x3 그리드에서 정답 클래스 이미지 4개를 찾아 4개 슬롯에 순서대로 배치하는
행동 패턴을 분석하여 사람/봇을 구별하는 모델

특징(Features) 19개:
- 시간 관련 (6개): 전체시간, 첫동작시간, 드래그시간, 드롭간격 등
- 경로 관련 (8개): 직선도, 속도, 경로길이, 방향전환 등
- 정확도 관련 (4개): 오답시도, 되돌리기, 드래그횟수, 배치정확도
- 결과 (1개): 최종정확도

권장 환경변수:
  MLFLOW_TRACKING_URI=http://61.109.238.4:5000
  MLFLOW_EXPERIMENT_NAME=captcha-phase-b
  DATA_COLLECTED_DIR=/home/ubuntu/tcurity-ai/data/phase_b_trainset
  MODEL_OUTPUT_ROOT=/home/ubuntu/tcurity-ai/models/phase_b
  DATASET_VERSION=v001
  SAVE_DIAGNOSTICS=0|1
  KEEP_BACKUPS=0|1|2|3...
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Feature Definition
# =============================================================================

FEATURE_NAMES = [
    # 시간 특징 (6개)
    'total_time',                # 전체 완료 시간 (ms)
    'first_action_time',         # 첫 동작까지 시간 (ms)
    'avg_drag_duration',         # 평균 드래그 지속 시간 (ms)
    'drag_duration_std',         # 드래그 지속 시간 표준편차
    'avg_time_between_drops',    # 평균 드롭 간 시간 간격 (ms)
    'time_interval_std',         # 시간 간격 표준편차
    
    # 경로 특징 (8개)
    'avg_straightness',          # 평균 경로 직선도 (1=완벽한 직선)
    'straightness_std',          # 직선도 표준편차
    'avg_speed',                 # 평균 드래그 속도 (px/ms)
    'speed_std',                 # 속도 표준편차
    'avg_speed_variation',       # 평균 속도 변화 계수
    'avg_path_length',           # 평균 경로 길이 (px)
    'path_length_std',           # 경로 길이 표준편차
    'avg_direction_changes',     # 평균 방향 전환 횟수
    
    # 정확도/행동 특징 (4개)
    'wrong_attempts',            # 잘못된 이미지 선택 횟수
    'backtrack_count',           # 되돌리기 횟수
    'total_drags',               # 총 드래그 횟수
    'correct_placement_ratio',   # 정확한 슬롯 배치 비율
    
    # 결과 (1개)
    'final_accuracy',            # 최종 정답률 (0~1)
]


# =============================================================================
# Utilities
# =============================================================================

def _to_tracking_uri(raw: str) -> str:
    """MLflow tracking URI 정규화"""
    s = (raw or "").strip()
    if not s:
        return "file://" + str(Path("mlruns").resolve())
    if s.lower().startswith(("file:", "sqlite:", "http://", "https://", "databricks")):
        return s
    return "file://" + Path(s).expanduser().resolve().as_posix()


def _atomic_replace(src: Path, dst: Path) -> None:
    """Atomic file replacement"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.replace(src, dst)


def _maybe_backup(dst_dir: Path, keep: int, files: list[str]) -> None:
    """선택적 백업 생성"""
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
        try:
            bdir.rmdir()
        except Exception:
            pass
        return

    # 오래된 백업 정리
    all_b = sorted([d for d in backups_dir.iterdir() if d.is_dir()], reverse=True)
    for d in all_b[keep:]:
        shutil.rmtree(d, ignore_errors=True)


def _log_model_compat(model):
    """MLflow 버전 호환 모델 로깅"""
    try:
        mlflow.sklearn.log_model(model, artifact_path="rf_model")
    except TypeError:
        mlflow.sklearn.log_model(model, "rf_model")


# =============================================================================
# Data Loading
# =============================================================================

def load_collected_data(data_dir: str, recursive: bool = True) -> tuple[list, list]:
    """수집된 JSON 데이터 로드"""
    data_path = Path(data_dir).expanduser().resolve()
    
    if not data_path.exists():
        print(f"[WARN] 데이터 폴더 없음: {data_path}")
        return [], []
    
    pattern = "**/*.json" if recursive else "*.json"
    json_files = sorted(data_path.glob(pattern))
    
    if not json_files:
        print(f"[WARN] JSON 파일 없음: {data_path}")
        return [], []
    
    print(f"[INFO] {len(json_files)}개 JSON 파일 발견")
    
    human_data = []
    bot_data = []
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
        
        # phase_b_behavior_*.json 형식 (human/bot 키)
        if 'human' in obj:
            for item in obj['human']:
                if 'features' in item:
                    human_data.append(item['features'])
        
        if 'bot' in obj:
            for item in obj['bot']:
                if 'features' in item:
                    bot_data.append(item['features'])
        
        # 개별 파일 형식 (label 키)
        if 'label' in obj and 'features' in obj:
            if obj['label'] == 'human':
                human_data.append(obj['features'])
            elif obj['label'] == 'bot':
                bot_data.append(obj['features'])
    
    if bad_files:
        print(f"[WARN] {bad_files}개 파일 스킵됨 (파싱 실패)")
    
    print(f"[INFO] Human: {len(human_data)}개, Bot: {len(bot_data)}개 로드됨")
    
    return human_data, bot_data


# =============================================================================
# Synthetic Bot Generators
# =============================================================================

def generate_bot_data(num: int, seed: int, mode: str = "mixed") -> list[dict]:
    """
    다양한 봇 행동 패턴 시뮬레이션
    
    봇 타입:
    0: fast     - 매우 빠름, 높은 직선도, 완벽한 정확도
    1: sequential - 그리드 순서대로 선택, 일정한 패턴
    2: precise  - 느리지만 완벽, 매우 일정한 간격
    3: disguised - 사람 흉내, 약간의 변동과 실수
    """
    rng = np.random.default_rng(seed)
    
    mode = (mode or "mixed").lower()
    bot_types = {
        "fast": [0],
        "sequential": [1],
        "precise": [2],
        "disguised": [3],
        "mixed": [0, 1, 2, 3],
        "hard": [0, 1, 2, 3],  # alias
    }
    
    available_types = bot_types.get(mode, [0, 1, 2, 3])
    
    bots = []
    type_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
    for i in range(num):
        bt = available_types[i % len(available_types)]
        type_counts[bt] += 1
        
        if bt == 0:  # fast bot
            bot = {
                'total_time': rng.uniform(800, 2000),
                'first_action_time': rng.uniform(30, 150),
                'avg_drag_duration': rng.uniform(50, 150),
                'drag_duration_std': rng.uniform(5, 20),
                'avg_time_between_drops': rng.uniform(100, 300),
                'time_interval_std': rng.uniform(10, 40),
                
                'avg_straightness': rng.uniform(0.94, 0.99),
                'straightness_std': rng.uniform(0.005, 0.02),
                'avg_speed': rng.uniform(1.5, 3.0),
                'speed_std': rng.uniform(0.1, 0.3),
                'avg_speed_variation': rng.uniform(0.02, 0.08),
                'avg_path_length': rng.uniform(180, 280),
                'path_length_std': rng.uniform(10, 30),
                'avg_direction_changes': rng.uniform(0, 2),
                
                'wrong_attempts': 0,
                'backtrack_count': 0,
                'total_drags': 4,
                'correct_placement_ratio': 1.0,
                'final_accuracy': 1.0,
                '_bot_type': 'fast',
            }
            
        elif bt == 1:  # sequential bot
            bot = {
                'total_time': rng.uniform(2000, 4000),
                'first_action_time': rng.uniform(100, 400),
                'avg_drag_duration': rng.uniform(200, 400),
                'drag_duration_std': rng.uniform(15, 40),
                'avg_time_between_drops': rng.uniform(400, 700),
                'time_interval_std': rng.uniform(20, 60),
                
                'avg_straightness': rng.uniform(0.90, 0.97),
                'straightness_std': rng.uniform(0.01, 0.03),
                'avg_speed': rng.uniform(0.8, 1.5),
                'speed_std': rng.uniform(0.1, 0.25),
                'avg_speed_variation': rng.uniform(0.05, 0.12),
                'avg_path_length': rng.uniform(220, 320),
                'path_length_std': rng.uniform(15, 40),
                'avg_direction_changes': rng.uniform(0, 3),
                
                'wrong_attempts': 0,
                'backtrack_count': 0,
                'total_drags': 4,
                'correct_placement_ratio': 1.0,
                'final_accuracy': 1.0,
                '_bot_type': 'sequential',
            }
            
        elif bt == 2:  # precise bot
            bot = {
                'total_time': rng.uniform(3500, 6000),
                'first_action_time': rng.uniform(200, 600),
                'avg_drag_duration': rng.uniform(350, 550),
                'drag_duration_std': rng.uniform(10, 30),
                'avg_time_between_drops': rng.uniform(700, 1200),
                'time_interval_std': rng.uniform(15, 50),
                
                'avg_straightness': rng.uniform(0.88, 0.96),
                'straightness_std': rng.uniform(0.01, 0.03),
                'avg_speed': rng.uniform(0.5, 1.0),
                'speed_std': rng.uniform(0.05, 0.15),
                'avg_speed_variation': rng.uniform(0.03, 0.10),
                'avg_path_length': rng.uniform(250, 350),
                'path_length_std': rng.uniform(10, 35),
                'avg_direction_changes': rng.uniform(0, 3),
                
                'wrong_attempts': 0,
                'backtrack_count': 0,
                'total_drags': 4,
                'correct_placement_ratio': 1.0,
                'final_accuracy': 1.0,
                '_bot_type': 'precise',
            }
            
        else:  # disguised bot
            bot = {
                'total_time': rng.uniform(6000, 12000),
                'first_action_time': rng.uniform(500, 1500),
                'avg_drag_duration': rng.uniform(400, 800),
                'drag_duration_std': rng.uniform(80, 200),
                'avg_time_between_drops': rng.uniform(1200, 2500),
                'time_interval_std': rng.uniform(150, 400),
                
                'avg_straightness': rng.uniform(0.70, 0.88),
                'straightness_std': rng.uniform(0.04, 0.10),
                'avg_speed': rng.uniform(0.3, 0.7),
                'speed_std': rng.uniform(0.1, 0.25),
                'avg_speed_variation': rng.uniform(0.12, 0.30),
                'avg_path_length': rng.uniform(280, 420),
                'path_length_std': rng.uniform(40, 100),
                'avg_direction_changes': rng.uniform(2, 8),
                
                'wrong_attempts': int(rng.integers(0, 2)),
                'backtrack_count': int(rng.integers(0, 1)),
                'total_drags': int(rng.integers(4, 6)),
                'correct_placement_ratio': float(rng.uniform(0.7, 1.0)),
                'final_accuracy': float(rng.uniform(0.75, 1.0)),
                '_bot_type': 'disguised',
            }
        
        bots.append(bot)
    
    print(f"[INFO] 봇 생성 완료: fast={type_counts[0]}, sequential={type_counts[1]}, "
          f"precise={type_counts[2]}, disguised={type_counts[3]}")
    
    return bots


def generate_human_data(num: int, seed: int) -> list[dict]:
    """사람 행동 시뮬레이션 (테스트/보조용)"""
    rng = np.random.default_rng(seed)
    
    humans = []
    for _ in range(num):
        human = {
            'total_time': rng.uniform(5000, 20000),
            'first_action_time': rng.uniform(300, 2500),
            'avg_drag_duration': rng.uniform(300, 1000),
            'drag_duration_std': rng.uniform(100, 400),
            'avg_time_between_drops': rng.uniform(800, 3000),
            'time_interval_std': rng.uniform(250, 800),
            
            'avg_straightness': rng.uniform(0.45, 0.82),
            'straightness_std': rng.uniform(0.08, 0.22),
            'avg_speed': rng.uniform(0.2, 0.6),
            'speed_std': rng.uniform(0.1, 0.3),
            'avg_speed_variation': rng.uniform(0.20, 0.55),
            'avg_path_length': rng.uniform(320, 550),
            'path_length_std': rng.uniform(60, 180),
            'avg_direction_changes': rng.uniform(4, 15),
            
            'wrong_attempts': int(rng.integers(0, 4)),
            'backtrack_count': int(rng.integers(0, 3)),
            'total_drags': int(rng.integers(4, 8)),
            'correct_placement_ratio': float(rng.uniform(0.3, 1.0)),
            'final_accuracy': float(rng.uniform(0.5, 1.0)),
        }
        humans.append(human)
    
    return humans


# =============================================================================
# Feature Extraction
# =============================================================================

def extract_features(data_list: list[dict]) -> np.ndarray:
    """행동 데이터 → 특징 벡터"""
    X = []
    for d in data_list:
        features = [float(d.get(name, 0)) for name in FEATURE_NAMES]
        X.append(features)
    return np.array(X) if X else np.empty((0, len(FEATURE_NAMES)))


# =============================================================================
# Visualization
# =============================================================================

def save_score_distribution(path: Path, human_scores: np.ndarray, bot_scores: np.ndarray):
    """Human/Bot 확률 분포 시각화"""
    plt.figure(figsize=(10, 6))
    
    if human_scores.size:
        plt.hist(human_scores, bins=50, alpha=0.6, label='Human', color='green')
    if bot_scores.size:
        plt.hist(bot_scores, bins=50, alpha=0.6, label='Bot', color='red')
    
    plt.axvline(0.5, color='black', linewidth=2, linestyle='--', label='Threshold (0.5)')
    plt.xlabel('Human Probability')
    plt.ylabel('Count')
    plt.title('Phase B: Human Probability Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_roc_curve(path: Path, y_true: np.ndarray, y_prob: np.ndarray, auc_score: float):
    """ROC 곡선 시각화"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='blue', linewidth=2, label=f'ROC (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Phase B: ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_feature_importance(path: Path, importances: list[tuple[str, float]]):
    """Feature Importance 시각화"""
    names = [x[0] for x in importances]
    values = [x[1] for x in importances]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(names)), values, color='steelblue')
    plt.yticks(range(len(names)), names)
    plt.xlabel('Importance')
    plt.title('Phase B: Feature Importance')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# =============================================================================
# Main Training
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase B - Random Forest 행동 패턴 분류기 학습",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 데이터/출력 경로
    parser.add_argument("--data-dir", default=None,
                        help="학습 데이터 디렉토리 (우선순위: arg > $DATA_COLLECTED_DIR)")
    parser.add_argument("--output-root", default=None,
                        help="모델 출력 루트 (우선순위: arg > $MODEL_OUTPUT_ROOT)")
    parser.add_argument("--recursive", action="store_true",
                        help="하위 폴더까지 스캔")
    
    # MLflow
    parser.add_argument("--experiment", default=None,
                        help="MLflow experiment (우선순위: arg > $MLFLOW_EXPERIMENT_NAME)")
    parser.add_argument("--run-name", default=None,
                        help="MLflow run 이름")
    parser.add_argument("--dataset-version", default=None,
                        help="데이터셋 버전 태그 (우선순위: arg > $DATASET_VERSION)")
    
    # 데이터 분할
    parser.add_argument("--test-ratio", type=float, default=0.20,
                        help="테스트 데이터 비율")
    parser.add_argument("--random-state", type=int, default=42,
                        help="랜덤 시드")
    
    # 봇 생성
    parser.add_argument("--bot-num", type=int, default=500,
                        help="생성할 봇 데이터 수")
    parser.add_argument("--bot-mode", type=str, default="mixed",
                        help="봇 모드: fast, sequential, precise, disguised, mixed")
    
    # 사람 시뮬레이션 (실제 데이터 없을 때)
    parser.add_argument("--human-sim-num", type=int, default=0,
                        help="시뮬레이션 사람 데이터 수 (0=실제 데이터만 사용)")
    
    # 모델 파라미터
    parser.add_argument("--n-estimators", type=int, default=200,
                        help="Random Forest 트리 수")
    parser.add_argument("--max-depth", type=int, default=15,
                        help="최대 트리 깊이")
    parser.add_argument("--min-samples-split", type=int, default=5,
                        help="분할 최소 샘플 수")
    parser.add_argument("--min-samples-leaf", type=int, default=2,
                        help="리프 최소 샘플 수")
    
    # 교차검증
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="교차검증 폴드 수")
    
    args = parser.parse_args()
    
    # =========================================================================
    # Environment Setup
    # =========================================================================
    
    tracking_uri = _to_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", ""))
    experiment_name = args.experiment or os.environ.get("MLFLOW_EXPERIMENT_NAME") or "captcha-phase-b"
    run_name = args.run_name or f"phaseB_rf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dataset_version = args.dataset_version or os.environ.get("DATASET_VERSION") or "dev"
    
    save_diag = os.environ.get("SAVE_DIAGNOSTICS", "0").strip() == "1"
    keep_backups = int(os.environ.get("KEEP_BACKUPS", "0").strip() or "0")
    
    script_dir = Path(__file__).resolve().parent
    data_dir = Path(
        args.data_dir
        or os.environ.get("DATA_COLLECTED_DIR")
        or (script_dir / "data" if (script_dir / "data").exists() else script_dir / "data_collected")
    ).expanduser().resolve()
    
    output_root = Path(
        args.output_root
        or os.environ.get("MODEL_OUTPUT_ROOT")
        or (script_dir / "models" / "phase_b")
    ).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    
    # 고정 파일명
    required_files = [
        "model_rf.pkl",
        "scaler_rf.pkl",
        "config_rf.json",
    ]
    optional_files = [
        "score_distribution.png",
        "roc_curve.png",
        "feature_importance.png",
        "classification_report.json",
    ]
    
    print("\n" + "=" * 70)
    print("Phase B - Random Forest Training")
    print("=" * 70)
    print(f"[data_dir]         {data_dir}")
    print(f"[output_root]      {output_root}")
    print(f"[mlflow]           {tracking_uri}")
    print(f"[experiment]       {experiment_name}")
    print(f"[dataset_version]  {dataset_version}")
    print(f"[save_diagnostics] {save_diag}")
    print(f"[keep_backups]     {keep_backups}")
    print()
    
    # =========================================================================
    # MLflow Setup
    # =========================================================================
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("phase", "B")
        mlflow.set_tag("model_name", "random_forest")
        mlflow.set_tag("dataset_version", dataset_version)
        mlflow.set_tag("bot_mode", args.bot_mode)
        
        # =====================================================================
        # Data Loading
        # =====================================================================
        
        print("[1/6] 데이터 로드...")
        human_data, bot_data = load_collected_data(str(data_dir), recursive=args.recursive)
        
        # 사람 시뮬레이션 추가 (필요시)
        if args.human_sim_num > 0:
            print(f"[INFO] 사람 시뮬레이션 {args.human_sim_num}개 추가")
            human_sim = generate_human_data(args.human_sim_num, args.random_state)
            human_data.extend(human_sim)
        
        # 데이터 없으면 전체 시뮬레이션
        if not human_data:
            print("[WARN] 실제 사람 데이터 없음 → 시뮬레이션 모드")
            human_data = generate_human_data(100, args.random_state)
        
        # 봇 데이터 생성
        if not bot_data or len(bot_data) < args.bot_num:
            print(f"[INFO] 봇 데이터 생성: {args.bot_num}개")
            bot_data = generate_bot_data(args.bot_num, args.random_state, args.bot_mode)
        
        print(f"[INFO] 최종 데이터: Human={len(human_data)}, Bot={len(bot_data)}")
        
        # =====================================================================
        # Feature Extraction
        # =====================================================================
        
        print("[2/6] 특징 추출...")
        X_human = extract_features(human_data)
        X_bot = extract_features(bot_data)
        
        X = np.vstack([X_human, X_bot])
        y = np.array([1] * len(human_data) + [0] * len(bot_data))  # 1=Human, 0=Bot
        
        print(f"[INFO] X shape: {X.shape}, y shape: {y.shape}")
        
        # =====================================================================
        # Preprocessing
        # =====================================================================
        
        print("[3/6] 전처리 (StandardScaler)...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # =====================================================================
        # Train/Test Split
        # =====================================================================
        
        print("[4/6] 데이터 분할...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y,
            test_size=args.test_ratio,
            random_state=args.random_state,
            stratify=y
        )
        
        print(f"[INFO] Train: {len(X_train)}, Test: {len(X_test)}")
        
        # =====================================================================
        # Model Training
        # =====================================================================
        
        print("[5/6] 모델 학습...")
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_state,
            n_jobs=-1,
            verbose=1
        )
        
        model.fit(X_train, y_train)
        print("[INFO] 학습 완료!")
        
        # =====================================================================
        # Evaluation
        # =====================================================================
        
        print("[6/6] 평가...")
        
        # Predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_prob = model.predict_proba(X_train)[:, 1]
        test_prob = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        train_auc = roc_auc_score(y_train, train_prob)
        test_auc = roc_auc_score(y_test, test_prob)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_scaled, y, cv=args.cv_folds, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, test_pred)
        tn, fp, fn_count, tp = cm.ravel()
        
        # Classification Report
        cls_report = classification_report(y_test, test_pred, target_names=['Bot', 'Human'], output_dict=True)
        
        # Feature Importance
        importances = sorted(
            zip(FEATURE_NAMES, model.feature_importances_),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Bot type별 성능 (가능한 경우)
        bot_metrics = {}
        if bot_data and '_bot_type' in bot_data[0]:
            X_bot_scaled = scaler.transform(X_bot)
            bot_types = [b.get('_bot_type', 'unknown') for b in bot_data]
            bot_pred = model.predict(X_bot_scaled)
            
            for bt in set(bot_types):
                mask = np.array([t == bt for t in bot_types])
                if mask.any():
                    block_rate = float((bot_pred[mask] == 0).mean())
                    bot_metrics[f"bot_block_{bt}"] = block_rate
        
        # =====================================================================
        # Save Artifacts
        # =====================================================================
        
        # 백업
        _maybe_backup(output_root, keep_backups, required_files + (optional_files if save_diag else []))
        
        # 임시 디렉토리
        tmp_dir = output_root / f".tmp_{run_name}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        
        # 필수 파일 저장
        joblib.dump(model, tmp_dir / "model_rf.pkl")
        joblib.dump(scaler, tmp_dir / "scaler_rf.pkl")
        
        config = {
            "feature_names": FEATURE_NAMES,
            "n_features": len(FEATURE_NAMES),
            "model_params": {
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "min_samples_split": args.min_samples_split,
                "min_samples_leaf": args.min_samples_leaf,
            },
            "threshold": 0.5,
            "created_at": datetime.now().isoformat(),
            "dataset_version": dataset_version,
        }
        (tmp_dir / "config_rf.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        
        # 선택적 진단 파일
        if save_diag:
            # Score distribution
            human_test_prob = test_prob[y_test == 1]
            bot_test_prob = test_prob[y_test == 0]
            save_score_distribution(tmp_dir / "score_distribution.png", human_test_prob, bot_test_prob)
            
            # ROC curve
            save_roc_curve(tmp_dir / "roc_curve.png", y_test, test_prob, test_auc)
            
            # Feature importance
            save_feature_importance(tmp_dir / "feature_importance.png", importances)
            
            # Classification report
            (tmp_dir / "classification_report.json").write_text(
                json.dumps(cls_report, indent=2),
                encoding="utf-8"
            )
        
        # Atomic swap
        for fn in required_files:
            _atomic_replace(tmp_dir / fn, output_root / fn)
        
        if save_diag:
            for fn in optional_files:
                src = tmp_dir / fn
                if src.exists():
                    _atomic_replace(src, output_root / fn)
        
        # Cleanup
        shutil.rmtree(tmp_dir, ignore_errors=True)
        
        # =====================================================================
        # MLflow Logging
        # =====================================================================
        
        mlflow.log_params({
            "data_dir": str(data_dir),
            "output_root": str(output_root),
            "dataset_version": dataset_version,
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_split": args.min_samples_split,
            "min_samples_leaf": args.min_samples_leaf,
            "random_state": args.random_state,
            "test_ratio": args.test_ratio,
            "cv_folds": args.cv_folds,
            "bot_num": args.bot_num,
            "bot_mode": args.bot_mode,
            "human_sim_num": args.human_sim_num,
            "save_diagnostics": int(save_diag),
        })
        
        mlflow.log_metrics({
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "train_auc": train_auc,
            "test_auc": test_auc,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "n_features": len(FEATURE_NAMES),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_human": len(human_data),
            "n_bot": len(bot_data),
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn_count),
            "true_positive": int(tp),
            "human_precision": cls_report['Human']['precision'],
            "human_recall": cls_report['Human']['recall'],
            "human_f1": cls_report['Human']['f1-score'],
            "bot_precision": cls_report['Bot']['precision'],
            "bot_recall": cls_report['Bot']['recall'],
            "bot_f1": cls_report['Bot']['f1-score'],
        })
        
        # Bot type별 메트릭
        for name, value in bot_metrics.items():
            mlflow.log_metric(name, value)
        
        # Feature importance 로깅
        for name, imp in importances[:5]:
            mlflow.log_metric(f"importance_{name}", float(imp))
        
        # Artifacts 로깅
        for fn in required_files:
            mlflow.log_artifact(str(output_root / fn))
        if save_diag:
            for fn in optional_files:
                fpath = output_root / fn
                if fpath.exists():
                    mlflow.log_artifact(str(fpath))
        
        # 모델 로깅
        _log_model_compat(model)
        
        # =====================================================================
        # Print Results
        # =====================================================================
        
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        
        print(f"\n📊 성능:")
        print(f"   Train Accuracy: {train_acc*100:.2f}%")
        print(f"   Test Accuracy:  {test_acc*100:.2f}%")
        print(f"   Train AUC:      {train_auc:.4f}")
        print(f"   Test AUC:       {test_auc:.4f}")
        print(f"   CV Mean:        {cv_mean*100:.2f}% (±{cv_std*100:.2f}%)")
        
        print(f"\n📋 Confusion Matrix:")
        print(f"              Pred:Bot  Pred:Human")
        print(f"   Real Bot:     {tn:4d}       {fp:4d}")
        print(f"   Real Human:   {fn_count:4d}       {tp:4d}")
        
        print(f"\n🔍 Top 5 Features:")
        for i, (name, imp) in enumerate(importances[:5], 1):
            bar = '█' * int(imp * 40)
            print(f"   {i}. {name:28s} {imp:.4f} ({imp*100:5.1f}%) {bar}")
        
        if bot_metrics:
            print(f"\n🤖 Bot Block Rate by Type:")
            for name, rate in bot_metrics.items():
                print(f"   {name}: {rate*100:.1f}%")
        
        print(f"\n💾 Saved to: {output_root}")
        for fn in required_files:
            print(f"   - {fn}")
        if save_diag:
            for fn in optional_files:
                if (output_root / fn).exists():
                    print(f"   - {fn}")
        
        print("\n" + "=" * 70)
        print("✅ Training Complete!")
        print("=" * 70)


if __name__ == "__main__":
    main()