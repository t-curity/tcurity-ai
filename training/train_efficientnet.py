#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
[train_efficientnet.py - EfficientNet-B0 모델 학습 스크립트]

권장 환경변수:
  MLFLOW_TRACKING_URI=http://61.109.238.4:5000
  MLFLOW_EXPERIMENT_NAME=captcha-phase-b-image
  IMAGE_DATA_ROOT=/data/images
  MODEL_OUTPUT_ROOT=/models
  DATASET_VERSION=v001 (선택)

사용 예:
  source .env/mlflow.env
  source .env/phase_b.env

  python train_efficientnet.py
  (or python train_efficientnet.py --batch_size 128 --learning_rate 0.0006 --epochs 50 --patience 5)

"""

import os
import time
import argparse
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import Counter

import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature

from captcha_dataset import CAPTCHADataset



# ----------------------------
# MLflow compat logger
# ----------------------------
def _log_pytorch_model_compat(model, artifact_name: str, signature=None):
    """MLflow 버전 차이를 흡수하면서 PyTorch 모델 로깅"""
    try:
        # 일부 버전: name= 지원
        mlflow.pytorch.log_model(model, name=artifact_name, signature=signature)
    except TypeError:
        # 보편적으로 artifact_path 지원
        mlflow.pytorch.log_model(model, artifact_path=artifact_name, signature=signature)


def _get_tracking_uri() -> str:
    """환경변수 기반 tracking uri (없으면 로컬 file:// fallback)"""
    raw = (os.environ.get("MLFLOW_TRACKING_URI") or "").strip()
    if raw:
        return raw
    # fallback: 프로젝트 루트 기준 ./mlruns
    return "file://" + str(Path("mlruns").resolve())


def main():
    parser = argparse.ArgumentParser(description="EfficientNet CAPTCHA Training Script")

    # data/output: env fallback 허용
    parser.add_argument("--data_dir", type=str, default=None,
                        help="학습 데이터(images) 경로 (예: ./images). 미지정 시 $IMAGE_DATA_ROOT/$DATA_DIR 사용")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="모델 저장 폴더. 미지정 시 $MODEL_OUTPUT_ROOT/$MODEL_OUTPUT_DIR 사용")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)

    # MLflow: env 기본 + 필요 시 override
    parser.add_argument("--experiment", type=str, default=None,
                        help="MLflow experiment 이름. 미지정 시 $MLFLOW_EXPERIMENT_NAME 사용")
    parser.add_argument("--run_name", type=str, default=None,
                        help="MLflow run 이름. 미지정 시 자동 생성")
    parser.add_argument("--dataset_version", type=str, default=None,
                        help="데이터셋 버전 태그 (예: v001). 미지정 시 $DATASET_VERSION 사용")

    args = parser.parse_args()

    # ------------------------
    # data/output resolve
    # ------------------------
    data_dir = args.data_dir or os.environ.get("IMAGE_DATA_ROOT") or os.environ.get("DATA_DIR")

    base_output_dir = (
        args.output_dir
        or os.environ.get("MODEL_OUTPUT_ROOT")
        or os.environ.get("MODEL_OUTPUT_DIR")
        or "./models"
    )

    if not data_dir:
        raise ValueError("data_dir이 필요합니다. --data_dir 또는 $IMAGE_DATA_ROOT/$DATA_DIR 를 설정하세요.")

    data_dir = str(Path(data_dir).expanduser().resolve())

    # 🔽 Phase B 전용 디렉토리
    output_dir = Path(base_output_dir).expanduser().resolve() / "phase_b"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_save_path = output_dir / "best_model.pth"

    # ------------------------
    # MLflow 설정 (env 기반)
    # ------------------------
    tracking_uri = _get_tracking_uri()
    experiment_name = args.experiment or os.environ.get("MLFLOW_EXPERIMENT_NAME") or "captcha-phase-b"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    dataset_version = args.dataset_version or os.environ.get("DATASET_VERSION") or "dev"

    run_name = args.run_name or f"effnetb0_lr{args.learning_rate}_bs{args.batch_size}_{datetime.now().strftime('%Y%m%d_%H%M')}"

    # ------------------------
    # 전처리
    # ------------------------
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.05),
        
        transforms.RandomAffine(
            degrees=10,
            translate=(0.05, 0.05),
            scale=(0.9, 1.1)
        ),

        transforms.RandomPerspective(
            distortion_scale=0.15,
            p=0.25
        ),

        transforms.ColorJitter(
        brightness=0.2,
            contrast=0.2,
            saturation=0.15,
            hue=0.05
        ),

        transforms.RandomApply([
            transforms.GaussianBlur(3, (0.1, 1.0))
        ], p=0.3),

        transforms.RandomErasing(
            p=0.25,
            scale=(0.02, 0.1),
            ratio=(0.3, 3.3)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ------------------------
    # 데이터 로드
    # ------------------------
    print("=" * 60)
    print("데이터셋 로드 중...")
    print("=" * 60)

    full_dataset = CAPTCHADataset(data_dir, transform=None)

    train_ratio = 0.8
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size

    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])
    train_subset.dataset.transform = train_transform
    val_subset.dataset.transform = val_transform

    train_loader = DataLoader(
    train_subset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    # persistent_workers=True
    )

    val_loader = DataLoader(
    val_subset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    # persistent_workers=True
    )

    print(f"총 데이터: {total_size}개")
    print(f"학습 데이터: {train_size}개")
    print(f"검증 데이터: {val_size}개\n")

    # ------------------------
    # 모델
    # ------------------------
    print("=" * 60)
    print("모델 설정 중...")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"사용 장치: {device}\n")

    model = models.efficientnet_b0(weights="IMAGENET1K_V1")

    NUM_CLASSES = len(set(full_dataset.labels))
    print(f"감지된 실제 클래스 수: {NUM_CLASSES}")

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    model = model.to(device)

    # features 동결 + 일부만 fine-tuning
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.features[-2:].parameters():
        p.requires_grad = True

    print(f"✅ EfficientNet-B0 로드 완료 ({NUM_CLASSES}개 클래스)")
    print(f"   Fine-tuning: features[-2:] + classifier\n")
    
    # 1. 자동 계산
    class_counts_dict = Counter(full_dataset.labels)
    class_counts = torch.tensor(
        [class_counts_dict[i] for i in range(NUM_CLASSES)],
        dtype=torch.float
    )
    # 2. 가중치 계산 (ChatGPT 방법)
    num_classes = len(class_counts)
    total_samples = class_counts.sum()
    class_weights = total_samples / (num_classes * class_counts)
    class_weights = torch.clamp(class_weights, max=3.0)  # 과도한 영향 방지

    print("📊 클래스 가중치:")
    for i, w in enumerate(class_weights):
        print(f"  Class {i}: {w:.2f}x")

    # 3. 손실함수 (가중치 적용)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # 기존 방식 - 가중치 균등화
    # criterion = nn.CrossEntropyLoss(label_smoothing=0.1) 

    optimizer = torch.optim.AdamW([
        {"params": model.features[-2:].parameters(), "lr": args.learning_rate * 0.05, "weight_decay": 1e-5},
        {"params": model.classifier.parameters(), "lr": args.learning_rate, "weight_decay": 1e-4},
    ])

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # signature
    example_input = torch.randn(1, 3, 128, 128).to(device)
    with torch.no_grad():
        example_output = model(example_input)
    signature = infer_signature(example_input.cpu().numpy(), example_output.cpu().numpy())

    # ------------------------
    # 학습 + MLflow
    # ------------------------
    print("=" * 60)
    print("학습 시작")
    print("=" * 60 + "\n")

    start_total = time.time()
    best_accuracy = 0.0
    patience_counter = 0

    with mlflow.start_run(run_name=run_name):
        # tags (Phase/모델 확장 대비)
        mlflow.set_tag("phase", "B")
        mlflow.set_tag("model_family", "cnn")
        mlflow.set_tag("model_name", "efficientnet_b0")
        mlflow.set_tag("dataset_version", dataset_version)

        # params
        mlflow.log_param("run_name", run_name)
        mlflow.log_param("data_dir", data_dir)
        mlflow.log_param("output_dir", output_dir)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("patience", args.patience)
        mlflow.log_param("train_samples", train_size)
        mlflow.log_param("val_samples", val_size)
        mlflow.log_param("num_classes", NUM_CLASSES)

        for epoch in range(args.epochs):
            epoch_start = time.time()

            # Train
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, (imgs, labels) in enumerate(train_loader):
                imgs, labels = imgs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

                if (batch_idx + 1) % 10 == 0:
                    print(f"  [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

            avg_train_loss = train_loss / max(1, len(train_loader))
            train_accuracy = train_correct / max(1, train_total)

            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            avg_val_loss = val_loss / max(1, len(val_loader))
            val_accuracy = val_correct / max(1, val_total)

            scheduler.step()
            epoch_time = time.time() - epoch_start

            print(f"[Epoch {epoch+1}/{args.epochs}]")
            print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2%}")
            print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_accuracy:.2%}")
            print(f"  → Epoch Time: {epoch_time:.2f} sec")

            gap = train_accuracy - val_accuracy
            print(f"  {'⚠️ 과적합 경고' if gap >= 0.10 else '✓ 정상'} (차이: {gap:.2%})")

            # metrics
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("train_acc", train_accuracy, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_accuracy, step=epoch)
            mlflow.log_metric("lr_head", optimizer.param_groups[1]["lr"], step=epoch)
            mlflow.log_metric("lr_backbone", optimizer.param_groups[0]["lr"], step=epoch)

            # Early Stopping
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                patience_counter = 0

                torch.save(model.state_dict(), model_save_path)
                print(f"  ✅ 최고 모델 저장! (Val_Acc: {val_accuracy:.2%})")

                _log_pytorch_model_compat(model, artifact_name="best_model", signature=signature)
            else:
                patience_counter += 1
                print(f"  ⏳ Patience: {patience_counter}/{args.patience}")
                if patience_counter >= args.patience:
                    print("\n🛑 Early Stopping 발동!")
                    break

            print()

        mlflow.log_metric("best_val_acc", float(best_accuracy))
        mlflow.log_artifact(model_save_path)

    end_total = time.time()
    print("=" * 60)
    print("학습 완료!")
    print(f"최고 검증 정확도: {best_accuracy:.2%}")
    print(f"전체 학습 시간: {end_total - start_total:.2f} sec")
    print(f"모델 저장 경로: {model_save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
