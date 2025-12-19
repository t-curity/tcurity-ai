"""
app/inference/classify_images.py

[classify_images.py - 정답 데이터셋 생성 (운영/관리용)]

- 학습된 EfficientNet 모델을 사용하여
  이미지 자동 분류 및 전처리된 이미지를 저장한다. 
- 클래스별 평균 confidence 계산 및 JSON 통계 생성한다.

- 입력: data/imageset/ (라벨 없는 이미지)
- 출력: data/processed_images/[클래스]/ (128x128 리사이즈 이미지)
        data/processed_images/unknown/ (confidence 미달 이미지)
- 모델: models/phase_b/best_model.pth

"""

import os
import json
from pathlib import Path
from typing import List, Dict

import torch
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv


# =====================================================
# 환경변수 로드
# =====================================================
env_file = Path(__file__).parent.parent / ".env" / "phase_b_image.env"
if env_file.exists():
    load_dotenv(env_file)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGESET_DIR = PROJECT_ROOT / "data" / "imageset"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed_images"
MODEL_PATH = PROJECT_ROOT / "models" / "phase_b" / "best_model.pth"

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

CLASSES: List[str] = [
    "Animals", "Birds", "Building", "Devices", "Fashion",
    "Food", "Instrument", "Nature", "Sports", "Vehicle",
]


# =====================================================
# 분류기 클래스
# =====================================================
class ImageClassifier:
    def __init__(self, model_path: Path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.classes = CLASSES

        print(f"🔄 모델 로딩 중... ({self.device})")

        # 모델 생성 (학습 코드와 동일)
        self.model = efficientnet_b0(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = torch.nn.Linear(
            in_features, len(self.classes)
        )

        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

        # 예측용 전처리 (Normalize 포함)
        self.predict_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # 저장용 전처리
        self.save_transform = transforms.Resize((128, 128))

        print("✅ 모델 로드 완료\n")

    def predict(self, img: Image.Image):
        x = self.predict_transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]

        conf, idx = torch.max(probs, dim=0)
        return self.classes[idx.item()], conf.item()

    def process(
        self,
        input_dir: Path,
        output_dir: Path,
        min_confidence: float = 0.7,
    ):
        # 출력 폴더 생성
        output_dir.mkdir(exist_ok=True)
        for cls in self.classes:
            (output_dir / cls).mkdir(parents=True, exist_ok=True)
        (output_dir / "unknown").mkdir(exist_ok=True)

        # ===============================
        # 통계 초기화
        # ===============================
        stats = {
            "total": 0,
            "processed": 0,
            "low_confidence": 0,
            "failed": 0,
            "class_distribution": {c: 0 for c in self.classes},
            "class_confidence_avg": {},  # ← 추가됨
            "details": [],
        }

        # 평균 confidence 계산용 누적 변수
        conf_sum: Dict[str, float] = {c: 0.0 for c in self.classes}
        conf_count: Dict[str, int] = {c: 0 for c in self.classes}

        images = [
            p for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]

        print(f"📁 입력 이미지: {len(images)}개\n")

        for idx, img_path in enumerate(tqdm(images, desc="분류 중")):
            stats["total"] += 1

            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                stats["failed"] += 1
                continue

            label, conf = self.predict(img)
            resized = self.save_transform(img)

            # ---------- confidence 미달 ----------
            if conf < min_confidence:
                out_path = (
                    output_dir / "unknown" / f"unknown_{idx:06d}.jpg"
                )
                resized.save(out_path, format="JPEG", quality=95)

                stats["low_confidence"] += 1
                stats["details"].append({
                    "original": str(img_path),
                    "output": str(out_path),
                    "label": "unknown",
                    "confidence": round(conf, 4),
                })
                continue

            # ---------- 정상 분류 ----------
            out_path = (
                output_dir / label / f"{label.lower()}_{idx:06d}.jpg"
            )
            resized.save(out_path, format="JPEG", quality=95)

            stats["processed"] += 1
            stats["class_distribution"][label] += 1

            # confidence 누적
            conf_sum[label] += conf
            conf_count[label] += 1

            stats["details"].append({
                "original": str(img_path),
                "output": str(out_path),
                "label": label,
                "confidence": round(conf, 4),
            })

        # ===============================
        # 클래스별 평균 confidence 계산
        # ===============================
        for cls in self.classes:
            if conf_count[cls] > 0:
                stats["class_confidence_avg"][cls] = round(
                    conf_sum[cls] / conf_count[cls], 4
                )
            else:
                stats["class_confidence_avg"][cls] = 0.0

        # 결과 JSON 저장
        with open(
            output_dir / "classification_results.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        # 요약 출력
        print("\n" + "=" * 60)
        print("✅ 처리 완료")
        print(f"총 이미지: {stats['total']}")
        print(f"정상 분류: {stats['processed']}")
        print(f"unknown: {stats['low_confidence']}")
        print(f"실패: {stats['failed']}")
        print("\n📊 클래스별 평균 confidence")
        for cls, avg in stats["class_confidence_avg"].items():
            print(f"- {cls:10s}: {avg}")
        print("=" * 60)


# =====================================================
# 실행부
# =====================================================
def main():
    if not IMAGESET_DIR.exists():
        raise RuntimeError(f"입력 폴더 없음: {IMAGESET_DIR}")
    if not MODEL_PATH.exists():
        raise RuntimeError(f"모델 파일 없음: {MODEL_PATH}")

    classifier = ImageClassifier(MODEL_PATH)
    classifier.process(
        input_dir=IMAGESET_DIR,
        output_dir=OUTPUT_DIR,
        min_confidence=0.7,
    )


if __name__ == "__main__":
    main()
