"""
app/inference/phase_b_image_verifier.py

[phase_b_image_verifier.py - 서비스용 추론 모듈]

- 이 파일은 API 서버에서 실제로 호출되는
  Phase B CAPTCHA 이미지 검증 로직이다.

- EfficientNet-B0 모델을 로드하여
  사용자가 선택한 이미지들이
  특정 target_label에 해당하는지 판단한다.
- 선택 결과를 기반으로 PASS / FAIL을 반환한다.

※ 이 파일은 서비스 코드이며, CLI 실행이나 디버깅 목적이 아님

"""

import os
from pathlib import Path
from typing import Dict, List, Union

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

ImageInput = Union[Image.Image, str, Path]


class PhaseBImageVerifier:
    """
    EfficientNet-B0 기반 이미지 분류기

    - model_path: 학습된 .pth 경로
        · 미지정 시 $MODEL_OUTPUT_ROOT/best_model.pth 사용
    - label_map:
        · {0: "Animals", 1: "Birds", ...}
        · 반드시 학습 시 사용한 클래스 순서와 동일해야 함
    """

    def __init__(
        self,
        label_map: Dict[int, str],
        model_path: str | None = None,
        device: str | None = None,
    ):
        # =====================================================
        # 1) Device 설정
        # =====================================================
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        # =====================================================
        # 2) Label mapping
        # =====================================================
        self.label_map = label_map
        self.num_classes = len(label_map)

        # "라벨 문자열" → "모델 출력 인덱스"
        self.label_to_idx = {v: k for k, v in label_map.items()}

        # =====================================================
        # 3) Model path 결정 (Phase B 기준)
        # =====================================================
        if model_path is None:
            model_root = os.environ.get("MODEL_OUTPUT_ROOT")
            if not model_root:
                raise RuntimeError(
                    "model_path가 없고 MODEL_OUTPUT_ROOT 환경변수도 없습니다."
                )
        
            # ✅ Phase B 디렉토리 고정
            model_path = Path(model_root) / "phase_b" / "best_model.pth"

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"모델 파일 없음: {model_path}")

        # =====================================================
        # 4) 모델 생성
        # =====================================================
        self.model = models.efficientnet_b0(weights=None)

        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, self.num_classes)

        # =====================================================
        # 5) 가중치 로드
        # =====================================================
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

        # =====================================================
        # 6) 전처리 (학습과 반드시 동일)
        # =====================================================
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    # =====================================================
    # 내부 유틸
    # =====================================================
    def _load_image(self, image: ImageInput) -> Image.Image:
        if isinstance(image, (str, Path)):
            p = Path(image)
            if not p.exists():
                raise FileNotFoundError(f"이미지 파일 없음: {p}")
            return Image.open(p).convert("RGB")

        return image.convert("RGB")

    def _infer_probs(self, image: Image.Image) -> torch.Tensor:
        x = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]
        return probs.cpu()

    # =====================================================
    # Public API
    # =====================================================
    def predict(self, image: ImageInput) -> Dict:
        """
        단일 이미지 분류

        반환 예:
        {
            "predicted_label": "Animals",
            "confidence": 0.82,
            "probabilities": {
                "Animals": 0.82,
                "Food": 0.05,
                ...
            }
        }
        """
        img = self._load_image(image)
        probs = self._infer_probs(img)

        pred_idx = int(probs.argmax())
        confidence = float(probs[pred_idx])

        return {
            "predicted_label": self.label_map[pred_idx],
            "confidence": confidence,
            "probabilities": {
                self.label_map[i]: float(probs[i])
                for i in range(self.num_classes)
            },
        }

    def verify_selected(
        self,
        selected_images: List[ImageInput],
        target_label: str,
        threshold: float = 0.5,
    ) -> Dict:
        """
        CAPTCHA 선택 검증

        규칙:
        - 선택된 이미지들 중
        - target_label 확률의 "최솟값"이 threshold 이상이면 PASS

        반환 예:
        {
            "status": "PASS",
            "target_label": "Animals",
            "threshold": 0.5,
            "average_confidence": 0.76,
            "min_confidence": 0.62,
            "max_confidence": 0.88,
            "scores": [0.62, 0.88, 0.79, 0.75]
        }
        """
        if target_label not in self.label_to_idx:
            raise ValueError(f"존재하지 않는 클래스: {target_label}")

        if len(selected_images) == 0:
            raise ValueError("선택된 이미지가 없습니다")

        target_idx = self.label_to_idx[target_label]

        scores: List[float] = []
        for img in selected_images:
            image = self._load_image(img)
            probs = self._infer_probs(image)
            scores.append(float(probs[target_idx]))

        return {
            "status": "PASS" if min(scores) >= threshold else "FAIL",
            "target_label": target_label,
            "threshold": threshold,
            "average_confidence": sum(scores) / len(scores),
            "min_confidence": min(scores),
            "max_confidence": max(scores),
            "scores": scores,
        }
