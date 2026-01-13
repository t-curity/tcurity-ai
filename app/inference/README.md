# 📕 AI 모듈 Inference 가이드 (행동·이미지 기반 AI 통합 안내)

본 문서는 CAPTCHA 서비스에서 사용되는 AI 추론(Inference) 기능을 설명합니다.
우리 서비스의 AI는 총 3가지 모델로 구성됩니다:

## 📗 전체 AI 구성 개요

| 모델 종류                          | 용도                   | 설명                       |
| -------------------------------- | --------------------- | ------------------------- |
| **Isolation Forest (행동 기반 ①)** | 마우스·드래그 행동 이상 탐지 |  비지도 학습 기반 이상 행동 감지  |
| **Random Forest (행동 기반 ②)**    | 시퀀스 기반 행동 패턴 분류  |  지도 학습 기반 행동 특징 분류   |
| **EfficientNet-B0 (이미지 기반)**   | 이미지 분류 (동물/사물)    | 사용자가 올린 이미지의 대그룹 분류 |


### 📘 AI Inference 가이드

#### 1️⃣ Isolation Forest Inference 가이드

추후 작성 예정

#### 2️⃣ Random Forest Inference 가이드

추후 작성 예정

#### 3️⃣ EfficientNet Inference 가이드
EfficientNet inference 스크립트는 이미지 기반 CAPTCHA의 분류 모델만을 수행합니다.
- 지원 기능
  - 자동 랜덤 이미지 선택 또는 특정 이미지 직접 넣어서 분류
  - 데이터셋 구조를 기반으로 라벨 자동 생성 (animal, object 등)

#### 3.1 실행 방법
AI 추론 스크립트는 반드시 다음 디렉토리에서 실행해야 합니다:
```
cd /home/ubuntu/tcurity-ai
```
이 위치에서만 `./images`, `./models`, `training/`, `inference/` 등의 상대 경로가 정상적으로 연결됩니다.


- (1) 환경변수 로드 
```
source .env/mlflow.env
source .env/phase_b_image.env
```

- (2)-a. 이미지 랜덤 선택 추론
```
python app/inference/image_classifier.py
```

- (2)-b. 특정 이미지 직접 선택 추론
```
python app/inference/image_classifier.py \
  --image_path ./data/images/Animals/elephant/elephant_22.jpeg
```
