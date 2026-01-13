# 🧠 AI Training Module
본 디렉토리는 CAPTCHA 서비스의 AI 학습 파이프라인을 관리하는 공간이며, 아래 두 가지 범주의 모델을 포함할 수 있도록 확장성을 고려해 설계되어 있습니다.

- 행동 기반 모델
  - Isolation Forest — 이상 행동 탐지 (Ticket Slice)
  - Random Forest — 드래그 패턴 기반 행동 분류 (Drag & Drop)
    ※ 행동 기반 모델은 별도 스크립트로 추가 예정
- 이미지 기반 모델
  - EfficientNet-B0 기반 이미지 분류 모델 (Drag & Drop)

본 폴더는 이러한 모델들의 전처리, 학습, 모델 관리 기능을 통합적으로 제공하며, 확장성을 고려하여 설계되어 있습니다.

---

# 📂 Directory Structure
```
tcurity-ai/
 ├── training/
 │     ├── train_efficientnet.py     # 이미지 분류 학습 메인 스크립트
 │     ├── captcha_dataset.py        # 2단계 구조 Dataset 자동 라벨링
 │     ├── name_changer.py           # 이미지 파일명 정규화 스크립트
 │     ├── islocation_forest_ai.py.  # 추후 추가 예정 
 │     ├── random_forest_ai.py.      # 추후 추가 예정
 │     └── README.md
 │
 └── inference/
       ├── image_classifier.py       # 단일 이미지 분류 (랜덤 선택 포함)
       └── README.md

```

# ✂️ Ticket Slice (행동 기반 검증 AI)
## 🌳 Isolation Forest
→ 추가 예정

# 🖱️ Drag & Drop (인지 기반 검증 AI)
## 🏕️ Random Forest
→ 추가 예정

## 🎆 EfficientNet-B0 AI 
### 💿 Image Data Structure 
학습 데이터는 반드시 아래와 같은 2단계 폴더 구조를 따라야 합니다.
```
images/
 ├── animal/
 │     ├── cheetah/
 │     ├── dog/
 │     └── ...
 ├── object/
 │     ├── toaster/
 │     ├── gloves/
 │     └── ...
```
- Layer 1: 대그룹 (예: animal, object)
- Layer 2: 세부 클래스 폴더 (예: cheetah / toaster 등)
- 실제 이미지 파일(.jpg/.png)은 2단계 폴더 안에 위치
📍 대그룹 기준으로 label이 자동 생성되며, 클래스 수(NUM_CLASSES)는 학습 시 자동 계산됩니다.

---

### 🚀 Training Workflow

#### 0. 실행 전 위치 설정 
AI 학습 스크립트는 반드시 아래 경로에서 실행해야 합니다.
```bash
cd /home/ubuntu/tcurity-ai/
```
이 위치를 기준으로 `./images`, `./models`, `training/`, `inference/` 등의 상대 경로가 정상적으로 동작합니다.

#### 1. MLflow Tracking Server 실행
⚠️ 학습을 시작하기 전에 반드시 MLflow 서버를 먼저 실행해야 합니다.
MLflow 서버가 실행되지 않으면 학습 중 Tracking 단계에서 오류가 발생합니다.
✅ MLflow 서버 실행 명령어
```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --host 0.0.0.0 \
  --port 5000 \
  --cors-allowed-origins="*" \
  --allowed-hosts="*"
```
- MLflow Dashboard 접속 주소: `http://<SERVER_IP>:5000`

#### 2. 파일명 정규화 – name_changer.py
다양한 원본 이미지 이름을 다음처럼 규칙적으로 정리합니다.
```
e.g. 
cheetah_1.jpg
cheetah_2.jpg
...
toaster_1.jpg
toaster_2.jpg
...
```
✅ 실행 명령어
```bash
source .env/phase_b_image.env
python training/name_changer.py
```

#### 3. Dataset 자동 라벨링 — captcha_dataset.py
대그룹 폴더명 기준으로 label 자동 생성합니다. 
```
e.g.
- Animal = 0  
- Object = 1  
```
- 전체 이미지 개수 출력
- 클래스별 이미지 개수 출력
- NUM_CLASSES 자동 계산 → 모델에 직접 반영됨

#### 4. EfficientNet 학습 — train_efficientnet.py

#### 4.1 기본(default) 파라미터 실행
✅ 실행 명령어
```bash
source .env/mlflow.env
source .env/phase_b_image.env 
python training/train_efficientnet.py 
```
→ batch_size, learning_rate, epochs 등은 스크립트 내부 default 사용

#### 4.2 하이퍼파라미터 직접 지정 실행
✅ 실행 명령어 
```bash
python training/train_efficientnet.py \
  --batch_size 64 \
  --learning_rate 0.0004 \
  --epochs 50 \
  --patience 5
```

#### 5. Output
학습 완료 시 아래 파일이 생성됩니다.
```
/models/best_model.pth
```
- Validation Accuracy 기준 최고 성능 모델
- Early Stopping 적용
- Inference 서버(ai/inference/)에서 해당 모델을 로드하여 사용

---

### 📊 MLflow Tracking
학습 과정에서 MLflow로 다음 항목들이 자동 기록됩니다.
- Hyper Parameters (batch_size, learning_rate, epochs, patience)
- Train / Validation Loss
- Train / Validation Accuracy
- Learning Rate 변화
- Best Model Artifact
- Model Signature (Inference용)

---

### 🎯 Summary
- EfficientNet-B0 기반 이미지 CAPTCHA 분류 모델
- Dataset 구조 변경 시에도 NUM_CLASSES 자동 대응
- MLflow Tracking Server 사전 실행 필수
- train_efficientnet.py는 학습 + 기록을 담당하는 메인 스크립트
- best_model.pth는 API / Inference 서버에서 직접 사용 가능