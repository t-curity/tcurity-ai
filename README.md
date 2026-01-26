# T:CURITY AI Server

2-Phase CAPTCHA 검증을 위한 GPU 기반 AI 추론 서버입니다.

## ✨ 주요 기능

- **Phase A 분석**: Isolation Forest 기반 드래그 행동 패턴 분석
- **Phase B 검증**: Random Forest 기반 이미지 선택 행동 분석
- **이미지 분류**: 도형/객체 이미지 분류 및 문제 생성
- **실시간 추론**: FastAPI + GPU 가속 처리

## 🛠 기술 스택

- **Framework**: FastAPI
- **ML**: scikit-learn, ONNX Runtime
- **GPU**: CUDA (Optional)
- **Server**: Uvicorn

## 🚀 시작하기

### 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 실행

```bash
uvicorn app.main:app --host 0.0.0.0 --port 9000
```

## 📁 프로젝트 구조

```
tcurity-ai/
├── app/
│   ├── main.py                    # FastAPI 엔트리포인트
│   ├── inference/
│   │   ├── phase_a_service.py     # Phase A 드래그 분석
│   │   ├── phase_b_service.py     # Phase B 이미지 검증
│   │   ├── image_classifier.py    # 이미지 분류기
│   │   └── phase_b_problem_generator.py  # 문제 생성
│   ├── endpoints/                 # API 라우터
│   ├── core/                      # 환경 설정
│   └── utils/                     # 공용 유틸리티
├── training/                      # 모델 학습 스크립트
├── tests/                         # 테스트 코드
└── requirements/                  # 환경별 의존성
```

## 🔌 API 엔드포인트

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/inference/phase-a` | 드래그 행동 분석 |
| POST | `/inference/phase-b/problem` | Phase B 문제 생성 |
| POST | `/inference/phase-b/verify` | Phase B 정답 검증 |
| GET | `/health` | 서버 상태 확인 |

## ⚙️ 배포

- GPU 서버 (10.0.83.48:9000)에서 실행
- Main 서버 Backend와 내부 통신

## 🧩 개발 규칙

- 개발: `/home/<user>/workspace/tcurity-ai`
- 운영: `/srv/inference-server` (수정 금지)
- 모델 변경 시 PR 필수

## 📄 라이선스

MIT License - Copyright (c) 2025 T:CURITY