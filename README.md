# T-CURITY AI (Inference Server)

GPU 기반 CAPTCHA 모델 추론 서버 초기 템플릿입니다.

---

## 📦 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🚀 실행
```bash
uvicorn app.main:app --host 0.0.0.0 --port 9000
```

---

## 📁 구조
```bash
app/
 ├── main.py            # FastAPI 엔트리
 ├── inference/         # 모델 추론 로직
 ├── core/              # 환경 설정
 └── utils/             # 공용 유틸
models/                 # 개발자가 임시 저장하는 모델 (.onnx)
```

---

## 🧩 개발 규칙
- 개발은 /home/<user>/workspace/tcurity-ai에서 진행
- 운영 서버 /srv/inference-server는 절대 수정 금지
- 모델 변경 시 PR 필수
- main 브랜치에는 배포된 코드만 유지