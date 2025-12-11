from fastapi import FastAPI
from app.inference.runner import run_inference

app = FastAPI(
    title="T-CURITY Inference Server",
    description="GPU-based inference server for CAPTCHA",
    version="1.0.0"
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/inference")
def inference(payload: dict):
    """
    payload 예:
    {
        "behavior": [...],
        "phase": "A"
    }
    """
    result = run_inference(payload)
    return {"result": result}
