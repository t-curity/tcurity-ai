from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from app.inference.phase_a_service import PhaseAInfer, coerce_points

app = FastAPI()

# ✅ 서버 프로세스 시작 시 1회 로드
phase_a = PhaseAInfer(model_dir="/home/ubuntu/tcurity-ai/models/phase_a")


class DragPayload(BaseModel):
    points: List[Dict[str, Any]]


@app.post("/phase-a/verify")
def phase_a_verify(payload: Dict[str, Any]):
    """
    응답은 오직 사람/봇만
    """
    points = coerce_points(payload)
    result = phase_a.infer_human_bot(points)
    return result
