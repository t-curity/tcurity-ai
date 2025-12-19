from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from app.inference.phase_a_service import PhaseAInfer, coerce_points

app = FastAPI()

phase_a = PhaseAInfer(model_dir="/home/ubuntu/tcurity-ai/models/phase_a")


class DragPayload(BaseModel):
    points: List[Dict[str, Any]]


@app.post("/phase-a/verify")
def phase_a_verify(payload: Dict[str, Any]):
    """
    응답은 오직 사람/봇만
    """
    try:
        points = coerce_points(payload)
        
        result = phase_a.infer_human_bot(points, return_score=True)
        
        is_human = result.get("pass", False)
        score = result.get("score")
        threshold = result.get("threshold")
        
        return {
            "pass": is_human,
            "label": "사람" if is_human else "봇",
            "score": score,
            "threshold": threshold
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))