from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List
import random

# ===================================================
# Phase A
# ===================================================
from app.inference.phase_a_service import PhaseAInfer, coerce_points

# ===================================================
# Phase B
# ===================================================
from app.inference.phase_b_problem_generator import (
    generate_phase_b_problem,
    PHASE_B_RULES,
)
from app.inference.phase_b_service import PhaseBInfer, coerce_features

app = FastAPI()

# =====================================================
# 서버 시작 시 1회 로드 (AI 모델들)
# =====================================================
phase_a = PhaseAInfer(model_dir="/home/ubuntu/tcurity-ai/models/phase_a")
# 임시 버전 -- 추후 수정 예정
phase_b_ai = PhaseBInfer(model_dir="/home/ubuntu/tcurity-ai/models/phase_b")


# =====================================================
# Phase A Payload
# =====================================================
class DragPayload(BaseModel):
    points: List[Dict[str, Any]]


# =====================================================
# Phase A 드래그 검증 API 
# =====================================================
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


# =====================================================
# Phase B 문제 "재료" 생성 API
# =====================================================
@app.get("/phase-b/problem/generate")
def phase_b_problem_generate():
    """
    Phase B 문제 재료 생성 (GPU 서버)

    반환:
    - target_class: 문제로 제시할 대분류 (한글)
    - images: 이미지 path + 정답 여부
    """
    target_class = random.choice(list(PHASE_B_RULES.keys()))
    problem = generate_phase_b_problem(target_class)

    return {
        "target_class": problem["target_class"],
        "images": [
            {
                "path": img["path"],
                "is_target": img["is_target"],
            }
            for img in problem["images"]
        ],
    }


# =====================================================
# Phase B 행동 검증 API (추가, 임시 버전)
# =====================================================
@app.post("/phase-b/behavior/verify")
def phase_b_behavior_verify(payload: Dict[str, Any]):
    """
    Phase B 행동 기반 AI 추론
    (정답 판정 ❌, 행동만 판단)
    """
    try:
        features = coerce_features(payload)
        return phase_b_ai.infer_human_bot(
            features,
            return_score=False
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
