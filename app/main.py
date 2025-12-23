from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
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

# ====================================================
# Phase B - 행동 기반 AI (RandomForest)
# ====================================================
# from app.inference.phase_b_service import PhaseBInfer, coerce_features

app = FastAPI()

# =====================================================
# 서버 시작 시 1회 로드 (AI 모델들)
# =====================================================
phase_a = PhaseAInfer(model_dir="/home/ubuntu/tcurity-ai/models/phase_a")
# 임시 버전 -- 추후 수정 예정
# phase_b_ai = PhaseBInfer(model_dir="/home/ubuntu/tcurity-ai/models/phase_b")


# =====================================================
# Phase A Payload
# =====================================================
class DragPayload(BaseModel):
    points: List[Dict[str, Any]]

class PhaseBGeneratePayload(BaseModel):
    target_class: Optional[str] = None


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
        return phase_a.infer_human_bot(points, return_score=False)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# =====================================================
# Phase B 문제 생성 API (POST 유지 + base64)
# =====================================================
@app.post("/phase-b/generate")
def phase_b_problem_generate(payload: PhaseBGeneratePayload):
    try:
        target_class = payload.target_class
        if target_class is None:
            target_class = random.choice(list(PHASE_B_RULES.keys()))

        problem = generate_phase_b_problem(target_class)

        # ✅ Backend가 받아야 할 모든 정보 반환
        return {
            "question": problem["question"],
            "target_class": problem["target_class"],
            "display_class": problem["display_class"],

            # ✅ FE에 보여줄 이미지들
            "images": [
                {
                    "image_id": img["uuid"],
                    "image_base64": img["image_base64"],
                }
                for img in problem["images"]
            ],

            # ✅ Backend 전용 (FE로는 전달 ❌)
            "answer_uuids": problem["answer_uuids"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# # =====================================================
# # Phase B 행동 검증 -- 임시 버전 !!! 추후 변경 예정
# # =====================================================
# @app.post("/phase-b/behavior/verify")
# def phase_b_behavior_verify(payload: PhaseBBehaviorPayload):
#     """
#     Phase B 행동 기반 AI 추론
#     (정답 판정 ❌, 행동만 판단)
#     """
#     try:
#         features = coerce_features(payload.behavior)
#         return phase_b_ai.infer_human_bot(
#             features,
#             return_score=False
#         )
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))