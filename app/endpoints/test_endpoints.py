# app/endpoints/test_endpoints.py
from fastapi import APIRouter

router = APIRouter(prefix="/test", tags=["Test"])

@router.post("/phase_a")
def test_phase_a(payload: dict):
    from app.inference.phase_a_service import PhaseAInfer, coerce_points
    infer = PhaseAInfer()
    points = coerce_points(payload)
    return infer.infer_human_bot(points, return_score=True)

@router.post("/phase_b")
def test_phase_b(payload: dict):
    from app.inference.phase_b_service import get_instance
    infer = get_instance()
    return infer.infer_from_payload(payload, return_score=True, return_features=True)