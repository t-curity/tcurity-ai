# app/endpoints/test_endpoints.py

from fastapi import APIRouter
from app.inference.phase_a_service import get_instance as get_phase_a_instance
from app.inference.phase_b_service import get_instance as get_phase_b_instance

router = APIRouter(prefix="/test", tags=["Test"])

@router.post("/phase_a")
def test_phase_a(payload: dict):
    """Phase A 단독 테스트 (개발용)"""
    infer = get_phase_a_instance()
    return infer.infer_from_payload(payload, return_score=True, return_features=True)

@router.post("/phase_b")
def test_phase_b(payload: dict):
    """Phase B 단독 테스트 (개발용)"""
    infer = get_phase_b_instance()
    return infer.infer_from_payload(payload, return_score=True, return_features=True)