from fastapi import APIRouter
from app.inference.phase_b_service import get_instance

router = APIRouter(prefix="/test", tags=["Test"])

@router.post("/phase_b")
def test_phase_b(payload: dict):
    """Phase B 단독 테스트 (개발용)"""
    infer = get_instance()
    return infer.infer_from_payload(payload, return_score=True, return_features=True)