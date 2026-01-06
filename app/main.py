from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Tuple, Union
from fastapi import FastAPI, HTTPException
import random
import logging

# ===================================================
# Phase A
# ===================================================
from app.inference.phase_a_service import PhaseAInfer, coerce_points

# ====================================================
# Phase B - 문제 생성
# ====================================================
from app.inference.phase_b_problem_generator import (
    generate_phase_b_problem,
    PHASE_B_RULES,
)

# ====================================================
# Phase B - 행동 기반 AI (RandomForest)
# ====================================================
from app.inference.phase_b_service import PhaseBInfer, save_phase_b_sample

app = FastAPI()
logger = logging.getLogger("uvicorn.error")

# =====================================================
# 서버 시작 시 1회 로드 (AI 모델들)
# =====================================================
phase_a = PhaseAInfer()
phase_b_ai = PhaseBInfer()  # phase_b_rf_matched.pkl 등 자동 탐색


class PhaseBGeneratePayload(BaseModel):
    target_class: Optional[str] = None


# -----------------------------
# Phase B payload coerce
# -----------------------------
def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _to_int(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return default


def _normalize_time_to_relative_ms(points: List[Dict[str, Any]]) -> None:
    """
    t 값이 epoch(ms/sec/ns)처럼 큰 값이면 첫 점 기준 상대시간(ms)으로 변환.
    """
    if not points:
        return

    t0 = _to_int(points[0].get("t", 0), 0)

    # epoch 판단: 보통 ms epoch는 1e12 근처(= 2001~2286년), sec epoch는 1e9 근처
    # ns epoch는 1e18 근처
    # → 상대시간(ms)로 맞춰줌
    if t0 >= 10**17:
        # ns로 들어왔다고 가정
        base = t0
        for p in points:
            p["t"] = (_to_int(p.get("t", 0), 0) - base) // 10**6
        return

    if t0 >= 10**12:
        # ms epoch로 들어왔다고 가정
        base = t0
        for p in points:
            p["t"] = _to_int(p.get("t", 0), 0) - base
        return

    if t0 >= 10**9:
        # sec epoch로 들어왔다고 가정 → ms로 변환 후 상대시간
        base = t0
        for p in points:
            p["t"] = (_to_int(p.get("t", 0), 0) - base) * 1000
        return

    # 이미 상대시간(ms)로 들어온 케이스는 그대로 둠
    return


def coerce_phase_b_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    FE/BE에서 넘어오는 다양한 포맷을 PhaseBInfer가 학습 때 쓰던 형태로 정규화.
    기대 결과:
      {
        "points": [{"x": "...", "y": "...", "t": 0, "action": "move"}, ...],
        "metadata": {"deviceType": "...", "res": {"w":..., "h":...}, ...}
      }
    """
    # 1) behavior wrapper 처리
    data = payload
    if isinstance(payload, dict) and isinstance(payload.get("behavior"), dict):
        data = payload["behavior"]

    raw_points = data.get("points", [])
    points: List[Dict[str, Any]] = []

    if raw_points and isinstance(raw_points[0], (list, tuple)):
        # 예: [[x,y,t,"move"], ...]
        for p in raw_points:
            if len(p) < 3:
                continue
            x, y, t = p[0], p[1], p[2]
            action = p[3] if len(p) >= 4 else None
            points.append(
                {
                    "x": f"{_to_float(x):.4f}",
                    "y": f"{_to_float(y):.4f}",
                    "t": _to_int(t, 0),
                    "action": action,
                }
            )
    elif raw_points and isinstance(raw_points[0], dict):
        # 예: [{"x":..,"y":..,"t":..}, ...]
        for p in raw_points:
            x = p.get("x")
            y = p.get("y")
            t = p.get("t", 0)
            action = p.get("action") or p.get("a") or p.get("type")
            points.append(
                {
                    "x": f"{_to_float(x):.4f}",
                    "y": f"{_to_float(y):.4f}",
                    "t": _to_int(t, 0),
                    "action": action,
                }
            )

    # 2) 시간 보정 (epoch → relative ms)
    _normalize_time_to_relative_ms(points)

    # 3) metadata 정규화 (screenWidth/screenHeight → res.w/res.h)
    md = data.get("metadata", {}) or {}
    w = md.get("screenWidth") or md.get("w") or (md.get("res") or {}).get("w")
    h = md.get("screenHeight") or md.get("h") or (md.get("res") or {}).get("h")

    out_md = dict(md)
    out_md.setdefault("deviceType", md.get("deviceType", "unknown"))
    out_md["res"] = {"w": _to_int(w, 0), "h": _to_int(h, 0)}

    # 4) 기타 필드가 있으면 유지(학습엔 안 쓰더라도)
    out: Dict[str, Any] = {
        "points": points,
        "metadata": out_md,
    }
    # correct_count/is_perfect 같은 게 있으면 같이 넘겨도 됨(현재는 use_result_features=False였지만)
    for k in ("correct_count", "is_perfect", "target", "timestamp", "label"):
        if k in data:
            out[k] = data[k]

    return out


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
# Phase B 문제 생성 API
# =====================================================
@app.post("/phase-b/generate")
def phase_b_problem_generate(payload: PhaseBGeneratePayload):
    try:
        target_class = payload.target_class
        if target_class is None:
            if not PHASE_B_RULES:
                raise ValueError("PHASE_B_RULES is empty (dataset/rules load failed)")
            target_class = random.choice(list(PHASE_B_RULES.keys()))

        if target_class not in PHASE_B_RULES:
            raise ValueError(f"unknown target_class={target_class}")

        problem = generate_phase_b_problem(target_class)

        for k in ["question", "target_class", "display_class", "images", "answer_uuids"]:
            if k not in problem:
                raise ValueError(f"generate_phase_b_problem missing key: {k}")

        return {
            "question": problem["question"],
            "target_class": problem["target_class"],
            "display_class": problem["display_class"],
            "images": [{"image_id": img["uuid"], "image_base64": img["image_base64"]} for img in problem["images"]],
            "answer_uuids": problem["answer_uuids"],
        }

    except Exception as e:
        logger.exception("phase-b/generate failed")
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# Phase B 행동 검증 API
# =====================================================
@app.post("/phase-b/verify")
def phase_b_behavior_verify(payload: Dict[str, Any]):
    try:
        data = coerce_phase_b_payload(payload)

        infer_full = phase_b_ai.infer_from_payload(
            data,
            return_score=True,
            return_features=True,   # 용량 부담되면 False로
        )

        try:
            save_phase_b_sample(normalized_payload=data, infer=infer_full)
        except Exception:
            logger.exception("phase-b sample save failed")

        # 응답은 기존처럼
        return {"pass": infer_full["pass"], "label": infer_full["label"]}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("phase-b/verify failed")
        raise HTTPException(status_code=500, detail=str(e))

