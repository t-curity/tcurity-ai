import os
import random
import base64
from pathlib import Path
from typing import Dict, List, Tuple


# =====================================================
# Phase B 문제 출제 규칙
# =====================================================
PHASE_B_RULES: Dict[str, List[str]] = {
    "Animals": ["Building", "Devices", "Fashion", "Vehicle"],
    "Birds": ["Building", "Devices", "Fashion", "Food"],
    "Building": ["Nature", "Devices", "Fashion", "Sports"],
    "Devices": ["Instrument", "Building", "Fashion", "Food"],
    "Fashion": ["Devices", "Instrument", "Building", "Food"],
    "Food": ["Instrument", "Devices", "Fashion", "Nature"],
    "Instrument": ["Building", "Devices", "Fashion", "Nature"],
    "Nature": ["Building", "Devices", "Instrument", "Vehicle"],
    "Sports": ["Fashion", "Building", "Devices", "Food"],
    "Vehicle": ["Building", "Nature", "Devices", "Instrument"],
}


CLASS_KO_MAP: Dict[str, str] = {
    "Animals": "동물",
    "Birds": "새",
    "Building": "건물",
    "Devices": "전자기기",
    "Fashion": "패션",
    "Food": "음식",
    "Nature": "자연",
    "Sports": "스포츠",
    "Vehicle": "탈 것",
    "Instrument": "악기",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


# =====================================================
# 내부 유틸
# =====================================================
def _collect_images(class_dir: Path) -> List[Path]:
    if not class_dir.exists():
        return []
    return [
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]


def _image_to_base64(image_path: Path) -> str:
    with image_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _extract_uuid(image_path: Path) -> str:
    # <uuid>.jpg → uuid
    return image_path.stem


# =====================================================
# Public API
# =====================================================
def generate_phase_b_problem(
    target_class: str,
    *,
    num_target: int = 4,
    total_images: int = 9,
    rules: Dict[str, List[str]] = PHASE_B_RULES,
) -> Dict:
    """
    Phase B CAPTCHA 문제 생성 (UUID + base64 기반)
    """

    image_root = os.environ.get("PHASE_B_PROBLEM_IMAGE_ROOT")
    if not image_root:
        raise RuntimeError("PHASE_B_PROBLEM_IMAGE_ROOT not set")

    image_root = Path(image_root)

    if target_class not in rules:
        raise ValueError(f"Invalid target_class: {target_class}")

    display_class = CLASS_KO_MAP.get(target_class, target_class)

    # -------------------------------
    # 1) 정답 이미지
    # -------------------------------
    target_dir = image_root / target_class
    target_pool = _collect_images(target_dir)

    if len(target_pool) < num_target:
        raise RuntimeError("Not enough target images")

    target_imgs = random.sample(target_pool, num_target)

    # -------------------------------
    # 2) 오답 이미지
    # -------------------------------
    num_wrong = total_images - num_target
    wrong_pool: List[Path] = []

    for cls in rules[target_class]:
        wrong_pool.extend(_collect_images(image_root / cls))

    if len(wrong_pool) < num_wrong:
        raise RuntimeError("Not enough wrong images")

    wrong_imgs = random.sample(wrong_pool, num_wrong)

    # -------------------------------
    # 3) 문제 구성
    # -------------------------------
    images = []
    answer_uuids = set()

    for img in target_imgs:
        uid = _extract_uuid(img)
        answer_uuids.add(uid)
        images.append({
            "uuid": uid,
            "image_base64": _image_to_base64(img),
        })

    for img in wrong_imgs:
        uid = _extract_uuid(img)
        images.append({
            "uuid": uid,
            "image_base64": _image_to_base64(img),
        })

    random.shuffle(images)

    # -------------------------------
    # 4) 반환
    # -------------------------------
    return {
    "question": f"{display_class}에 해당하는 이미지를 순서대로 모두 고르세요.",
    "target_class": target_class,
    "display_class": display_class,
    "images": images,                    # FE용
    "answer_uuids": list(answer_uuids),  # 서버용
    }
