import os
import random
import base64
from pathlib import Path
from typing import Dict, List, Tuple


# =====================================================
# Phase B 문제 출제 규칙
# =====================================================
PHASE_B_RULES: Dict[str, List[str]] = {
    "Bird": ["Dog", "Nature", "Food", "Building"],
    "Building": ["Vehicle", "Nature", "Device", "Sports"],
    "Cat": ["Dog", "Bird", "Nature", "Food"],
    "Device": ["Vehicle", "Building", "Sports", "Food"],
    "Dog": ["Bird", "Nature", "Food", "Building"],
    "Food": ["Nature", "Bird", "Dog", "Building"],
    "Nature": ["Bird", "Food", "Dog", "Building"],
    "Sports": ["Vehicle", "Device", "Building", "Nature"],
    "Turtle": ["Dog", "Bird", "Nature", "Food"],
    "Vehicle": ["Device", "Building", "Sports", "Nature"],
}


CLASS_KO_MAP: Dict[str, str] = {
    "Bird": "새",
    "Building": "건물",
    "Cat": "고양이",
    "Device": "전자기기",
    "Dog": "강아지",
    "Food": "음식",
    "Nature": "자연",
    "Sports": "스포츠",
    "Turtle": "거북이",
    "Vehicle": "탈것",
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
    "question": f"{display_class} 이미지를 번호 순서대로 아래 칸에 드래그하세요.",
    "target_class": target_class,
    "display_class": display_class,
    "images": images,                    # FE용
    "answer_uuids": list(answer_uuids),  # 서버용
    }
