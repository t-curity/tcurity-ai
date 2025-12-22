"""
app/inference/phase_b_problem_generator.py

[Phase B 문제 출제 모듈]

- Phase B 이미지 선택 CAPTCHA 문제 생성 전용
- processed_images 기준 (소분류 없음)
- 정답: target_class 폴더에서 랜덤 4장
- 오답: rules 기반 클래스에서 랜덤 5장

※ 내부 로직은 영어 클래스명 사용
※ 사용자 노출 텍스트만 한국어 매핑
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Tuple


# =====================================================
# Phase B 문제 출제 규칙 (대분류 기준, 내부 로직용)
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


# =====================================================
# 사용자 노출용 한국어 클래스 매핑 (Furniture 없음)
# =====================================================
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


# =====================================================
# 내부 유틸
# =====================================================
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _collect_images_in_class_dir(class_dir: Path) -> List[Path]:
    """
    processed_images/<Class>/ 아래의 이미지 파일 수집
    """
    if not class_dir.exists():
        return []

    return [
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]


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
    Phase B CAPTCHA 문제 1개 생성

    Returns:
    {
        "question": "동물 이미지를 모두 고르시오",
        "target_class": "Animals",       # 내부 로직용
        "display_class": "동물",          # 사용자 노출용
        "images": [
            {
                "path": "...",
                "label": "Animals",       # 내부 검증용
                "is_target": True
            },
            ...
        ]
    }
    """

    # -------------------------------------------------
    # 0) 이미지 루트 (환경변수)
    # -------------------------------------------------
    image_root = os.environ.get("PHASE_B_PROBLEM_IMAGE_ROOT")
    if not image_root:
        raise RuntimeError(
            "PHASE_B_PROBLEM_IMAGE_ROOT 환경변수가 설정되지 않았습니다"
        )

    image_root = Path(image_root)

    if target_class not in rules:
        raise ValueError(f"정의되지 않은 target_class: {target_class}")

    target_dir = image_root / target_class
    if not target_dir.exists():
        raise FileNotFoundError(f"타겟 클래스 폴더 없음: {target_dir}")

    # -------------------------------------------------
    # 1) 사용자 노출용 한국어 클래스명
    # -------------------------------------------------
    display_class = CLASS_KO_MAP.get(target_class, target_class)

    # -------------------------------------------------
    # 2) 정답 이미지 4장
    # -------------------------------------------------
    target_pool = _collect_images_in_class_dir(target_dir)

    if len(target_pool) < num_target:
        raise RuntimeError(
            f"{target_class} 정답 이미지 부족: "
            f"{len(target_pool)} < {num_target}"
        )

    target_images = random.sample(target_pool, num_target)

    # -------------------------------------------------
    # 3) 오답 이미지 5장 (rules 기반)
    # -------------------------------------------------
    num_wrong = total_images - num_target
    wrong_pool: List[Tuple[str, Path]] = []

    for cls in rules[target_class]:
        cls_dir = image_root / cls
        imgs = _collect_images_in_class_dir(cls_dir)
        for img in imgs:
            wrong_pool.append((cls, img))

    if len(wrong_pool) < num_wrong:
        raise RuntimeError(
            f"오답 이미지 풀 부족: {len(wrong_pool)} < {num_wrong}"
        )

    selected_wrong_images = random.sample(wrong_pool, num_wrong)

    # -------------------------------------------------
    # 4) 문제 이미지 구성
    # -------------------------------------------------
    problem_images = []

    for img in target_images:
        problem_images.append({
            "path": str(img),
            "label": target_class,   # 내부 검증용
            "is_target": True,
        })

    for cls, img in selected_wrong_images:
        problem_images.append({
            "path": str(img),
            "label": cls,            # 내부 검증용
            "is_target": False,
        })

    random.shuffle(problem_images)

    # -------------------------------------------------
    # 5) 반환
    # -------------------------------------------------
    return {
        "question": f"{display_class}에 해당하는 이미지를 모두 고르세요.",
        "target_class": target_class,
        "display_class": display_class,
        "images": problem_images,
    }


# =====================================================
# 로컬 테스트
# =====================================================
if __name__ == "__main__":
    target = random.choice(list(PHASE_B_RULES.keys()))
    problem = generate_phase_b_problem(target)

    print(problem["question"])
    for img in problem["images"]:
        print(
            img["label"],
            "✔" if img["is_target"] else "✘",
            img["path"],
        )
