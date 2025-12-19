"""
app/inference/phase_b_test_verifier.py

[test_verifier.py - 개발자용 Phase B CAPTCHA 검증 테스트 스크립트]

- [phase_b_image_verifier.py(PhaseBImageVerifier)]를 사용해
  Phase B 이미지 선택 검증 로직이 정상 동작하는지 테스트한다.
- 정답 이미지 선택 - [PASS 기대] / 오답 섞인 선택 - [FAIL 기대]
  두 가지 시나리오를 로컬에서 실행한다.

※ 이 파일은 API에서 호출되지 않음.
※ 서비스용 로직은 [phase_b_image_verifier.py]에 있으며, 이 파일은 검증/테스트 목적의 실행 스크립트임.

"""


import os
import random
from pathlib import Path
from pprint import pprint

from phase_b_image_verifier import PhaseBImageVerifier


# =====================================
# 프로젝트 루트 계산 (유지)
# =====================================
HERE = Path(__file__).resolve()
INFERENCE_DIR = HERE.parent
APP_DIR = INFERENCE_DIR.parent
PROJECT_ROOT = APP_DIR.parent


# =====================================
# 환경변수 기반 경로 (★ 핵심 ★)
# =====================================
IMAGE_DATA_ROOT = os.environ.get("IMAGE_DATA_ROOT")
MODEL_OUTPUT_ROOT = os.environ.get("MODEL_OUTPUT_ROOT")

if not IMAGE_DATA_ROOT:
    raise RuntimeError("IMAGE_DATA_ROOT 환경변수가 설정되지 않았습니다.")

if not MODEL_OUTPUT_ROOT:
    raise RuntimeError("MODEL_OUTPUT_ROOT 환경변수가 설정되지 않았습니다.")

IMAGES_DIR = Path(IMAGE_DATA_ROOT).resolve()

# ✅ Phase B 모델 경로 고정
MODEL_PATH = (Path(MODEL_OUTPUT_ROOT) / "phase_b" / "best_model.pth").resolve()


# =====================================
# label_map (학습 기준과 반드시 동일)
# =====================================
LABEL_MAP = {
    0: "Animals",
    1: "Birds",
    2: "Building",
    3: "Devices",
    4: "Fashion",
    5: "Food",
    6: "Instrument",
    7: "Nature",
    8: "Sports",
    9: "Vehicle",
}


# =====================================
# 이미지 수집 유틸
# =====================================
def collect_images(group: str):
    exts = {".jpg", ".jpeg", ".png"}
    base = IMAGES_DIR / group
    if not base.exists():
        raise RuntimeError(f"그룹 폴더 없음: {base}")
    return [p for p in base.rglob("*") if p.suffix.lower() in exts]


def pick_from_group(group: str, n: int):
    imgs = collect_images(group)
    if len(imgs) < n:
        raise RuntimeError(f"{group} 이미지 부족: {len(imgs)}장")
    return random.sample(imgs, n)


def pick_mixed(n: int):
    pool = []
    for g in LABEL_MAP.values():
        pool.extend(collect_images(g))
    return random.sample(pool, n)


# =====================================
# 메인 테스트
# =====================================
def main():
    random.seed()

    classifier = PhaseBImageVerifier(
        model_path=str(MODEL_PATH),
        label_map=LABEL_MAP,
    )

    target_label = random.choice(list(LABEL_MAP.values()))
    threshold = 0.5
    n_selected = 4

    print("\n===============================")
    print("랜덤 CAPTCHA 검증 테스트")
    print("target_label:", target_label)
    print("===============================")

    # -------- TEST A (정답만 선택 → PASS 기대) --------
    selected_pass = pick_from_group(target_label, n_selected)

    print("\n[TEST A - PASS 기대]")
    for p in selected_pass:
        print(" -", p)

    result_a = classifier.verify_selected(
        selected_images=[str(p) for p in selected_pass],
        target_label=target_label,
        threshold=threshold,
    )
    pprint(result_a)

    # -------- TEST B (섞어서 선택 → FAIL 기대) --------
    selected_fail = pick_mixed(n_selected)

    print("\n[TEST B - FAIL 기대]")
    for p in selected_fail:
        print(" -", p)

    result_b = classifier.verify_selected(
        selected_images=[str(p) for p in selected_fail],
        target_label=target_label,
        threshold=threshold,
    )
    pprint(result_b)


if __name__ == "__main__":
    main()
