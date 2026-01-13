import os
import argparse
import uuid

IMG_EXTS = (".jpg", ".jpeg", ".png")

def rename_images(root: str) -> None:
    root = os.path.abspath(root)
    print(f"[시작] 이미지 루트 경로: {root}")

    if not os.path.isdir(root):
        raise NotADirectoryError(f"유효하지 않은 폴더입니다: {root}")

    for group in sorted(os.listdir(root)):
        group_path = os.path.join(root, group)
        if not os.path.isdir(group_path):
            continue

        class_count = 0
        renamed_count = 0

        for cls in sorted(os.listdir(group_path)):
            cls_path = os.path.join(group_path, cls)
            if not os.path.isdir(cls_path):
                continue

            files = [
                f for f in sorted(os.listdir(cls_path))
                if f.lower().endswith(IMG_EXTS)
            ]
            if not files:
                continue

            class_count += 1

            # 1) 임시 이름
            temp_names = []
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                tmp = f"__tmp__{uuid.uuid4().hex}{ext}"
                os.rename(os.path.join(cls_path, f), os.path.join(cls_path, tmp))
                temp_names.append(tmp)

            # 2) 최종 이름
            for i, tmp in enumerate(sorted(temp_names), start=1):
                ext = os.path.splitext(tmp)[1].lower()
                final_name = f"{cls}_{i}{ext}"
                os.rename(os.path.join(cls_path, tmp), os.path.join(cls_path, final_name))
                renamed_count += 1

        print(f"[그룹 처리 완료] {group} (클래스 {class_count}개, 변경 {renamed_count}개)")

    print("\n✔ 전체 리네이밍 작업 완료!")


def parse_args():
    parser = argparse.ArgumentParser(description="2단계 이미지 폴더 파일명 정규화")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="images 최상위 경로 (미지정 시 $IMAGE_DATA_ROOT 사용)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    data_dir = (
        args.data_dir
        or os.environ.get("IMAGE_DATA_ROOT")
    )

    if not data_dir:
        raise ValueError(
            "--data_dir 또는 IMAGE_DATA_ROOT 환경변수를 설정하세요."
        )

    rename_images(data_dir)
