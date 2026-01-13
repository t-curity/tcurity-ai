import os
from PIL import Image
from torch.utils.data import Dataset


class CAPTCHADataset(Dataset):
    """
    2단계 폴더 구조 지원 Dataset
    images/
      ├── animal/
      │       ├── cheetah/
      │       ├── dog/
      └── object/
              ├── gloves/
              ├── toaster/
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # 대그룹 자동 감지 (animal, object)
        group_folders = sorted(
            [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
        )

        # group → label 자동 매핑
        self.group_to_label = {group: idx for idx, group in enumerate(group_folders)}

        # 이미지 로딩
        for group_name, class_id in self.group_to_label.items():
            group_path = os.path.join(root_dir, group_name)

            for class_folder in sorted(os.listdir(group_path)):
                class_path = os.path.join(group_path, class_folder)
                if not os.path.isdir(class_path):
                    continue

                for img_name in sorted(os.listdir(class_path)):
                    if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.image_paths.append(os.path.join(class_path, img_name))
                        self.labels.append(class_id)

        print(f"총 {len(self.image_paths)}개 이미지 로딩 완료")
        for group, cid in self.group_to_label.items():
            print(f"{group}: {self.labels.count(cid)}개")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
