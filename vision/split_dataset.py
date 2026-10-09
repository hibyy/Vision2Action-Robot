"""
Splits synthetic_dataset into train/val subsets in YOLO's expected layout:

synthetic_dataset/
├── images/train/  images/val/
├── labels/train/  labels/val/
└── data.yaml

Run:
    python vision/split_dataset.py
"""

import os
import random
import shutil

CLASSES = ["red_obj", "green_obj", "blue_obj", "yellow_obj"]
VAL_FRACTION = 0.15


def split(dataset_dir="synthetic_dataset", seed=42):
    img_dir = os.path.join(dataset_dir, "images")
    lbl_dir = os.path.join(dataset_dir, "labels")

    all_stems = sorted(
        os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith(".png")
    )
    random.Random(seed).shuffle(all_stems)

    n_val = int(len(all_stems) * VAL_FRACTION)
    val_stems = set(all_stems[:n_val])

    for split_name in ("train", "val"):
        os.makedirs(os.path.join(img_dir, split_name), exist_ok=True)
        os.makedirs(os.path.join(lbl_dir, split_name), exist_ok=True)

    for stem in all_stems:
        split_name = "val" if stem in val_stems else "train"
        shutil.move(os.path.join(img_dir, stem + ".png"),
                    os.path.join(img_dir, split_name, stem + ".png"))
        shutil.move(os.path.join(lbl_dir, stem + ".txt"),
                    os.path.join(lbl_dir, split_name, stem + ".txt"))

    yaml_content = f"""path: .
train: images/train
val: images/val

names:
"""
    for i, cls in enumerate(CLASSES):
        yaml_content += f"  {i}: {cls}\n"

    with open(os.path.join(dataset_dir, "data.yaml"), "w") as f:
        f.write(yaml_content)

    print(f"Train: {len(all_stems) - n_val} images, Val: {n_val} images")
    print(f"Wrote {dataset_dir}/data.yaml")


if __name__ == "__main__":
    split()