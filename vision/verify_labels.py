"""
Sanity check: draw the generated YOLO labels back onto their images so you
can visually confirm the boxes actually land on the right objects before
spending time training on them.

Run:
    python vision/verify_labels.py --n 5
"""

import argparse
import os
import random

import cv2

CLASSES = ["red_obj", "green_obj", "blue_obj", "yellow_obj"]
COLORS = {
    "red_obj": (0, 0, 255),
    "green_obj": (0, 200, 0),
    "blue_obj": (255, 0, 0),
    "yellow_obj": (0, 220, 220),
}


def draw_labels(img_path, lbl_path, out_path):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    if os.path.exists(lbl_path):
        with open(lbl_path) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    else:
        lines = []

    for line in lines:
        cls_id, cx, cy, bw, bh = line.split()
        cls_id = int(cls_id)
        cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
        x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
        x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
        label = CLASSES[cls_id]
        color = COLORS[label]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        cv2.putText(img, label, (x1, max(y1 - 4, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.imwrite(out_path, img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="synthetic_dataset")
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    img_dir = os.path.join(args.dataset, "images")
    lbl_dir = os.path.join(args.dataset, "labels")
    out_dir = os.path.join(args.dataset, "verify")
    os.makedirs(out_dir, exist_ok=True)

    all_imgs = sorted(os.listdir(img_dir))
    sample = random.sample(all_imgs, min(args.n, len(all_imgs)))

    for fname in sample:
        stem = os.path.splitext(fname)[0]
        img_path = os.path.join(img_dir, fname)
        lbl_path = os.path.join(lbl_dir, stem + ".txt")
        out_path = os.path.join(out_dir, fname)
        draw_labels(img_path, lbl_path, out_path)
        print(f"  {fname} -> {out_path}")

    print(f"\nOpen the images in {out_dir} and check the boxes visually.")