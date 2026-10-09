import os, sys
import cv2
sys.path.insert(0, os.path.dirname(__file__))
from detector import _get_model, LABEL_MAP

model = _get_model()

CLASSES = ["red_obj", "green_obj", "blue_obj", "yellow_obj"]
img_dir = "synthetic_dataset/images/val"
lbl_dir = "synthetic_dataset/labels/val"

for fname in sorted(os.listdir(img_dir))[:5]:
    stem = os.path.splitext(fname)[0]
    img_path = os.path.join(img_dir, fname)
    lbl_path = os.path.join(lbl_dir, stem + ".txt")

    rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    results = model.predict(rgb, conf=0.3, verbose=False)[0]

    with open(lbl_path) as f:
        gt_classes = sorted(set(CLASSES[int(line.split()[0])] for line in f if line.strip()))

    pred_classes = sorted(set(model.names[int(b.cls[0])] for b in results.boxes))

    print(f"{fname}: ground_truth={gt_classes}  predicted={pred_classes}")