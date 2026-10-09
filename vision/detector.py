"""
Phase 3 (v1): Object detector.

Pretrained YOLO (COCO weights) doesn't know what a "red cube" or "blue box"
is -- COCO's 80 classes are everyday objects (person, cup, chair...), not
plain colored primitives. So for the MVP we detect objects the classical
way: HSV color thresholding + contours. This is deterministic, needs no
training data, and is enough to prove out the rest of the pipeline
(3D localization -> motion planning -> grasping).

detect_by_color() returns a list of detections in the same *shape* YOLO
would (label, confidence, bbox, pixel centroid), so swapping in a trained
YOLO model later (see notes/synthetic_dataset_plan.md) is a drop-in
replacement -- nothing downstream needs to change.
"""

import os

import cv2
import numpy as np


# HSV ranges tuned for the cube/box colors defined in simulation/environment.py
COLOR_RANGES = {
    "red_cube": {
        "lower": np.array([0, 120, 80]),
        "upper": np.array([10, 255, 255]),
        "bgr_draw": (0, 0, 255),
    },
    "blue_box": {
        "lower": np.array([100, 120, 80]),
        "upper": np.array([130, 255, 255]),
        "bgr_draw": (255, 0, 0),
    },
}


def detect_by_color(rgb_image, min_area=15, max_area=2500, min_fill=0.6):
    """
    rgb_image: HxWx3 uint8, RGB order (as returned by OverheadCamera.capture)
    Returns: list of dicts:
        {"label": str, "confidence": float, "bbox": (x, y, w, h),
         "centroid_px": (cx, cy)}

    max_area / min_fill exist specifically to reject large, sparse blobs
    like the robot arm's own light-blue plastic, which is a similar hue to
    blue_box but forms a big, irregular (low bbox-fill) region rather than
    a small solid square.
    """
    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    detections = []
    for label, spec in COLOR_RANGES.items():
        mask = cv2.inRange(hsv, spec["lower"], spec["upper"])
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w / 2, y + h / 2

            # confidence: how "filled" the bbox is by the mask -- a solid
            # cube fills its bbox almost completely; a stray arm/limb
            # silhouette does not.
            confidence = float(area / (w * h + 1e-6))
            if confidence < min_fill:
                continue

            detections.append(
                {
                    "label": label,
                    "confidence": round(min(confidence, 0.99), 2),
                    "bbox": (x, y, w, h),
                    "centroid_px": (cx, cy),
                }
            )
    return detections


# Class names the trained model outputs -> names the rest of the pipeline
# (pick_place.py, COLOR_RANGES) uses. Add entries here if you retrain.
LABEL_ALIASES = {"red_obj": "red_cube", "blue_obj": "blue_box"}

DEFAULT_WEIGHTS = os.path.join(os.path.dirname(__file__), "weights", "best.pt")


def load_yolo(weights_path=DEFAULT_WEIGHTS):
    """Load the trained YOLO model once and reuse it (loading is slow)."""
    from ultralytics import YOLO

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")
    model = YOLO(weights_path)
    print(f"[detector] loaded {weights_path} with classes: {model.names}")
    return model


def detect_by_yolo(rgb_image, model, conf=0.25, imgsz=640):
    """
    rgb_image: HxWx3 uint8 in RGB order (as returned by OverheadCamera.capture).

    IMPORTANT: Ultralytics treats a raw numpy array as BGR (OpenCV order),
    but PyBullet gives RGB. Passing it straight in swaps red and blue, which
    makes a model trained on correct images call the red cube "blue_box" and
    vice versa. We convert to BGR here so the model sees what it saw in
    training. (Passing a file path works in a notebook only because the
    file loader does this conversion for you.)

    Returns the same dict format as detect_by_color().
    """
    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    result = model.predict(bgr, conf=conf, imgsz=imgsz, verbose=False)[0]

    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x, y = int(x1), int(y1)
        w, h = int(x2 - x1), int(y2 - y1)
        label = result.names[int(box.cls[0])]
        label = LABEL_ALIASES.get(label, label)
        detections.append(
            {
                "label": label,
                "confidence": round(float(box.conf[0]), 2),
                "bbox": (x, y, w, h),
                "centroid_px": ((x1 + x2) / 2, (y1 + y2) / 2),
            }
        )
    return detections


def check_label_colors(rgb_image, detections):
    """
    Sanity check: warn if a detection's label disagrees with the dominant
    red/blue colour inside its box. Catches RGB/BGR mix-ups immediately.
    """
    for det in detections:
        x, y, w, h = det["bbox"]
        patch = rgb_image[max(y, 0):y + h, max(x, 0):x + w]
        if patch.size == 0:
            continue
        r, b = patch[..., 0].mean(), patch[..., 2].mean()
        seen = "red" if r > b else "blue"
        expected = "red" if det["label"].startswith("red") else "blue"
        if seen != expected:
            print(
                f"[detector] WARNING: '{det['label']}' box looks {seen} "
                f"(R={r:.0f}, B={b:.0f}) - check channel order / weights"
            )


_model = None


def detect_objects(rgb_image, use_yolo=True, weights_path=DEFAULT_WEIGHTS, **kw):
    """
    Single entry point for the pipeline. Uses YOLO if the weights exist,
    otherwise falls back to the HSV colour detector.
    """
    global _model
    if use_yolo and os.path.exists(weights_path):
        if _model is None:
            _model = load_yolo(weights_path)
        dets = detect_by_yolo(rgb_image, _model, **kw)
        check_label_colors(rgb_image, dets)
        return dets
    return detect_by_color(rgb_image)


def draw_detections(rgb_image, detections):
    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR).copy()
    for det in detections:
        x, y, w, h = det["bbox"]
        color = COLOR_RANGES.get(det["label"], {}).get("bgr_draw", (0, 255, 0))
        cv2.rectangle(bgr, (x, y), (x + w, y + h), color, 2)
        text = f"{det['label']} {det['confidence']:.2f}"
        cv2.putText(
            bgr, text, (x, max(y - 6, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )
    return bgr


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulation"))
    from environment import WarehouseEnv
    from camera import OverheadCamera

    env = WarehouseEnv(gui=True)
    cam = OverheadCamera()
    rgb, depth, seg = cam.capture()

    detections = detect_objects(rgb)
    print(f"Found {len(detections)} detections:")
    for d in detections:
        print(f"  {d['label']:10s} conf={d['confidence']:.2f} "
              f"bbox={d['bbox']} centroid_px={d['centroid_px']}")

    annotated = draw_detections(rgb, detections)
    out_dir = os.path.join(os.path.dirname(__file__), "..", "experiments")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "phase3_detections.png")
    cv2.imwrite(out_path, annotated)

    env.close()
    print(f"Phase 3 OK: saved {out_path}")