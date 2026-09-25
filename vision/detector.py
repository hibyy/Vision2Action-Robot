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


def draw_detections(rgb_image, detections):
    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR).copy()
    for det in detections:
        x, y, w, h = det["bbox"]
        color = COLOR_RANGES[det["label"]]["bgr_draw"]
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

    env = WarehouseEnv(gui=False)
    cam = OverheadCamera()
    rgb, depth, seg = cam.capture()

    detections = detect_by_color(rgb)
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