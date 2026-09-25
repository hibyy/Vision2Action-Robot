"""
Phase 4: Detection (2D pixel) -> 3D world coordinate.

Combines:
- vision/detector.py     (pixel bbox + centroid per object)
- simulation/camera.py   (depth map + pixel_to_world projection)

to produce, for each detected object, a real (x, y, z) position the robot
can actually plan a motion to.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulation"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vision"))

from camera import OverheadCamera          # noqa: E402
from detector import detect_by_color       # noqa: E402


def localize_objects(rgb, depth, camera: OverheadCamera):
    """
    Returns a list of dicts, one per detected object:
        {"label": str, "confidence": float,
         "pixel_centroid": (cx, cy), "world_position": (x, y, z)}
    """
    detections = detect_by_color(rgb)
    results = []
    for det in detections:
        cx, cy = det["centroid_px"]
        px, py = int(round(cx)), int(round(cy))

        # Use the median depth over a small patch around the centroid --
        # more robust than a single noisy pixel.
        patch = depth[max(py - 2, 0):py + 3, max(px - 2, 0):px + 3]
        depth_m = float(patch.mean())

        world_pos = camera.pixel_to_world(cx, cy, depth_m)

        results.append(
            {
                "label": det["label"],
                "confidence": det["confidence"],
                "pixel_centroid": (cx, cy),
                "world_position": world_pos,
            }
        )
    return results


if __name__ == "__main__":
    from environment import WarehouseEnv  # noqa: E402

    env = WarehouseEnv(gui=False)
    cam = OverheadCamera()
    rgb, depth, seg = cam.capture()

    objects = localize_objects(rgb, depth, cam)

    print("Estimated 3D positions vs. ground truth:\n")
    for obj in objects:
        gt = env.object_position(obj["label"])
        est = obj["world_position"]
        err = (
            (gt[0] - est[0]) ** 2
            + (gt[1] - est[1]) ** 2
            + (gt[2] - est[2]) ** 2
        ) ** 0.5
        print(f"{obj['label']:10s} "
              f"estimated=({est[0]:+.3f}, {est[1]:+.3f}, {est[2]:+.3f})  "
              f"ground_truth=({gt[0]:+.3f}, {gt[1]:+.3f}, {gt[2]:+.3f})  "
              f"error={err*100:.1f} cm")

    env.close()