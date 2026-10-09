"""
Synthetic dataset generator for YOLO training.

Spawns randomized scenes (random object count, class, position, rotation)
in PyBullet, captures an overhead RGB image, and derives YOLO-format
bounding box labels directly from PyBullet's segmentation mask -- no
manual annotation needed.

v2 changes (fixing a real-scene detection bug found after the first
training run):
- Objects are now sometimes spawned axis-aligned (yaw=0), not always
  rotated -- environment.py's actual objects are always axis-aligned,
  and v1's all-rotated training data underrepresented that case.
- The robot arm is now sometimes posed at environment.py's exact
  HOME_POSE (not just random joint angles) -- guarantees the model has
  seen the precise arm configuration used at rest in the real pipeline,
  while still seeing random poses for when the arm is mid-motion.
The arm is still never labeled (see build_random_scene) -- it's a
visual-only background distractor.

Run:
    python vision/dataset_gen.py --n 200
"""

import argparse
import os
import random

import cv2
import numpy as np
import pybullet as p
import pybullet_data

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulation"))
from camera import OverheadCamera  # noqa: E402


CLASSES = ["red_obj", "green_obj", "blue_obj", "yellow_obj"]
CLASS_RGBA = {
    "red_obj": [0.8, 0.1, 0.1, 1],
    "green_obj": [0.1, 0.7, 0.1, 1],
    "blue_obj": [0.1, 0.2, 0.8, 1],
    "yellow_obj": [0.9, 0.8, 0.1, 1],
}

TABLE_TOP_Z = 0.62
TABLE_X_RANGE = (0.30, 0.70)
TABLE_Y_RANGE = (-0.35, 0.35)

# The real arm home pose from simulation/environment.py -- kept in sync
# manually since this is a separate, standalone data-gen script.
HOME_POSE = [0.0, 0.26, 0.0, -0.64, 0.0, 2.16, 0.0]


def build_random_scene(min_objects=3, max_objects=8):
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")

    table_half = [0.35, 0.45, 0.02]
    table_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=table_half)
    table_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=table_half,
                                     rgbaColor=[0.55, 0.4, 0.25, 1])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=table_col,
                       baseVisualShapeIndex=table_vis,
                       basePosition=[0.5, 0, TABLE_TOP_Z - table_half[2]])

    # Robot arm as a visual-only background distractor. It gets NO label --
    # the point is teaching the model to see it and ignore it. 40% of the
    # time it's posed at the exact real HOME_POSE (so the model has
    # definitely seen the real resting configuration); otherwise a random
    # pose (so it also generalizes to the arm mid-motion during a pick).
    robot_id = p.loadURDF("kuka_iiwa/model.urdf", basePosition=[0, 0, 0],
                           useFixedBase=True)

    if random.random() < 0.4:
        for j, target in enumerate(HOME_POSE):
            p.resetJointState(robot_id, j, target)
    else:
        for j in range(p.getNumJoints(robot_id)):
            info = p.getJointInfo(robot_id, j)
            lo, hi = info[8], info[9]
            p.resetJointState(robot_id, j, random.uniform(lo, hi))

    n_objects = random.randint(min_objects, max_objects)
    body_to_class = {}

    for _ in range(n_objects):
        cls = random.choice(CLASSES)
        half = random.uniform(0.02, 0.035)
        x = random.uniform(*TABLE_X_RANGE)
        y = random.uniform(*TABLE_Y_RANGE)

        # 70% rotated (general case), 30% axis-aligned (yaw=0) -- matches
        # environment.py's real objects, which are always axis-aligned and
        # were underrepresented in the first (all-rotated) training set.
        yaw = random.uniform(0, 3.14159) if random.random() < 0.7 else 0.0

        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half, half, half])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[half, half, half],
                                   rgbaColor=CLASS_RGBA[cls])
        body_id = p.createMultiBody(
            baseMass=0.05,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[x, y, TABLE_TOP_Z + half + 0.01],
            baseOrientation=p.getQuaternionFromEuler([0, 0, yaw]),
        )
        body_to_class[body_id] = cls

    for _ in range(120):
        p.stepSimulation()

    return body_to_class


def seg_to_yolo_labels(seg, body_to_class, img_w, img_h):
    """Derive a YOLO-format label line per visible object from the
    segmentation mask: class_id x_center y_center width height (all
    normalized 0-1)."""
    lines = []
    for body_id, cls in body_to_class.items():
        mask = (seg == body_id)
        if not mask.any():
            continue  # object occluded/out of frame -- skip, don't fake a box
        ys, xs = np.where(mask)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        # Skip slivers (a handful of visible pixels isn't a usable training box)
        if (x_max - x_min) < 3 or (y_max - y_min) < 3:
            continue

        cx = (x_min + x_max) / 2 / img_w
        cy = (y_min + y_max) / 2 / img_h
        w = (x_max - x_min) / img_w
        h = (y_max - y_min) / img_h

        class_id = CLASSES.index(cls)
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def generate_dataset(n_scenes, out_dir):
    img_dir = os.path.join(out_dir, "images")
    lbl_dir = os.path.join(out_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    with open(os.path.join(out_dir, "classes.txt"), "w") as f:
        f.write("\n".join(CLASSES))

    for i in range(n_scenes):
        body_to_class = build_random_scene()
        cam = OverheadCamera()
        rgb, depth, seg = cam.capture()

        labels = seg_to_yolo_labels(seg, body_to_class, cam.width, cam.height)

        img_path = os.path.join(img_dir, f"scene_{i:05d}.png")
        lbl_path = os.path.join(lbl_dir, f"scene_{i:05d}.txt")
        cv2.imwrite(img_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        with open(lbl_path, "w") as f:
            f.write("\n".join(labels))

        p.disconnect()

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{n_scenes} scenes generated")

    print(f"Done: {n_scenes} scenes -> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--out", type=str, default="synthetic_dataset")
    args = parser.parse_args()

    generate_dataset(args.n, args.out)