"""
Phase 8: Spatial reasoning.

Turns raw (x, y, z) positions -- from the vision pipeline (Phases 3-4), NOT
ground truth -- into natural-language spatial relations between objects.
This is deliberately plain Python logic, not a learned model: the roadmap
calls for starting simple here and only reaching for a learned spatial
model later if the rule-based version proves insufficient.

Reference frame (robot base at the world origin):
    +x = farther from the robot (deeper onto the table)
    -x = nearer to the robot
    +y = to the robot's LEFT
    -y = to the robot's RIGHT

This choice is arbitrary but fixed and documented -- it does NOT match
"left/right as a camera or bystander would see it" (that depends on
camera orientation). Pick whichever frame your consumer (Phase 9's
LLM parser, a UI, etc.) actually expects, and keep it consistent.
"""

import math
import os
import sys


# Minimum difference (m) along an axis before we call it a relation at all,
# rather than "roughly aligned" -- avoids claiming "A is left of B" when
# they're 2mm apart on that axis (noise, not a real spatial fact).
AXIS_THRESHOLD = 0.03

# Distance (m) below which two objects are described as "next to" each
# other rather than just "near" or "far apart". Calibrated against this
# system's measured placement accuracy (~5-8cm real-world error from
# IK/reach imprecision, see pick_place.py) rather than picked arbitrarily
# -- a tighter threshold would flag physically successful placements as
# failures due to precision the arm doesn't actually have.
NEXT_TO_THRESHOLD = 0.15
NEAR_THRESHOLD = 0.30


def relation_between(pos_a, pos_b, label_a="object A", label_b="object B"):
    """
    Returns a dict describing the spatial relationship of A relative to B:
        {
            "horizontal": "left" | "right" | None,
            "depth": "farther" | "nearer" | None,
            "distance_m": float,
            "proximity": "next_to" | "near" | "far",
            "sentence": "<natural language description>",
        }
    """
    dx = pos_a[0] - pos_b[0]
    dy = pos_a[1] - pos_b[1]
    dist = math.sqrt(dx ** 2 + dy ** 2 + (pos_a[2] - pos_b[2]) ** 2)

    horizontal = None
    if dy > AXIS_THRESHOLD:
        horizontal = "left"
    elif dy < -AXIS_THRESHOLD:
        horizontal = "right"

    depth = None
    if dx > AXIS_THRESHOLD:
        depth = "farther"
    elif dx < -AXIS_THRESHOLD:
        depth = "nearer"

    if dist < NEXT_TO_THRESHOLD:
        proximity = "next_to"
    elif dist < NEAR_THRESHOLD:
        proximity = "near"
    else:
        proximity = "far"

    sentence = _build_sentence(label_a, label_b, horizontal, depth, proximity)

    return {
        "horizontal": horizontal,
        "depth": depth,
        "distance_m": round(dist, 3),
        "proximity": proximity,
        "sentence": sentence,
    }


def _build_sentence(label_a, label_b, horizontal, depth, proximity):
    if proximity == "next_to":
        return f"{label_a} is right next to {label_b}"

    clauses = []
    if horizontal:
        clauses.append(f"to the {horizontal} of {label_b}")
    if depth:
        depth_phrase = (
            f"farther from the robot than {label_b}"
            if depth == "farther"
            else f"closer to the robot than {label_b}"
        )
        clauses.append(depth_phrase)

    if not clauses:
        return f"{label_a} is at roughly the same spot as {label_b}"

    return f"{label_a} is " + " and ".join(clauses)


def describe_scene(objects):
    """
    objects: list of {"label": str, "world_position": (x, y, z)}
             (this is exactly the shape localize_objects() from Phase 4
             returns, so vision output plugs straight in)

    Returns a list of relation sentences, one per unordered pair.
    """
    sentences = []
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            a, b = objects[i], objects[j]
            rel = relation_between(a["world_position"], b["world_position"],
                                    a["label"], b["label"])
            sentences.append(rel["sentence"])
    return sentences


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulation"))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vision"))

    from environment import WarehouseEnv   # noqa: E402
    from camera import OverheadCamera      # noqa: E402
    from coordinates import localize_objects  # noqa: E402

    env = WarehouseEnv(gui=True)
    cam = OverheadCamera()
    rgb, depth, seg = cam.capture()

    # Vision-derived positions, same as Phase 6 -- NOT ground truth
    objects = localize_objects(rgb, depth, cam)
    print("Vision-detected objects:")
    for obj in objects:
        pos = obj["world_position"]
        print(f"  {obj['label']:10s} at ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})")

    print("\nSpatial relations:")
    for sentence in describe_scene(objects):
        print(f"  - {sentence}")

    env.close()
    print("\nPhase 8 OK.")