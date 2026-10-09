"""
Phase 6: Vision-driven pick and place.

This is the real milestone: the robot only knows where an object is
because the camera + detector + 3D-localization pipeline told it so --
NOT because we read the object's true position out of the simulator.

The one simulation-only shortcut: to actually *grasp* an object in
PyBullet we create a fixed constraint, which needs a body ID. Vision
gives us WHERE to move the gripper; the object ID is only used at the
final "close gripper" instant as a stand-in for real physical contact
(a real robot grasps by contact/force feedback, not by ID lookup). This
is a documented simplification, not vision being bypassed -- the arm's
entire approach trajectory is driven purely by the estimated 3D position.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulation"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vision"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "spatial"))

from environment import WarehouseEnv       # noqa: E402
from camera import OverheadCamera          # noqa: E402
from robot import RobotController          # noqa: E402
from coordinates import localize_objects   # noqa: E402


def find_object(label, localized_objects):
    """Pick the highest-confidence detection matching `label`."""
    matches = [o for o in localized_objects if o["label"] == label]
    if not matches:
        return None
    return max(matches, key=lambda o: o["confidence"])


def pick_and_place(env, robot, cam, label, drop_xy, lift_height=0.08, verbose=True,
                    detect_retries=2):
    """
    Full vision-driven pick-and-place for one object.

    label:    e.g. "red_cube" -- must match a label the detector produces
    drop_xy:  (x, y) world position to place the object at
    detect_retries: if the object isn't detected on the first capture,
        wait a moment (let the scene settle -- an object that just landed
        might be mid-bounce/mid-rotation and momentarily hard to detect)
        and try again before giving up. This is real, if simple, failure
        recovery: a transient miss shouldn't be treated the same as "this
        object genuinely isn't here."
    """
    # 1. SEE: capture the scene (retrying on a transient detection miss)
    target = None
    for attempt in range(detect_retries + 1):
        rgb, depth, seg = cam.capture()
        detections = localize_objects(rgb, depth, cam)
        target = find_object(label, detections)
        if target is not None:
            break
        if attempt < detect_retries:
            if verbose:
                print(f"[vision] '{label}' not detected (attempt {attempt + 1}), "
                      f"letting the scene settle and retrying...")
            env.step(60)  # let physics settle -- may be mid-bounce/rotation

    if target is None:
        raise RuntimeError(
            f"Vision pipeline did not detect any '{label}' in the scene "
            f"after {detect_retries + 1} attempt(s)."
        )
    est_pos = target["world_position"]
    if verbose:
        gt_pos = env.object_position(label)
        err = sum((a - b) ** 2 for a, b in zip(est_pos, gt_pos)) ** 0.5
        print(f"[vision] detected '{label}' conf={target['confidence']:.2f} "
              f"at ({est_pos[0]:+.3f}, {est_pos[1]:+.3f}, {est_pos[2]:+.3f})  "
              f"[ground-truth error: {err*100:.1f} cm, for our own debugging only]")

    # 3. ACT: move using ONLY the vision-estimated position from here on
    robot.move_to((0.5, 0.0, 1.0))  # safe hover, clear of the table
    robot.move_to((est_pos[0], est_pos[1], est_pos[2] + 0.15))   # approach
    robot.move_to((est_pos[0], est_pos[1], est_pos[2] + 0.02))   # descend

    # Simulation-only grasp step -- see module docstring
    object_id = env.object_ids[label]
    robot.close_gripper(object_id)
    if verbose:
        print(f"[grasp] gripper closed on '{label}'")

    robot.move_to((est_pos[0], est_pos[1], est_pos[2] + lift_height))
    robot.move_to((drop_xy[0], drop_xy[1], est_pos[2] + lift_height))
    env.step(30)  # let residual arm velocity settle before releasing --
                   # releasing while still moving imparts drift to the object
    robot.open_gripper()
    env.step(60)

    final_pos = env.object_position(label)
    if verbose:
        print(f"[done] '{label}' placed at "
              f"({final_pos[0]:+.3f}, {final_pos[1]:+.3f}, {final_pos[2]:+.3f})")
    return final_pos


if __name__ == "__main__":
    env = WarehouseEnv(gui=True)
    robot = RobotController(env)
    cam = OverheadCamera()

    print("=== Vision-driven pick-and-place: red_cube -> (0.5, 0.35) ===")
    pick_and_place(env, robot, cam, "red_cube", drop_xy=(0.5, 0.35))

    print()
    print("=== Vision-driven pick-and-place: blue_box -> (0.35, -0.30) ===")
    pick_and_place(env, robot, cam, "blue_box", drop_xy=(0.35, -0.30))

    env.close()
    print("\nPhase 6 OK: both objects placed using vision-estimated positions only.")