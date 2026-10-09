"""
Phase 9: Natural-language pick-and-place, end to end.

    "put the red cube next to the blue box"
        -> parsed by task_parser.py
        -> target position found via the vision pipeline (Phase 3-4)
        -> a valid drop position computed from the requested relation
        -> executed via pick_and_place() (Phase 6)
        -> the RESULT is re-observed by vision and checked with
           spatial/reasoning.py (Phase 8) against what was actually asked

That last step matters: it's the difference between "the robot did
something" and "the robot did what was asked" -- closing the loop with
the same reasoning module that describes scenes, rather than just
trusting the geometry math worked.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulation"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vision"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "spatial"))

from environment import WarehouseEnv        # noqa: E402
from camera import OverheadCamera           # noqa: E402
from robot import RobotController           # noqa: E402
from coordinates import localize_objects    # noqa: E402
from reasoning import relation_between      # noqa: E402
from task_parser import parse_instruction, ParseError  # noqa: E402
from pick_place import pick_and_place, find_object      # noqa: E402


# How far (m) to place the moved object from the target when satisfying
# a relation. Small enough to be unambiguous, large enough that the arm/
# gripper geometry doesn't cause the objects to collide on placement.
RELATION_OFFSET = 0.15


def compute_drop_position(relation, target_pos):
    """
    Given the requested relation ("next_to" / "left" / "right") and the
    TARGET object's current (x, y, z), compute an (x, y) drop position for
    the object being moved that will satisfy the relation.

    Uses the exact same frame convention as spatial/reasoning.py:
    +y = the robot's left, +x = farther from the robot.
    """
    tx, ty = target_pos[0], target_pos[1]

    if relation == "next_to":
        # Offset along y (keeping x at the target's own position). Root
        # cause of earlier placement drift was transport height pushing
        # near the arm's reach limit (fixed in pick_and_place's
        # lift_height) -- with that fixed, a modest offset here is enough.
        return (tx, ty + RELATION_OFFSET * 0.6)
    elif relation == "left":
        return (tx, ty + RELATION_OFFSET)
    elif relation == "right":
        return (tx, ty - RELATION_OFFSET)
    else:
        raise ValueError(f"Unknown relation: {relation}")


def run_instruction(env, robot, cam, instruction, parser=None, verbose=True):
    """
    parser: callable(instruction_text, known_objects) -> {"object",
    "relation", "target"}. Defaults to the rule-based parser (which
    ignores known_objects). Pass llm_parse_instruction from
    llm_task_parser.py to swap in the LLM version -- nothing else in
    this function changes.

    known_objects is determined by a live vision detection pass, not a
    hardcoded list -- this is what makes the LLM parser actually more
    flexible than the rule-based one: it only ever sees the objects
    genuinely present in the scene right now, so it adapts automatically
    if you add/remove objects in simulation without touching any code
    here.
    """
    if parser is None:
        parser = lambda text, known_objects: parse_instruction(text)

    if verbose:
        print(f'Instruction: "{instruction}"')

    rgb0, depth0, seg0 = cam.capture()
    scene_now = localize_objects(rgb0, depth0, cam)
    known_objects = sorted({o["label"] for o in scene_now})

    command = parser(instruction, known_objects)
    if verbose:
        print(f"  Parsed: {command}")

    # Find the TARGET object's current position via vision (not ground truth)
    rgb, depth, seg = cam.capture()
    detections = localize_objects(rgb, depth, cam)
    target_obj = find_object(command["target"], detections)
    if target_obj is None:
        raise RuntimeError(f"Vision could not find target object '{command['target']}'")

    drop_xy = compute_drop_position(command["relation"], target_obj["world_position"])
    if verbose:
        print(f"  Target '{command['target']}' seen at "
              f"({target_obj['world_position'][0]:+.3f}, {target_obj['world_position'][1]:+.3f}); "
              f"computed drop position ({drop_xy[0]:+.3f}, {drop_xy[1]:+.3f})")

    pick_and_place(env, robot, cam, command["object"], drop_xy, verbose=verbose)
    robot.park()  # clear the camera's view before re-observing (see park() docstring)

    # VERIFY: re-observe the scene and check the actual relation achieved
    rgb2, depth2, seg2 = cam.capture()
    final_detections = localize_objects(rgb2, depth2, cam)
    moved_obj = find_object(command["object"], final_detections)
    target_obj_final = find_object(command["target"], final_detections)

    achieved = relation_between(
        moved_obj["world_position"], target_obj_final["world_position"],
        command["object"], command["target"],
    )

    requested_satisfied = (
        achieved["proximity"] == "next_to" if command["relation"] == "next_to"
        else achieved["horizontal"] == command["relation"]
    )

    if verbose:
        print(f"  Verification: {achieved['sentence']}")
        print(f"  Instruction satisfied: {requested_satisfied}")
        print()

    return requested_satisfied


if __name__ == "__main__":
    env = WarehouseEnv(gui=True)
    robot = RobotController(env)
    cam = OverheadCamera()

    instructions = [
        "put the red cube next to the blue box",
        "place red cube to the left of the blue box",
        "move the box to the right of the cube",
    ]

    results = []
    for instr in instructions:
        try:
            ok = run_instruction(env, robot, cam, instr)
            results.append(ok)
        except (ParseError, RuntimeError) as e:
            print(f'Instruction: "{instr}"\n  FAILED: {e}\n')
            results.append(False)

    env.close()
    n_ok = sum(results)
    print(f"Phase 9 OK: {n_ok}/{len(results)} instructions correctly satisfied.")