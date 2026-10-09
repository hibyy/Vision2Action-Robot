"""
Phase 10: Evaluation.

Runs a batch of instructions through the full pipeline and reports real
numbers -- success rate, placement error, timing -- instead of eyeballing
individual runs. This is the difference between "it worked when I tried
it" and a result you can actually put a number on for a report/CV.

Also demonstrates basic failure recovery: a single failed vision capture
(camera returns a frame where detection missed, e.g. mid-motion blur in a
real system, or a transient miss here) is retried a bounded number of
times before being counted as a real failure, rather than crashing the
whole batch on one bad frame.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulation"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tasks"))

from environment import WarehouseEnv        # noqa: E402
from robot import RobotController           # noqa: E402
from camera import OverheadCamera           # noqa: E402
from nl_pick_place import run_instruction   # noqa: E402
from task_parser import ParseError          # noqa: E402


def run_with_retry(env, robot, cam, instruction, parser=None, max_attempts=2, verbose=True):
    """
    Wraps run_instruction with a bounded retry: if it fails with a
    RuntimeError (e.g. "vision could not find target object" -- a
    transient detection miss, not a logic error), try again up to
    max_attempts times before giving up. ParseError is NOT retried --
    if the instruction genuinely can't be parsed, trying the exact same
    text again won't help; that's a real failure, not a transient one.
    """
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            success = run_instruction(env, robot, cam, instruction, parser=parser, verbose=verbose)
            return success, attempt, None
        except RuntimeError as e:
            last_error = str(e)
            if verbose and attempt < max_attempts:
                print(f"  [retry] attempt {attempt} failed ({e}), retrying...")
        except ParseError as e:
            return False, attempt, str(e)  # not retried, see docstring

    return False, max_attempts, last_error


def evaluate(instructions, parser=None, gui=False, verbose=False):
    """
    Runs every instruction in a FRESH environment each time (so results
    for instruction N aren't affected by wherever instruction N-1 left
    objects) and returns a results list plus a summary.
    """
    results = []

    for instr in instructions:
        env = WarehouseEnv(gui=gui)
        robot = RobotController(env)
        cam = OverheadCamera()

        start = time.time()
        success, attempts, error = run_with_retry(
            env, robot, cam, instr, parser=parser, verbose=verbose
        )
        elapsed = time.time() - start

        env.close()

        results.append({
            "instruction": instr,
            "success": success,
            "attempts": attempts,
            "error": error,
            "elapsed_s": round(elapsed, 2),
        })

    return results


def summarize(results):
    n = len(results)
    n_success = sum(1 for r in results if r["success"])
    avg_time = sum(r["elapsed_s"] for r in results) / n if n else 0
    retried = sum(1 for r in results if r["attempts"] > 1)

    lines = [
        f"Evaluated {n} instructions",
        f"Success rate: {n_success}/{n} ({100 * n_success / n:.0f}%)" if n else "No instructions run",
        f"Average time per instruction: {avg_time:.1f}s",
        f"Instructions that needed a retry: {retried}",
        "",
        "Per-instruction results:",
    ]
    for r in results:
        status = "OK" if r["success"] else "FAIL"
        detail = f" ({r['error']})" if r["error"] else ""
        lines.append(f"  [{status}] \"{r['instruction']}\" "
                      f"-- {r['attempts']} attempt(s), {r['elapsed_s']}s{detail}")

    return "\n".join(lines)


if __name__ == "__main__":
    test_instructions = [
        "put the red cube next to the blue box",
        "place red cube to the left of the blue box",
        "move the box to the right of the cube",
        "put the red cube somewhere nice",   # deliberately invalid -- should fail cleanly
    ]

    results = evaluate(test_instructions, verbose=True)
    print()
    print(summarize(results))