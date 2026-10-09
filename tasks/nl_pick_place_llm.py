"""
Phase 9b demo: full pick-and-place loop using the LOCAL LLM parser
(Ollama) instead of the rule-based one.

Same run_instruction() from nl_pick_place.py, same robot/vision/planning
pipeline -- only the parser is swapped. This is the concrete proof that
the parser is a true drop-in: nothing else in the project needed to
change.

Requires Ollama running locally with the model pulled:
    ollama pull qwen3:4b
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulation"))

from environment import WarehouseEnv           # noqa: E402
from robot import RobotController              # noqa: E402
from camera import OverheadCamera              # noqa: E402
from nl_pick_place import run_instruction       # noqa: E402
from task_parser import ParseError              # noqa: E402
from llm_task_parser import llm_parse_instruction  # noqa: E402


if __name__ == "__main__":
    env = WarehouseEnv(gui=True)
    robot = RobotController(env)
    cam = OverheadCamera()

    # More varied phrasing than the rule-based parser could ever handle --
    # this is the actual point of swapping to an LLM.
    instructions = [
        "put the red cube next to the blue box",
        "grab the small red one and move it to the left of the box",
    ]

    results = []
    for instr in instructions:
        try:
            ok = run_instruction(env, robot, cam, instr, parser=llm_parse_instruction)
            results.append(ok)
        except (ParseError, RuntimeError) as e:
            print(f'Instruction: "{instr}"\n  FAILED: {e}\n')
            results.append(False)

    env.close()
    n_ok = sum(results)
    print(f"Phase 9b (local LLM) OK: {n_ok}/{len(results)} instructions correctly satisfied.")