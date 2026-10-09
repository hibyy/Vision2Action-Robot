"""
Phase 9b: Local LLM-based instruction parsing with Ollama.

Drop-in replacement for task_parser.parse_instruction().

Uses a local Ollama model instead of a paid external API.

Input:
    - natural-language instruction
    - objects currently detected in the scene
    - allowed spatial relations

Output:
    {"object": "...", "relation": "...", "target": "..."}

The robot never receives direct instructions from the LLM.
The LLM only converts natural language into a validated structured command.

BUG FIX vs. the previous version: the original OUTPUT_SCHEMA required
{"object", "relation", "target"} and nothing else. Because format=SCHEMA
uses constrained decoding, the model is structurally FORCED to always
produce that exact shape -- it has no way to ever emit {"error": "..."}
instead, no matter what the system prompt says. That's why "make the
robot dance" silently produced a fabricated command instead of failing:
the schema made failure impossible to express, not just unlikely.

Fix: the schema now always requires a `valid` boolean plus a `reason`
string, alongside object/relation/target (which are set to empty strings
when valid=false). This keeps every response schema-valid while still
giving the model an actual way to say "I can't map this."
"""

import json

from ollama import chat

from task_parser import ParseError  # noqa: E402  -- SAME exception type as the
# rule-based parser. Using a locally-defined ParseError here was a real bug:
# nl_pick_place_llm.py's except clause only catches task_parser.ParseError,
# so a locally-raised one of the same NAME but different CLASS slips right
# past it and crashes the whole script instead of being handled as a
# failed instruction. Two classes named the same thing are not the same
# exception to Python's except -- this is why the crash happened even
# though the except clause "looked" like it should have caught it.


MODEL = "qwen3:4b"

SYSTEM_PROMPT = """You are a robot instruction parser.

Your ONLY job is to convert a natural-language pick-and-place instruction
into a structured command.

You must ALWAYS return all five fields: valid, object, relation, target, reason.

If the instruction IS a valid pick-and-place request:
- valid: true
- object: the object to move (must be from the provided object list)
- relation: the spatial relation (must be from the provided relation list)
- target: the reference object (must be from the provided object list, and
  different from `object`)
- reason: "" (empty string)

If the instruction is NOT a valid pick-and-place request (wrong/missing
object, no clear spatial relation, object and target would be the same,
or it's not a pick-and-place instruction at all):
- valid: false
- object: "" (empty string)
- relation: "" (empty string)
- target: "" (empty string)
- reason: a short explanation of why it could not be parsed

IMPORTANT:
- Only use objects from the provided object list. Never invent one.
- Only use relations from the provided relation list. Never invent one.
- When in doubt, set valid to false rather than guessing.

Return ONLY valid JSON matching the schema. No other text.
"""


OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "valid": {"type": "boolean"},
        "object": {"type": "string"},
        "relation": {"type": "string"},
        "target": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": [
        "valid",
        "object",
        "relation",
        "target",
        "reason",
    ],
}


def llm_parse_instruction(
    text,
    known_objects,
    known_relations=("next_to", "left", "right"),
    model=MODEL,
):
    """
    Parse a natural-language robot instruction using a local Ollama model.

    known_objects:
        Object labels currently detected in the scene.

    known_relations:
        Spatial relations currently supported by the robot.

    Returns:
        {
            "object": str,
            "relation": str,
            "target": str
        }

    Raises:
        ParseError on invalid input, malformed output, LLM-declared
        invalid instruction, or LLM/connection failure.
    """

    if not text or not text.strip():
        raise ParseError("Instruction is empty.")

    if not known_objects:
        raise ParseError("No objects are currently available in the scene.")

    user_content = (
        f"Instruction: {text}\n"
        f"Objects present: {list(known_objects)}\n"
        f"Relations understood: {list(known_relations)}"
    )

    try:
        response = chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            format=OUTPUT_SCHEMA,
            think=False,
        )

    except Exception as e:
        raise ParseError(f"Local LLM failed: {e}")

    raw_text = response.message.content.strip()

    try:
        parsed = json.loads(raw_text)

    except json.JSONDecodeError:
        raise ParseError(
            f"LLM did not return valid JSON: {raw_text!r}"
        )

    required_keys = ("valid", "object", "relation", "target", "reason")

    missing = [
        key for key in required_keys
        if key not in parsed
    ]

    if missing:
        raise ParseError(
            f"LLM response missing keys {missing}: {parsed}"
        )

    # -------------------------
    # The model's own validity signal -- this is the actual fix. Checked
    # BEFORE any other validation, since object/relation/target are
    # meaningless (empty strings, by design) when valid=false.
    # -------------------------
    if not parsed["valid"]:
        reason = parsed["reason"] or "instruction could not be mapped to a command"
        raise ParseError(f"LLM declined to parse instruction: {reason}")

    object_label = parsed["object"]
    relation = parsed["relation"]
    target = parsed["target"]

    # -------------------------
    # Deterministic validation (still needed even when valid=true --
    # the model can still hallucinate an object/relation outside the
    # provided lists despite instructions not to)
    # -------------------------

    if object_label not in known_objects:
        raise ParseError(
            f"LLM returned an object not in the scene: "
            f"'{object_label}'"
        )

    if target not in known_objects:
        raise ParseError(
            f"LLM returned a target not in the scene: "
            f"'{target}'"
        )

    if relation not in known_relations:
        raise ParseError(
            f"LLM returned an unsupported relation: "
            f"'{relation}'"
        )

    if object_label == target:
        raise ParseError(
            f"Object and target are the same: '{object_label}'"
        )

    return {
        "object": object_label,
        "relation": relation,
        "target": target,
    }


if __name__ == "__main__":

    test_instructions = [
        "put the red cube next to the blue box",
        "grab the red cube and place it to the left of the blue box",
        "move the blue box to the right of the red cube",
        "make the robot dance",
        "put the red cube somewhere nice",  # no clear relation -- should also fail
    ]

    known_objects = [
        "red_cube",
        "blue_box",
    ]

    print("=== Local LLM instruction parser ===")
    print(f"Model: {MODEL}")
    print()

    for instruction in test_instructions:

        print(f"Instruction: {instruction}")

        try:
            result = llm_parse_instruction(
                instruction,
                known_objects,
            )

            print(f"Parsed: {result}")

        except ParseError as e:

            print(f"PARSE FAILED: {e}")

        print()