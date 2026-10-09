"""
Phase 9: Natural language instruction parsing (rule-based).

Turns a plain-English instruction into the structured command
pick_and_place() (Phase 6) already knows how to execute:

    "put the red cube next to the blue box"
    -> {"object": "red_cube", "relation": "next_to", "target": "blue_box"}

This is intentionally rule-based (keyword/regex matching), not an LLM
call -- per the original roadmap, natural language is meant to be added
AFTER the rest of the pipeline works, and a rule-based parser keeps the
whole project self-contained with no API key / cost / network dependency.
Swapping this module for an LLM-based parser later is a drop-in
replacement: LLM_parse_instruction() would just need to return the same
{"object", "relation", "target"} dict shape.

Only understands the vocabulary the rest of the project already uses
(object labels from vision, relations from spatial/reasoning.py) -- it
is not a general NLU system, and says so honestly when it can't parse
something rather than guessing.
"""

import re


# Maps the words a person might actually say to the exact labels the
# vision pipeline produces (vision/detector.py's COLOR_RANGES keys).
OBJECT_ALIASES = {
    "red cube": "red_cube",
    "red_cube": "red_cube",
    "the cube": "red_cube",
    "cube": "red_cube",
    "blue box": "blue_box",
    "blue_box": "blue_box",
    "the box": "blue_box",
    "box": "blue_box",
}

# Maps relation phrases to the same relation vocabulary produced by
# spatial/reasoning.py's relation_between(), so a parsed instruction and
# a described scene always speak the same language.
RELATION_PATTERNS = [
    (r"\bnext to\b", "next_to"),
    (r"\bnear\b", "next_to"),
    (r"\bbeside\b", "next_to"),
    (r"\bto the left of\b", "left"),
    (r"\bleft of\b", "left"),
    (r"\bto the right of\b", "right"),
    (r"\bright of\b", "right"),
]

ACTION_WORDS = ["put", "place", "move", "pick"]


class ParseError(Exception):
    """Raised when the instruction can't be confidently parsed -- the
    parser is honest about its limits rather than guessing."""
    pass


def _find_object(text, exclude_span=None):
    """Finds the first object alias mentioned in text, longest alias first
    so "red cube" matches before the bare "cube" would."""
    for alias in sorted(OBJECT_ALIASES, key=len, reverse=True):
        match = re.search(re.escape(alias), text)
        if match:
            if exclude_span and match.start() >= exclude_span[0] and match.start() < exclude_span[1]:
                continue
            return OBJECT_ALIASES[alias], match.span()
    return None, None


def parse_instruction(text):
    """
    Returns {"object": str, "relation": str, "target": str}

    Raises ParseError with a clear reason if the instruction doesn't
    contain a recognizable action + object + relation + target.
    """
    original = text
    text = text.lower().strip()

    if not any(word in text for word in ACTION_WORDS):
        raise ParseError(
            f"No recognized action (put/place/move/pick) in: '{original}'"
        )

    relation = None
    relation_span = None
    for pattern, rel_name in RELATION_PATTERNS:
        match = re.search(pattern, text)
        if match:
            relation = rel_name
            relation_span = match.span()
            break

    if relation is None:
        raise ParseError(
            f"No recognized spatial relation (next to / left of / right of) in: '{original}'"
        )

    # The object being moved is mentioned BEFORE the relation phrase;
    # the target it's being moved relative to is mentioned AFTER it.
    before_text = text[:relation_span[0]]
    after_text = text[relation_span[1]:]

    obj_label, _ = _find_object(before_text)
    target_label, _ = _find_object(after_text)

    if obj_label is None:
        raise ParseError(f"Could not identify which object to move in: '{original}'")
    if target_label is None:
        raise ParseError(f"Could not identify the reference object in: '{original}'")
    if obj_label == target_label:
        raise ParseError(
            f"Object and target resolved to the same thing ('{obj_label}') in: '{original}'"
        )

    return {"object": obj_label, "relation": relation, "target": target_label}


if __name__ == "__main__":
    test_instructions = [
        "put the red cube next to the blue box",
        "place red cube to the left of the blue box",
        "move the box to the right of the cube",
        "pick the cube and put it near the box",
        "make the robot dance",              # should fail: no relation
        "put the red cube somewhere nice",     # should fail: no relation/target
    ]

    for instr in test_instructions:
        print(f"'{instr}'")
        try:
            result = parse_instruction(instr)
            print(f"   -> {result}")
        except ParseError as e:
            print(f"   -> PARSE FAILED: {e}")
        print()