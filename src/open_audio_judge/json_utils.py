from __future__ import annotations

import json
import re
from typing import Any

from open_audio_judge.models import JudgeOutput


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fence_match = _FENCE_RE.match(cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = json.loads(_first_balanced_object(cleaned))

    if not isinstance(parsed, dict):
        raise ValueError("Judge output must be a JSON object.")
    return parsed


def parse_judge_output(text: str) -> JudgeOutput:
    data = extract_json_object(text)

    score = (
        data.get("overall_score")
        if "overall_score" in data
        else data.get("overallScore", data.get("score"))
    )
    reason = data.get("reason", data.get("rationale", data.get("explanation")))

    if score is None:
        raise ValueError("Judge JSON is missing overall_score.")
    if reason is None:
        raise ValueError("Judge JSON is missing reason.")

    return JudgeOutput(overall_score=int(score), reason=str(reason).strip())


def _first_balanced_object(text: str) -> str:
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in judge output.")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    raise ValueError("No balanced JSON object found in judge output.")
