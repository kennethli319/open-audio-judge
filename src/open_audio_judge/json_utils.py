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

    raw_score = (
        data.get("overall_score")
        if "overall_score" in data
        else data.get("overallScore", data.get("score"))
    )
    raw_reason = data.get("reason", data.get("rationale", data.get("explanation")))

    if raw_score is None:
        raise ValueError("Judge JSON is missing overall_score.")
    score = _parse_score(raw_score)

    if raw_reason is None:
        raise ValueError("Judge JSON is missing reason.")
    reason = str(raw_reason).strip()
    if not reason:
        raise ValueError("Judge JSON reason must not be empty.")

    return JudgeOutput(
        overall_score=score,
        reason=reason,
        judge_transcript=_optional_string(
            data.get("judge_transcript", data.get("judgeTranscript", data.get("own_transcript")))
        ),
        meaning_preservation=_optional_string(
            data.get("meaning_preservation", data.get("meaningPreservation"))
        ),
        semantic_error_summary=_optional_string(
            data.get("semantic_error_summary", data.get("semanticErrorSummary"))
        ),
        key_differences=_string_list(data.get("key_differences", data.get("keyDifferences"))),
        error_categories=_string_list(data.get("error_categories", data.get("errorCategories"))),
        researcher_notes=_string_list(data.get("researcher_notes", data.get("researcherNotes"))),
    )


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _parse_score(value: Any) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Judge JSON overall_score must be an integer from 1 to 100.") from exc

    if not 1 <= score <= 100:
        raise ValueError("Judge JSON overall_score must be an integer from 1 to 100.")
    return score


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
