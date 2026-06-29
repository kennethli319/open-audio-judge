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


def parse_judge_output(text: str, response_schema: dict[str, Any] | None = None) -> JudgeOutput:
    data = extract_json_object(text)
    if response_schema:
        _validate_response_schema(data, response_schema)

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
    if isinstance(value, bool):
        raise ValueError("Judge JSON overall_score must be an integer from 1 to 100.")

    if isinstance(value, int):
        score = value
    elif isinstance(value, str):
        cleaned = value.strip()
        if not re.fullmatch(r"[+-]?\d+", cleaned):
            raise ValueError("Judge JSON overall_score must be an integer from 1 to 100.")
        score = int(cleaned)
    else:
        raise ValueError("Judge JSON overall_score must be an integer from 1 to 100.")

    if not 1 <= score <= 100:
        raise ValueError("Judge JSON overall_score must be an integer from 1 to 100.")
    return score


def _validate_response_schema(data: dict[str, Any], schema: dict[str, Any]) -> None:
    _validate_object_schema("response", data, schema)


def _validate_object_schema(field: str, data: dict[str, Any], schema: dict[str, Any]) -> None:
    required = schema.get("required", [])
    if isinstance(required, list):
        missing = [field for field in required if isinstance(field, str) and field not in data]
        if missing:
            joined = ", ".join(missing)
            if field == "response":
                raise ValueError(f"Judge JSON is missing required field(s): {joined}.")
            raise ValueError(f"Judge JSON {field} is missing required field(s): {joined}.")

    if schema.get("additionalProperties") is False:
        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            extra = sorted(key for key in data if key not in properties)
            if extra:
                joined = ", ".join(extra)
                if field == "response":
                    raise ValueError(f"Judge JSON includes unsupported field(s): {joined}.")
                raise ValueError(f"Judge JSON {field} includes unsupported field(s): {joined}.")

    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return
    for field, field_schema in properties.items():
        if field not in data or not isinstance(field_schema, dict):
            continue
        _validate_field_type(field, data[field], field_schema)


def _validate_field_type(field: str, value: Any, schema: dict[str, Any]) -> None:
    if "enum" in schema and value not in schema["enum"]:
        allowed = ", ".join(str(item) for item in schema["enum"])
        raise ValueError(f"Judge JSON {field} must be one of: {allowed}.")

    if "anyOf" in schema and isinstance(schema["anyOf"], list):
        if any(_matches_schema_type(value, option) for option in schema["anyOf"]):
            return
        raise ValueError(f"Judge JSON {field} does not match the response schema.")

    if not _matches_schema_type(value, schema):
        expected_type = schema.get("type", "the response schema")
        raise ValueError(f"Judge JSON {field} must match {expected_type}.")

    if schema.get("type") == "object" and isinstance(value, dict):
        _validate_object_schema(field, value, schema)

    _validate_numeric_bounds(field, value, schema)


def _matches_schema_type(value: Any, schema: dict[str, Any]) -> bool:
    expected_type = schema.get("type")
    if expected_type is None:
        return True
    if expected_type == "null":
        return value is None
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "array":
        if not isinstance(value, list):
            return False
        item_schema = schema.get("items")
        if not isinstance(item_schema, dict):
            return True
        return all(_matches_schema_type(item, item_schema) for item in value)
    if expected_type == "object":
        return isinstance(value, dict)
    return True


def _validate_numeric_bounds(field: str, value: Any, schema: dict[str, Any]) -> None:
    if schema.get("type") != "integer" or not isinstance(value, int) or isinstance(value, bool):
        return

    minimum = schema.get("minimum")
    if isinstance(minimum, int | float) and value < minimum:
        raise ValueError(f"Judge JSON {field} must be at least {minimum}.")

    maximum = schema.get("maximum")
    if isinstance(maximum, int | float) and value > maximum:
        raise ValueError(f"Judge JSON {field} must be at most {maximum}.")


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
