from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from open_audio_judge.models import EvaluationCase


DEFAULT_TTS_CATEGORY_HINTS = {
    "ambiguity_clarification",
    "benign_safety_helpfulness",
    "calibration_confidence",
    "context_synthesis",
    "cross_lingual_transfer",
    "function_calling",
    "instruction_constraints",
    "long_context_retrieval",
    "multilingual_understanding",
    "multi_turn_state",
    "privacy_redaction",
    "quantitative_math",
    "safety_refusal",
    "structured_output",
    "temporal_reasoning",
    "unknown_handling",
}

DEFAULT_TTS_KEYWORD_HINTS = {
    "arithmetic",
    "calendar",
    "code",
    "date",
    "format",
    "json",
    "long_context",
    "memory",
    "multilingual",
    "number",
    "privacy",
    "punctuation",
    "safety",
    "translation",
    "time",
}


@dataclass(frozen=True)
class TtsCaseSummary:
    total_cases: int
    by_slice: dict[str, int]
    by_source_category: dict[str, int]
    requires_synthesis: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_cases": self.total_cases,
            "by_slice": self.by_slice,
            "by_source_category": self.by_source_category,
            "requires_synthesis": self.requires_synthesis,
        }


def load_evalset_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            records.append(record)
    return records


def build_tts_cases(
    records: Iterable[dict[str, Any]],
    *,
    source_name: str,
    limit: int | None = None,
    category_filter: set[str] | None = None,
    slice_filter: set[str] | None = None,
    per_slice_limit: int | None = None,
) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    slice_counts: Counter[str] = Counter()
    for record in records:
        if category_filter and str(record.get("category", "")) not in category_filter:
            continue
        if not is_tts_slice_candidate(record):
            continue
        case = tts_case_from_evalset_record(record, source_name=source_name)
        if case is None:
            continue
        tts_slice = str(case.metadata.get("tts_slice", "general_response"))
        if slice_filter and tts_slice not in slice_filter:
            continue
        if per_slice_limit is not None and slice_counts[tts_slice] >= per_slice_limit:
            continue
        cases.append(case)
        slice_counts[tts_slice] += 1
        if limit is not None and len(cases) >= limit:
            break
    return cases


def is_tts_slice_candidate(record: dict[str, Any]) -> bool:
    category = str(record.get("category", ""))
    if category in DEFAULT_TTS_CATEGORY_HINTS:
        return True

    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    tags = metadata.get("tags", []) if isinstance(metadata, dict) else []
    joined_hints = " ".join(
        str(value).lower()
        for value in [
            category,
            record.get("task", ""),
            record.get("ideal_answer", ""),
            *tags,
        ]
    )
    return any(keyword in joined_hints for keyword in DEFAULT_TTS_KEYWORD_HINTS)


def tts_case_from_evalset_record(
    record: dict[str, Any],
    *,
    source_name: str,
) -> EvaluationCase | None:
    target_text = str(record.get("ideal_answer") or "").strip()
    if not target_text:
        return None

    source_id = str(record.get("id") or "unknown")
    turns = record.get("turns") if isinstance(record.get("turns"), list) else []
    normalized_turns = [
        {
            "role": str(turn.get("role", "user")),
            "content": str(turn.get("content", "")),
        }
        for turn in turns
        if isinstance(turn, dict) and str(turn.get("content", "")).strip()
    ]
    if not normalized_turns:
        normalized_turns = [{"role": "user", "content": "Read the target text aloud exactly."}]

    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    tags = metadata.get("tags", []) if isinstance(metadata, dict) else []
    case_metadata = {
        "source": source_name,
        "source_id": source_id,
        "source_version": record.get("version"),
        "source_category": record.get("category"),
        "source_task": record.get("task"),
        "source_tags": tags,
        "tts_slice": classify_tts_slice(record, target_text),
        "requires_synthesis": True,
    }

    return EvaluationCase(
        id=f"tts-{source_name}-{_slugify(source_id)}",
        task="tts_naturalness",
        turns=normalized_turns,
        reference_text=target_text,
        metadata=case_metadata,
    )


def write_cases_jsonl(cases: Iterable[EvaluationCase], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case.model_dump(exclude_none=True), ensure_ascii=False) + "\n")
    return path


def summarize_tts_cases(cases: Iterable[EvaluationCase]) -> TtsCaseSummary:
    case_list = list(cases)
    return TtsCaseSummary(
        total_cases=len(case_list),
        by_slice=_sorted_counts(
            str(case.metadata.get("tts_slice") or "unknown")
            for case in case_list
        ),
        by_source_category=_sorted_counts(
            str(case.metadata.get("source_category") or "unknown")
            for case in case_list
        ),
        requires_synthesis=sum(1 for case in case_list if case.metadata.get("requires_synthesis") is True),
    )


def write_tts_summary_json(cases: Iterable[EvaluationCase], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_tts_cases(cases)
    path.write_text(json.dumps(summary.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def classify_tts_slice(record: dict[str, Any], target_text: str) -> str:
    category = str(record.get("category", "")).lower()
    task = str(record.get("task", "")).lower()
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    tags = [str(tag).lower() for tag in metadata.get("tags", [])] if metadata else []
    hints = " ".join([category, task, *tags, target_text.lower()])

    if (
        "multilingual" in hints
        or "cross_lingual" in hints
        or "translation" in hints
        or "spanish" in hints
        or "chinese" in hints
    ):
        return "multilingual"
    if "json" in hints or "code" in hints or _looks_code_like(target_text):
        return "code_like"
    if "date" in hints or "time" in hints or re.search(r"\b\d{1,2}:\d{2}\b", target_text):
        return "dates_times"
    if "number" in hints or re.search(r"\d", target_text):
        return "numbers"
    if "punctuation" in hints or "format" in hints or "\n" in target_text:
        return "punctuation_format"
    if "long_context" in hints or len(target_text) > 220:
        return "long_context"
    if "safety" in hints or "privacy" in hints:
        return "safety_privacy"
    return "general_response"


def _looks_code_like(text: str) -> bool:
    return bool(re.search(r"[{}()[\]_;=]|```|</?[a-zA-Z][^>]*>", text))


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "unknown"


def _sorted_counts(values: Iterable[str]) -> dict[str, int]:
    counts = Counter(values)
    return {key: counts[key] for key in sorted(counts)}
