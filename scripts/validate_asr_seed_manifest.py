from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from open_audio_judge.models import EvaluationCase  # noqa: E402
from open_audio_judge.runner import load_cases  # noqa: E402


DEFAULT_CASES = ROOT / "examples" / "asr_research_cases.jsonl"
REQUIRED_METADATA_FIELDS = (
    "language",
    "eval_category",
    "asr_slice",
    "source",
    "source_basis",
    "expected_error_focus",
    "requires_audio_materialization",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the public-safe ASR seed manifest before audio materialization.",
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument(
        "--expected-cases-per-category",
        type=int,
        default=5,
        help="Require exactly this many seed cases in every ASR eval category.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        help="Optional JSON summary artifact for CI or cron logs.",
    )
    args = parser.parse_args()

    cases = load_cases(args.cases)
    summary = validate_asr_seed_manifest(
        cases,
        cases_path=args.cases,
        expected_cases_per_category=args.expected_cases_per_category,
    )
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        "Validated "
        f"{summary['total_cases']} ASR seed cases across {summary['category_count']} categories "
        f"({args.expected_cases_per_category} each)."
    )


def validate_asr_seed_manifest(
    cases: list[EvaluationCase],
    *,
    cases_path: Path,
    expected_cases_per_category: int,
) -> dict[str, Any]:
    if expected_cases_per_category < 1:
        raise ValueError("expected_cases_per_category must be at least 1.")
    if not cases:
        raise ValueError("ASR seed manifest is empty.")

    ids = [case.id for case in cases]
    duplicate_ids = sorted(case_id for case_id, count in Counter(ids).items() if count > 1)
    if duplicate_ids:
        raise ValueError(f"Duplicate ASR seed case ids: {duplicate_ids}")

    issues: list[str] = []
    categories: Counter[str] = Counter()
    slices: Counter[str] = Counter()
    languages: Counter[str] = Counter()

    for case in cases:
        _validate_case_contract(case, issues)
        metadata = case.metadata
        categories[str(metadata.get("eval_category") or "")] += 1
        slices[str(metadata.get("asr_slice") or "")] += 1
        languages[str(metadata.get("language") or "")] += 1

    bad_counts = {
        category: count
        for category, count in sorted(categories.items())
        if category and count != expected_cases_per_category
    }
    if bad_counts:
        issues.append(
            "Expected exactly "
            f"{expected_cases_per_category} cases per eval_category; observed {bad_counts}."
        )

    missing_categories = [category for category in categories if not category]
    if missing_categories:
        issues.append("Every case must include metadata.eval_category.")

    if issues:
        raise ValueError("ASR seed manifest validation failed: " + " ".join(issues))

    return {
        "status": "complete",
        "cases_path": _repo_relative(cases_path),
        "total_cases": len(cases),
        "category_count": len(categories),
        "expected_cases_per_category": expected_cases_per_category,
        "categories": dict(sorted(categories.items())),
        "asr_slices": dict(sorted(slices.items())),
        "languages": dict(sorted(languages.items())),
        "requires_audio_materialization": sum(
            1 for case in cases if case.metadata.get("requires_audio_materialization") is True
        ),
    }


def _validate_case_contract(case: EvaluationCase, issues: list[str]) -> None:
    if case.task != "asr_error":
        issues.append(f"{case.id}: task must be asr_error.")
    if not case.reference_text:
        issues.append(f"{case.id}: reference_text is required.")
    if case.audio_path or case.audio_url:
        issues.append(f"{case.id}: seed manifest must not point to materialized audio.")
    if case.candidate_text:
        issues.append(f"{case.id}: seed manifest must not include candidate_text.")

    metadata = case.metadata
    for field in REQUIRED_METADATA_FIELDS:
        value = metadata.get(field)
        if value is None or value == "":
            issues.append(f"{case.id}: metadata.{field} is required.")
    if metadata.get("source") != "research-backed-asr-demo":
        issues.append(f"{case.id}: metadata.source must be research-backed-asr-demo.")
    if metadata.get("requires_audio_materialization") is not True:
        issues.append(f"{case.id}: metadata.requires_audio_materialization must be true.")


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    main()
