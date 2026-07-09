import importlib.util
import sys
from pathlib import Path

import pytest

from open_audio_judge.models import EvaluationCase


SCRIPT = Path("scripts/validate_asr_seed_manifest.py")


def load_script_module():
    spec = importlib.util.spec_from_file_location("validate_asr_seed_manifest", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def seed_case(case_id: str, category: str, asr_slice: str) -> EvaluationCase:
    return EvaluationCase(
        id=case_id,
        task="asr_error",
        reference_text="Read back the verified public-safe seed sentence.",
        metadata={
            "language": "en",
            "eval_category": category,
            "asr_slice": asr_slice,
            "source": "research-backed-asr-demo",
            "source_basis": "Unit test source basis.",
            "expected_error_focus": "Unit test focus.",
            "requires_audio_materialization": True,
        },
    )


def test_validate_asr_seed_manifest_summarizes_balanced_cases() -> None:
    module = load_script_module()
    cases = [
        seed_case("case-a-1", "category_a", "slice_a_1"),
        seed_case("case-a-2", "category_a", "slice_a_2"),
        seed_case("case-b-1", "category_b", "slice_b_1"),
        seed_case("case-b-2", "category_b", "slice_b_2"),
    ]

    summary = module.validate_asr_seed_manifest(
        cases,
        cases_path=Path("examples/asr_research_cases.jsonl"),
        expected_cases_per_category=2,
    )

    assert summary["status"] == "complete"
    assert summary["total_cases"] == 4
    assert summary["category_count"] == 2
    assert summary["categories"] == {"category_a": 2, "category_b": 2}
    assert summary["requires_audio_materialization"] == 4


def test_validate_asr_seed_manifest_rejects_unbalanced_categories() -> None:
    module = load_script_module()
    cases = [
        seed_case("case-a-1", "category_a", "slice_a_1"),
        seed_case("case-a-2", "category_a", "slice_a_2"),
        seed_case("case-b-1", "category_b", "slice_b_1"),
    ]

    with pytest.raises(ValueError, match="Expected exactly 2 cases per eval_category"):
        module.validate_asr_seed_manifest(
            cases,
            cases_path=Path("examples/asr_research_cases.jsonl"),
            expected_cases_per_category=2,
        )


def test_validate_asr_seed_manifest_rejects_audio_and_candidate_text() -> None:
    module = load_script_module()
    case = seed_case("case-a-1", "category_a", "slice_a_1").model_copy(
        update={
            "audio_path": "runs/asr-research-audio/case-a-1.wav",
            "candidate_text": "candidate transcript",
        }
    )

    with pytest.raises(ValueError, match="must not point to materialized audio"):
        module.validate_asr_seed_manifest(
            [case],
            cases_path=Path("examples/asr_research_cases.jsonl"),
            expected_cases_per_category=1,
        )
