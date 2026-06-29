from pathlib import Path

from open_audio_judge.prompting import load_prompt
from open_audio_judge.providers.mock import MockProvider
from open_audio_judge.runner import evaluate_cases, load_cases


def test_evaluate_cases_with_mock(tmp_path: Path) -> None:
    cases = load_cases(Path("examples/asr_cases.jsonl"))
    prompt = load_prompt("asr_error")
    results = evaluate_cases(cases, prompt, MockProvider(), tmp_path)

    assert len(results) == len(cases)
    assert (tmp_path / "results.jsonl").exists()
    assert (tmp_path / "report.html").exists()
    assert all(result.status == "ok" for result in results)
    by_id = {result.case_id: result for result in results}
    assert by_id["asr-demo-002"].overall_score <= 55
    assert "number_error" in by_id["asr-demo-002"].error_categories
    assert by_id["asr-calibration-negation-001"].overall_score <= 40
    assert "negation_error" in by_id["asr-calibration-negation-001"].error_categories
    assert "entity_error" in by_id["asr-calibration-entity-001"].error_categories
    assert "date_time_error" in by_id["asr-calibration-date-001"].error_categories
    assert "date_time_error" in by_id["asr-calibration-time-001"].error_categories
    assert "unit_error" in by_id["asr-calibration-unit-001"].error_categories
