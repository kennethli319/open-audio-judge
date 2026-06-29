from pathlib import Path

from open_audio_judge.prompting import load_prompt
from open_audio_judge.providers.mock import MockProvider
from open_audio_judge.runner import evaluate_cases, load_cases


def test_evaluate_cases_with_mock(tmp_path: Path) -> None:
    cases = load_cases(Path("examples/asr_cases.jsonl"))
    prompt = load_prompt("asr_error")
    results = evaluate_cases(cases, prompt, MockProvider(), tmp_path)

    assert len(results) == 2
    assert (tmp_path / "results.jsonl").exists()
    assert (tmp_path / "report.html").exists()
    assert all(result.status == "ok" for result in results)
    assert results[1].overall_score <= 55
    assert "number_error" in results[1].error_categories
