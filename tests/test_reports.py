from pathlib import Path

from open_audio_judge.models import EvaluationResult
from open_audio_judge.reports import label_for_score, write_html_report


def test_label_for_score() -> None:
    assert label_for_score(92) == "accurate"
    assert label_for_score(71) == "needs_review"
    assert label_for_score(12) == "inaccurate"


def test_write_html_report(tmp_path: Path) -> None:
    result = EvaluationResult(
        case_id="case-1",
        task="asr_error",
        judge_id="asr_error",
        judge_version="0.1.0",
        provider="mock",
        overall_score=82,
        reason="Mostly correct.",
        label="accurate",
    )
    output = write_html_report([result], tmp_path / "report.html")
    html = output.read_text(encoding="utf-8")

    assert "Open Audio Judge Report" in html
    assert "Mostly correct." in html
