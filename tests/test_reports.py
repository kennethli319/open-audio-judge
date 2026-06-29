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
        judge_transcript="Transfer fifteen dollars.",
        meaning_preservation="preserved",
        semantic_error_summary="Meaning is preserved.",
        key_differences=["No meaningful difference."],
        error_categories=["no_error"],
        researcher_notes=["No action needed."],
        label="accurate",
    )
    output = write_html_report([result], tmp_path / "report.html")
    html = output.read_text(encoding="utf-8")

    assert "Open Audio Judge Report" in html
    assert "Mostly correct." in html
    assert "Meaning is preserved." in html
    assert "Meaning Preservation" in html
    assert "preserved" in html
    assert "Error Categories" in html
    assert "no error" in html
    assert "High-Impact Errors" in html
    assert "No high-impact semantic errors" in html
    assert "No action needed." in html


def test_write_html_report_highlights_priority_semantic_cases(tmp_path: Path) -> None:
    results = [
        EvaluationResult(
            case_id="low-wer-negation",
            task="asr_error",
            judge_id="asr_error",
            judge_version="0.1.0",
            provider="mock",
            overall_score=40,
            reason="A negation changed.",
            meaning_preservation="major_loss",
            semantic_error_summary="The transcript reverses the operational instruction.",
            error_categories=["negation_error"],
            label="inaccurate",
        ),
        EvaluationResult(
            case_id="wording-only",
            task="asr_error",
            judge_id="asr_error",
            judge_version="0.1.0",
            provider="mock",
            overall_score=91,
            reason="Small wording change.",
            meaning_preservation="preserved",
            semantic_error_summary="Meaning is preserved.",
            error_categories=["no_error"],
            label="accurate",
        ),
    ]

    output = write_html_report(results, tmp_path / "report.html")
    html = output.read_text(encoding="utf-8")

    assert "Priority Cases" in html
    assert "low-wer-negation" in html
    assert "negation error" in html
    assert "High-Impact Errors" in html
    assert "The transcript reverses the operational instruction." in html
