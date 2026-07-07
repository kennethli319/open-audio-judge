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
    assert "Actionable Notes" in html
    assert "Calibration Checks" in html
    assert "All calibration expectations matched" in html
    assert "No action needed." in html


def test_write_html_report_aggregates_researcher_notes(tmp_path: Path) -> None:
    results = [
        EvaluationResult(
            case_id="amount-1",
            task="asr_error",
            judge_id="asr_error",
            judge_version="0.1.0",
            provider="mock",
            overall_score=52,
            reason="The amount changed.",
            researcher_notes=["Improve numeric robustness.", "Audit payment-domain numbers."],
            label="inaccurate",
        ),
        EvaluationResult(
            case_id="amount-2",
            task="asr_error",
            judge_id="asr_error",
            judge_version="0.1.0",
            provider="mock",
            overall_score=55,
            reason="The date changed.",
            researcher_notes=["Improve numeric robustness."],
            label="inaccurate",
        ),
    ]

    output = write_html_report(results, tmp_path / "report.html")
    html = output.read_text(encoding="utf-8")

    assert "Actionable Notes" in html
    assert "Improve numeric robustness." in html
    assert "<strong>2</strong>" in html
    assert "Audit payment-domain numbers." in html


def test_write_html_report_aggregates_tts_candidate_metadata(tmp_path: Path) -> None:
    results = [
        EvaluationResult(
            case_id="tts-date-local-tts",
            task="tts_naturalness",
            judge_id="tts_naturalness",
            judge_version="0.1.0",
            provider="gemini",
            overall_score=70,
            reason="Audible pacing issue around the date.",
            semantic_error_summary="Date phrasing sounded hesitant.",
            error_categories=["prosody_issue"],
            label="needs_review",
            metadata={
                "eval_category": "information_tuning",
                "tts_slice": "dates_times",
                "synthesis_model": "mlx-community/chatterbox-turbo-6bit",
                "synthesis_voice": "af_heart",
                "synthesis_lang_code": "en",
                "sample_kind": "local_synthetic_tts",
                "source_case_id": "tts-date",
                "judge_sample_scores": [65, 75],
                "judge_sample_average": 70.0,
                "source_basis": (
                    "Seed-TTS-Eval intelligibility checks and TTSDS intelligibility/prosody "
                    "dimensions"
                ),
            },
        ),
        EvaluationResult(
            case_id="tts-code-local-tts",
            task="tts_naturalness",
            judge_id="tts_naturalness",
            judge_version="0.1.0",
            provider="gemini",
            overall_score=91,
            reason="Natural enough.",
            semantic_error_summary="No audible issues.",
            error_categories=["no_error"],
            label="accurate",
            metadata={
                "eval_category": "instruction_following",
                "tts_slice": "code_like",
                "synthesis_model": "mlx-community/chatterbox-turbo-6bit",
                "synthesis_voice": "af_heart",
                "language": "en-US",
                "sample_kind": "local_synthetic_tts",
                "source_case_id": "tts-code",
            },
        ),
        EvaluationResult(
            case_id="tts-kokoro-date-local-tts",
            task="tts_naturalness",
            judge_id="tts_naturalness",
            judge_version="0.1.0",
            provider="gemini",
            overall_score=45,
            reason="The number and date were hard to understand.",
            semantic_error_summary="Dense facts became unclear.",
            error_categories=["intelligibility_issue", "number_error"],
            label="inaccurate",
            metadata={
                "eval_category": "information_tuning",
                "tts_slice": "dates_times",
                "synthesis_model": "mlx-community/Kokoro-82M-4bit",
                "synthesis_voice": "af_heart",
                "synthesis_lang_code": "en",
                "sample_kind": "local_synthetic_tts",
                "source_case_id": "tts-date",
                "source_basis": (
                    "Seed-TTS-Eval intelligibility checks and TTSDS intelligibility/prosody "
                    "dimensions"
                ),
            },
        ),
        EvaluationResult(
            case_id="tts-json-local-tts",
            task="tts_naturalness",
            judge_id="tts_naturalness",
            judge_version="0.1.0",
            provider="gemini",
            overall_score=1,
            reason="Evaluation failed: upstream timeout",
            error_categories=[],
            label="inaccurate",
            status="provider_error",
            error="upstream timeout",
            metadata={
                "eval_category": "storytelling",
                "tts_slice": "structured_json",
                "synthesis_model": "mlx-community/chatterbox-turbo-6bit",
                "synthesis_voice": "af_heart",
                "synthesis_lang_code": "en",
                "sample_kind": "local_synthetic_tts",
            },
        ),
    ]

    output = write_html_report(results, tmp_path / "report.html")
    html = output.read_text(encoding="utf-8")

    assert "Candidate Metadata" in html
    assert "TTS Slice" in html
    assert "dates times" in html
    assert "code like" in html
    assert "Synthesis Model" in html
    assert "mlx-community/chatterbox-turbo-6bit" in html
    assert "Synthesis Voice" in html
    assert "af heart" in html
    assert "Language" in html
    assert "en-US" in html
    assert "Evaluation Category" in html
    assert "information tuning" in html
    assert "instruction following" in html
    assert "Sample Kind" in html
    assert "local synthetic tts" in html
    assert "Issues By Category" in html
    assert "information tuning / prosody issue" in html
    assert "Issues By TTS Slice" in html
    assert "Failures By TTS Slice" in html
    assert "Failures By Category" in html
    assert "storytelling / provider error" in html
    assert "structured json / provider error" in html
    assert "Issues By Model" in html
    assert "mlx-community/chatterbox-turbo-6bit / prosody issue" in html
    assert "Issues By Voice" in html
    assert "af heart / prosody issue" in html
    assert "Issues By Language" in html
    assert "en / prosody issue" in html
    assert "Scores By TTS Slice" in html
    assert "Scores By Category" in html
    assert "Scores By Model" in html
    assert "Scores By Voice" in html
    assert "Scores By Language" in html
    assert "avg 57.5 / n 2 / 45-70" in html
    assert "avg 80.5 / n 2 / 70-91" in html
    assert "Failures By Model" in html
    assert "mlx-community/chatterbox-turbo-6bit / provider error" in html
    assert "Failures By Voice" in html
    assert "af heart / provider error" in html
    assert "Failures By Language" in html
    assert "en / provider error" in html
    assert "Failures By Sample Kind" in html
    assert "local synthetic tts / provider error" in html
    assert "dates times / prosody issue" in html
    assert "judge samples: 65, 75; avg 70.00" in html
    assert "Weakest Segments" in html
    assert "Model-Category Action Matrix" in html
    assert "Category Guidance" in html
    assert "numbers, dates, units, ordered steps, and safety-critical wording" in html
    assert "numeric/date normalization" in html
    assert "entity and unit pronunciation" in html
    assert "Seed-TTS-Eval intelligibility checks" in html
    assert "mlx-community/Kokoro-82M-4bit" in html
    assert "intelligibility" in html
    assert "text faithfulness" in html
    assert "information tuning" in html
    assert "avg 45.0" in html
    assert "tts-kokoro-date-local-tts" in html
    assert 'id="case-search"' in html
    assert 'id="case-label-filter"' in html
    assert 'id="case-status-filter"' in html
    assert 'id="case-model-filter"' in html
    assert 'id="case-category-filter"' in html
    assert 'id="case-slice-filter"' in html
    assert 'id="case-visible-count"' in html
    assert 'data-sort="score"' in html
    assert 'data-sort="case"' in html
    assert 'data-score="45"' in html
    assert 'data-label="inaccurate"' in html
    assert 'data-status="provider_error"' in html
    assert 'data-model="mlx-community/Kokoro-82M-4bit"' in html
    assert 'data-category="information_tuning"' in html
    assert 'data-slice="dates_times"' in html
    assert "All categories" in html
    assert "All slices" in html
    assert "The number and date were hard to understand." in html
    assert "Baseline Model Deltas" in html
    assert "Compared Model" in html
    assert "Wins / Ties / Losses" in html
    assert "-25.0" in html
    assert "-25" in html
    assert "tts-date" in html
    assert "45 vs baseline 70" in html
    assert "Baseline Regression Slices" in html
    assert "Evaluation Category: information tuning" in html
    assert "TTS Slice: dates times" in html
    assert "Regression Examples" in html


def test_write_html_report_shows_sample_provenance_per_row(tmp_path: Path) -> None:
    result = EvaluationResult(
        case_id="tts-date-local-tts",
        task="tts_naturalness",
        judge_id="tts_naturalness",
        judge_version="0.1.0",
        provider="gemini",
        overall_score=72,
        reason="Mostly natural with a slight pause.",
        error_categories=["prosody_issue"],
        label="needs_review",
        metadata={
            "tts_slice": "dates_times",
            "synthesis_model": "mlx-community/chatterbox-turbo-6bit",
            "synthesis_voice": "af_heart",
            "synthesis_lang_code": "en",
            "sample_kind": "local_synthetic_tts",
            "source_case_id": "tts-evalset-row-001",
            "reference_text_sha256": "abc123",
            "audio_duration_seconds": 1.234,
            "audio_bytes": 4096,
        },
    )

    output = write_html_report([result], tmp_path / "report.html")
    html = output.read_text(encoding="utf-8")

    assert "<th>Provenance</th>" in html
    assert 'data-label="Provenance"' in html
    assert "dates times" in html
    assert "mlx-community/chatterbox-turbo-6bit" in html
    assert "af heart" in html
    assert "local synthetic tts" in html
    assert "tts-evalset-row-001" in html
    assert "abc123" in html
    assert "1.23s" in html
    assert "4096" in html


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
        EvaluationResult(
            case_id="dosage-unit",
            task="asr_error",
            judge_id="asr_error",
            judge_version="0.1.0",
            provider="mock",
            overall_score=50,
            reason="The dosage unit changed.",
            meaning_preservation="partial_loss",
            semantic_error_summary="The transcript changes the dosage unit.",
            error_categories=["unit_error"],
            label="inaccurate",
        ),
    ]

    output = write_html_report(results, tmp_path / "report.html")
    html = output.read_text(encoding="utf-8")

    assert "Priority Cases" in html
    assert "low-wer-negation" in html
    assert "negation error" in html
    assert "unit error" in html
    assert "High-Impact Errors" in html
    assert "The transcript reverses the operational instruction." in html


def test_write_html_report_flags_calibration_mismatches(tmp_path: Path) -> None:
    result = EvaluationResult(
        case_id="calibration-number",
        task="asr_error",
        judge_id="asr_error",
        judge_version="0.1.0",
        provider="mock",
        overall_score=88,
        reason="Looks mostly correct.",
        meaning_preservation="preserved",
        semantic_error_summary="Meaning is preserved.",
        error_categories=["no_error"],
        label="accurate",
        metadata={
            "calibration_focus": "low_wer_high_semantic_severity",
            "expected_meaning_preservation": "partial_loss",
            "expected_error_categories": ["number_error"],
        },
    )

    output = write_html_report([result], tmp_path / "report.html")
    html = output.read_text(encoding="utf-8")

    assert "Calibration Checks" in html
    assert "calibration-number" in html
    assert "low wer high semantic severity" in html
    assert "expected meaning partial loss, got preserved" in html
    assert "missing categories: number error" in html
