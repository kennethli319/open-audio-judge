from open_audio_judge.asr_semantics import analyze_reference_candidate


def test_number_change_is_capped_as_semantic_error() -> None:
    diff = analyze_reference_candidate(
        "Please transfer fifteen dollars to Maya before Friday.",
        "Please transfer fifty dollars to Maya before Friday.",
    )

    assert diff.score <= 55
    assert diff.meaning_preservation in {"partial_loss", "major_loss"}
    assert "number_error" in diff.error_categories


def test_negation_change_is_high_impact() -> None:
    diff = analyze_reference_candidate(
        "Do not take two tablets daily.",
        "Do take two tablets daily.",
    )

    assert diff.score <= 40
    assert "negation_error" in diff.error_categories


def test_equivalent_digit_and_word_number_is_not_high_impact() -> None:
    diff = analyze_reference_candidate(
        "Please transfer fifteen dollars to Maya before Friday.",
        "Please transfer 15 dollars to Maya before Friday.",
    )

    assert diff.score > 80
    assert "number_error" not in diff.error_categories


def test_multiword_number_change_is_high_impact() -> None:
    diff = analyze_reference_candidate(
        "The invoice total is twenty one dollars.",
        "The invoice total is twenty two dollars.",
    )

    assert diff.score <= 55
    assert "number_error" in diff.error_categories


def test_weekday_change_is_high_impact_temporal_error() -> None:
    diff = analyze_reference_candidate(
        "The cardiology appointment moved from Monday morning to Friday afternoon.",
        "The cardiology appointment moved from Tuesday morning to Friday afternoon.",
    )

    assert diff.score <= 55
    assert "date_time_error" in diff.error_categories


def test_time_marker_change_is_high_impact_temporal_error() -> None:
    diff = analyze_reference_candidate(
        "Start the batch job at 9 am after the backup completes.",
        "Start the batch job at 9 pm after the backup completes.",
    )

    assert diff.score <= 55
    assert "date_time_error" in diff.error_categories
