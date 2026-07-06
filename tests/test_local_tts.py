import json
from pathlib import Path

import pytest

from open_audio_judge.local_tts import (
    LocalTtsConfig,
    _output_path_from_stdout,
    synthesize_cases_with_local_tts,
    write_local_tts_summary_json,
)
from open_audio_judge.models import EvaluationCase


def test_synthesize_cases_with_local_tts_dry_run_writes_relative_audio_metadata(
    tmp_path: Path,
) -> None:
    case = EvaluationCase(
        id="tts sample/001",
        task="tts_naturalness",
        reference_text="Call me at 09:45.",
        metadata={"tts_slice": "dates_times", "requires_synthesis": True},
    )

    synthesized = synthesize_cases_with_local_tts(
        [case],
        out_dir=tmp_path / "synthesis",
        config=LocalTtsConfig(
            tts_bin=Path("/missing/local-tts-speak"),
            model="mlx-community/chatterbox-turbo-6bit",
            dry_run=True,
            keep_text_sidecars=True,
        ),
    )

    assert synthesized[0].id == "tts sample/001-local-tts"
    assert synthesized[0].audio_path == "audio/tts-sample-001.wav"
    assert synthesized[0].reference_text == "Call me at 09:45."
    assert synthesized[0].metadata["sample_kind"] == "local_synthetic_tts"
    assert synthesized[0].metadata["synthesis_provider"] == "local_chatterbox"
    assert synthesized[0].metadata["synthesis_model"] == "mlx-community/chatterbox-turbo-6bit"
    assert synthesized[0].metadata["source_case_id"] == "tts sample/001"
    assert synthesized[0].metadata["requires_synthesis"] is False
    assert synthesized[0].metadata["text_sidecar_path"] == "text/tts-sample-001.txt"


def test_synthesize_cases_with_local_tts_requires_reference_text(tmp_path: Path) -> None:
    case = EvaluationCase(id="tts-empty", task="tts_naturalness")

    with pytest.raises(ValueError, match="require reference_text"):
        synthesize_cases_with_local_tts(
            [case],
            out_dir=tmp_path / "synthesis",
            config=LocalTtsConfig(dry_run=True),
        )


def test_output_path_from_stdout_accepts_progress_before_json() -> None:
    output = _output_path_from_stdout(
        "\x1b[94mText:\x1b[0m hello\n"
        "S3 Token -> Mel Inference...\n"
        '{"output": "/tmp/example.wav", "synthesis_ms": 123.4}\n'
    )

    assert output is not None
    assert output.name == "example.wav"


def test_write_local_tts_summary_json(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    case = EvaluationCase(
        id="tts-1-local-tts",
        task="tts_naturalness",
        audio_path="audio/tts-1.wav",
        reference_text="Hello.",
        metadata={
            "synthesis_voice": "af_heart",
            "synthesis_audio_format": "wav",
            "tts_slice": "general",
        },
    )

    write_local_tts_summary_json(
        [case],
        summary,
        source_cases=Path("examples/tts_cases.jsonl"),
        model="mlx-community/chatterbox-turbo-6bit",
    )

    data = json.loads(summary.read_text(encoding="utf-8"))
    assert data["candidate_model"] == "mlx-community/chatterbox-turbo-6bit"
    assert data["candidate_generator"] == "local_chatterbox"
    assert data["cases_with_audio_path"] == 1
    assert data["by_synthesis_voice"] == {"af_heart": 1}
