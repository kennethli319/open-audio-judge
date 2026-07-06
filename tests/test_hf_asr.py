import json
from pathlib import Path

import pytest

from open_audio_judge.hf_asr import (
    transcribe_case_with_hf_asr,
    transcribe_cases_with_hf_asr,
    write_hf_asr_summary_json,
)
from open_audio_judge.models import EvaluationCase


def test_transcribe_cases_with_hf_asr_adds_candidate_metadata(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF")
    case = EvaluationCase(
        id="case-1",
        task="asr_error",
        audio_path=str(audio),
        reference_text="Transfer fifteen dollars.",
    )

    cases = transcribe_cases_with_hf_asr(
        [case],
        model="openai/whisper-tiny",
        pipeline_factory=lambda **_: lambda _audio_path: {"text": "Transfer fifty dollars."},
    )

    assert cases[0].candidate_text == "Transfer fifty dollars."
    assert cases[0].metadata["candidate_model"] == "openai/whisper-tiny"
    assert cases[0].metadata["candidate_transcriber"] == "huggingface-transformers-asr"


def test_transcribe_case_with_hf_asr_can_join_chunk_text(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF")
    case = EvaluationCase(id="case-1", task="asr_error", audio_path=str(audio))

    transcript = transcribe_case_with_hf_asr(
        case,
        lambda _audio_path: {"chunks": [{"text": "hello"}, {"text": "world"}]},
    )

    assert transcript.text == "hello world"
    assert transcript.raw["chunk_count"] == 2


def test_transcribe_case_with_hf_asr_requires_audio_path() -> None:
    case = EvaluationCase(id="case-1", task="asr_error")

    with pytest.raises(ValueError, match="requires a local audio_path"):
        transcribe_case_with_hf_asr(case, lambda _audio_path: {"text": "hello"})


def test_write_hf_asr_summary_json(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    case = EvaluationCase(
        id="case-1",
        task="asr_error",
        candidate_text="hello",
    )

    write_hf_asr_summary_json(
        [case],
        summary,
        source_cases=Path("examples/asr_cases.jsonl"),
        model="openai/whisper-tiny",
    )

    data = json.loads(summary.read_text(encoding="utf-8"))
    assert data["candidate_model"] == "openai/whisper-tiny"
    assert data["candidate_transcriber"] == "huggingface-transformers-asr"
    assert data["cases_with_candidate_text"] == 1
