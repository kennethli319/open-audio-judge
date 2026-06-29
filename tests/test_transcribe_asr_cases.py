import json
from pathlib import Path

from open_audio_judge.models import EvaluationCase
from scripts.transcribe_asr_cases import read_whisper_json, transcribe_cases


def test_read_whisper_json_uses_top_level_text(tmp_path: Path) -> None:
    output = tmp_path / "sample.json"
    output.write_text(
        json.dumps({"text": " hello world ", "language": "en", "segments": [{"text": "hello"}]}),
        encoding="utf-8",
    )

    transcript = read_whisper_json(output)

    assert transcript.text == "hello world"
    assert transcript.language == "en"
    assert transcript.segments == 1


def test_read_whisper_json_falls_back_to_segment_text(tmp_path: Path) -> None:
    output = tmp_path / "sample.json"
    output.write_text(
        json.dumps({"segments": [{"text": " hello"}, {"text": "world "}] }),
        encoding="utf-8",
    )

    transcript = read_whisper_json(output)

    assert transcript.text == "hello world"


def test_transcribe_cases_uses_fake_whisper_cli(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF....WAVE")
    fake_whisper = tmp_path / "fake_whisper.py"
    fake_whisper.write_text(
        """#!/usr/bin/env python3
import json
import sys
from pathlib import Path

audio = Path(sys.argv[1])
out_dir = Path(sys.argv[sys.argv.index("--output_dir") + 1])
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / f"{audio.stem}.json").write_text(
    json.dumps({"text": "hello from fake whisper", "language": "en", "segments": [{"text": "hello"}]})
)
""",
        encoding="utf-8",
    )
    fake_whisper.chmod(0o755)

    [case] = transcribe_cases(
        [
            EvaluationCase(
                id="asr one",
                task="asr_error",
                audio_path=str(audio),
                reference_text="hello from fake whisper",
            )
        ],
        transcripts_dir=tmp_path / "transcripts",
        whisper_bin=str(fake_whisper),
        model="tiny",
        language="en",
        device="cpu",
        threads="1",
    )

    assert case.candidate_text == "hello from fake whisper"
    assert case.metadata["candidate_model"] == "whisper-tiny"
    assert case.metadata["candidate_text_source"] == "whisper_cli_json"
    assert case.metadata["whisper_segment_count"] == 1
