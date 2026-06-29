import json
import subprocess
from pathlib import Path

from scripts.synthesize_tts_cases import synthesize_cases


def test_synthesize_cases_dry_run_writes_local_audio_manifest(tmp_path: Path) -> None:
    cases_path = tmp_path / "tts_cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "id": "tts eval/001",
                "task": "tts_naturalness",
                "turns": [{"role": "user", "content": "Read the answer aloud."}],
                "reference_text": "Call me at 09:45.",
                "metadata": {"source_id": "private-001", "tts_slice": "dates_times"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    derived = synthesize_cases(
        cases_path=cases_path,
        out_dir=tmp_path / "out",
        tts_bin=Path("/missing/local-tts-speak"),
        model="mlx-community/chatterbox-turbo-6bit",
        voice="af_heart",
        lang_code="en",
        audio_format="wav",
        dry_run=True,
    )

    manifest = tmp_path / "out" / "tts_audio_cases.jsonl"
    written = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]

    assert derived == written
    assert written[0]["id"] == "tts eval/001-local-tts"
    assert written[0]["audio_path"] == "audio/tts-eval-001.wav"
    assert written[0]["reference_text"] == "Call me at 09:45."
    assert written[0]["metadata"]["sample_kind"] == "local_synthetic_tts"
    assert written[0]["metadata"]["source_case_id"] == "tts eval/001"
    assert (tmp_path / "out" / "text" / "tts-eval-001.txt").read_text(encoding="utf-8") == "Call me at 09:45."


def test_synthesize_cases_uses_reported_tts_output_path(
    tmp_path: Path, monkeypatch
) -> None:
    cases_path = tmp_path / "tts_cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "id": "tts-001",
                "task": "tts_naturalness",
                "reference_text": "Synthetic sample.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    generated = tmp_path / "out" / "audio" / "tts-001_0001.wav"

    def fake_run(*args, **kwargs):
        generated.parent.mkdir(parents=True, exist_ok=True)
        generated.write_bytes(b"RIFF....WAVE")
        return subprocess.CompletedProcess(args[0], 0, stdout=json.dumps({"output": str(generated)}))

    monkeypatch.setattr("scripts.synthesize_tts_cases.subprocess.run", fake_run)

    synthesize_cases(
        cases_path=cases_path,
        out_dir=tmp_path / "out",
        tts_bin=Path("/fake/local-tts-speak"),
        model="mlx-community/chatterbox-turbo-6bit",
        voice="af_heart",
        lang_code="en",
        audio_format="wav",
    )

    written = [
        json.loads(line)
        for line in (tmp_path / "out" / "tts_audio_cases.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert written[0]["audio_path"] == "audio/tts-001_0001.wav"
