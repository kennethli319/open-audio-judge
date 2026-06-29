import json
import subprocess
import wave
from pathlib import Path

import pytest

from scripts.synthesize_tts_cases import (
    summarize_synthesized_cases,
    synthesize_cases,
    write_synthesis_summary_json,
)


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
    assert written[0]["metadata"]["reference_text_sha256"] == (
        "836983522ec15bbf2ce214aa8c1cdb1d3dc4dc7bbce25cc950909cc5dfaa56bf"
    )
    assert "audio_sha256" not in written[0]["metadata"]
    assert (tmp_path / "out" / "text" / "tts-eval-001.txt").read_text(encoding="utf-8") == "Call me at 09:45."


def test_write_synthesis_summary_is_metadata_only_for_dry_run(tmp_path: Path) -> None:
    cases_path = tmp_path / "tts_cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "id": "tts-private",
                "task": "tts_naturalness",
                "reference_text": "Private phrase: call me at 09:45.",
                "metadata": {
                    "source_category": "instruction_constraints",
                    "tts_slice": "dates_times",
                },
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
    summary_path = write_synthesis_summary_json(derived, tmp_path / "out" / "summary.json")
    summary_text = summary_path.read_text(encoding="utf-8")
    summary = json.loads(summary_text)

    assert summary == {
        "audio_bytes": {"average": None, "max": None, "min": None, "total": None},
        "audio_duration_seconds": {"average": None, "max": None, "min": None, "total": None},
        "by_sample_kind": {"local_synthetic_tts": 1},
        "by_slice": {"dates_times": 1},
        "by_source_category": {"instruction_constraints": 1},
        "total_cases": 1,
        "with_audio_sha256": 0,
    }
    assert "Private phrase" not in summary_text
    assert "09:45" not in summary_text


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
    tts_bin = tmp_path / "bin" / "local-tts-speak"
    tts_bin.parent.mkdir()
    tts_bin.write_text("#!/bin/sh\n", encoding="utf-8")

    def fake_run(*args, **kwargs):
        generated.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(generated), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b"\x00\x00" * 8000)
        return subprocess.CompletedProcess(args[0], 0, stdout=json.dumps({"output": str(generated)}))

    monkeypatch.setattr("scripts.synthesize_tts_cases.subprocess.run", fake_run)

    synthesize_cases(
        cases_path=cases_path,
        out_dir=tmp_path / "out",
        tts_bin=tts_bin,
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
    assert written[0]["metadata"]["audio_bytes"] == generated.stat().st_size
    assert written[0]["metadata"]["audio_duration_seconds"] == 0.5
    assert written[0]["metadata"]["audio_sha256"]
    assert summarize_synthesized_cases(written)["audio_duration_seconds"] == {
        "average": 0.5,
        "max": 0.5,
        "min": 0.5,
        "total": 0.5,
    }


def test_synthesize_cases_rejects_reported_audio_outside_output_dir(
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
    generated = tmp_path / "elsewhere" / "tts-001.wav"
    tts_bin = tmp_path / "bin" / "local-tts-speak"
    tts_bin.parent.mkdir()
    tts_bin.write_text("#!/bin/sh\n", encoding="utf-8")

    def fake_run(*args, **kwargs):
        generated.parent.mkdir(parents=True, exist_ok=True)
        generated.write_bytes(b"RIFF....WAVE")
        return subprocess.CompletedProcess(args[0], 0, stdout=json.dumps({"output": str(generated)}))

    monkeypatch.setattr("scripts.synthesize_tts_cases.subprocess.run", fake_run)

    with pytest.raises(ValueError, match="under the output directory"):
        synthesize_cases(
            cases_path=cases_path,
            out_dir=tmp_path / "out",
            tts_bin=tts_bin,
            model="mlx-community/chatterbox-turbo-6bit",
            voice="af_heart",
            lang_code="en",
            audio_format="wav",
        )

    assert not (tmp_path / "out" / "tts_audio_cases.jsonl").exists()


def test_synthesize_cases_rejects_cases_without_reference_text(tmp_path: Path) -> None:
    cases_path = tmp_path / "tts_cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "id": "tts-missing-text",
                "task": "tts_naturalness",
                "turns": [{"role": "user", "content": "Read this aloud."}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="TTS synthesis cases require reference_text"):
        synthesize_cases(
            cases_path=cases_path,
            out_dir=tmp_path / "out",
            tts_bin=Path("/missing/local-tts-speak"),
            model="mlx-community/chatterbox-turbo-6bit",
            voice="af_heart",
            lang_code="en",
            audio_format="wav",
            dry_run=True,
        )

    assert not (tmp_path / "out" / "tts_audio_cases.jsonl").exists()


def test_synthesize_cases_keeps_slug_collision_outputs_distinct(tmp_path: Path) -> None:
    cases_path = tmp_path / "tts_cases.jsonl"
    cases_path.write_text(
        "\n".join(
            json.dumps(case)
            for case in [
                {
                    "id": "tts eval/001",
                    "task": "tts_naturalness",
                    "reference_text": "First sample.",
                },
                {
                    "id": "tts eval 001",
                    "task": "tts_naturalness",
                    "reference_text": "Second sample.",
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    synthesize_cases(
        cases_path=cases_path,
        out_dir=tmp_path / "out",
        tts_bin=Path("/missing/local-tts-speak"),
        model="mlx-community/chatterbox-turbo-6bit",
        voice="af_heart",
        lang_code="en",
        audio_format="wav",
        dry_run=True,
    )

    written = [
        json.loads(line)
        for line in (tmp_path / "out" / "tts_audio_cases.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert written[0]["audio_path"] == "audio/tts-eval-001.wav"
    assert written[1]["audio_path"].startswith("audio/tts-eval-001-")
    assert written[1]["audio_path"].endswith(".wav")
    assert (tmp_path / "out" / "text" / "tts-eval-001.txt").read_text(encoding="utf-8") == "First sample."
    assert len(list((tmp_path / "out" / "text").glob("tts-eval-001*.txt"))) == 2


def test_synthesize_cases_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    cases_path = tmp_path / "tts_cases.jsonl"
    cases_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": "tts-duplicate",
                    "task": "tts_naturalness",
                    "reference_text": text,
                }
            )
            for text in ["First sample.", "Second sample."]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unique case ids"):
        synthesize_cases(
            cases_path=cases_path,
            out_dir=tmp_path / "out",
            tts_bin=Path("/missing/local-tts-speak"),
            model="mlx-community/chatterbox-turbo-6bit",
            voice="af_heart",
            lang_code="en",
            audio_format="wav",
            dry_run=True,
        )

    assert not (tmp_path / "out").exists()


def test_synthesize_cases_preflights_missing_tts_binary_before_writing_text(
    tmp_path: Path,
) -> None:
    cases_path = tmp_path / "tts_cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "id": "tts-private-text",
                "task": "tts_naturalness",
                "reference_text": "Keep this local.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="local TTS binary not found"):
        synthesize_cases(
            cases_path=cases_path,
            out_dir=tmp_path / "out",
            tts_bin=Path("/missing/local-tts-speak"),
            model="mlx-community/chatterbox-turbo-6bit",
            voice="af_heart",
            lang_code="en",
            audio_format="wav",
        )

    assert not (tmp_path / "out" / "text").exists()
    assert not (tmp_path / "out" / "tts_audio_cases.jsonl").exists()
