import hashlib
import json
import subprocess
import wave
from pathlib import Path

import pytest

from open_audio_judge.runner import load_cases
from scripts.synthesize_tts_cases import (
    summarize_synthesized_cases,
    summarize_validation_issues,
    synthesize_cases,
    validate_synthesized_manifest,
    write_synthesis_summary_json,
    write_validation_summary_json,
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
                "metadata": {
                    "source_id": "private-001",
                    "tts_slice": "dates_times",
                    "requires_synthesis": True,
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

    manifest = tmp_path / "out" / "tts_audio_cases.jsonl"
    written = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]

    assert derived == written
    assert written[0]["id"] == "tts eval/001-local-tts"
    assert written[0]["audio_path"] == "audio/tts-eval-001.wav"
    assert written[0]["reference_text"] == "Call me at 09:45."
    assert written[0]["metadata"]["sample_kind"] == "local_synthetic_tts"
    assert written[0]["metadata"]["source_case_id"] == "tts eval/001"
    assert written[0]["metadata"]["text_context_fields"] == ["reference_text", "turns"]
    assert written[0]["metadata"]["requires_synthesis"] is False
    assert written[0]["metadata"]["turn_count"] == 1
    assert written[0]["metadata"]["turn_roles"] == ["user"]
    assert written[0]["metadata"]["text_sidecar_written"] is True
    assert written[0]["metadata"]["reference_text_sha256"] == (
        "836983522ec15bbf2ce214aa8c1cdb1d3dc4dc7bbce25cc950909cc5dfaa56bf"
    )
    assert "audio_sha256" not in written[0]["metadata"]
    assert (tmp_path / "out" / "text" / "tts-eval-001.txt").read_text(encoding="utf-8") == "Call me at 09:45."
    assert (
        validate_synthesized_manifest(
            cases_path=manifest,
            require_local_audio=False,
            require_text_context_metadata=True,
            require_synthesis_metadata=True,
        )
        == []
    )


def test_validate_synthesized_manifest_accepts_relative_audio_and_text_context(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "out" / "audio" / "sample.wav"
    audio_path.parent.mkdir(parents=True)
    with wave.open(str(audio_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 1600)
    cases_path = tmp_path / "out" / "tts_audio_cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "id": "tts-audio",
                "task": "tts_naturalness",
                "audio_path": "audio/sample.wav",
                "reference_text": "Synthetic sample.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    issues = validate_synthesized_manifest(cases_path=cases_path)

    assert issues == []
    assert summarize_validation_issues(issues) == {
        "by_reason": {},
        "case_ids": [],
        "issue_count": 0,
        "valid": True,
    }


def test_validate_synthesized_manifest_accepts_matching_audio_metadata(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "out" / "audio" / "sample.wav"
    audio_path.parent.mkdir(parents=True)
    with wave.open(str(audio_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 1600)
    cases_path = tmp_path / "out" / "tts_audio_cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "id": "tts-audio",
                "task": "tts_naturalness",
                "audio_path": "audio/sample.wav",
                "reference_text": "Synthetic sample.",
                "metadata": {
                    "audio_bytes": audio_path.stat().st_size,
                    "audio_duration_seconds": 0.1,
                    "audio_sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert validate_synthesized_manifest(cases_path=cases_path) == []


def test_validate_synthesized_manifest_reports_stale_audio_metadata(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "out" / "audio" / "sample.wav"
    audio_path.parent.mkdir(parents=True)
    with wave.open(str(audio_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 1600)
    cases_path = tmp_path / "out" / "tts_audio_cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "id": "tts-audio",
                "task": "tts_naturalness",
                "audio_path": "audio/sample.wav",
                "reference_text": "Synthetic sample.",
                "metadata": {
                    "audio_bytes": 1,
                    "audio_duration_seconds": 9.9,
                    "audio_sha256": "stale",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    issues = validate_synthesized_manifest(cases_path=cases_path)

    assert [(issue.case_id, issue.reason) for issue in issues] == [
        (
            "tts-audio",
            f"metadata.audio_bytes does not match audio_path file: expected {audio_path.stat().st_size}, got 1.",
        ),
        ("tts-audio", "metadata.audio_sha256 does not match audio_path file."),
        (
            "tts-audio",
            "metadata.audio_duration_seconds does not match audio_path file: expected 0.1, got 9.9.",
        ),
    ]


def test_validate_synthesized_manifest_reports_missing_audio_and_text_contract(
    tmp_path: Path,
) -> None:
    cases_path = tmp_path / "tts_audio_cases.jsonl"
    cases_path.write_text(
        "\n".join(
            json.dumps(case)
            for case in [
                {
                    "id": "missing-audio",
                    "task": "tts_naturalness",
                    "audio_path": "audio/missing.wav",
                    "reference_text": "Synthetic sample.",
                },
                {
                    "id": "missing-text",
                    "task": "tts_naturalness",
                    "audio_url": "https://example.test/audio.wav",
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    issues = validate_synthesized_manifest(cases_path=cases_path)
    summary_path = write_validation_summary_json(issues, tmp_path / "summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert [(issue.case_id, issue.reason) for issue in issues] == [
        ("missing-audio", "audio_path file not found: audio/missing.wav"),
        (
            "missing-text",
            "Audio judge cases require textual context via reference_text, candidate_text, or turns.",
        ),
    ]
    assert summary == {
        "by_reason": {
            "Audio judge cases require textual context via reference_text, candidate_text, or turns.": 1,
            "audio_path file not found: audio/missing.wav": 1,
        },
        "case_ids": ["missing-audio", "missing-text"],
        "issue_count": 2,
        "valid": False,
    }


def test_validation_summary_can_hash_private_case_ids(tmp_path: Path) -> None:
    issues = [
        validate_issue
        for validate_issue in validate_synthesized_manifest(
            cases_path=_write_private_invalid_manifest(tmp_path)
        )
    ]

    summary_path = write_validation_summary_json(
        issues,
        tmp_path / "summary.json",
        redact_case_ids=True,
    )
    summary_text = summary_path.read_text(encoding="utf-8")
    summary = json.loads(summary_text)

    expected_hash = hashlib.sha256("private-session-2026-06-29-row-001".encode("utf-8")).hexdigest()
    assert summary["case_ids"] == [f"case-{expected_hash[:12]}"]
    assert "private-session" not in summary_text
    assert summarize_validation_issues(issues)["case_ids"] == [
        "private-session-2026-06-29-row-001"
    ]


def test_validation_summary_can_include_metadata_only_manifest_coverage(
    tmp_path: Path,
) -> None:
    cases_path = tmp_path / "tts_audio_cases.jsonl"
    cases_path.write_text(
        "\n".join(
            json.dumps(case)
            for case in [
                {
                    "id": "private-session-2026-06-29-row-001-local-tts",
                    "task": "tts_naturalness",
                    "audio_path": "audio/missing.wav",
                    "reference_text": "Private phrase: call me at 09:45.",
                    "metadata": {
                        "sample_kind": "local_synthetic_tts",
                        "source_category": "instruction_constraints",
                        "source_case_id": "private-session-2026-06-29-row-001",
                        "synthesis_provider": "local_chatterbox",
                        "text_context_fields": ["reference_text", "turns"],
                        "tts_slice": "dates_times",
                        "turn_count": 2,
                        "turn_roles": ["user", "assistant"],
                    },
                    "turns": [
                        {"role": "user", "content": "Read the private phrase aloud."},
                        {"role": "assistant", "content": "Private phrase: call me at 09:45."},
                    ],
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cases = [case.model_dump(exclude_none=True) for case in load_cases(cases_path)]
    issues = validate_synthesized_manifest(cases_path=cases_path)

    summary_path = write_validation_summary_json(
        issues,
        tmp_path / "summary.json",
        redact_case_ids=True,
        cases=cases,
    )
    summary_text = summary_path.read_text(encoding="utf-8")
    summary = json.loads(summary_text)

    expected_hash = hashlib.sha256(
        "private-session-2026-06-29-row-001-local-tts".encode("utf-8")
    ).hexdigest()
    assert summary["case_ids"] == [f"case-{expected_hash[:12]}"]
    assert summary["manifest"]["by_sample_kind"] == {"local_synthetic_tts": 1}
    assert summary["manifest"]["by_slice"] == {"dates_times": 1}
    assert summary["manifest"]["by_source_category"] == {"instruction_constraints": 1}
    assert summary["manifest"]["by_text_context_fields"] == {"reference_text+turns": 1}
    assert summary["manifest"]["by_turn_role_sequence"] == {"user+assistant": 1}
    assert summary["manifest"]["multi_turn_cases"] == 1
    assert "Private phrase" not in summary_text
    assert "09:45" not in summary_text
    assert "private-session" not in summary_text


def test_validate_synthesized_manifest_can_allow_missing_dry_run_audio(
    tmp_path: Path,
) -> None:
    cases_path = tmp_path / "tts_audio_cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "id": "dry-run-case",
                "task": "tts_naturalness",
                "audio_path": "audio/future.wav",
                "reference_text": "Future synthetic sample.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert validate_synthesized_manifest(cases_path=cases_path, require_local_audio=False) == []


def test_validate_synthesized_manifest_rejects_url_only_audio(
    tmp_path: Path,
) -> None:
    cases_path = tmp_path / "tts_audio_cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "id": "url-only-case",
                "task": "tts_naturalness",
                "audio_url": "https://example.test/audio.wav",
                "reference_text": "Synthetic sample.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    issues = validate_synthesized_manifest(cases_path=cases_path, require_local_audio=False)

    assert [(issue.case_id, issue.reason) for issue in issues] == [
        ("url-only-case", "Synthesized TTS manifests require local audio_path.")
    ]


def test_validate_synthesized_manifest_rejects_mixed_local_and_url_audio(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "out" / "audio" / "sample.wav"
    audio_path.parent.mkdir(parents=True)
    with wave.open(str(audio_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 1600)
    cases_path = tmp_path / "out" / "tts_audio_cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "id": "mixed-audio-case",
                "task": "tts_naturalness",
                "audio_path": "audio/sample.wav",
                "audio_url": "https://example.test/audio.wav",
                "reference_text": "Synthetic sample.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    issues = validate_synthesized_manifest(cases_path=cases_path)

    assert [(issue.case_id, issue.reason) for issue in issues] == [
        ("mixed-audio-case", "Synthesized TTS manifests must not include audio_url.")
    ]


def test_validate_synthesized_manifest_reports_stale_text_context_metadata(
    tmp_path: Path,
) -> None:
    cases_path = tmp_path / "tts_audio_cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "id": "stale-context-fields",
                "task": "tts_naturalness",
                "audio_path": "audio/future.wav",
                "reference_text": "Synthetic sample.",
                "candidate_text": "Synthetic sample.",
                "metadata": {"text_context_fields": ["reference_text"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    issues = validate_synthesized_manifest(cases_path=cases_path, require_local_audio=False)

    assert [(issue.case_id, issue.reason) for issue in issues] == [
        (
            "stale-context-fields",
            "metadata.text_context_fields does not match case text context: "
            "expected reference_text+candidate_text, got reference_text.",
        )
    ]


def test_validate_synthesized_manifest_can_require_text_context_metadata(
    tmp_path: Path,
) -> None:
    cases_path = tmp_path / "tts_audio_cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "id": "missing-context-fields",
                "task": "tts_naturalness",
                "audio_path": "audio/future.wav",
                "reference_text": "Synthetic sample.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    issues = validate_synthesized_manifest(
        cases_path=cases_path,
        require_local_audio=False,
        require_text_context_metadata=True,
    )

    assert [(issue.case_id, issue.reason) for issue in issues] == [
        ("missing-context-fields", "metadata.text_context_fields is missing.")
    ]


def test_validate_synthesized_manifest_can_require_synthesis_metadata(
    tmp_path: Path,
) -> None:
    cases_path = tmp_path / "tts_audio_cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "id": "missing-synthesis-metadata-local-tts",
                "task": "tts_naturalness",
                "audio_path": "audio/future.wav",
                "reference_text": "Synthetic sample.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    issues = validate_synthesized_manifest(
        cases_path=cases_path,
        require_local_audio=False,
        require_synthesis_metadata=True,
    )

    assert [(issue.case_id, issue.reason) for issue in issues] == [
        ("missing-synthesis-metadata-local-tts", "metadata.sample_kind is missing."),
        ("missing-synthesis-metadata-local-tts", "metadata.synthesis_provider is missing."),
        ("missing-synthesis-metadata-local-tts", "metadata.source_case_id is missing."),
        ("missing-synthesis-metadata-local-tts", "metadata.reference_text_sha256 is missing."),
    ]


def test_validate_synthesized_manifest_reports_stale_synthesis_metadata(
    tmp_path: Path,
) -> None:
    cases_path = tmp_path / "tts_audio_cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "id": "tts-source-local-tts",
                "task": "tts_naturalness",
                "audio_path": "audio/future.wav",
                "reference_text": "Updated synthetic sample.",
                "metadata": {
                    "sample_kind": "human_recording",
                    "synthesis_provider": "other",
                    "source_case_id": "different-source",
                    "reference_text_sha256": "stale",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    issues = validate_synthesized_manifest(cases_path=cases_path, require_local_audio=False)

    assert [(issue.case_id, issue.reason) for issue in issues] == [
        (
            "tts-source-local-tts",
            "metadata.sample_kind has unexpected value: expected local_synthetic_tts, got human_recording.",
        ),
        (
            "tts-source-local-tts",
            "metadata.synthesis_provider has unexpected value: expected local_chatterbox, got other.",
        ),
        (
            "tts-source-local-tts",
            "metadata.source_case_id does not match synthesized case id: "
            "expected tts-source, got different-source.",
        ),
        (
            "tts-source-local-tts",
            "metadata.reference_text_sha256 does not match reference_text.",
        ),
    ]


def test_validate_synthesized_manifest_rejects_unsynthesized_draft_marker(
    tmp_path: Path,
) -> None:
    cases_path = tmp_path / "tts_audio_cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "id": "tts-source-local-tts",
                "task": "tts_naturalness",
                "audio_path": "audio/future.wav",
                "reference_text": "Synthetic sample.",
                "metadata": {
                    "requires_synthesis": True,
                    "sample_kind": "local_synthetic_tts",
                    "synthesis_provider": "local_chatterbox",
                    "source_case_id": "tts-source",
                    "reference_text_sha256": (
                        "0d988c7d994421014d1fa0145748fafbef20b4c64112207c609a449b1feeb739"
                    ),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    issues = validate_synthesized_manifest(cases_path=cases_path, require_local_audio=False)

    assert [(issue.case_id, issue.reason) for issue in issues] == [
        (
            "tts-source-local-tts",
            "metadata.requires_synthesis must be removed or set false after synthesis.",
        )
    ]


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
        "by_text_context_fields": {"reference_text": 1},
        "by_turn_role_sequence": {"none": 1},
        "multi_turn_cases": 0,
        "total_cases": 1,
        "with_audio_sha256": 0,
    }
    assert "Private phrase" not in summary_text
    assert "09:45" not in summary_text


def test_synthesize_cases_can_skip_text_sidecars_in_dry_run(tmp_path: Path) -> None:
    cases_path = tmp_path / "tts_cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "id": "tts-private",
                "task": "tts_naturalness",
                "reference_text": "Do not persist this text sidecar.",
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
        keep_text_sidecars=False,
        dry_run=True,
    )

    assert derived[0]["metadata"]["text_sidecar_written"] is False
    assert not (tmp_path / "out" / "text" / "tts-private.txt").exists()
    assert (tmp_path / "out" / "tts_audio_cases.jsonl").exists()


def test_synthesis_summary_counts_text_context_field_combinations(
    tmp_path: Path,
) -> None:
    cases_path = tmp_path / "tts_cases.jsonl"
    cases_path.write_text(
        "\n".join(
            json.dumps(case)
            for case in [
                {
                    "id": "reference-only",
                    "task": "tts_naturalness",
                    "reference_text": "Read this.",
                },
                {
                    "id": "candidate-and-turns",
                    "task": "tts_naturalness",
                    "turns": [{"role": "user", "content": "Answer tersely."}],
                    "reference_text": "Yes.",
                    "candidate_text": "Yes.",
                },
            ]
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

    assert summarize_synthesized_cases(derived)["by_text_context_fields"] == {
        "candidate_text+reference_text+turns": 1,
        "reference_text": 1,
    }


def test_synthesis_summary_counts_turn_role_sequences(tmp_path: Path) -> None:
    cases_path = tmp_path / "tts_cases.jsonl"
    cases_path.write_text(
        "\n".join(
            json.dumps(case)
            for case in [
                {
                    "id": "reference-only",
                    "task": "tts_naturalness",
                    "reference_text": "Read this.",
                },
                {
                    "id": "multi-turn",
                    "task": "tts_naturalness",
                    "turns": [
                        {"role": "user", "content": "Remember 7294."},
                        {"role": "assistant", "content": "Stored."},
                        {"role": "user", "content": "Read it back."},
                    ],
                    "reference_text": "The code is 7294.",
                },
            ]
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

    summary = summarize_synthesized_cases(derived)

    assert derived[0]["metadata"]["turn_count"] == 0
    assert derived[0]["metadata"]["turn_roles"] == []
    assert derived[1]["metadata"]["turn_count"] == 3
    assert derived[1]["metadata"]["turn_roles"] == ["user", "assistant", "user"]
    assert summary["by_turn_role_sequence"] == {
        "none": 1,
        "user+assistant+user": 1,
    }
    assert summary["multi_turn_cases"] == 1


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


def test_synthesize_cases_can_delete_text_sidecars_after_synthesis(
    tmp_path: Path, monkeypatch
) -> None:
    cases_path = tmp_path / "tts_cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "id": "tts-private",
                "task": "tts_naturalness",
                "reference_text": "Temporary text file only.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    generated = tmp_path / "out" / "audio" / "tts-private.wav"
    tts_bin = tmp_path / "bin" / "local-tts-speak"
    tts_bin.parent.mkdir()
    tts_bin.write_text("#!/bin/sh\n", encoding="utf-8")

    def fake_run(*args, **kwargs):
        text_path = Path(args[0][args[0].index("--text-file") + 1])
        assert text_path.read_text(encoding="utf-8") == "Temporary text file only."
        generated.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(generated), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b"\x00\x00" * 8000)
        return subprocess.CompletedProcess(args[0], 0, stdout=json.dumps({"output": str(generated)}))

    monkeypatch.setattr("scripts.synthesize_tts_cases.subprocess.run", fake_run)

    derived = synthesize_cases(
        cases_path=cases_path,
        out_dir=tmp_path / "out",
        tts_bin=tts_bin,
        model="mlx-community/chatterbox-turbo-6bit",
        voice="af_heart",
        lang_code="en",
        audio_format="wav",
        keep_text_sidecars=False,
    )

    assert derived[0]["metadata"]["text_sidecar_written"] is False
    assert not (tmp_path / "out" / "text" / "tts-private.txt").exists()
    assert (tmp_path / "out" / "audio" / "tts-private.wav").exists()


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


def _write_private_invalid_manifest(tmp_path: Path) -> Path:
    cases_path = tmp_path / "tts_audio_cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "id": "private-session-2026-06-29-row-001",
                "task": "tts_naturalness",
                "audio_path": "audio/missing.wav",
                "reference_text": "Synthetic sample.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return cases_path
