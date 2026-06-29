from __future__ import annotations

import json
from pathlib import Path

import pytest

from open_audio_judge.runner import load_cases
from scripts.gemini_sample_records import missing_records, record_issues


OPEN_SAMPLE_MANIFESTS = [
    Path("examples/asr_open_samples.jsonl"),
    Path("examples/tts_open_samples.jsonl"),
]


def test_open_sample_manifests_load_as_cases() -> None:
    for manifest in OPEN_SAMPLE_MANIFESTS:
        cases = load_cases(manifest)

        assert len(cases) >= 3
        assert len({case.id for case in cases}) == len(cases)
        assert all(case.audio_url and case.audio_url.startswith("https://") for case in cases)
        assert all(case.audio_path is None for case in cases)
        assert all(case.metadata.get("sample_kind") == "open_development_audio" for case in cases)
        assert all(case.metadata.get("source_page") for case in cases)
        assert all(case.metadata.get("license_note") for case in cases)


def test_open_sample_docs_list_every_case_id() -> None:
    sample_docs = Path("docs/sample-audio.md").read_text(encoding="utf-8")
    case_ids = {
        case.id
        for manifest in OPEN_SAMPLE_MANIFESTS
        for case in load_cases(manifest)
    }

    for case_id in case_ids:
        assert case_id in sample_docs


def test_gemini_sample_records_are_current() -> None:
    assert missing_records(
        Path("examples/gemini_sample_records.jsonl"),
        provider="gemini",
        model="gemini-3.5-flash",
    ) == []


def test_gemini_sample_records_detect_prompt_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    def changed_fingerprints(provider: str, model: str) -> dict[str, str]:
        assert provider == "gemini"
        assert model == "gemini-3.5-flash"
        return {"asr-open-armstrong-small-step": "changed"}

    monkeypatch.setattr(
        "scripts.gemini_sample_records.expected_fingerprints",
        changed_fingerprints,
    )

    assert missing_records(
        Path("examples/gemini_sample_records.jsonl"),
        provider="gemini",
        model="gemini-3.5-flash",
    ) == ["asr-open-armstrong-small-step"]


def test_gemini_sample_record_issues_explain_rerun_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def changed_fingerprints(provider: str, model: str) -> dict[str, str]:
        assert provider == "gemini"
        assert model == "gemini-3.5-flash"
        return {
            "asr-open-armstrong-small-step": "changed",
            "asr-open-jfk-moon": "missing",
        }

    monkeypatch.setattr(
        "scripts.gemini_sample_records.expected_fingerprints",
        changed_fingerprints,
    )

    issues = record_issues(
        Path("examples/gemini_sample_records.jsonl"),
        provider="gemini",
        model="gemini-3.5-flash",
    )

    assert [(issue.case_id, issue.reason) for issue in issues] == [
        ("asr-open-armstrong-small-step", "changed_fingerprint"),
        ("asr-open-jfk-moon", "missing_record"),
    ]


def test_gemini_sample_record_issues_detect_duplicate_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records_path = tmp_path / "records.jsonl"
    record = {
        "base_case_id": "asr-open-armstrong-small-step",
        "case_id": "asr-open-armstrong-small-step-wav",
        "provider": "gemini",
        "model": "gemini-3.5-flash",
        "sample_fingerprint": "current",
        "status": "ok",
    }
    records_path.write_text(
        "\n".join([json.dumps(record), json.dumps(record)]) + "\n",
        encoding="utf-8",
    )

    def current_fingerprints(provider: str, model: str) -> dict[str, str]:
        assert provider == "gemini"
        assert model == "gemini-3.5-flash"
        return {"asr-open-armstrong-small-step": "current"}

    monkeypatch.setattr(
        "scripts.gemini_sample_records.expected_fingerprints",
        current_fingerprints,
    )

    issues = record_issues(
        records_path,
        provider="gemini",
        model="gemini-3.5-flash",
    )

    assert [(issue.case_id, issue.reason) for issue in issues] == [
        ("asr-open-armstrong-small-step", "duplicate_record"),
    ]
