from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from open_audio_judge.runner import load_cases
from scripts.gemini_sample_records import missing_records, record_issues, update_records


OPEN_SAMPLE_MANIFESTS = [
    Path("examples/asr_open_samples.jsonl"),
    Path("examples/tts_open_samples.jsonl"),
]

TTS_MULTITURN_MANIFEST = Path("examples/tts_multiturn_cases.jsonl")
TTS_RESEARCH_CATEGORIES = {
    "paralinguistics",
    "instruction_following",
    "information_tuning",
    "storytelling_dialogue",
    "speech_steerability",
    "robustness_intelligibility",
    "speaker_voice_consistency",
    "multilingual_code_switching",
    "long_form_discourse",
    "text_normalization",
    "acoustic_contexts",
    "spontaneous_conversation",
    "affective_transitions",
    "punctuation_prosody",
    "domain_terminology",
    "heteronym_disambiguation",
    "formatting_markup_robustness",
    "nonverbal_paralinguistic_cues",
    "voice_conversion_similarity",
    "accent_dialect_handling",
    "artifact_suppression",
    "temporal_rhythm_control",
    "safety_privacy_delivery",
    "semantic_contrast_focus",
    "dialogue_turn_management",
    "compositional_style_control",
    "named_entity_pronunciation",
    "disfluency_repair_control",
    "lexical_stress_disambiguation",
    "pragmatic_intent_delivery",
    "symbolic_math_reading",
    "multi_speaker_attribution",
    "structured_enumeration_delivery",
}
TTS_REQUIRED_METADATA = {
    "language",
    "eval_category",
    "tts_slice",
    "voice",
    "source",
    "style_prompt",
    "expected_style",
    "expected_instruction",
    "source_basis",
}
TTS_SOURCE_BASIS_TERMS = (
    "Seed-TTS",
    "Seed-TTS-Eval",
    "InstructTTSEval",
    "VoiceBench",
    "VocalBench",
    "TTSDS",
    "Discrete-token SLM",
    "prosodic variation",
    "long-form intelligibility",
)


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


def test_tts_multiturn_manifest_has_research_metadata_contract() -> None:
    cases = load_cases(TTS_MULTITURN_MANIFEST)

    assert len(cases) == 165
    assert len({case.id for case in cases}) == len(cases)
    assert len({case.metadata["tts_slice"] for case in cases}) == len(cases)
    assert {case.metadata["eval_category"] for case in cases} == TTS_RESEARCH_CATEGORIES

    for case in cases:
        metadata = case.metadata
        missing = sorted(key for key in TTS_REQUIRED_METADATA if not metadata.get(key))
        assert missing == [], f"{case.id} missing metadata: {missing}"
        assert metadata["source"] == "research-backed-tts-demo"
        assert metadata["language"] == "en"
        assert case.reference_text
        assert case.turns[-1].content == case.reference_text
        assert any(term in metadata["source_basis"] for term in TTS_SOURCE_BASIS_TERMS)


def test_tts_multiturn_manifest_keeps_five_cases_per_category() -> None:
    cases = load_cases(TTS_MULTITURN_MANIFEST)
    counts: dict[str, int] = {}
    for case in cases:
        category = str(case.metadata["eval_category"])
        counts[category] = counts.get(category, 0) + 1

    assert counts == {category: 5 for category in TTS_RESEARCH_CATEGORIES}


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


def test_gemini_sample_record_update_preserves_unrelated_current_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records_path = tmp_path / "records.jsonl"
    existing_asr = {
        "base_case_id": "asr-open-armstrong-small-step",
        "case_id": "asr-open-armstrong-small-step-wav",
        "provider": "gemini",
        "model": "gemini-3.5-flash",
        "sample_fingerprint": "asr-current",
        "status": "ok",
    }
    stale_tts = {
        "base_case_id": "tts-open-en-us-hello",
        "case_id": "tts-open-en-us-hello-wav",
        "provider": "gemini",
        "model": "gemini-3.5-flash",
        "sample_fingerprint": "tts-stale",
        "status": "ok",
    }
    records_path.write_text(
        "".join(json.dumps(record) + "\n" for record in [existing_asr, stale_tts]),
        encoding="utf-8",
    )
    results_path = tmp_path / "results.jsonl"
    results_path.write_text(
        json.dumps(
            {
                "case_id": "tts-open-en-us-hello-wav",
                "task": "tts_naturalness",
                "judge_id": "tts_naturalness",
                "judge_version": "0.2.0",
                "status": "ok",
                "overall_score": 95,
                "label": "accurate",
                "reason": "Natural.",
                "semantic_error_summary": "No issues.",
                "key_differences": [],
                "error_categories": ["no_error"],
                "researcher_notes": [],
                "created_at": "2026-06-29T00:00:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "scripts.gemini_sample_records.expected_fingerprints",
        lambda provider, model: {
            "asr-open-armstrong-small-step": "asr-current",
            "tts-open-en-us-hello": "tts-current",
        },
    )
    monkeypatch.setattr(
        "scripts.gemini_sample_records.sample_cases_by_id",
        lambda: {
            "tts-open-en-us-hello": SimpleNamespace(
                metadata={"source_page": "https://example.test", "sample_kind": "open"}
            )
        },
    )

    update_records([results_path], records_path, provider="gemini", model="gemini-3.5-flash")

    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    assert {record["base_case_id"] for record in records} == {
        "asr-open-armstrong-small-step",
        "tts-open-en-us-hello",
    }
    assert [record for record in records if record["base_case_id"].startswith("asr-")] == [
        existing_asr
    ]
    tts_record = next(record for record in records if record["base_case_id"] == "tts-open-en-us-hello")
    assert tts_record["sample_fingerprint"] == "tts-current"
    assert tts_record["semantic_error_summary"] == "No issues."
