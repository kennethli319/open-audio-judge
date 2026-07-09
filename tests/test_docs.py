import json
from html.parser import HTMLParser
from pathlib import Path


class StrictEnoughHtmlParser(HTMLParser):
    def error(self, message: str) -> None:
        raise AssertionError(message)


def test_tts_leaderboard_demo_page_documents_workflow() -> None:
    page = Path("docs/tts-leaderboard-demo.html")
    html = page.read_text(encoding="utf-8")

    parser = StrictEnoughHtmlParser()
    parser.feed(html)
    parser.close()

    required_text = [
        "Open Audio Judge TTS Leaderboard",
        "Multiple MLX Community TTS models generate the same eval set",
        "oaj autojudge-local-tts",
        "--judge-provider gemini",
        "--judge-samples 3",
        "Expected Output Files",
        "synthesis/tts_audio_cases.jsonl",
        "model_summary.json",
        "judge-report/results.jsonl",
        "judge-report/report.html",
        "Sample Report Preview",
        "Demo controls",
        "data-view=\"demo\"",
        "data-view=\"usage\"",
        "data-view=\"reference\"",
        "Demo Run view is selected.",
        "let activeView = Object.prototype.hasOwnProperty.call(sectionViews, requestedView) ? requestedView : \"demo\"",
        "id=\"report-search\"",
        "id=\"model-filter\"",
        "id=\"sample-breakdown-table\"",
        "function applyFilters()",
        "function sortRows()",
        "Eval Set Samples",
        "The public demo eval set contains 440 synthetic, public-safe cases",
        "Paralinguistics",
        "Instruction Following",
        "Speaker Voice Consistency",
        "tts-paralinguistics-reassurance-001",
        "tts-information-tuning-dense-numbers-001",
        "tts-multilingual-code-switching-bilingual-direction-001",
        "tts-long-form-discourse-endurance-001",
        "tts-text-normalization-version-001",
        "tts-acoustic-contexts-radio-001",
        "tts-domain-terminology-science-001",
        "tts-heteronym-disambiguation-read-001",
        "tts-formatting-markup-identifiers-001",
        "tts-safety-privacy-delivery-secret-warning-001",
        "tts-nonverbal-paralinguistic-cues-whisper-001",
        "tts-accent-dialect-handling-indian-support-001",
        "tts-artifact-suppression-repeat-loop-001",
        "tts-temporal-rhythm-control-speed-ramp-001",
        "tts-pragmatic-intent-delivery-pol-refusal-001",
        "tts-compositional-style-control-quiet-urgent-clinic-001",
        "tts-disfluency-repair-control-false-start-001",
        "tts-lexical-stress-disambiguation-record-001",
        "tts-multi-speaker-attribution-quote-boundary-001",
        "tts-phonetic-confusability-minimal-pair-001",
        "tts-referential-cohesion-former-latter-001",
        "tts-measurement-unit-disambiguation-medication-label-001",
        "tts-contextual-abbreviation-expansion-saint-street-001",
        "tts-noise-resilience-delivery-commute-alert-001",
        "tts-uncertainty-calibration-delivery-probability-001",
        "tts-audience-register-adaptation-child-safety-001",
        "tts-real-time-streaming-delivery-first-token-001",
        "tts-dialogue-act-prosody-confirmation-001",
        "tts-address-wayfinding-delivery-street-suite-001",
        "tts-pause-breath-control-brief-long-pause-001",
        "tts-citation-reference-delivery-academic-001",
        "tts-digital-locator-delivery-email-alias-001",
        "tts-conditional-logic-delivery-unless-exception-001",
        "tts-priority-escalation-delivery-critical-alert-001",
        "tts-contrastive-pair-delivery-option-ab-001",
        "tts-statistical-notation-delivery-p-value-ci-001",
        "tts-readback-confirmation-delivery-order-repeat-001",
        "tts-focus-particle-scope-delivery-only-shift-001",
        "tts-slot-value-pairing-delivery-intake-form-001",
        "tts-commitment-scope-delivery-apology-boundary-001",
        "tts-ordinal-ranking-delivery-triage-rerank-001",
        "tts-temporal-relation-delivery-migration-window-001",
        "tts-acronym-initialism-delivery-nasa-eta-001",
        "tts-currency-financial-delivery-usd-cad-001",
        "tts-tool-result-state-delivery-success-warning-001",
        "tts-status-code-delivery-http-429-001",
        "tts-locale-format-disambiguation-us-date-001",
        "Model Leaderboard",
        "Category Leaderboard",
        "no error: 167",
        "style/instruction mismatch: 11",
        "voice drift: 3",
        "Scores By Category",
        "Baseline Model Deltas",
        "Wins / Ties / Losses",
        "42 / 31 / 222",
        "Weakest Segments",
        "conditional_logic_delivery",
        "critical_alert_room_action_escalation",
        "Likely fix areas",
        "priority_escalation_delivery",
        "960 generated",
        "Category Guidance",
        "Focus: unless, only-if, if/otherwise, nested conditions, and exception boundaries.",
        "Source basis: VoiceBench safety/robustness behavior and Seed-TTS-Eval intelligibility checks",
        "searchable/sortable sample-by-sample details",
        "Sample-By-Sample Breakdown",
        "representative rows across models, categories, slices, and score bands",
        "tts-storytelling-bedtime-001-chatterbox",
        "tts-speech-steerability-slow-001-kokoro",
        "tts-information-tuning-dates-times-001-qwen3-tts",
        "tts-paralinguistics-urgency-001-qwen3-tts",
        "Representative Result JSON",
        "judge_sample_scores",
        "mlx-community/chatterbox-turbo-6bit",
        "mlx-community/Kokoro-82M-4bit",
        "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit",
    ]
    for text in required_text:
        assert text in html


def test_asr_leaderboard_demo_page_documents_workflow() -> None:
    page = Path("docs/asr-leaderboard-demo.html")
    html = page.read_text(encoding="utf-8")

    parser = StrictEnoughHtmlParser()
    parser.feed(html)
    parser.close()

    required_text = [
        "Open Audio Judge ASR Leaderboard",
        "Three MLX Community ASR models transcribe the same research-guided eval set",
        "oaj autojudge-mlx-asr",
        "--judge-provider gemini",
        "--judge-samples 3",
        "Expected Output Files",
        "candidate_cases.jsonl",
        "model_summary.json",
        "judge-report/results.jsonl",
        "judge-report/report.html",
        "combined/report.html",
        "transcription_accuracy_wer",
        "entity_factual_integrity",
        "numeric_unit_integrity",
        "negation_modality_scope",
        "temporal_scheduling_accuracy",
        "semantic_paraphrase_preservation",
        "acoustic_noise_robustness",
        "runs/asr-research-audio/tts_audio_cases.jsonl",
        "mlx-community/whisper-large-v3-turbo-asr-fp16",
        "mlx-community/Qwen3-ASR-1.7B-8bit",
        "mlx-community/VibeVoice-ASR-4bit",
        "examples/asr_research_cases.jsonl",
        "docs/asr-eval-taxonomy.md",
        "docs/asr-leaderboard-run-manifest.json",
        "Representative Result JSON",
    ]
    for text in required_text:
        assert text in html


def test_asr_research_docs_list_categories() -> None:
    docs = Path("docs/asr-eval-taxonomy.md").read_text(encoding="utf-8")
    manifest = [
        json.loads(line)
        for line in Path("examples/asr_research_cases.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    categories = {record["metadata"]["eval_category"] for record in manifest}

    for category in categories:
        assert category in docs


def test_tts_multiturn_examples_cover_requested_categories() -> None:
    records = [
        json.loads(line)
        for line in Path("examples/tts_multiturn_cases.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    categories: dict[str, int] = {}
    for record in records:
        category = record["metadata"]["eval_category"]
        categories[category] = categories.get(category, 0) + 1

    assert len(records) == 440
    assert categories == {
        "paralinguistics": 5,
        "instruction_following": 5,
        "information_tuning": 5,
        "storytelling_dialogue": 5,
        "speech_steerability": 5,
        "robustness_intelligibility": 5,
        "speaker_voice_consistency": 5,
        "multilingual_code_switching": 5,
        "long_form_discourse": 5,
        "text_normalization": 5,
        "acoustic_contexts": 5,
        "spontaneous_conversation": 5,
        "affective_transitions": 5,
        "punctuation_prosody": 5,
        "domain_terminology": 5,
        "heteronym_disambiguation": 5,
        "formatting_markup_robustness": 5,
        "nonverbal_paralinguistic_cues": 5,
        "voice_conversion_similarity": 5,
        "accent_dialect_handling": 5,
        "artifact_suppression": 5,
        "temporal_rhythm_control": 5,
        "safety_privacy_delivery": 5,
        "semantic_contrast_focus": 5,
        "dialogue_turn_management": 5,
        "compositional_style_control": 5,
        "named_entity_pronunciation": 5,
        "disfluency_repair_control": 5,
        "lexical_stress_disambiguation": 5,
        "pragmatic_intent_delivery": 5,
        "symbolic_math_reading": 5,
        "multi_speaker_attribution": 5,
        "structured_enumeration_delivery": 5,
        "phonetic_confusability": 5,
        "referential_cohesion": 5,
        "measurement_unit_disambiguation": 5,
        "contextual_abbreviation_expansion": 5,
        "noise_resilience_delivery": 5,
        "audience_register_adaptation": 5,
        "uncertainty_calibration_delivery": 5,
        "real_time_streaming_delivery": 5,
        "numeric_identifier_delivery": 5,
        "sentence_boundary_inference": 5,
        "cross_lingual_name_pronunciation": 5,
        "speech_mode_stability": 5,
        "dialogue_act_prosody": 5,
        "address_wayfinding_delivery": 5,
        "repair_sensitive_delivery": 5,
        "pause_breath_control": 5,
        "citation_reference_delivery": 5,
        "digital_locator_delivery": 5,
        "conditional_logic_delivery": 5,
        "priority_escalation_delivery": 5,
        "contrastive_pair_delivery": 5,
        "statistical_notation_delivery": 5,
        "readback_confirmation_delivery": 5,
        "focus_particle_scope_delivery": 5,
        "slot_value_pairing_delivery": 5,
        "commitment_scope_delivery": 5,
        "ordinal_ranking_delivery": 5,
        "temporal_relation_delivery": 5,
        "syntactic_attachment_delivery": 5,
        "quantifier_scope_delivery": 5,
        "acronym_initialism_delivery": 5,
        "spatial_relation_delivery": 5,
        "homograph_number_format_delivery": 5,
        "currency_financial_delivery": 5,
        "medication_dosage_delivery": 5,
        "morphosyntactic_marker_delivery": 5,
        "operator_precedence_delivery": 5,
        "discourse_marker_intonation": 5,
        "deictic_reference_delivery": 5,
        "ellipsis_fragment_delivery": 5,
        "quoted_reported_speech_delivery": 5,
        "modal_negation_scope_delivery": 5,
        "compound_proper_noun_delivery": 5,
        "range_interval_delivery": 5,
        "email_thread_context_delivery": 5,
        "table_matrix_reading": 5,
        "calendar_schedule_delivery": 5,
        "instruction_conflict_resolution_delivery": 5,
        "tool_result_state_delivery": 5,
        "authorization_access_delivery": 5,
        "policy_clause_delivery": 5,
        "status_code_delivery": 5,
        "menu_option_navigation_delivery": 5,
        "locale_format_disambiguation": 5,
        "accessibility_cue_delivery": 5,
    }
    assert all(record["turns"] for record in records)
    assert all(record["reference_text"] for record in records)
    assert all(record["metadata"]["tts_slice"] for record in records)
    assert all(record["metadata"]["source"] == "research-backed-tts-demo" for record in records)
    assert all(record["metadata"]["source_basis"] for record in records)
    assert all(record["metadata"]["style_prompt"] for record in records)
    assert all(record["metadata"]["expected_style"] for record in records)
    assert all(record["metadata"]["expected_instruction"] for record in records)


def test_tts_leaderboard_demo_lists_every_eval_case() -> None:
    html = Path("docs/tts-leaderboard-demo.html").read_text(encoding="utf-8")
    records = [
        json.loads(line)
        for line in Path("examples/tts_multiturn_cases.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    for record in records:
        assert record["id"] in html
        assert record["metadata"]["tts_slice"] in html
        assert record["reference_text"] in html


def test_readme_links_chatterbox_gemini_sample_page() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "docs/tts-leaderboard-demo.html" in readme
    assert "https://kennethli319.github.io/open-audio-judge/tts-leaderboard-demo.html" in readme
    assert "TTS model leaderboard judged by Gemini" in readme
    assert "docs/tts-eval-taxonomy.md" in readme
