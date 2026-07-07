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
        "The public demo eval set contains 60 synthetic, public-safe cases",
        "Paralinguistics",
        "Instruction Following",
        "Speaker Voice Consistency",
        "tts-paralinguistics-reassurance-001",
        "tts-information-tuning-dense-numbers-001",
        "tts-multilingual-code-switching-bilingual-direction-001",
        "tts-long-form-discourse-endurance-001",
        "tts-text-normalization-version-001",
        "tts-acoustic-contexts-radio-001",
        "Model Leaderboard",
        "Category Leaderboard",
        "no error: 104",
        "style/instruction mismatch: 7",
        "voice drift: 2",
        "Scores By Category",
        "Baseline Model Deltas",
        "Wins / Ties / Losses",
        "Weakest Segments",
        "Likely fix areas",
        "Category Guidance",
        "Category focus: numbers, dates, units, ordered steps, and safety-critical wording.",
        "Source basis: Seed-TTS-Eval intelligibility checks",
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

    assert len(records) == 60
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
