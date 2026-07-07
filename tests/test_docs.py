from html.parser import HTMLParser
from pathlib import Path


class StrictEnoughHtmlParser(HTMLParser):
    def error(self, message: str) -> None:
        raise AssertionError(message)


def test_chatterbox_gemini_sample_page_documents_workflow() -> None:
    page = Path("docs/chatterbox-gemini-sample.html")
    html = page.read_text(encoding="utf-8")

    parser = StrictEnoughHtmlParser()
    parser.feed(html)
    parser.close()

    required_text = [
        "Open Audio Judge Sample",
        "Chatterbox TTS generation scored by the Gemini audio judge",
        "oaj autojudge-local-tts",
        "--judge-provider gemini",
        "Expected Output Files",
        "synthesis/tts_audio_cases.jsonl",
        "model_summary.json",
        "judge-report/results.jsonl",
        "judge-report/report.html",
        "Sample Report Preview",
        "Sample-By-Sample Breakdown",
        "Representative Result JSON",
        "mlx-community/chatterbox-turbo-6bit",
    ]
    for text in required_text:
        assert text in html


def test_readme_links_chatterbox_gemini_sample_page() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "docs/chatterbox-gemini-sample.html" in readme
