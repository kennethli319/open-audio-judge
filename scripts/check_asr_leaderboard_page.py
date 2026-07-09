from __future__ import annotations

import argparse
import html
import json
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.update_asr_leaderboard_demo import END_MARKER, START_MARKER  # noqa: E402


DEFAULT_PAGE = ROOT / "docs" / "asr-leaderboard-demo.html"
DEFAULT_SUMMARY = ROOT / "docs" / "asr-leaderboard-summary.json"


class StrictEnoughHtmlParser(HTMLParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the generated ASR leaderboard demo page and summary artifact.",
    )
    parser.add_argument("--page", type=Path, default=DEFAULT_PAGE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    summary = check_asr_leaderboard_page(args.page, summary_path=args.summary)
    print(
        "Validated ASR leaderboard page "
        f"{summary['page']} against {summary['summary_path']} "
        f"({summary['total_results']} results, {summary['model_count']} models, "
        f"{summary['category_count']} categories)."
    )


def check_asr_leaderboard_page(page: Path, *, summary_path: Path) -> dict[str, Any]:
    if not page.exists():
        raise FileNotFoundError(f"Missing ASR leaderboard page: {page}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing ASR leaderboard summary: {summary_path}")

    html = page.read_text(encoding="utf-8")
    parser = StrictEnoughHtmlParser()
    parser.feed(html)
    parser.close()
    if START_MARKER not in html or END_MARKER not in html:
        raise ValueError(f"{page} must contain generated ASR leaderboard markers.")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _validate_summary(summary, summary_path=summary_path)
    required_text = _required_page_text(summary)
    missing = [text for text in required_text if text not in html]
    if missing:
        raise ValueError(f"{page} is missing required ASR leaderboard text: {missing}")

    return {
        "status": "complete",
        "page": _repo_relative(page),
        "summary_path": _repo_relative(summary_path),
        "total_results": summary["total_results"],
        "model_count": summary["model_count"],
        "category_count": summary["category_count"],
    }


def _validate_summary(summary: Any, *, summary_path: Path) -> None:
    if not isinstance(summary, dict):
        raise ValueError(f"{summary_path} must contain a JSON object.")

    required_keys = (
        "results_path",
        "report_path",
        "total_results",
        "model_count",
        "category_count",
        "expected_cases_per_model",
        "models",
        "categories",
        "refresh_workflow",
    )
    missing_keys = [key for key in required_keys if key not in summary]
    if missing_keys:
        raise ValueError(f"{summary_path} is missing required keys: {missing_keys}")

    if summary["total_results"] < 1:
        raise ValueError("ASR leaderboard summary must include at least one result.")
    if summary["model_count"] != len(summary["models"]):
        raise ValueError("ASR leaderboard summary model_count does not match models.")
    if summary["category_count"] != len(summary["categories"]):
        raise ValueError("ASR leaderboard summary category_count does not match categories.")

    _validate_referenced_artifacts(summary, summary_path=summary_path)


def _validate_referenced_artifacts(summary: dict[str, Any], *, summary_path: Path) -> None:
    artifact_keys = ("results_path", "report_path")
    missing = []
    for key in artifact_keys:
        raw_path = summary.get(key)
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"{summary_path} has invalid {key}: {raw_path!r}")
        path = _resolve_summary_path(raw_path)
        if not path.exists():
            missing.append((key, raw_path))

    raw_source_paths = summary.get("source_result_paths", [])
    if not isinstance(raw_source_paths, list):
        raise ValueError(f"{summary_path} source_result_paths must be a list.")
    for raw_path in raw_source_paths:
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"{summary_path} has invalid source_result_paths entry: {raw_path!r}")
        path = _resolve_summary_path(raw_path)
        if not path.exists():
            missing.append(("source_result_paths", raw_path))

    if missing:
        formatted = ", ".join(f"{key}={path}" for key, path in missing)
        raise ValueError(f"{summary_path} references missing ASR artifact(s): {formatted}")


def _required_page_text(summary: dict[str, Any]) -> list[str]:
    model_names = [
        str(model["model"])
        for model in summary.get("models", [])
        if isinstance(model, dict) and model.get("model")
    ]
    category_names = [
        str(category["category"])
        for category in summary.get("categories", [])
        if isinstance(category, dict) and category.get("category")
    ]
    workflow = summary.get("refresh_workflow", {})
    commands = []
    if isinstance(workflow, dict):
        commands = [
            _rendered_command_text(command)
            for key, command in workflow.items()
            if key
            in {
                "seed_manifest_validation_command",
                "audio_materialization_command",
                "model_run_template",
                "manifest_refresh_command",
                "page_validation_command",
                "hosted_artifact_command",
            }
            if isinstance(command, list)
        ]

    return [
        "Open Audio Judge ASR Leaderboard",
        "Verified Leaderboard Results",
        "Category Breakdown",
        "Generated Refresh Workflow",
        "Generated Artifacts",
        f"{summary['total_results']} judged transcripts",
        *model_names,
        *category_names,
        *commands,
    ]


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_summary_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return ROOT / path


def _rendered_command_text(command: list[object]) -> str:
    text = " ".join(str(part) for part in command)
    if "<" in text or ">" in text:
        return html.escape(text)
    return text


if __name__ == "__main__":
    main()
