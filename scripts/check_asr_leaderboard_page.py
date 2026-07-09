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
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=ROOT,
        help="Root used to resolve relative artifact paths from the summary.",
    )
    parser.add_argument(
        "--path-map",
        action="append",
        default=[],
        metavar="FROM=TO",
        help=(
            "Rewrite a summary artifact path prefix before resolving it. "
            "Repeat for hosted layouts, for example docs/= and runs/asr-leaderboard/=asr-leaderboard/."
        ),
    )
    parser.add_argument(
        "--allow-missing-source-results",
        action="store_true",
        help="Allow source_result_paths to be absent when validating a hosted artifact mirror.",
    )
    args = parser.parse_args()

    summary = check_asr_leaderboard_page(
        args.page,
        summary_path=args.summary,
        artifact_root=args.artifact_root,
        path_maps=parse_path_maps(args.path_map),
        allow_missing_source_results=args.allow_missing_source_results,
    )
    print(
        "Validated ASR leaderboard page "
        f"{summary['page']} against {summary['summary_path']} "
        f"({summary['total_results']} results, {summary['model_count']} models, "
        f"{summary['category_count']} categories)."
    )


def check_asr_leaderboard_page(
    page: Path,
    *,
    summary_path: Path,
    artifact_root: Path = ROOT,
    path_maps: list[tuple[str, str]] | None = None,
    allow_missing_source_results: bool = False,
) -> dict[str, Any]:
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
    _validate_summary(
        summary,
        summary_path=summary_path,
        artifact_root=artifact_root,
        path_maps=path_maps or [],
        allow_missing_source_results=allow_missing_source_results,
    )
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
        "output_artifact_count": len(summary["output_artifacts"]),
    }


def _validate_summary(
    summary: Any,
    *,
    summary_path: Path,
    artifact_root: Path,
    path_maps: list[tuple[str, str]],
    allow_missing_source_results: bool,
) -> None:
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

    _validate_referenced_artifacts(
        summary,
        summary_path=summary_path,
        artifact_root=artifact_root,
        path_maps=path_maps,
        allow_missing_source_results=allow_missing_source_results,
    )
    _validate_run_manifest_artifact(
        summary,
        summary_path=summary_path,
        artifact_root=artifact_root,
        path_maps=path_maps,
    )
    _validate_output_artifacts(
        summary,
        summary_path=summary_path,
        artifact_root=artifact_root,
        path_maps=path_maps,
    )
    _validate_status_artifact(
        summary,
        key="manifest_validation_path",
        summary_path=summary_path,
        artifact_root=artifact_root,
        path_maps=path_maps,
        expected_fields={
            "total_results": summary["total_results"],
            "model_count": summary["model_count"],
            "category_count": summary["category_count"],
            "expected_cases_per_model": summary["expected_cases_per_model"],
        },
    )
    _validate_status_artifact(
        summary,
        key="seed_manifest_validation_path",
        summary_path=summary_path,
        artifact_root=artifact_root,
        path_maps=path_maps,
    )


def _validate_referenced_artifacts(
    summary: dict[str, Any],
    *,
    summary_path: Path,
    artifact_root: Path,
    path_maps: list[tuple[str, str]],
    allow_missing_source_results: bool,
) -> None:
    artifact_keys = (
        "results_path",
        "report_path",
        "run_manifest_path",
        "manifest_validation_path",
        "seed_manifest_validation_path",
    )
    missing = []
    for key in artifact_keys:
        raw_path = summary.get(key)
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"{summary_path} has invalid {key}: {raw_path!r}")
        path = _resolve_summary_path(raw_path, artifact_root=artifact_root, path_maps=path_maps)
        if not path.exists():
            missing.append((key, raw_path))

    raw_source_paths = summary.get("source_result_paths", [])
    if not isinstance(raw_source_paths, list):
        raise ValueError(f"{summary_path} source_result_paths must be a list.")
    for raw_path in raw_source_paths:
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"{summary_path} has invalid source_result_paths entry: {raw_path!r}")
        path = _resolve_summary_path(raw_path, artifact_root=artifact_root, path_maps=path_maps)
        if not path.exists() and not allow_missing_source_results:
            missing.append(("source_result_paths", raw_path))

    if missing:
        formatted = ", ".join(f"{key}={path}" for key, path in missing)
        raise ValueError(f"{summary_path} references missing ASR artifact(s): {formatted}")


def _validate_run_manifest_artifact(
    summary: dict[str, Any],
    *,
    summary_path: Path,
    artifact_root: Path,
    path_maps: list[tuple[str, str]],
) -> None:
    raw_path = summary.get("run_manifest_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{summary_path} has invalid run_manifest_path: {raw_path!r}")
    path = _resolve_summary_path(raw_path, artifact_root=artifact_root, path_maps=path_maps)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} must contain valid JSON: {exc}") from exc

    if not isinstance(manifest, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    if manifest.get("expected_cases_per_model") != summary["expected_cases_per_model"]:
        raise ValueError(
            f"{path} expected_cases_per_model={manifest.get('expected_cases_per_model')!r} "
            f"does not match {summary_path} expected_cases_per_model="
            f"{summary['expected_cases_per_model']!r}."
        )

    result_paths = manifest.get("result_paths")
    runs = manifest.get("runs")
    if not isinstance(result_paths, list) or not result_paths:
        raise ValueError(f"{path} must include a non-empty result_paths list.")
    if not all(isinstance(result_path, str) and result_path for result_path in result_paths):
        raise ValueError(f"{path} result_paths entries must be non-empty strings.")
    if not isinstance(runs, list) or not runs:
        raise ValueError(f"{path} must include a non-empty runs list.")

    summary_sources = summary.get("source_result_paths", [])
    if summary_sources:
        if result_paths != summary_sources:
            raise ValueError(
                f"{path} result_paths do not match {summary_path} source_result_paths."
            )

    summary_models = {
        str(model["model"])
        for model in summary.get("models", [])
        if isinstance(model, dict) and model.get("model")
    }
    run_paths = []
    counts_by_model: dict[str, int] = {}
    ok_counts_by_model: dict[str, int] = {}
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise ValueError(f"{path} runs[{index}] must be an object.")
        model = run.get("model")
        results_path = run.get("results_path")
        result_count = run.get("result_count")
        ok_count = run.get("ok_count")
        category_counts = run.get("category_counts")
        if not isinstance(model, str) or not model:
            raise ValueError(f"{path} runs[{index}] has invalid model: {model!r}")
        if summary_models and model not in summary_models:
            raise ValueError(f"{path} runs[{index}] model is not present in the summary: {model}")
        if not isinstance(results_path, str) or not results_path:
            raise ValueError(f"{path} runs[{index}] has invalid results_path: {results_path!r}")
        if not isinstance(result_count, int) or result_count < 1:
            raise ValueError(f"{path} runs[{index}] has invalid result_count: {result_count!r}")
        if not isinstance(ok_count, int) or ok_count < 0 or ok_count > result_count:
            raise ValueError(f"{path} runs[{index}] has invalid ok_count: {ok_count!r}")
        if not isinstance(category_counts, dict) or not category_counts:
            raise ValueError(f"{path} runs[{index}] must include non-empty category_counts.")
        if not all(isinstance(count, int) and count >= 0 for count in category_counts.values()):
            raise ValueError(f"{path} runs[{index}] has invalid category_counts: {category_counts!r}")
        if sum(category_counts.values()) != result_count:
            raise ValueError(f"{path} runs[{index}] category_counts do not sum to result_count.")

        run_paths.append(results_path)
        counts_by_model[model] = counts_by_model.get(model, 0) + result_count
        ok_counts_by_model[model] = ok_counts_by_model.get(model, 0) + ok_count

    if run_paths != result_paths:
        raise ValueError(f"{path} runs results_path values do not match result_paths.")

    for model in summary.get("models", []):
        if not isinstance(model, dict):
            continue
        model_name = model.get("model")
        result_count = model.get("result_count")
        ok_count = model.get("ok_count")
        if not isinstance(model_name, str) or not model_name:
            continue
        if result_count is not None and counts_by_model.get(model_name) != result_count:
            raise ValueError(
                f"{path} aggregated result_count for {model_name}="
                f"{counts_by_model.get(model_name)!r} does not match summary result_count="
                f"{result_count!r}."
            )
        if ok_count is not None and ok_counts_by_model.get(model_name) != ok_count:
            raise ValueError(
                f"{path} aggregated ok_count for {model_name}="
                f"{ok_counts_by_model.get(model_name)!r} does not match summary ok_count="
                f"{ok_count!r}."
            )


def _validate_output_artifacts(
    summary: dict[str, Any],
    *,
    summary_path: Path,
    artifact_root: Path,
    path_maps: list[tuple[str, str]],
) -> None:
    artifacts = summary.get("output_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError(f"{summary_path} must include a non-empty output_artifacts list.")

    missing = []
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            raise ValueError(f"{summary_path} output_artifacts[{index}] must be an object.")
        raw_path = artifact.get("path")
        purpose = artifact.get("purpose")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"{summary_path} output_artifacts[{index}] has invalid path: {raw_path!r}")
        if not isinstance(purpose, str) or not purpose:
            raise ValueError(f"{summary_path} output_artifacts[{index}] has invalid purpose: {purpose!r}")
        if not _resolve_summary_path(raw_path, artifact_root=artifact_root, path_maps=path_maps).exists():
            missing.append(raw_path)

    if missing:
        formatted = ", ".join(missing)
        raise ValueError(f"{summary_path} output_artifacts reference missing ASR artifact(s): {formatted}")


def _validate_status_artifact(
    summary: dict[str, Any],
    *,
    key: str,
    summary_path: Path,
    artifact_root: Path,
    path_maps: list[tuple[str, str]],
    expected_fields: dict[str, object] | None = None,
) -> None:
    raw_path = summary.get(key)
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{summary_path} has invalid {key}: {raw_path!r}")
    path = _resolve_summary_path(raw_path, artifact_root=artifact_root, path_maps=path_maps)
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} must contain valid JSON: {exc}") from exc

    if not isinstance(artifact, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    if artifact.get("status") != "complete":
        raise ValueError(f"{path} status must be complete.")

    for field, expected in (expected_fields or {}).items():
        if artifact.get(field) != expected:
            raise ValueError(
                f"{path} {field}={artifact.get(field)!r} does not match "
                f"{summary_path} {field}={expected!r}."
            )


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
    artifacts = summary.get("output_artifacts", [])
    artifact_text = []
    if isinstance(artifacts, list):
        for artifact in artifacts:
            if isinstance(artifact, dict):
                raw_path = artifact.get("path")
                purpose = artifact.get("purpose")
                if isinstance(raw_path, str) and raw_path:
                    artifact_text.append(raw_path)
                if isinstance(purpose, str) and purpose:
                    artifact_text.append(purpose)

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
        *artifact_text,
    ]


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def parse_path_maps(raw_maps: list[str]) -> list[tuple[str, str]]:
    path_maps = []
    for raw_map in raw_maps:
        if "=" not in raw_map:
            raise ValueError(f"Invalid --path-map value: {raw_map!r}; expected FROM=TO.")
        source, destination = raw_map.split("=", 1)
        if not source:
            raise ValueError(f"Invalid --path-map value: {raw_map!r}; FROM must be non-empty.")
        path_maps.append((source, destination))
    return path_maps


def _resolve_summary_path(
    raw_path: str,
    *,
    artifact_root: Path,
    path_maps: list[tuple[str, str]],
) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    mapped = raw_path
    for source, destination in path_maps:
        if mapped == source.rstrip("/"):
            mapped = destination.rstrip("/")
            break
        if mapped.startswith(source):
            mapped = destination + mapped[len(source):]
            break
    return artifact_root / mapped


def _rendered_command_text(command: list[object]) -> str:
    text = " ".join(str(part) for part in command)
    if "<" in text or ">" in text:
        return html.escape(text)
    return text


if __name__ == "__main__":
    main()
