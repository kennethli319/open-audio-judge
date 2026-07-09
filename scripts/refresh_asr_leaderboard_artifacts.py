from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from open_audio_judge.reports import write_html_report  # noqa: E402
from open_audio_judge.runner import load_results_jsonl, write_results_jsonl  # noqa: E402
from scripts.update_asr_leaderboard_demo import (  # noqa: E402
    DEFAULT_PAGE,
    DEFAULT_SUMMARY,
    render_generated_sections,
    replace_generated_block,
    write_summary_artifact,
)


DEFAULT_COMBINED_OUT = ROOT / "runs" / "asr-leaderboard" / "full-35-combined"
DEFAULT_RUN_MANIFEST = ROOT / "docs" / "asr-leaderboard-run-manifest.json"
FULL_RUN_RESULT_PATHS = [
    ROOT
    / "runs"
    / "asr-leaderboard"
    / "whisper-large-v3-turbo-full-gap"
    / "judge-report"
    / "results.jsonl",
    ROOT
    / "runs"
    / "asr-leaderboard"
    / "qwen3-asr-1.7b-full-gap"
    / "judge-report"
    / "results.jsonl",
    ROOT
    / "runs"
    / "asr-leaderboard"
    / "vibevoice-asr-full-gap"
    / "judge-report"
    / "results.jsonl",
]
SEGMENTED_MODEL_RUNS = [
    "whisper-large-v3-turbo-smoke",
    "whisper-large-v3-turbo-full-gap",
    "whisper-large-v3-turbo-semantic-smoke",
    "whisper-large-v3-turbo-entity-smoke",
    "whisper-large-v3-turbo-paraphrase-smoke",
    "whisper-large-v3-turbo-noise-smoke",
    "qwen3-asr-1.7b-smoke",
    "qwen3-asr-1.7b-full-gap",
    "qwen3-asr-1.7b-semantic-smoke",
    "qwen3-asr-1.7b-entity-smoke",
    "qwen3-asr-1.7b-paraphrase-smoke",
    "qwen3-asr-1.7b-noise-smoke",
    "vibevoice-asr-smoke",
    "vibevoice-asr-full-gap",
    "vibevoice-asr-semantic-smoke",
    "vibevoice-asr-entity-smoke",
    "vibevoice-asr-paraphrase-smoke",
    "vibevoice-asr-noise-smoke",
]
SEGMENTED_RESULT_PATHS = [
    ROOT / "runs" / "asr-leaderboard" / run / "judge-report" / "results.jsonl"
    for run in SEGMENTED_MODEL_RUNS
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combine verified ASR model result files, rewrite the combined report, "
            "and refresh the hosted ASR leaderboard demo artifacts."
        ),
    )
    parser.add_argument(
        "--results",
        action="append",
        type=Path,
        default=[],
        help=(
            "Model results.jsonl path or model run directory. Repeat for each run. "
            "Defaults to complete full-run files when available, otherwise the segmented full-35 runs."
        ),
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_COMBINED_OUT)
    parser.add_argument("--page", type=Path, default=DEFAULT_PAGE)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--run-manifest",
        type=Path,
        default=DEFAULT_RUN_MANIFEST,
        help=(
            "JSON manifest listing verified ASR result files. Used when --results is omitted; "
            "set to a missing path to fall back to built-in historical run names."
        ),
    )
    parser.add_argument(
        "--hosted-dir",
        type=Path,
        help=(
            "Optional kennethli319.github.io/open-audio-judge directory. When set, copy the "
            "refreshed ASR demo page, summary JSON, and run manifest there."
        ),
    )
    parser.add_argument(
        "--expected-cases-per-model",
        type=int,
        default=35,
        help="Fail unless every model has this many ok judged results.",
    )
    args = parser.parse_args()

    result_paths = (
        [_normalize_results_path(path) for path in args.results]
        if args.results
        else _default_result_paths(
            args.expected_cases_per_model,
            run_manifest=args.run_manifest,
        )
    )
    refresh_asr_leaderboard_artifacts(
        result_paths,
        out=args.out,
        page=args.page,
        summary_out=args.summary_out,
        hosted_dir=args.hosted_dir,
        expected_cases_per_model=args.expected_cases_per_model,
    )


def refresh_asr_leaderboard_artifacts(
    result_paths: list[Path],
    *,
    out: Path,
    page: Path,
    summary_out: Path,
    expected_cases_per_model: int,
    hosted_dir: Path | None = None,
) -> None:
    result_paths = [_normalize_results_path(path) for path in result_paths]
    for path in result_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing ASR result file: {path}")

    combined_results = [
        result
        for path in result_paths
        for result in load_results_jsonl(path)
    ]
    if not combined_results:
        raise ValueError("No ASR evaluation results were loaded.")

    combined_results_path = out / "results.jsonl"
    generated = render_generated_sections(
        combined_results,
        results_path=combined_results_path,
        expected_cases_per_model=expected_cases_per_model,
    )

    out.mkdir(parents=True, exist_ok=True)
    combined_report_path = out / "report.html"
    write_results_jsonl(combined_results, combined_results_path)
    write_html_report(combined_results, combined_report_path)
    replace_generated_block(page, generated)
    write_summary_artifact(
        combined_results,
        summary_out,
        results_path=combined_results_path,
        expected_cases_per_model=expected_cases_per_model,
        source_result_paths=result_paths,
    )
    copied_hosted_paths = (
        copy_hosted_asr_artifacts(
            hosted_dir,
            page=page,
            summary_out=summary_out,
            run_manifest=DEFAULT_RUN_MANIFEST,
        )
        if hosted_dir
        else []
    )

    print(f"Combined {len(combined_results)} ASR results from {len(result_paths)} files")
    print(f"Results: {combined_results_path}")
    print(f"Report:  {combined_report_path}")
    print(f"Page:    {page}")
    print(f"Summary: {summary_out}")
    for copied_path in copied_hosted_paths:
        print(f"Hosted:  {copied_path}")


def copy_hosted_asr_artifacts(
    hosted_dir: Path,
    *,
    page: Path = DEFAULT_PAGE,
    summary_out: Path = DEFAULT_SUMMARY,
    run_manifest: Path = DEFAULT_RUN_MANIFEST,
) -> list[Path]:
    hosted_dir.mkdir(parents=True, exist_ok=True)
    copied_paths = []
    for source in (page, summary_out, run_manifest):
        if not source.exists():
            raise FileNotFoundError(f"Missing hosted ASR source artifact: {source}")
        destination = hosted_dir / source.name
        shutil.copyfile(source, destination)
        copied_paths.append(destination)
    return copied_paths


def _normalize_results_path(path: Path) -> Path:
    if path.is_dir():
        direct = path / "results.jsonl"
        nested = path / "judge-report" / "results.jsonl"
        return direct if direct.exists() else nested
    return path


def _default_result_paths(expected_cases_per_model: int, *, run_manifest: Path) -> list[Path]:
    errors = []
    if run_manifest.exists():
        try:
            paths = _result_paths_from_run_manifest(run_manifest)
            _validate_candidate_paths(paths, expected_cases_per_model=expected_cases_per_model)
        except Exception as exc:
            errors.append(f"manifest {run_manifest}: {exc}")
        else:
            return paths

    for label, paths in (
        ("full-run", FULL_RUN_RESULT_PATHS),
        ("segmented", SEGMENTED_RESULT_PATHS),
    ):
        try:
            _validate_candidate_paths(paths, expected_cases_per_model=expected_cases_per_model)
        except Exception as exc:
            errors.append(f"{label}: {exc}")
        else:
            return paths
    raise ValueError(
        "Default ASR result discovery did not find complete coverage. "
        + " ".join(errors)
        + " Pass --results for each verified model result file or run directory."
    )


def _result_paths_from_run_manifest(manifest_path: Path) -> list[Path]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Run manifest must be a JSON object.")

    raw_paths = data.get("result_paths")
    if raw_paths is None:
        runs = data.get("runs")
        if not isinstance(runs, list):
            raise ValueError("Run manifest must include result_paths or runs.")
        raw_paths = [
            run.get("results_path")
            for run in runs
            if isinstance(run, dict)
        ]

    if not isinstance(raw_paths, list) or not raw_paths:
        raise ValueError("Run manifest did not list any result paths.")

    paths = []
    for raw_path in raw_paths:
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"Invalid result path in run manifest: {raw_path!r}")
        path = Path(raw_path)
        if not path.is_absolute():
            path = ROOT / path
        paths.append(_normalize_results_path(path))
    return paths


def _validate_candidate_paths(paths: list[Path], *, expected_cases_per_model: int) -> None:
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    combined_results = [
        result
        for path in paths
        for result in load_results_jsonl(path)
    ]
    render_generated_sections(
        combined_results,
        results_path=DEFAULT_COMBINED_OUT / "results.jsonl",
        expected_cases_per_model=expected_cases_per_model,
    )


if __name__ == "__main__":
    main()
