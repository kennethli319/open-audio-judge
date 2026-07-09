from __future__ import annotations

import argparse
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
        "--expected-cases-per-model",
        type=int,
        default=35,
        help="Fail unless every model has this many ok judged results.",
    )
    args = parser.parse_args()

    result_paths = (
        [_normalize_results_path(path) for path in args.results]
        if args.results
        else _default_result_paths(args.expected_cases_per_model)
    )
    refresh_asr_leaderboard_artifacts(
        result_paths,
        out=args.out,
        page=args.page,
        summary_out=args.summary_out,
        expected_cases_per_model=args.expected_cases_per_model,
    )


def refresh_asr_leaderboard_artifacts(
    result_paths: list[Path],
    *,
    out: Path,
    page: Path,
    summary_out: Path,
    expected_cases_per_model: int,
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
    )

    print(f"Combined {len(combined_results)} ASR results from {len(result_paths)} files")
    print(f"Results: {combined_results_path}")
    print(f"Report:  {combined_report_path}")
    print(f"Page:    {page}")
    print(f"Summary: {summary_out}")


def _normalize_results_path(path: Path) -> Path:
    if path.is_dir():
        direct = path / "results.jsonl"
        nested = path / "judge-report" / "results.jsonl"
        return direct if direct.exists() else nested
    return path


def _default_result_paths(expected_cases_per_model: int) -> list[Path]:
    errors = []
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
