from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from open_audio_judge.reports import write_html_report  # noqa: E402
from open_audio_judge.runner import load_cases, load_results_jsonl, write_results_jsonl  # noqa: E402
from scripts.check_asr_leaderboard_page import check_asr_leaderboard_page  # noqa: E402
from scripts.update_asr_leaderboard_demo import (  # noqa: E402
    DEFAULT_PAGE,
    DEFAULT_REFRESH_REPORT,
    DEFAULT_SEED_MANIFEST_VALIDATION,
    DEFAULT_SUMMARY,
    render_generated_sections,
    replace_generated_block,
    write_refresh_report,
    write_summary_artifact,
)
from scripts.validate_asr_seed_manifest import (  # noqa: E402
    DEFAULT_CASES,
    validate_asr_seed_manifest,
)


DEFAULT_COMBINED_OUT = ROOT / "runs" / "asr-leaderboard" / "full-35-combined"
DEFAULT_RUN_MANIFEST = ROOT / "docs" / "asr-leaderboard-run-manifest.json"
DEFAULT_MANIFEST_VALIDATION = ROOT / "docs" / "asr-leaderboard-manifest-validation.json"
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
    parser.add_argument("--refresh-report-out", type=Path, default=DEFAULT_REFRESH_REPORT)
    parser.add_argument(
        "--manifest-validation-out",
        type=Path,
        default=DEFAULT_MANIFEST_VALIDATION,
        help="Write a machine-readable validation summary for the ASR run manifest.",
    )
    parser.add_argument(
        "--seed-cases",
        type=Path,
        default=DEFAULT_CASES,
        help="ASR seed manifest to validate as part of the refresh.",
    )
    parser.add_argument(
        "--seed-manifest-validation-out",
        type=Path,
        default=DEFAULT_SEED_MANIFEST_VALIDATION,
        help="Write seed-manifest validation summary for the ASR demo artifact set.",
    )
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
        "--update-run-manifest",
        action="store_true",
        help="Rewrite --run-manifest from the verified result files before validation and hosted sync.",
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
        refresh_report_out=args.refresh_report_out,
        manifest_validation_out=args.manifest_validation_out,
        seed_cases=args.seed_cases,
        seed_manifest_validation_out=args.seed_manifest_validation_out,
        run_manifest=args.run_manifest,
        update_run_manifest=args.update_run_manifest,
        hosted_dir=args.hosted_dir,
        expected_cases_per_model=args.expected_cases_per_model,
    )


def refresh_asr_leaderboard_artifacts(
    result_paths: list[Path],
    *,
    out: Path,
    page: Path,
    summary_out: Path,
    refresh_report_out: Path,
    manifest_validation_out: Path,
    run_manifest: Path,
    expected_cases_per_model: int,
    update_run_manifest: bool = False,
    seed_cases: Path = DEFAULT_CASES,
    seed_manifest_validation_out: Path = DEFAULT_SEED_MANIFEST_VALIDATION,
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
    if update_run_manifest:
        write_run_manifest_artifact(
            result_paths,
            run_manifest,
            expected_cases_per_model=expected_cases_per_model,
        )
    replace_generated_block(page, generated)
    write_summary_artifact(
        combined_results,
        summary_out,
        results_path=combined_results_path,
        expected_cases_per_model=expected_cases_per_model,
        source_result_paths=result_paths,
    )
    write_refresh_report(
        combined_results,
        refresh_report_out,
        results_path=combined_results_path,
        expected_cases_per_model=expected_cases_per_model,
        source_result_paths=result_paths,
    )
    write_manifest_validation_artifact(
        combined_results,
        manifest_validation_out,
        result_paths=result_paths,
        run_manifest=run_manifest,
        expected_cases_per_model=expected_cases_per_model,
    )
    write_seed_manifest_validation_artifact(
        seed_cases,
        seed_manifest_validation_out,
    )
    copied_hosted_paths = (
        copy_hosted_asr_artifacts(
            hosted_dir,
            page=page,
            summary_out=summary_out,
            refresh_report_out=refresh_report_out,
            run_manifest=run_manifest,
            manifest_validation_out=manifest_validation_out,
            seed_manifest_validation_out=seed_manifest_validation_out,
            combined_results_path=combined_results_path,
            combined_report_path=combined_report_path,
        )
        if hosted_dir
        else []
    )
    if hosted_dir:
        check_asr_leaderboard_page(
            hosted_dir / page.name,
            summary_path=hosted_dir / summary_out.name,
            artifact_root=hosted_dir,
            path_maps=[
                ("docs/", ""),
                ("runs/asr-leaderboard/", "asr-leaderboard/"),
            ],
            allow_missing_source_results=True,
        )

    print(f"Combined {len(combined_results)} ASR results from {len(result_paths)} files")
    print(f"Results: {combined_results_path}")
    print(f"Report:  {combined_report_path}")
    print(f"Page:    {page}")
    print(f"Summary: {summary_out}")
    print(f"Refresh report: {refresh_report_out}")
    print(f"Manifest validation: {manifest_validation_out}")
    print(f"Seed manifest validation: {seed_manifest_validation_out}")
    for copied_path in copied_hosted_paths:
        print(f"Hosted:  {copied_path}")


def write_run_manifest_artifact(
    result_paths: list[Path],
    output_path: Path,
    *,
    expected_cases_per_model: int,
) -> None:
    runs = []
    for path in result_paths:
        results = load_results_jsonl(path)
        if not results:
            raise ValueError(f"Cannot add empty result file to run manifest: {path}")
        models = sorted(
            {
                str(result.metadata.get("candidate_model") or "")
                for result in results
            }
        )
        if len(models) != 1 or not models[0]:
            raise ValueError(
                f"Run manifest source must contain exactly one model with metadata.candidate_model: {path}"
            )
        categories = Counter(str(result.metadata.get("eval_category") or "") for result in results)
        if any(not category for category in categories):
            raise ValueError(f"Run manifest source has missing metadata.eval_category: {path}")
        runs.append(
            {
                "run_name": _run_name_from_results_path(path),
                "model": models[0],
                "results_path": _repo_relative(path),
                "result_count": len(results),
                "ok_count": sum(1 for result in results if result.status == "ok"),
                "category_counts": dict(sorted(categories.items())),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "description": "Verified ASR leaderboard source result files for the full 35-case MLX ASR demo.",
                "version": 2,
                "expected_cases_per_model": expected_cases_per_model,
                "generated_audio_manifest": "runs/asr-research-audio/tts_audio_cases.jsonl",
                "result_paths": [run["results_path"] for run in runs],
                "runs": runs,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _run_name_from_results_path(path: Path) -> str:
    if path.name == "results.jsonl" and path.parent.name == "judge-report":
        return path.parent.parent.name
    if path.name == "results.jsonl":
        return path.parent.name
    return path.stem


def copy_hosted_asr_artifacts(
    hosted_dir: Path,
    *,
    page: Path = DEFAULT_PAGE,
    summary_out: Path = DEFAULT_SUMMARY,
    refresh_report_out: Path = DEFAULT_REFRESH_REPORT,
    run_manifest: Path = DEFAULT_RUN_MANIFEST,
    manifest_validation_out: Path = DEFAULT_MANIFEST_VALIDATION,
    seed_manifest_validation_out: Path = DEFAULT_SEED_MANIFEST_VALIDATION,
    combined_results_path: Path | None = None,
    combined_report_path: Path | None = None,
) -> list[Path]:
    hosted_dir.mkdir(parents=True, exist_ok=True)
    copied_paths = []
    source_destinations = (
        (page, {page.name, DEFAULT_PAGE.name}),
        (summary_out, {summary_out.name, DEFAULT_SUMMARY.name}),
        (refresh_report_out, {refresh_report_out.name, DEFAULT_REFRESH_REPORT.name}),
        (run_manifest, {run_manifest.name, DEFAULT_RUN_MANIFEST.name}),
        (manifest_validation_out, {manifest_validation_out.name, DEFAULT_MANIFEST_VALIDATION.name}),
        (seed_manifest_validation_out, {seed_manifest_validation_out.name, DEFAULT_SEED_MANIFEST_VALIDATION.name}),
    )
    for source, destination_names in source_destinations:
        if not source.exists():
            raise FileNotFoundError(f"Missing hosted ASR source artifact: {source}")
        for destination_name in sorted(destination_names):
            destination = hosted_dir / destination_name
            shutil.copyfile(source, destination)
            copied_paths.append(destination)

    if combined_results_path or combined_report_path:
        hosted_combined_dir = hosted_dir / "asr-leaderboard" / "full-35-combined"
        hosted_combined_dir.mkdir(parents=True, exist_ok=True)
        for source in (combined_results_path, combined_report_path):
            if source is None:
                continue
            if not source.exists():
                raise FileNotFoundError(f"Missing hosted ASR combined artifact: {source}")
            destination = hosted_combined_dir / source.name
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


def write_manifest_validation_artifact(
    results: list,
    output_path: Path,
    *,
    result_paths: list[Path],
    run_manifest: Path,
    expected_cases_per_model: int,
) -> None:
    validation = build_manifest_validation(
        results,
        result_paths=result_paths,
        run_manifest=run_manifest,
        expected_cases_per_model=expected_cases_per_model,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_seed_manifest_validation_artifact(
    cases_path: Path,
    output_path: Path,
    *,
    expected_cases_per_category: int = 5,
) -> None:
    cases = load_cases(cases_path)
    validation = validate_asr_seed_manifest(
        cases,
        cases_path=cases_path,
        expected_cases_per_category=expected_cases_per_category,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_manifest_validation(
    results: list,
    *,
    result_paths: list[Path],
    run_manifest: Path,
    expected_cases_per_model: int,
) -> dict[str, object]:
    model_category_counts: dict[str, dict[str, int]] = {}
    model_counts: dict[str, int] = {}
    model_ok_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    manifest_models_by_path = _manifest_declared_models_by_result_path(run_manifest)

    for result in results:
        model = str(result.metadata.get("candidate_model") or "")
        category = str(result.metadata.get("eval_category") or "")
        model_counts[model] = model_counts.get(model, 0) + 1
        if result.status == "ok":
            model_ok_counts[model] = model_ok_counts.get(model, 0) + 1
        category_counts[category] = category_counts.get(category, 0) + 1
        category_map = model_category_counts.setdefault(model, {})
        category_map[category] = category_map.get(category, 0) + 1

    categories = sorted(category_counts)
    expected_cases_per_category = (
        expected_cases_per_model // len(categories)
        if categories and expected_cases_per_model % len(categories) == 0
        else None
    )

    models = []
    for model in sorted(model_counts):
        category_map = model_category_counts[model]
        models.append(
            {
                "model": model,
                "result_count": model_counts[model],
                "ok_count": model_ok_counts.get(model, 0),
                "category_counts": {
                    category: category_map.get(category, 0)
                    for category in categories
                },
                "complete": (
                    model_counts[model] == expected_cases_per_model
                    and model_ok_counts.get(model, 0) == model_counts[model]
                    and (
                        expected_cases_per_category is None
                        or all(
                            category_map.get(category, 0) == expected_cases_per_category
                            for category in categories
                        )
                    )
                ),
            }
        )

    result_file_checks = [
        _result_file_manifest_check(path, manifest_models_by_path)
        for path in result_paths
    ]
    complete = all(model["complete"] for model in models) and all(
        check["model_match"] for check in result_file_checks
    )

    return {
        "status": "complete" if complete else "incomplete",
        "run_manifest": _repo_relative(run_manifest),
        "result_file_count": len(result_paths),
        "result_paths": [_repo_relative(path) for path in result_paths],
        "result_file_checks": result_file_checks,
        "total_results": len(results),
        "model_count": len(models),
        "category_count": len(categories),
        "expected_cases_per_model": expected_cases_per_model,
        "expected_cases_per_category": expected_cases_per_category,
        "category_counts": {
            category: category_counts[category]
            for category in categories
        },
        "models": models,
    }


def _manifest_declared_models_by_result_path(manifest_path: Path) -> dict[Path, str]:
    if not manifest_path.exists():
        return {}
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = data.get("runs") if isinstance(data, dict) else None
    if not isinstance(runs, list):
        return {}

    models_by_path = {}
    for run in runs:
        if not isinstance(run, dict):
            continue
        raw_path = run.get("results_path")
        model = run.get("model")
        if not isinstance(raw_path, str) or not raw_path:
            continue
        if not isinstance(model, str) or not model:
            continue
        path = Path(raw_path)
        if not path.is_absolute():
            path = ROOT / path
        models_by_path[_normalize_results_path(path).resolve()] = model
    return models_by_path


def _result_file_manifest_check(path: Path, manifest_models_by_path: dict[Path, str]) -> dict[str, object]:
    file_results = load_results_jsonl(path)
    actual_models = sorted(
        {
            str(result.metadata.get("candidate_model") or "")
            for result in file_results
        }
    )
    declared_model = manifest_models_by_path.get(path.resolve())
    model_match = (
        declared_model is None
        or actual_models == [declared_model]
    )
    return {
        "path": _repo_relative(path),
        "declared_model": declared_model,
        "actual_models": actual_models,
        "model_match": model_match,
    }


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    main()
