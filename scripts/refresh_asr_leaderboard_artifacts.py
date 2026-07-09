from __future__ import annotations

import argparse
import hashlib
import json
import os
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
    ASR_LEADERBOARD_MODELS,
    DEFAULT_PAGE,
    DEFAULT_ARTIFACT_INDEX,
    DEFAULT_REFRESH_COMMANDS,
    DEFAULT_REFRESH_REPORT,
    DEFAULT_SEED_MANIFEST_VALIDATION,
    DEFAULT_SUMMARY,
    DEFAULT_HOSTED_MANIFEST,
    DEFAULT_NEXT_RUNS,
    build_next_run_plan,
    build_output_artifact_index,
    render_generated_sections,
    replace_generated_block,
    write_refresh_report,
    write_refresh_commands_script,
    write_summary_artifact,
)
from scripts.validate_asr_seed_manifest import (  # noqa: E402
    DEFAULT_CASES,
    validate_asr_seed_manifest,
)


DEFAULT_COMBINED_OUT = ROOT / "runs" / "asr-leaderboard" / "full-35-combined"
DEFAULT_RUN_MANIFEST = ROOT / "docs" / "asr-leaderboard-run-manifest.json"
DEFAULT_MANIFEST_VALIDATION = ROOT / "docs" / "asr-leaderboard-manifest-validation.json"
DEFAULT_HOSTED_DIR_ENV = "ASR_LEADERBOARD_HOSTED_DIR"
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
    parser.add_argument("--refresh-commands-out", type=Path, default=DEFAULT_REFRESH_COMMANDS)
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
        "--next-runs-out",
        type=Path,
        default=DEFAULT_NEXT_RUNS,
        help="Write a machine-readable next-refresh plan for missing ASR model/category cells.",
    )
    parser.add_argument(
        "--hosted-manifest-out",
        type=Path,
        default=DEFAULT_HOSTED_MANIFEST,
        help="Write a machine-readable manifest of ASR demo artifacts mirrored to Pages.",
    )
    parser.add_argument(
        "--artifact-index-out",
        type=Path,
        default=DEFAULT_ARTIFACT_INDEX,
        help="Write a machine-readable index of the ASR leaderboard artifact bundle.",
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
        "--discover-complete-model-runs",
        action="store_true",
        help=(
            "When --results is omitted, scan --runs-root for complete result files and "
            "select the newest complete run for each primary ASR leaderboard model."
        ),
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=ROOT / "runs" / "asr-leaderboard",
        help="Root scanned by --discover-complete-model-runs.",
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
        "--hosted-dir-from-env",
        action="store_true",
        help=(
            f"Read the hosted Pages directory from ${DEFAULT_HOSTED_DIR_ENV}. "
            "Ignored when --hosted-dir is set."
        ),
    )
    parser.add_argument(
        "--hosted-dir-env",
        default=DEFAULT_HOSTED_DIR_ENV,
        help=(
            "Environment variable used with --hosted-dir-from-env "
            f"(default: {DEFAULT_HOSTED_DIR_ENV})."
        ),
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help=(
            "Validate selected ASR result sources, seed manifest, and existing demo artifacts "
            "without rewriting files or running live MLX/Gemini calls."
        ),
    )
    parser.add_argument(
        "--expected-cases-per-model",
        type=int,
        default=35,
        help="Fail unless every model has this many ok judged results.",
    )
    args = parser.parse_args()
    hosted_dir = args.hosted_dir
    if hosted_dir is None and args.hosted_dir_from_env:
        hosted_dir = _hosted_dir_from_env(args.hosted_dir_env)

    if args.results:
        result_paths = [_normalize_results_path(path) for path in args.results]
    elif args.discover_complete_model_runs:
        result_paths = _discover_or_default_result_paths(
            args.runs_root,
            expected_cases_per_model=args.expected_cases_per_model,
            run_manifest=args.run_manifest,
        )
    else:
        result_paths = _default_result_paths(
            args.expected_cases_per_model,
            run_manifest=args.run_manifest,
        )
    if args.check_only:
        check_summary = check_asr_leaderboard_refresh_inputs(
            result_paths,
            page=args.page,
            summary_out=args.summary_out,
            seed_cases=args.seed_cases,
            expected_cases_per_model=args.expected_cases_per_model,
        )
        print(
            "ASR refresh preflight OK: "
            f"{check_summary['total_results']} results, "
            f"{check_summary['model_count']} models, "
            f"{check_summary['category_count']} categories, "
            f"{check_summary['result_file_count']} source files."
        )
        return
    refresh_asr_leaderboard_artifacts(
        result_paths,
        out=args.out,
        page=args.page,
        summary_out=args.summary_out,
        refresh_report_out=args.refresh_report_out,
        refresh_commands_out=args.refresh_commands_out,
        manifest_validation_out=args.manifest_validation_out,
        seed_cases=args.seed_cases,
        seed_manifest_validation_out=args.seed_manifest_validation_out,
        next_runs_out=args.next_runs_out,
        hosted_manifest_out=args.hosted_manifest_out,
        artifact_index_out=args.artifact_index_out,
        run_manifest=args.run_manifest,
        update_run_manifest=args.update_run_manifest,
        hosted_dir=hosted_dir,
        expected_cases_per_model=args.expected_cases_per_model,
    )


def _hosted_dir_from_env(env_var: str) -> Path:
    raw_value = os.environ.get(env_var, "").strip()
    if not raw_value:
        raise ValueError(
            f"--hosted-dir-from-env requires ${env_var} to point to the "
            "kennethli319.github.io/open-audio-judge checkout."
        )
    return Path(raw_value).expanduser()


def check_asr_leaderboard_refresh_inputs(
    result_paths: list[Path],
    *,
    page: Path,
    summary_out: Path,
    seed_cases: Path,
    expected_cases_per_model: int,
    artifact_root: Path = ROOT,
    path_maps: list[tuple[str, str]] | None = None,
) -> dict[str, object]:
    result_paths = [_normalize_results_path(path) for path in result_paths]
    _validate_unique_result_paths(result_paths, context="ASR refresh preflight result sources")
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
    render_generated_sections(
        combined_results,
        results_path=DEFAULT_COMBINED_OUT / "results.jsonl",
        expected_cases_per_model=expected_cases_per_model,
        source_result_paths=result_paths,
    )
    seed_cases_data = load_cases(seed_cases)
    seed_validation = validate_asr_seed_manifest(
        seed_cases_data,
        cases_path=seed_cases,
        expected_cases_per_category=5,
    )
    if seed_validation.get("status") != "complete":
        raise ValueError(f"ASR seed manifest preflight is incomplete: {seed_validation}")
    page_validation = check_asr_leaderboard_page(
        page,
        summary_path=summary_out,
        artifact_root=artifact_root,
        path_maps=path_maps or [],
    )
    return {
        "status": "complete",
        "result_file_count": len(result_paths),
        "total_results": len(combined_results),
        "model_count": len(
            {
                str(result.metadata.get("candidate_model") or "")
                for result in combined_results
            }
        ),
        "category_count": len(
            {
                str(result.metadata.get("eval_category") or "")
                for result in combined_results
            }
        ),
        "seed_manifest_status": seed_validation["status"],
        "page_status": page_validation["status"],
    }


def refresh_asr_leaderboard_artifacts(
    result_paths: list[Path],
    *,
    out: Path,
    page: Path,
    summary_out: Path,
    refresh_report_out: Path,
    refresh_commands_out: Path,
    manifest_validation_out: Path,
    run_manifest: Path,
    expected_cases_per_model: int,
    update_run_manifest: bool = False,
    seed_cases: Path = DEFAULT_CASES,
    seed_manifest_validation_out: Path = DEFAULT_SEED_MANIFEST_VALIDATION,
    next_runs_out: Path = DEFAULT_NEXT_RUNS,
    hosted_manifest_out: Path = DEFAULT_HOSTED_MANIFEST,
    artifact_index_out: Path = DEFAULT_ARTIFACT_INDEX,
    hosted_dir: Path | None = None,
) -> None:
    result_paths = [_normalize_results_path(path) for path in result_paths]
    _validate_unique_result_paths(result_paths, context="ASR refresh result sources")
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
        source_result_paths=result_paths,
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
    write_refresh_commands_script(
        refresh_commands_out,
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
    write_next_runs_artifact(
        combined_results,
        next_runs_out,
        expected_cases_per_model=expected_cases_per_model,
    )
    write_artifact_index(
        artifact_index_out,
        results=combined_results,
        results_path=combined_results_path,
        report_path=combined_report_path,
        page=page,
        summary_out=summary_out,
        refresh_report_out=refresh_report_out,
        refresh_commands_out=refresh_commands_out,
        run_manifest=run_manifest,
        manifest_validation_out=manifest_validation_out,
        seed_manifest_validation_out=seed_manifest_validation_out,
        next_runs_out=next_runs_out,
        hosted_manifest_out=hosted_manifest_out,
        expected_cases_per_model=expected_cases_per_model,
    )
    write_hosted_manifest_artifact(
        hosted_manifest_out,
        page=page,
        summary_out=summary_out,
        refresh_report_out=refresh_report_out,
        refresh_commands_out=refresh_commands_out,
        run_manifest=run_manifest,
        manifest_validation_out=manifest_validation_out,
        seed_manifest_validation_out=seed_manifest_validation_out,
        next_runs_out=next_runs_out,
        artifact_index_out=artifact_index_out,
        combined_results_path=combined_results_path,
        combined_report_path=combined_report_path,
    )
    copied_hosted_paths = (
        copy_hosted_asr_artifacts(
            hosted_dir,
            page=page,
            summary_out=summary_out,
            refresh_report_out=refresh_report_out,
            refresh_commands_out=refresh_commands_out,
            run_manifest=run_manifest,
            manifest_validation_out=manifest_validation_out,
            seed_manifest_validation_out=seed_manifest_validation_out,
            next_runs_out=next_runs_out,
            hosted_manifest_out=hosted_manifest_out,
            artifact_index_out=artifact_index_out,
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
    print(f"Refresh commands: {refresh_commands_out}")
    print(f"Manifest validation: {manifest_validation_out}")
    print(f"Seed manifest validation: {seed_manifest_validation_out}")
    print(f"Next-refresh plan: {next_runs_out}")
    print(f"Hosted manifest: {hosted_manifest_out}")
    print(f"Artifact index: {artifact_index_out}")
    for copied_path in copied_hosted_paths:
        print(f"Hosted:  {copied_path}")


def write_run_manifest_artifact(
    result_paths: list[Path],
    output_path: Path,
    *,
    expected_cases_per_model: int,
) -> None:
    result_paths = [_normalize_results_path(path) for path in result_paths]
    _validate_unique_result_paths(result_paths, context="ASR run manifest sources")
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
    refresh_commands_out: Path = DEFAULT_REFRESH_COMMANDS,
    run_manifest: Path = DEFAULT_RUN_MANIFEST,
    manifest_validation_out: Path = DEFAULT_MANIFEST_VALIDATION,
    seed_manifest_validation_out: Path = DEFAULT_SEED_MANIFEST_VALIDATION,
    next_runs_out: Path = DEFAULT_NEXT_RUNS,
    hosted_manifest_out: Path = DEFAULT_HOSTED_MANIFEST,
    artifact_index_out: Path = DEFAULT_ARTIFACT_INDEX,
    combined_results_path: Path | None = None,
    combined_report_path: Path | None = None,
) -> list[Path]:
    hosted_dir.mkdir(parents=True, exist_ok=True)
    copied_paths = []
    source_destinations = (
        (page, {page.name, DEFAULT_PAGE.name}),
        (summary_out, {summary_out.name, DEFAULT_SUMMARY.name}),
        (refresh_report_out, {refresh_report_out.name, DEFAULT_REFRESH_REPORT.name}),
        (refresh_commands_out, {refresh_commands_out.name, DEFAULT_REFRESH_COMMANDS.name}),
        (run_manifest, {run_manifest.name, DEFAULT_RUN_MANIFEST.name}),
        (manifest_validation_out, {manifest_validation_out.name, DEFAULT_MANIFEST_VALIDATION.name}),
        (seed_manifest_validation_out, {seed_manifest_validation_out.name, DEFAULT_SEED_MANIFEST_VALIDATION.name}),
        (next_runs_out, {next_runs_out.name, DEFAULT_NEXT_RUNS.name}),
        (hosted_manifest_out, {hosted_manifest_out.name, DEFAULT_HOSTED_MANIFEST.name}),
        (artifact_index_out, {artifact_index_out.name, DEFAULT_ARTIFACT_INDEX.name}),
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


def write_hosted_manifest_artifact(
    output_path: Path,
    *,
    page: Path = DEFAULT_PAGE,
    summary_out: Path = DEFAULT_SUMMARY,
    refresh_report_out: Path = DEFAULT_REFRESH_REPORT,
    refresh_commands_out: Path = DEFAULT_REFRESH_COMMANDS,
    run_manifest: Path = DEFAULT_RUN_MANIFEST,
    manifest_validation_out: Path = DEFAULT_MANIFEST_VALIDATION,
    seed_manifest_validation_out: Path = DEFAULT_SEED_MANIFEST_VALIDATION,
    next_runs_out: Path = DEFAULT_NEXT_RUNS,
    artifact_index_out: Path = DEFAULT_ARTIFACT_INDEX,
    combined_results_path: Path,
    combined_report_path: Path,
) -> None:
    artifacts = []
    source_destinations = (
        (page, {page.name, DEFAULT_PAGE.name}),
        (summary_out, {summary_out.name, DEFAULT_SUMMARY.name}),
        (refresh_report_out, {refresh_report_out.name, DEFAULT_REFRESH_REPORT.name}),
        (refresh_commands_out, {refresh_commands_out.name, DEFAULT_REFRESH_COMMANDS.name}),
        (run_manifest, {run_manifest.name, DEFAULT_RUN_MANIFEST.name}),
        (manifest_validation_out, {manifest_validation_out.name, DEFAULT_MANIFEST_VALIDATION.name}),
        (seed_manifest_validation_out, {seed_manifest_validation_out.name, DEFAULT_SEED_MANIFEST_VALIDATION.name}),
        (next_runs_out, {next_runs_out.name, DEFAULT_NEXT_RUNS.name}),
        (artifact_index_out, {artifact_index_out.name, DEFAULT_ARTIFACT_INDEX.name}),
        (output_path, {output_path.name, DEFAULT_HOSTED_MANIFEST.name}),
        (combined_results_path, {"asr-leaderboard/full-35-combined/results.jsonl"}),
        (combined_report_path, {"asr-leaderboard/full-35-combined/report.html"}),
    )
    for source, hosted_paths in source_destinations:
        if source == output_path:
            continue
        if not source.exists():
            raise FileNotFoundError(f"Missing ASR hosted manifest source artifact: {source}")
        artifacts.append(
            {
                "source_path": _repo_relative(source),
                "hosted_paths": sorted(hosted_paths),
                "bytes": source.stat().st_size,
                "sha256": _sha256_file(source),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "description": "Generated manifest for ASR leaderboard artifacts copied to kennethli319.github.io/open-audio-judge.",
                "version": 1,
                "hosted_base_path": "open-audio-judge",
                "artifact_count": len(artifacts),
                "artifacts": artifacts,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_artifact_index(
    output_path: Path,
    *,
    results: list,
    results_path: Path,
    report_path: Path,
    page: Path,
    summary_out: Path,
    refresh_report_out: Path,
    refresh_commands_out: Path,
    run_manifest: Path,
    manifest_validation_out: Path,
    seed_manifest_validation_out: Path,
    next_runs_out: Path,
    hosted_manifest_out: Path,
    expected_cases_per_model: int,
) -> None:
    artifact_paths = {
        _repo_relative(results_path): results_path,
        _repo_relative(report_path): report_path,
        _repo_relative(page): page,
        _repo_relative(summary_out): summary_out,
        _repo_relative(refresh_report_out): refresh_report_out,
        _repo_relative(refresh_commands_out): refresh_commands_out,
        _repo_relative(run_manifest): run_manifest,
        _repo_relative(manifest_validation_out): manifest_validation_out,
        _repo_relative(seed_manifest_validation_out): seed_manifest_validation_out,
        _repo_relative(next_runs_out): next_runs_out,
        _repo_relative(hosted_manifest_out): hosted_manifest_out,
        _repo_relative(output_path): output_path,
        _repo_relative(DEFAULT_SUMMARY): summary_out,
        _repo_relative(DEFAULT_REFRESH_REPORT): refresh_report_out,
        _repo_relative(DEFAULT_REFRESH_COMMANDS): refresh_commands_out,
        _repo_relative(DEFAULT_RUN_MANIFEST): run_manifest,
        _repo_relative(DEFAULT_MANIFEST_VALIDATION): manifest_validation_out,
        _repo_relative(DEFAULT_SEED_MANIFEST_VALIDATION): seed_manifest_validation_out,
        _repo_relative(DEFAULT_NEXT_RUNS): next_runs_out,
        _repo_relative(DEFAULT_HOSTED_MANIFEST): hosted_manifest_out,
        _repo_relative(DEFAULT_ARTIFACT_INDEX): output_path,
    }
    artifact_index = build_output_artifact_index(results_path=results_path)
    indexed_paths = {artifact["path"] for artifact in artifact_index}
    indexed_paths.update({_repo_relative(page), _repo_relative(refresh_report_out)})
    indexed_paths.update(artifact_paths)
    records = []
    for raw_path in sorted(indexed_paths):
        path = artifact_paths.get(raw_path, ROOT / raw_path)
        is_generated_after_index = path in {output_path, hosted_manifest_out}
        is_stable_alias = raw_path != _repo_relative(path)
        digest_status = "ok"
        if is_generated_after_index:
            digest_status = "deferred_circular_reference"
        elif is_stable_alias:
            digest_status = "alias"
        elif not path.exists():
            digest_status = "missing"
        records.append(
            {
                "path": raw_path,
                "exists": True if is_generated_after_index else path.exists(),
                "bytes": None
                if is_generated_after_index or is_stable_alias or not path.exists()
                else path.stat().st_size,
                "sha256": None
                if is_generated_after_index or is_stable_alias or not path.exists()
                else _sha256_file(path),
                "digest_status": digest_status,
            }
        )

    models = sorted(
        {
            str(result.metadata.get("candidate_model") or "")
            for result in results
        }
    )
    categories = sorted(
        {
            str(result.metadata.get("eval_category") or "")
            for result in results
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "description": "Generated index for the ASR leaderboard demo artifact bundle.",
                "version": 1,
                "status": "complete" if all(record["exists"] for record in records) else "incomplete",
                "total_results": len(results),
                "model_count": len(models),
                "category_count": len(categories),
                "expected_cases_per_model": expected_cases_per_model,
                "artifacts": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    _validate_unique_result_paths(paths, context=f"ASR run manifest {manifest_path}")
    return paths


def discover_complete_model_result_paths(
    runs_root: Path,
    *,
    expected_cases_per_model: int,
    model_ids: list[str] | None = None,
) -> list[Path]:
    if not runs_root.exists():
        raise FileNotFoundError(f"Missing ASR runs root: {runs_root}")

    target_models = model_ids or [model for model, _ in ASR_LEADERBOARD_MODELS]
    candidates: dict[str, list[tuple[float, Path]]] = {model: [] for model in target_models}
    for path in sorted(runs_root.rglob("results.jsonl")):
        result_path = _normalize_results_path(path)
        try:
            results = load_results_jsonl(result_path)
            model = _complete_result_file_model(
                results,
                expected_cases_per_model=expected_cases_per_model,
            )
        except Exception:
            continue
        if model in candidates:
            candidates[model].append((result_path.stat().st_mtime, result_path))

    missing_models = [
        model
        for model, model_candidates in candidates.items()
        if not model_candidates
    ]
    if missing_models:
        raise ValueError(
            "No complete ASR result file found for model(s): "
            + ", ".join(missing_models)
            + f" under {_repo_relative(runs_root)}."
        )

    discovered_paths = [
        sorted(model_candidates, key=lambda item: (item[0], item[1].as_posix()))[-1][1]
        for model_candidates in candidates.values()
    ]
    _validate_candidate_paths(
        discovered_paths,
        expected_cases_per_model=expected_cases_per_model,
    )
    return discovered_paths


def _discover_or_default_result_paths(
    runs_root: Path,
    *,
    expected_cases_per_model: int,
    run_manifest: Path,
) -> list[Path]:
    try:
        return discover_complete_model_result_paths(
            runs_root,
            expected_cases_per_model=expected_cases_per_model,
        )
    except ValueError as exc:
        fallback_paths = _default_result_paths(
            expected_cases_per_model,
            run_manifest=run_manifest,
        )
        print(
            "No single complete ASR result file was discovered for every primary model; "
            "using the committed run manifest/segmented sources instead. "
            f"Discovery detail: {exc}",
            file=sys.stderr,
        )
        return fallback_paths


def _complete_result_file_model(
    results: list,
    *,
    expected_cases_per_model: int,
) -> str:
    if not results:
        raise ValueError("empty result file")
    models = sorted(
        {
            str(result.metadata.get("candidate_model") or "")
            for result in results
        }
    )
    if len(models) != 1 or not models[0]:
        raise ValueError("result file must contain exactly one model")
    render_generated_sections(
        results,
        results_path=DEFAULT_COMBINED_OUT / "results.jsonl",
        expected_cases_per_model=expected_cases_per_model,
    )
    return models[0]


def _validate_candidate_paths(paths: list[Path], *, expected_cases_per_model: int) -> None:
    _validate_unique_result_paths(paths, context="ASR result discovery")
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


def _validate_unique_result_paths(paths: list[Path], *, context: str) -> None:
    seen: set[Path] = set()
    duplicates = []
    for path in paths:
        resolved = _normalize_results_path(path).resolve()
        if resolved in seen:
            duplicates.append(_repo_relative(path))
        else:
            seen.add(resolved)
    if duplicates:
        raise ValueError(
            f"{context} contains duplicate result path(s): "
            + ", ".join(sorted(duplicates))
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


def write_next_runs_artifact(
    results: list,
    output_path: Path,
    *,
    expected_cases_per_model: int,
) -> None:
    next_runs = build_next_run_plan(
        results,
        expected_cases_per_model=expected_cases_per_model,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(next_runs, indent=2, sort_keys=True) + "\n",
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
