from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
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
    DEFAULT_AUDIO_CASES,
    DEFAULT_REPORT_INDEX,
    DEFAULT_REPORT_LINKS,
    DEFAULT_LIVE_REFRESH_SCRIPT,
    DEFAULT_REFRESH_COMMANDS,
    DEFAULT_REFRESH_WORKFLOW,
    DEFAULT_REFRESH_REPORT,
    DEFAULT_SEED_MANIFEST_VALIDATION,
    DEFAULT_SUMMARY,
    DEFAULT_HOSTED_MANIFEST,
    DEFAULT_RUNTIME_STATUS,
    DEFAULT_NEXT_RUNS,
    END_MARKER,
    GEMINI_SECRET_ENV,
    HOSTED_BASE_PATH,
    HOSTED_BASE_URL,
    START_MARKER,
    build_next_run_plan,
    build_output_artifact_index,
    build_refresh_runtime_status,
    render_generated_sections,
    replace_generated_block,
    summarize_source_result_files,
    write_next_run_plan_artifact,
    write_refresh_report,
    write_report_index,
    write_report_links_artifact,
    write_refresh_commands_script,
    write_refresh_workflow_artifact,
    write_live_refresh_script,
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
    parser.add_argument("--report-index-out", type=Path, default=DEFAULT_REPORT_INDEX)
    parser.add_argument("--report-links-out", type=Path, default=DEFAULT_REPORT_LINKS)
    parser.add_argument("--refresh-commands-out", type=Path, default=DEFAULT_REFRESH_COMMANDS)
    parser.add_argument("--refresh-workflow-out", type=Path, default=DEFAULT_REFRESH_WORKFLOW)
    parser.add_argument("--live-refresh-script-out", type=Path, default=DEFAULT_LIVE_REFRESH_SCRIPT)
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
        "--runtime-status-out",
        type=Path,
        default=DEFAULT_RUNTIME_STATUS,
        help="Write MLX ASR and Gemini readiness status for refresh automation.",
    )
    parser.add_argument(
        "--check-mlx-runtime",
        action="store_true",
        help="Run the bounded MLX ASR runtime preflight and include the result in runtime status.",
    )
    parser.add_argument(
        "--require-runtime-ready",
        action="store_true",
        help=(
            "With --check-only, fail unless local audio is ready, the Gemini secret file is "
            "present, and the bounded MLX ASR runtime preflight passes."
        ),
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
        "--check-summary-out",
        type=Path,
        help="With --check-only, write the preflight validation summary as JSON.",
    )
    parser.add_argument(
        "--require-generated-fresh",
        action="store_true",
        help=(
            "With --check-only, fail if the committed generated page block or summary "
            "does not match the selected ASR result sources."
        ),
    )
    parser.add_argument(
        "--require-audio-ready",
        action="store_true",
        help=(
            "With --check-only, fail if the materialized ASR audio manifest is missing, "
            "stale, or references missing local audio files."
        ),
    )
    parser.add_argument(
        "--require-hosted-current",
        action="store_true",
        help=(
            "With --check-only and --hosted-dir/--hosted-dir-from-env, fail unless every "
            "hosted Pages artifact matches the current local hosted manifest digests."
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
            refresh_report_out=args.refresh_report_out,
            report_index_out=args.report_index_out,
            report_links_out=args.report_links_out,
            refresh_commands_out=args.refresh_commands_out,
            refresh_workflow_out=args.refresh_workflow_out,
            live_refresh_script_out=args.live_refresh_script_out,
            next_runs_out=args.next_runs_out,
            seed_cases=args.seed_cases,
            expected_cases_per_model=args.expected_cases_per_model,
            hosted_dir=hosted_dir,
            require_generated_fresh=args.require_generated_fresh,
            require_audio_ready=args.require_audio_ready,
            require_hosted_current=args.require_hosted_current,
        )
        if args.check_mlx_runtime or args.require_runtime_ready:
            runtime_status = build_runtime_status_artifact_data(
                results=combined_results_from_paths(result_paths),
                results_path=DEFAULT_COMBINED_OUT / "results.jsonl",
                source_result_paths=result_paths,
                check_mlx_runtime=True,
            )
            write_runtime_status_data(args.runtime_status_out, runtime_status)
            enrich_check_summary_with_runtime_status(
                check_summary,
                runtime_status=runtime_status,
                runtime_status_out=args.runtime_status_out,
            )
            if args.require_runtime_ready:
                _validate_runtime_ready(runtime_status)
        if args.check_summary_out:
            args.check_summary_out.parent.mkdir(parents=True, exist_ok=True)
            args.check_summary_out.write_text(
                json.dumps(check_summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        message = (
            "ASR refresh preflight OK: "
            f"{check_summary['total_results']} results, "
            f"{check_summary['model_count']} models, "
            f"{check_summary['category_count']} categories, "
            f"{check_summary['result_file_count']} source files."
        )
        if check_summary.get("hosted_page_status"):
            message += f" Hosted mirror: {check_summary['hosted_page_status']}."
            message += (
                f" Hosted artifacts: {check_summary['hosted_artifact_count']} sources, "
                f"{check_summary['hosted_path_count']} paths."
            )
        print(message)
        return
    refresh_asr_leaderboard_artifacts(
        result_paths,
        out=args.out,
        page=args.page,
        summary_out=args.summary_out,
        refresh_report_out=args.refresh_report_out,
        report_index_out=args.report_index_out,
        report_links_out=args.report_links_out,
        refresh_commands_out=args.refresh_commands_out,
        refresh_workflow_out=args.refresh_workflow_out,
        live_refresh_script_out=args.live_refresh_script_out,
        manifest_validation_out=args.manifest_validation_out,
        seed_cases=args.seed_cases,
        seed_manifest_validation_out=args.seed_manifest_validation_out,
        next_runs_out=args.next_runs_out,
        hosted_manifest_out=args.hosted_manifest_out,
        artifact_index_out=args.artifact_index_out,
        runtime_status_out=args.runtime_status_out,
        run_manifest=args.run_manifest,
        update_run_manifest=args.update_run_manifest,
        hosted_dir=hosted_dir,
        check_mlx_runtime=args.check_mlx_runtime,
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
    refresh_report_out: Path = DEFAULT_REFRESH_REPORT,
    report_index_out: Path = DEFAULT_REPORT_INDEX,
    report_links_out: Path = DEFAULT_REPORT_LINKS,
    refresh_commands_out: Path = DEFAULT_REFRESH_COMMANDS,
    refresh_workflow_out: Path = DEFAULT_REFRESH_WORKFLOW,
    live_refresh_script_out: Path = DEFAULT_LIVE_REFRESH_SCRIPT,
    next_runs_out: Path = DEFAULT_NEXT_RUNS,
    artifact_root: Path = ROOT,
    path_maps: list[tuple[str, str]] | None = None,
    hosted_dir: Path | None = None,
    require_generated_fresh: bool = False,
    require_audio_ready: bool = False,
    require_hosted_current: bool = False,
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
    generated = render_generated_sections(
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
    if require_generated_fresh:
        _validate_generated_artifacts_fresh(
            combined_results,
            result_paths=result_paths,
            page=page,
            summary_out=summary_out,
            refresh_report_out=refresh_report_out,
            report_index_out=report_index_out,
            report_links_out=report_links_out,
            refresh_commands_out=refresh_commands_out,
            refresh_workflow_out=refresh_workflow_out,
            live_refresh_script_out=live_refresh_script_out,
            run_manifest=DEFAULT_RUN_MANIFEST,
            manifest_validation_out=DEFAULT_MANIFEST_VALIDATION,
            seed_cases=seed_cases,
            seed_manifest_validation_out=DEFAULT_SEED_MANIFEST_VALIDATION,
            next_runs_out=next_runs_out,
            hosted_manifest_out=DEFAULT_HOSTED_MANIFEST,
            artifact_index_out=DEFAULT_ARTIFACT_INDEX,
            runtime_status_out=DEFAULT_RUNTIME_STATUS,
            generated=generated,
            combined_results_path=DEFAULT_COMBINED_OUT / "results.jsonl",
            expected_cases_per_model=expected_cases_per_model,
        )
    audio_manifest = build_audio_manifest_status(
        seed_cases_path=seed_cases,
        audio_cases_path=DEFAULT_AUDIO_CASES,
    )
    if require_audio_ready and audio_manifest.get("status") != "complete":
        raise ValueError(
            "ASR audio manifest is not ready for live MLX ASR refresh: "
            + json.dumps(audio_manifest, sort_keys=True)
        )
    summary: dict[str, object] = {
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
        "audio_manifest_status": audio_manifest["status"],
        "audio_cases_path": audio_manifest["audio_cases_path"],
        "page_status": page_validation["status"],
        "source_result_paths": [_repo_relative(path) for path in result_paths],
    }
    if hosted_dir:
        hosted_validation = check_asr_leaderboard_page(
            hosted_dir / page.name,
            summary_path=hosted_dir / summary_out.name,
            artifact_root=hosted_dir,
            path_maps=[
                ("docs/", ""),
                ("runs/asr-leaderboard/", "asr-leaderboard/"),
            ],
            allow_missing_source_results=True,
        )
        summary["hosted_page_status"] = hosted_validation["status"]
        summary["hosted_artifact_count"] = hosted_validation["hosted_artifact_count"]
        summary["hosted_path_count"] = hosted_validation["hosted_path_count"]
        if require_hosted_current:
            hosted_current = validate_hosted_artifacts_current(
                hosted_dir,
                hosted_manifest_out=DEFAULT_HOSTED_MANIFEST,
            )
            summary["hosted_current_status"] = hosted_current["status"]
            summary["hosted_current_path_count"] = hosted_current["hosted_path_count"]
    elif require_hosted_current:
        raise ValueError("--require-hosted-current requires --hosted-dir or --hosted-dir-from-env.")
    return summary


def enrich_check_summary_with_runtime_status(
    summary: dict[str, object],
    *,
    runtime_status: dict[str, object],
    runtime_status_out: Path,
) -> dict[str, object]:
    summary["runtime_status_path"] = _repo_relative(runtime_status_out)
    summary["runtime_status"] = runtime_status
    try:
        _validate_runtime_ready(runtime_status)
    except ValueError as exc:
        summary["runtime_ready"] = False
        summary["runtime_ready_issue"] = str(exc)
    else:
        summary["runtime_ready"] = True
    return summary


def validate_hosted_artifacts_current(
    hosted_dir: Path,
    *,
    hosted_manifest_out: Path = DEFAULT_HOSTED_MANIFEST,
) -> dict[str, object]:
    if not hosted_dir.exists():
        raise FileNotFoundError(f"Missing hosted ASR directory: {hosted_dir}")
    if not hosted_manifest_out.exists():
        raise FileNotFoundError(f"Missing local ASR hosted manifest: {hosted_manifest_out}")

    manifest = json.loads(hosted_manifest_out.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError(f"{hosted_manifest_out} must include a non-empty artifacts list.")

    checked_paths = 0
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            raise ValueError(f"{hosted_manifest_out} artifacts[{index}] must be an object.")
        source_path = artifact.get("source_path")
        hosted_paths = artifact.get("hosted_paths")
        expected_bytes = artifact.get("bytes")
        expected_sha256 = artifact.get("sha256")
        if not isinstance(source_path, str) or not source_path:
            raise ValueError(
                f"{hosted_manifest_out} artifacts[{index}] has invalid source_path: "
                f"{source_path!r}"
            )
        if (
            not isinstance(hosted_paths, list)
            or not hosted_paths
            or not all(isinstance(path, str) and path for path in hosted_paths)
        ):
            raise ValueError(
                f"{hosted_manifest_out} artifacts[{index}] has invalid hosted_paths: "
                f"{hosted_paths!r}"
            )
        if not isinstance(expected_bytes, int) or expected_bytes < 0:
            raise ValueError(
                f"{hosted_manifest_out} artifacts[{index}] has invalid bytes: "
                f"{expected_bytes!r}"
            )
        if (
            not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(char not in "0123456789abcdef" for char in expected_sha256)
        ):
            raise ValueError(
                f"{hosted_manifest_out} artifacts[{index}] has invalid sha256: "
                f"{expected_sha256!r}"
            )

        source = ROOT / source_path
        if not source.exists():
            raise FileNotFoundError(
                f"Hosted current check source is missing: {source_path}"
            )
        if source.stat().st_size != expected_bytes or _sha256_file(source) != expected_sha256:
            raise ValueError(
                f"{_repo_relative(hosted_manifest_out)} is stale for source artifact: "
                f"{source_path}"
            )
        for hosted_path in hosted_paths:
            destination = hosted_dir / hosted_path
            if not destination.exists():
                raise FileNotFoundError(
                    f"Hosted Pages artifact is missing: {destination}"
                )
            if destination.stat().st_size != expected_bytes:
                raise ValueError(
                    f"Hosted Pages artifact byte size is stale: {destination}"
                )
            actual_sha256 = _sha256_file(destination)
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"Hosted Pages artifact sha256 is stale: {destination}"
                )
            checked_paths += 1

    return {
        "status": "complete",
        "hosted_artifact_count": len(artifacts),
        "hosted_path_count": checked_paths,
    }


def _validate_generated_artifacts_fresh(
    combined_results: list,
    *,
    result_paths: list[Path],
    page: Path,
    summary_out: Path,
    refresh_report_out: Path | None = None,
    report_index_out: Path | None = None,
    report_links_out: Path | None = None,
    refresh_commands_out: Path | None = None,
    refresh_workflow_out: Path | None = None,
    live_refresh_script_out: Path | None = None,
    run_manifest: Path | None = None,
    manifest_validation_out: Path | None = None,
    seed_cases: Path | None = None,
    seed_manifest_validation_out: Path | None = None,
    next_runs_out: Path | None = None,
    hosted_manifest_out: Path | None = None,
    artifact_index_out: Path | None = None,
    runtime_status_out: Path | None = None,
    generated: str,
    expected_cases_per_model: int,
    combined_results_path: Path = DEFAULT_COMBINED_OUT / "results.jsonl",
) -> None:
    existing_generated = _extract_generated_block(page)
    if existing_generated != generated:
        raise ValueError(
            f"{_repo_relative(page)} generated ASR leaderboard block is stale; "
            "run `.venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py`."
        )

    with tempfile.TemporaryDirectory(prefix="asr-leaderboard-fresh-") as raw_tmp_dir:
        tmp_dir = Path(raw_tmp_dir)
        expected_summary = tmp_dir / summary_out.name
        write_summary_artifact(
            combined_results,
            expected_summary,
            results_path=combined_results_path,
            expected_cases_per_model=expected_cases_per_model,
            source_result_paths=result_paths,
        )
        _compare_generated_text_artifact(summary_out, expected_summary)

        expected_combined_results = tmp_dir / combined_results_path.name
        write_results_jsonl(combined_results, expected_combined_results)
        _compare_generated_text_artifact(combined_results_path, expected_combined_results)

        expected_combined_report = tmp_dir / combined_results_path.with_name("report.html").name
        write_html_report(combined_results, expected_combined_report)
        _compare_generated_text_artifact(
            combined_results_path.with_name("report.html"),
            expected_combined_report,
        )

        if refresh_report_out is not None:
            expected_refresh_report = tmp_dir / refresh_report_out.name
            write_refresh_report(
                combined_results,
                expected_refresh_report,
                results_path=combined_results_path,
                expected_cases_per_model=expected_cases_per_model,
                source_result_paths=result_paths,
            )
            _compare_generated_text_artifact(refresh_report_out, expected_refresh_report)
        if refresh_commands_out is not None:
            expected_refresh_commands = tmp_dir / refresh_commands_out.name
            write_refresh_commands_script(
                expected_refresh_commands,
                source_result_paths=result_paths,
            )
            _compare_generated_text_artifact(refresh_commands_out, expected_refresh_commands)
        if refresh_workflow_out is not None:
            expected_refresh_workflow = tmp_dir / refresh_workflow_out.name
            write_refresh_workflow_artifact(
                expected_refresh_workflow,
                source_result_paths=result_paths,
            )
            _compare_generated_text_artifact(refresh_workflow_out, expected_refresh_workflow)
        if live_refresh_script_out is not None:
            expected_live_refresh_script = tmp_dir / live_refresh_script_out.name
            write_live_refresh_script(expected_live_refresh_script)
            _compare_generated_text_artifact(live_refresh_script_out, expected_live_refresh_script)
        if report_index_out is not None:
            expected_report_index = tmp_dir / report_index_out.name
            write_report_index(
                combined_results,
                expected_report_index,
                results_path=combined_results_path,
                expected_cases_per_model=expected_cases_per_model,
                source_result_paths=result_paths,
            )
            _compare_generated_text_artifact(report_index_out, expected_report_index)
        if report_links_out is not None:
            expected_report_links = tmp_dir / report_links_out.name
            write_report_links_artifact(
                combined_results,
                expected_report_links,
                results_path=combined_results_path,
                expected_cases_per_model=expected_cases_per_model,
                source_result_paths=result_paths,
            )
            _compare_generated_text_artifact(report_links_out, expected_report_links)
        if next_runs_out is not None:
            expected_next_runs = tmp_dir / next_runs_out.name
            write_next_run_plan_artifact(
                combined_results,
                expected_next_runs,
                expected_cases_per_model=expected_cases_per_model,
            )
            _compare_generated_text_artifact(next_runs_out, expected_next_runs)
        if hosted_manifest_out is not None and runtime_status_out is not None:
            expected_hosted_manifest = tmp_dir / hosted_manifest_out.name
            write_hosted_manifest_artifact(
                expected_hosted_manifest,
                page=page,
                summary_out=summary_out,
                refresh_report_out=refresh_report_out or DEFAULT_REFRESH_REPORT,
                report_index_out=report_index_out or DEFAULT_REPORT_INDEX,
                report_links_out=report_links_out or DEFAULT_REPORT_LINKS,
                refresh_commands_out=refresh_commands_out or DEFAULT_REFRESH_COMMANDS,
                refresh_workflow_out=(
                    refresh_workflow_out
                    or (refresh_commands_out.with_name(DEFAULT_REFRESH_WORKFLOW.name)
                        if refresh_commands_out is not None
                        else DEFAULT_REFRESH_WORKFLOW)
                ),
                live_refresh_script_out=live_refresh_script_out or DEFAULT_LIVE_REFRESH_SCRIPT,
                run_manifest=run_manifest or DEFAULT_RUN_MANIFEST,
                manifest_validation_out=manifest_validation_out or DEFAULT_MANIFEST_VALIDATION,
                seed_manifest_validation_out=(
                    seed_manifest_validation_out or DEFAULT_SEED_MANIFEST_VALIDATION
                ),
                next_runs_out=next_runs_out or DEFAULT_NEXT_RUNS,
                artifact_index_out=artifact_index_out or DEFAULT_ARTIFACT_INDEX,
                runtime_status_out=runtime_status_out,
                combined_results_path=combined_results_path,
                combined_report_path=combined_results_path.with_name("report.html"),
                source_result_paths=result_paths,
            )
            _compare_generated_text_artifact(hosted_manifest_out, expected_hosted_manifest)
        if artifact_index_out is not None:
            expected_artifact_index = tmp_dir / artifact_index_out.name
            expected_artifact_index.write_text(
                json.dumps(
                    build_artifact_index_data(
                        artifact_index_out,
                        results=combined_results,
                        results_path=combined_results_path,
                        report_path=combined_results_path.with_name("report.html"),
                        page=page,
                        summary_out=summary_out,
                        refresh_report_out=refresh_report_out or DEFAULT_REFRESH_REPORT,
                        report_index_out=report_index_out or DEFAULT_REPORT_INDEX,
                        report_links_out=report_links_out or DEFAULT_REPORT_LINKS,
                        refresh_commands_out=refresh_commands_out or DEFAULT_REFRESH_COMMANDS,
                        refresh_workflow_out=(
                            refresh_workflow_out
                            or (refresh_commands_out.with_name(DEFAULT_REFRESH_WORKFLOW.name)
                                if refresh_commands_out is not None
                                else DEFAULT_REFRESH_WORKFLOW)
                        ),
                        live_refresh_script_out=(
                            live_refresh_script_out or DEFAULT_LIVE_REFRESH_SCRIPT
                        ),
                        run_manifest=run_manifest or DEFAULT_RUN_MANIFEST,
                        manifest_validation_out=(
                            manifest_validation_out or DEFAULT_MANIFEST_VALIDATION
                        ),
                        seed_manifest_validation_out=(
                            seed_manifest_validation_out or DEFAULT_SEED_MANIFEST_VALIDATION
                        ),
                        next_runs_out=next_runs_out or DEFAULT_NEXT_RUNS,
                        hosted_manifest_out=hosted_manifest_out or DEFAULT_HOSTED_MANIFEST,
                        runtime_status_out=runtime_status_out or DEFAULT_RUNTIME_STATUS,
                        expected_cases_per_model=expected_cases_per_model,
                        source_result_paths=result_paths,
                    ),
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            _compare_generated_text_artifact(artifact_index_out, expected_artifact_index)
        if run_manifest is not None:
            expected_run_manifest = tmp_dir / run_manifest.name
            write_run_manifest_artifact(
                result_paths,
                expected_run_manifest,
                expected_cases_per_model=expected_cases_per_model,
            )
            _compare_generated_text_artifact(run_manifest, expected_run_manifest)
        if manifest_validation_out is not None and run_manifest is not None:
            expected_manifest_validation = tmp_dir / manifest_validation_out.name
            write_manifest_validation_artifact(
                combined_results,
                expected_manifest_validation,
                result_paths=result_paths,
                run_manifest=run_manifest,
                expected_cases_per_model=expected_cases_per_model,
            )
            _compare_generated_text_artifact(manifest_validation_out, expected_manifest_validation)
        if seed_manifest_validation_out is not None and seed_cases is not None:
            expected_seed_validation = tmp_dir / seed_manifest_validation_out.name
            write_seed_manifest_validation_artifact(
                seed_cases,
                expected_seed_validation,
            )
            _compare_generated_text_artifact(seed_manifest_validation_out, expected_seed_validation)
        if runtime_status_out is not None:
            expected_runtime_status = tmp_dir / runtime_status_out.name
            write_runtime_status_artifact(
                expected_runtime_status,
                results=combined_results,
                results_path=combined_results_path,
                source_result_paths=result_paths,
            )
            _compare_runtime_status_artifact(runtime_status_out, expected_runtime_status)


def _compare_generated_text_artifact(actual_path: Path, expected_path: Path) -> None:
    if not actual_path.exists():
        raise FileNotFoundError(f"Missing generated ASR leaderboard artifact: {actual_path}")
    actual = actual_path.read_text(encoding="utf-8")
    expected = expected_path.read_text(encoding="utf-8")
    if actual != expected:
        raise ValueError(
            f"{_repo_relative(actual_path)} is stale for selected ASR result sources; "
            "run `.venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py`."
        )


def _compare_runtime_status_artifact(actual_path: Path, expected_path: Path) -> None:
    if not actual_path.exists():
        raise FileNotFoundError(f"Missing generated ASR leaderboard artifact: {actual_path}")
    actual = json.loads(actual_path.read_text(encoding="utf-8"))
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    actual.pop("mlx_runtime_preflight", None)
    expected.pop("mlx_runtime_preflight", None)
    if actual != expected:
        raise ValueError(
            f"{_repo_relative(actual_path)} is stale for selected ASR result sources; "
            "run `.venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py`."
        )


def _extract_generated_block(page: Path) -> str:
    html_text = page.read_text(encoding="utf-8")
    if START_MARKER not in html_text or END_MARKER not in html_text:
        raise ValueError(f"{page} must contain {START_MARKER} and {END_MARKER}.")
    before, remainder = html_text.split(START_MARKER, 1)
    generated_inner, _ = remainder.split(END_MARKER, 1)
    return START_MARKER + generated_inner + END_MARKER


def refresh_asr_leaderboard_artifacts(
    result_paths: list[Path],
    *,
    out: Path,
    page: Path,
    summary_out: Path,
    refresh_report_out: Path,
    refresh_commands_out: Path,
    live_refresh_script_out: Path,
    manifest_validation_out: Path,
    run_manifest: Path,
    expected_cases_per_model: int,
    report_index_out: Path | None = None,
    report_links_out: Path | None = None,
    update_run_manifest: bool = False,
    seed_cases: Path = DEFAULT_CASES,
    seed_manifest_validation_out: Path = DEFAULT_SEED_MANIFEST_VALIDATION,
    next_runs_out: Path = DEFAULT_NEXT_RUNS,
    hosted_manifest_out: Path = DEFAULT_HOSTED_MANIFEST,
    artifact_index_out: Path = DEFAULT_ARTIFACT_INDEX,
    runtime_status_out: Path | None = None,
    refresh_workflow_out: Path | None = None,
    hosted_dir: Path | None = None,
    check_mlx_runtime: bool = False,
) -> None:
    report_index_out = report_index_out or refresh_report_out.with_name(DEFAULT_REPORT_INDEX.name)
    report_links_out = report_links_out or refresh_report_out.with_name(DEFAULT_REPORT_LINKS.name)
    runtime_status_out = runtime_status_out or artifact_index_out.with_name(DEFAULT_RUNTIME_STATUS.name)
    refresh_workflow_out = refresh_workflow_out or refresh_commands_out.with_name(
        DEFAULT_REFRESH_WORKFLOW.name
    )
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
    write_report_index(
        combined_results,
        report_index_out,
        results_path=combined_results_path,
        expected_cases_per_model=expected_cases_per_model,
        source_result_paths=result_paths,
    )
    write_report_links_artifact(
        combined_results,
        report_links_out,
        results_path=combined_results_path,
        expected_cases_per_model=expected_cases_per_model,
        source_result_paths=result_paths,
    )
    write_refresh_commands_script(
        refresh_commands_out,
        source_result_paths=result_paths,
    )
    write_refresh_workflow_artifact(
        refresh_workflow_out,
        source_result_paths=result_paths,
    )
    write_live_refresh_script(live_refresh_script_out)
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
    write_runtime_status_artifact(
        runtime_status_out,
        results=combined_results,
        results_path=combined_results_path,
        source_result_paths=result_paths,
        check_mlx_runtime=check_mlx_runtime,
    )
    write_artifact_index(
        artifact_index_out,
        results=combined_results,
        results_path=combined_results_path,
        report_path=combined_report_path,
        page=page,
        summary_out=summary_out,
        refresh_report_out=refresh_report_out,
        report_index_out=report_index_out,
        report_links_out=report_links_out,
        refresh_commands_out=refresh_commands_out,
        refresh_workflow_out=refresh_workflow_out,
        live_refresh_script_out=live_refresh_script_out,
        run_manifest=run_manifest,
        manifest_validation_out=manifest_validation_out,
        seed_manifest_validation_out=seed_manifest_validation_out,
        next_runs_out=next_runs_out,
        hosted_manifest_out=hosted_manifest_out,
        runtime_status_out=runtime_status_out,
        expected_cases_per_model=expected_cases_per_model,
        source_result_paths=result_paths,
    )
    write_hosted_manifest_artifact(
        hosted_manifest_out,
        page=page,
        summary_out=summary_out,
        refresh_report_out=refresh_report_out,
        report_index_out=report_index_out,
        report_links_out=report_links_out,
        refresh_commands_out=refresh_commands_out,
        refresh_workflow_out=refresh_workflow_out,
        live_refresh_script_out=live_refresh_script_out,
        run_manifest=run_manifest,
        manifest_validation_out=manifest_validation_out,
        seed_manifest_validation_out=seed_manifest_validation_out,
        next_runs_out=next_runs_out,
        artifact_index_out=artifact_index_out,
        runtime_status_out=runtime_status_out,
        combined_results_path=combined_results_path,
        combined_report_path=combined_report_path,
        source_result_paths=result_paths,
    )
    copied_hosted_paths = (
        copy_hosted_asr_artifacts(
            hosted_dir,
            page=page,
            summary_out=summary_out,
            refresh_report_out=refresh_report_out,
            report_index_out=report_index_out,
            report_links_out=report_links_out,
            refresh_commands_out=refresh_commands_out,
            refresh_workflow_out=refresh_workflow_out,
            live_refresh_script_out=live_refresh_script_out,
            run_manifest=run_manifest,
            manifest_validation_out=manifest_validation_out,
            seed_manifest_validation_out=seed_manifest_validation_out,
            next_runs_out=next_runs_out,
            hosted_manifest_out=hosted_manifest_out,
            artifact_index_out=artifact_index_out,
            runtime_status_out=runtime_status_out,
            combined_results_path=combined_results_path,
            combined_report_path=combined_report_path,
            source_result_paths=result_paths,
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
    print(f"Report index: {report_index_out}")
    print(f"Report links: {report_links_out}")
    print(f"Refresh commands: {refresh_commands_out}")
    print(f"Live refresh script: {live_refresh_script_out}")
    print(f"Manifest validation: {manifest_validation_out}")
    print(f"Seed manifest validation: {seed_manifest_validation_out}")
    print(f"Next-refresh plan: {next_runs_out}")
    print(f"Hosted manifest: {hosted_manifest_out}")
    print(f"Artifact index: {artifact_index_out}")
    print(f"Runtime status: {runtime_status_out}")
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
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
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
    report_index_out: Path = DEFAULT_REPORT_INDEX,
    report_links_out: Path = DEFAULT_REPORT_LINKS,
    refresh_commands_out: Path = DEFAULT_REFRESH_COMMANDS,
    refresh_workflow_out: Path = DEFAULT_REFRESH_WORKFLOW,
    live_refresh_script_out: Path = DEFAULT_LIVE_REFRESH_SCRIPT,
    run_manifest: Path = DEFAULT_RUN_MANIFEST,
    manifest_validation_out: Path = DEFAULT_MANIFEST_VALIDATION,
    seed_manifest_validation_out: Path = DEFAULT_SEED_MANIFEST_VALIDATION,
    next_runs_out: Path = DEFAULT_NEXT_RUNS,
    hosted_manifest_out: Path = DEFAULT_HOSTED_MANIFEST,
    artifact_index_out: Path = DEFAULT_ARTIFACT_INDEX,
    runtime_status_out: Path = DEFAULT_RUNTIME_STATUS,
    combined_results_path: Path | None = None,
    combined_report_path: Path | None = None,
    source_result_paths: list[Path] | None = None,
) -> list[Path]:
    hosted_dir.mkdir(parents=True, exist_ok=True)
    copied_paths = []
    source_destinations = (
        (page, {page.name, DEFAULT_PAGE.name}),
        (summary_out, {summary_out.name, DEFAULT_SUMMARY.name}),
        (refresh_report_out, {refresh_report_out.name, DEFAULT_REFRESH_REPORT.name}),
        (report_index_out, {report_index_out.name, DEFAULT_REPORT_INDEX.name}),
        (report_links_out, {report_links_out.name, DEFAULT_REPORT_LINKS.name}),
        (refresh_commands_out, {refresh_commands_out.name, DEFAULT_REFRESH_COMMANDS.name}),
        (refresh_workflow_out, {refresh_workflow_out.name, DEFAULT_REFRESH_WORKFLOW.name}),
        (live_refresh_script_out, {live_refresh_script_out.name, DEFAULT_LIVE_REFRESH_SCRIPT.name}),
        (run_manifest, {run_manifest.name, DEFAULT_RUN_MANIFEST.name}),
        (manifest_validation_out, {manifest_validation_out.name, DEFAULT_MANIFEST_VALIDATION.name}),
        (seed_manifest_validation_out, {seed_manifest_validation_out.name, DEFAULT_SEED_MANIFEST_VALIDATION.name}),
        (next_runs_out, {next_runs_out.name, DEFAULT_NEXT_RUNS.name}),
        (hosted_manifest_out, {hosted_manifest_out.name, DEFAULT_HOSTED_MANIFEST.name}),
        (artifact_index_out, {artifact_index_out.name, DEFAULT_ARTIFACT_INDEX.name}),
        (runtime_status_out, {runtime_status_out.name, DEFAULT_RUNTIME_STATUS.name}),
    )
    for source, destination_names in source_destinations:
        if not source.exists():
            raise FileNotFoundError(f"Missing hosted ASR source artifact: {source}")
        for destination_name in sorted(destination_names):
            destination = hosted_dir / destination_name
            shutil.copyfile(source, destination)
            copied_paths.append(destination)

    for source_result_path in source_result_paths or []:
        source_report = _normalize_results_path(source_result_path).with_name("report.html")
        if not source_report.exists():
            continue
        hosted_report_path = _hosted_report_path_for_source_report(source_report)
        destination = hosted_dir / hosted_report_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_report, destination)
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
    report_index_out: Path = DEFAULT_REPORT_INDEX,
    report_links_out: Path = DEFAULT_REPORT_LINKS,
    refresh_commands_out: Path = DEFAULT_REFRESH_COMMANDS,
    refresh_workflow_out: Path = DEFAULT_REFRESH_WORKFLOW,
    live_refresh_script_out: Path = DEFAULT_LIVE_REFRESH_SCRIPT,
    run_manifest: Path = DEFAULT_RUN_MANIFEST,
    manifest_validation_out: Path = DEFAULT_MANIFEST_VALIDATION,
    seed_manifest_validation_out: Path = DEFAULT_SEED_MANIFEST_VALIDATION,
    next_runs_out: Path = DEFAULT_NEXT_RUNS,
    artifact_index_out: Path = DEFAULT_ARTIFACT_INDEX,
    runtime_status_out: Path = DEFAULT_RUNTIME_STATUS,
    combined_results_path: Path,
    combined_report_path: Path,
    source_result_paths: list[Path] | None = None,
) -> None:
    artifacts = []
    source_destinations = (
        (page, {page.name, DEFAULT_PAGE.name}),
        (summary_out, {summary_out.name, DEFAULT_SUMMARY.name}),
        (refresh_report_out, {refresh_report_out.name, DEFAULT_REFRESH_REPORT.name}),
        (report_index_out, {report_index_out.name, DEFAULT_REPORT_INDEX.name}),
        (report_links_out, {report_links_out.name, DEFAULT_REPORT_LINKS.name}),
        (refresh_commands_out, {refresh_commands_out.name, DEFAULT_REFRESH_COMMANDS.name}),
        (refresh_workflow_out, {refresh_workflow_out.name, DEFAULT_REFRESH_WORKFLOW.name}),
        (live_refresh_script_out, {live_refresh_script_out.name, DEFAULT_LIVE_REFRESH_SCRIPT.name}),
        (run_manifest, {run_manifest.name, DEFAULT_RUN_MANIFEST.name}),
        (manifest_validation_out, {manifest_validation_out.name, DEFAULT_MANIFEST_VALIDATION.name}),
        (seed_manifest_validation_out, {seed_manifest_validation_out.name, DEFAULT_SEED_MANIFEST_VALIDATION.name}),
        (next_runs_out, {next_runs_out.name, DEFAULT_NEXT_RUNS.name}),
        (artifact_index_out, {artifact_index_out.name, DEFAULT_ARTIFACT_INDEX.name}),
        (runtime_status_out, {runtime_status_out.name, DEFAULT_RUNTIME_STATUS.name}),
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
    for source_result_path in source_result_paths or []:
        source_report = _normalize_results_path(source_result_path).with_name("report.html")
        if not source_report.exists():
            continue
        artifacts.append(
            {
                "source_path": _repo_relative(source_report),
                "hosted_paths": [_hosted_report_path_for_source_report(source_report)],
                "bytes": source_report.stat().st_size,
                "sha256": _sha256_file(source_report),
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


def _hosted_report_path_for_source_report(source_report: Path) -> str:
    raw_path = _repo_relative(source_report)
    prefix = "runs/asr-leaderboard/"
    if raw_path.startswith(prefix):
        return "asr-leaderboard/" + raw_path.removeprefix(prefix)
    return f"asr-leaderboard/source-reports/{source_report.parent.parent.name}/report.html"


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
    report_index_out: Path | None = None,
    report_links_out: Path | None = None,
    runtime_status_out: Path | None = None,
    live_refresh_script_out: Path | None = None,
    refresh_workflow_out: Path | None = None,
    source_result_paths: list[Path] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            build_artifact_index_data(
                output_path,
                results=results,
                results_path=results_path,
                report_path=report_path,
                page=page,
                summary_out=summary_out,
                refresh_report_out=refresh_report_out,
                refresh_commands_out=refresh_commands_out,
                refresh_workflow_out=refresh_workflow_out,
                live_refresh_script_out=live_refresh_script_out,
                run_manifest=run_manifest,
                manifest_validation_out=manifest_validation_out,
                seed_manifest_validation_out=seed_manifest_validation_out,
                next_runs_out=next_runs_out,
                hosted_manifest_out=hosted_manifest_out,
                expected_cases_per_model=expected_cases_per_model,
                report_index_out=report_index_out,
                report_links_out=report_links_out,
                runtime_status_out=runtime_status_out,
                source_result_paths=source_result_paths,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def build_artifact_index_data(
    output_path: Path,
    *,
    results: list,
    results_path: Path,
    report_path: Path,
    page: Path,
    summary_out: Path,
    refresh_report_out: Path,
    refresh_commands_out: Path,
    refresh_workflow_out: Path | None = None,
    run_manifest: Path,
    manifest_validation_out: Path,
    seed_manifest_validation_out: Path,
    next_runs_out: Path,
    hosted_manifest_out: Path,
    expected_cases_per_model: int,
    report_index_out: Path | None = None,
    report_links_out: Path | None = None,
    runtime_status_out: Path | None = None,
    live_refresh_script_out: Path | None = None,
    source_result_paths: list[Path] | None = None,
) -> dict[str, object]:
    report_index_out = report_index_out or refresh_report_out.with_name(DEFAULT_REPORT_INDEX.name)
    report_links_out = report_links_out or refresh_report_out.with_name(DEFAULT_REPORT_LINKS.name)
    runtime_status_out = runtime_status_out or output_path.with_name(DEFAULT_RUNTIME_STATUS.name)
    live_refresh_script_out = live_refresh_script_out or refresh_commands_out.with_name(
        DEFAULT_LIVE_REFRESH_SCRIPT.name
    )
    refresh_workflow_out = refresh_workflow_out or refresh_commands_out.with_name(
        DEFAULT_REFRESH_WORKFLOW.name
    )
    artifact_paths = {
        _repo_relative(results_path): results_path,
        _repo_relative(report_path): report_path,
        _repo_relative(page): page,
        _repo_relative(summary_out): summary_out,
        _repo_relative(refresh_report_out): refresh_report_out,
        _repo_relative(report_index_out): report_index_out,
        _repo_relative(report_links_out): report_links_out,
        _repo_relative(refresh_commands_out): refresh_commands_out,
        _repo_relative(refresh_workflow_out): refresh_workflow_out,
        _repo_relative(live_refresh_script_out): live_refresh_script_out,
        _repo_relative(run_manifest): run_manifest,
        _repo_relative(manifest_validation_out): manifest_validation_out,
        _repo_relative(seed_manifest_validation_out): seed_manifest_validation_out,
        _repo_relative(next_runs_out): next_runs_out,
        _repo_relative(hosted_manifest_out): hosted_manifest_out,
        _repo_relative(runtime_status_out): runtime_status_out,
        _repo_relative(output_path): output_path,
        _repo_relative(DEFAULT_SUMMARY): summary_out,
        _repo_relative(DEFAULT_REFRESH_REPORT): refresh_report_out,
        _repo_relative(DEFAULT_REPORT_INDEX): report_index_out,
        _repo_relative(DEFAULT_REPORT_LINKS): report_links_out,
        _repo_relative(DEFAULT_REFRESH_COMMANDS): refresh_commands_out,
        _repo_relative(DEFAULT_REFRESH_WORKFLOW): refresh_workflow_out,
        _repo_relative(DEFAULT_LIVE_REFRESH_SCRIPT): live_refresh_script_out,
        _repo_relative(DEFAULT_RUN_MANIFEST): run_manifest,
        _repo_relative(DEFAULT_MANIFEST_VALIDATION): manifest_validation_out,
        _repo_relative(DEFAULT_SEED_MANIFEST_VALIDATION): seed_manifest_validation_out,
        _repo_relative(DEFAULT_NEXT_RUNS): next_runs_out,
        _repo_relative(DEFAULT_HOSTED_MANIFEST): hosted_manifest_out,
        _repo_relative(DEFAULT_ARTIFACT_INDEX): output_path,
        _repo_relative(DEFAULT_RUNTIME_STATUS): runtime_status_out,
    }
    for source_result_path in source_result_paths or []:
        source_report = _normalize_results_path(source_result_path).with_name("report.html")
        if not source_report.exists():
            continue
        artifact_paths[_repo_relative(source_report)] = source_report
    artifact_index = build_output_artifact_index(results_path=results_path)
    artifact_purposes = {artifact["path"]: artifact["purpose"] for artifact in artifact_index}
    artifact_purposes[_repo_relative(page)] = "Generated ASR leaderboard demo HTML page."
    artifact_purposes[_repo_relative(refresh_report_out)] = (
        "Human-readable coverage, score, source-file, and command report."
    )
    for source_result_path in source_result_paths or []:
        source_report = _normalize_results_path(source_result_path).with_name("report.html")
        if source_report.exists():
            artifact_purposes[_repo_relative(source_report)] = (
                "Hosted source run HTML report linked from the ASR report index."
            )
    source_report_hosted_paths = {
        _repo_relative(_normalize_results_path(source_result_path).with_name("report.html")): [
            _hosted_report_path_for_source_report(
                _normalize_results_path(source_result_path).with_name("report.html")
            )
        ]
        for source_result_path in source_result_paths or []
        if _normalize_results_path(source_result_path).with_name("report.html").exists()
    }
    indexed_paths = set(artifact_purposes)
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
                "purpose": artifact_purposes.get(
                    raw_path,
                    "Generated ASR leaderboard support artifact.",
                ),
                "exists": True if is_generated_after_index else path.exists(),
                "bytes": None
                if is_generated_after_index or is_stable_alias or not path.exists()
                else path.stat().st_size,
                "sha256": None
                if is_generated_after_index or is_stable_alias or not path.exists()
                else _sha256_file(path),
                "digest_status": digest_status,
                "hosted_paths": source_report_hosted_paths.get(raw_path)
                or _hosted_paths_for_artifact(
                    raw_path,
                    results_path=results_path,
                    report_path=report_path,
                ),
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
    result_bundle = {
        "results_path": _repo_relative(results_path),
        "exists": results_path.exists(),
        "bytes": results_path.stat().st_size if results_path.exists() else None,
        "sha256": _sha256_file(results_path) if results_path.exists() else None,
        "total_results": len(results),
        "model_count": len(models),
        "category_count": len(categories),
        "expected_cases_per_model": expected_cases_per_model,
        "models": models,
        "categories": categories,
        "source_result_file_count": len(source_result_paths or []),
        "source_result_files": [
            {
                "path": _repo_relative(summary.path),
                "result_bytes": summary.result_bytes,
                "result_sha256": summary.result_sha256,
                "report_path": _repo_relative(summary.report_path),
                "report_exists": summary.report_exists,
                "report_bytes": summary.report_bytes,
                "report_sha256": summary.report_sha256,
                "hosted_report_paths": source_report_hosted_paths.get(
                    _repo_relative(summary.report_path),
                    [],
                ),
                "models": list(summary.models),
                "result_count": summary.result_count,
                "ok_count": summary.ok_count,
                "judge_samples": summary.judge_samples,
                "average_score": summary.average_score,
                "labels": dict(sorted(summary.labels.items())),
                "categories": dict(sorted(summary.categories.items())),
            }
            for summary in summarize_source_result_files(source_result_paths or [])
        ],
    }
    return {
        "description": "Generated index for the ASR leaderboard demo artifact bundle.",
        "version": 2,
        "hosted": {
            "base_path": HOSTED_BASE_PATH,
            "base_url": HOSTED_BASE_URL,
            "demo_page_url": f"{HOSTED_BASE_URL}/asr-leaderboard-demo.html",
            "combined_report_url": f"{HOSTED_BASE_URL}/asr-leaderboard/full-35-combined/report.html",
        },
        "status": "complete" if all(record["exists"] for record in records) else "incomplete",
        "result_bundle": result_bundle,
        "total_results": len(results),
        "model_count": len(models),
        "category_count": len(categories),
        "expected_cases_per_model": expected_cases_per_model,
        "artifacts": records,
    }


def _hosted_paths_for_artifact(
    raw_path: str,
    *,
    results_path: Path,
    report_path: Path,
) -> list[str]:
    hosted_names = {
        "docs/asr-leaderboard-demo.html": ["asr-leaderboard-demo.html"],
        "docs/asr-leaderboard-summary.json": ["asr-leaderboard-summary.json"],
        "docs/asr-leaderboard-refresh-report.md": ["asr-leaderboard-refresh-report.md"],
        "docs/asr-leaderboard-report-index.md": ["asr-leaderboard-report-index.md"],
        "docs/asr-leaderboard-report-links.json": ["asr-leaderboard-report-links.json"],
        "docs/asr-leaderboard-refresh-commands.sh": ["asr-leaderboard-refresh-commands.sh"],
        "docs/asr-leaderboard-refresh-workflow.json": ["asr-leaderboard-refresh-workflow.json"],
        "docs/asr-leaderboard-live-refresh.sh": ["asr-leaderboard-live-refresh.sh"],
        "docs/asr-leaderboard-run-manifest.json": ["asr-leaderboard-run-manifest.json"],
        "docs/asr-leaderboard-manifest-validation.json": [
            "asr-leaderboard-manifest-validation.json"
        ],
        "docs/asr-seed-manifest-validation.json": ["asr-seed-manifest-validation.json"],
        "docs/asr-leaderboard-next-runs.json": ["asr-leaderboard-next-runs.json"],
        "docs/asr-leaderboard-hosted-manifest.json": [
            "asr-leaderboard-hosted-manifest.json"
        ],
        "docs/asr-leaderboard-artifacts.json": ["asr-leaderboard-artifacts.json"],
        "docs/asr-leaderboard-runtime-status.json": ["asr-leaderboard-runtime-status.json"],
        _repo_relative(results_path): ["asr-leaderboard/full-35-combined/results.jsonl"],
        _repo_relative(report_path): ["asr-leaderboard/full-35-combined/report.html"],
    }
    return hosted_names.get(raw_path, [])


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


def combined_results_from_paths(result_paths: list[Path]) -> list:
    return [
        result
        for path in result_paths
        for result in load_results_jsonl(path)
    ]


def write_runtime_status_artifact(
    output_path: Path,
    *,
    results: list,
    results_path: Path | None = None,
    source_result_paths: list[Path] | None = None,
    check_mlx_runtime: bool = False,
) -> None:
    status = build_runtime_status_artifact_data(
        results=results,
        results_path=results_path,
        source_result_paths=source_result_paths,
        check_mlx_runtime=check_mlx_runtime,
    )
    write_runtime_status_data(output_path, status)


def build_runtime_status_artifact_data(
    *,
    results: list,
    results_path: Path | None = None,
    source_result_paths: list[Path] | None = None,
    check_mlx_runtime: bool = False,
) -> dict[str, object]:
    status = build_refresh_runtime_status(results)
    status["status"] = "complete"
    status["result_bundle"] = build_runtime_result_bundle_status(
        results,
        results_path=results_path,
        source_result_paths=source_result_paths or [],
    )
    status["mlx_runtime_preflight"] = (
        _run_mlx_runtime_preflight()
        if check_mlx_runtime
        else {
            "status": "not_checked",
            "command": _mlx_runtime_preflight_command(),
        }
    )
    status["gemini_secret"] = _gemini_secret_status()
    status["audio_manifest"] = build_audio_manifest_status()
    status["secret_handling"] = (
        "Gemini secrets are checked only for file presence; secret values are never "
        "printed or written into artifacts."
    )
    return status


def write_runtime_status_data(output_path: Path, status: dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_runtime_ready(status: dict[str, object]) -> None:
    failures = []
    audio_manifest = status.get("audio_manifest")
    if not isinstance(audio_manifest, dict) or audio_manifest.get("status") != "complete":
        failures.append("audio_manifest")
    gemini_secret = status.get("gemini_secret")
    if not isinstance(gemini_secret, dict) or gemini_secret.get("status") != "present":
        failures.append("gemini_secret")
    mlx_preflight = status.get("mlx_runtime_preflight")
    if not isinstance(mlx_preflight, dict) or mlx_preflight.get("status") != "ok":
        failures.append("mlx_runtime_preflight")
    if failures:
        raise ValueError(
            "ASR runtime is not ready for live MLX/Gemini refresh: "
            + ", ".join(failures)
        )


def build_runtime_result_bundle_status(
    results: list,
    *,
    results_path: Path | None = None,
    source_result_paths: list[Path] | None = None,
) -> dict[str, object]:
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
    source_files = []
    for path in source_result_paths or []:
        path = _normalize_results_path(path)
        source_files.append(
            {
                "path": _repo_relative(path),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else None,
                "sha256": _sha256_file(path) if path.exists() else None,
            }
        )
    return {
        "results_path": _repo_relative(results_path) if results_path is not None else None,
        "total_results": len(results),
        "model_count": len([model for model in models if model]),
        "category_count": len([category for category in categories if category]),
        "models": [model for model in models if model],
        "categories": [category for category in categories if category],
        "source_result_file_count": len(source_files),
        "source_result_files": source_files,
    }


def build_audio_manifest_status(
    *,
    seed_cases_path: Path = DEFAULT_CASES,
    audio_cases_path: Path = DEFAULT_AUDIO_CASES,
) -> dict[str, object]:
    command = [
        ".venv/bin/python",
        "scripts/synthesize_tts_cases.py",
        "--cases",
        _repo_relative(seed_cases_path),
        "--out",
        _repo_relative(audio_cases_path.parent),
        "--discard-text-sidecars",
        "--summary-out",
        _repo_relative(audio_cases_path.parent / "summary.json"),
    ]
    if not audio_cases_path.exists():
        return {
            "status": "missing",
            "seed_cases_path": _repo_relative(seed_cases_path),
            "audio_cases_path": _repo_relative(audio_cases_path),
            "command": command,
            "issue": "materialized ASR audio manifest is missing",
        }

    seed_cases = load_cases(seed_cases_path)
    audio_cases = load_cases(audio_cases_path)
    seed_case_ids = {case.id for case in seed_cases}
    audio_source_ids = {
        str(case.metadata.get("source_case_id") or "")
        for case in audio_cases
    }
    missing_source_ids = sorted(seed_case_ids - audio_source_ids)
    extra_source_ids = sorted(audio_source_ids - seed_case_ids)
    missing_audio_files = []
    for case in audio_cases:
        if not case.audio_path:
            missing_audio_files.append(case.id)
            continue
        audio_path = Path(case.audio_path)
        if not audio_path.is_absolute():
            audio_path = audio_cases_path.parent / audio_path
        if not audio_path.exists():
            missing_audio_files.append(case.id)

    status = (
        "complete"
        if not missing_source_ids and not extra_source_ids and not missing_audio_files
        else "stale"
    )
    return {
        "status": status,
        "seed_cases_path": _repo_relative(seed_cases_path),
        "audio_cases_path": _repo_relative(audio_cases_path),
        "command": command,
        "seed_case_count": len(seed_cases),
        "audio_case_count": len(audio_cases),
        "missing_source_case_ids": missing_source_ids,
        "extra_source_case_ids": extra_source_ids,
        "missing_audio_file_case_ids": missing_audio_files,
        "audio_manifest_sha256": _sha256_file(audio_cases_path),
    }


def _run_mlx_runtime_preflight() -> dict[str, object]:
    command = _mlx_runtime_preflight_command()
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    completed = subprocess.run(
        command[1:],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "status": "ok" if completed.returncode == 0 else "blocked",
        "command": command,
        "returncode": completed.returncode,
        "stdout": _bounded_output(completed.stdout),
        "stderr": _bounded_output(completed.stderr),
    }


def _mlx_runtime_preflight_command() -> list[str]:
    return [
        "PYTHONPATH=src",
        ".venv/bin/python",
        "-m",
        "open_audio_judge.cli",
        "check-mlx-asr-runtime",
        "--python-bin",
        ".venv/bin/python",
        "--model",
        ASR_LEADERBOARD_MODELS[0][0],
    ]


def _gemini_secret_status() -> dict[str, object]:
    secret_path = Path(GEMINI_SECRET_ENV)
    return {
        "status": "present" if secret_path.exists() and secret_path.stat().st_size > 0 else "missing",
        "path": GEMINI_SECRET_ENV,
    }


def _bounded_output(value: str, *, limit: int = 2000) -> str:
    value = value.strip()
    if len(value) <= limit:
        return value
    return value[:limit] + "\n[truncated]"


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
    manifest_runs_by_path = _manifest_declared_runs_by_result_path(run_manifest)

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
        _result_file_manifest_check(path, manifest_runs_by_path)
        for path in result_paths
    ]
    complete = all(model["complete"] for model in models) and all(
        check["model_match"] and check["digest_match"] for check in result_file_checks
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


def _manifest_declared_runs_by_result_path(manifest_path: Path) -> dict[Path, dict[str, object]]:
    if not manifest_path.exists():
        return {}
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = data.get("runs") if isinstance(data, dict) else None
    if not isinstance(runs, list):
        return {}

    runs_by_path = {}
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
        runs_by_path[_normalize_results_path(path).resolve()] = run
    return runs_by_path


def _result_file_manifest_check(path: Path, manifest_runs_by_path: dict[Path, dict[str, object]]) -> dict[str, object]:
    file_results = load_results_jsonl(path)
    actual_models = sorted(
        {
            str(result.metadata.get("candidate_model") or "")
            for result in file_results
        }
    )
    manifest_run = manifest_runs_by_path.get(path.resolve(), {})
    declared_model = manifest_run.get("model")
    model_match = (
        declared_model is None
        or actual_models == [declared_model]
    )
    declared_bytes = manifest_run.get("bytes")
    declared_sha256 = manifest_run.get("sha256")
    actual_bytes = path.stat().st_size
    actual_sha256 = _sha256_file(path)
    digest_match = (
        (declared_bytes is None or declared_bytes == actual_bytes)
        and (declared_sha256 is None or declared_sha256 == actual_sha256)
    )
    return {
        "path": _repo_relative(path),
        "declared_model": declared_model,
        "actual_models": actual_models,
        "model_match": model_match,
        "declared_bytes": declared_bytes,
        "actual_bytes": actual_bytes,
        "declared_sha256": declared_sha256,
        "actual_sha256": actual_sha256,
        "digest_match": digest_match,
    }


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    main()
