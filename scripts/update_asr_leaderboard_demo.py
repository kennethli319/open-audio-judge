from __future__ import annotations

import argparse
import html
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from open_audio_judge.models import EvaluationResult
from open_audio_judge.runner import load_results_jsonl


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "runs" / "asr-leaderboard" / "full-35-combined" / "results.jsonl"
DEFAULT_PAGE = ROOT / "docs" / "asr-leaderboard-demo.html"
DEFAULT_SUMMARY = ROOT / "docs" / "asr-leaderboard-summary.json"
DEFAULT_REFRESH_REPORT = ROOT / "docs" / "asr-leaderboard-refresh-report.md"
DEFAULT_REFRESH_COMMANDS = ROOT / "docs" / "asr-leaderboard-refresh-commands.sh"
DEFAULT_RUN_MANIFEST = ROOT / "docs" / "asr-leaderboard-run-manifest.json"
DEFAULT_MANIFEST_VALIDATION = ROOT / "docs" / "asr-leaderboard-manifest-validation.json"
DEFAULT_SEED_MANIFEST_VALIDATION = ROOT / "docs" / "asr-seed-manifest-validation.json"
DEFAULT_NEXT_RUNS = ROOT / "docs" / "asr-leaderboard-next-runs.json"
DEFAULT_HOSTED_MANIFEST = ROOT / "docs" / "asr-leaderboard-hosted-manifest.json"
DEFAULT_ARTIFACT_INDEX = ROOT / "docs" / "asr-leaderboard-artifacts.json"
DEFAULT_AUDIO_CASES = ROOT / "runs" / "asr-research-audio" / "tts_audio_cases.jsonl"
DEFAULT_SEED_CASES = ROOT / "examples" / "asr_research_cases.jsonl"
DEFAULT_HOSTED_DIR_ENV = "ASR_LEADERBOARD_HOSTED_DIR"
START_MARKER = "<!-- ASR_LEADERBOARD_GENERATED_START -->"
END_MARKER = "<!-- ASR_LEADERBOARD_GENERATED_END -->"

CATEGORY_COLUMNS = [
    ("transcription_accuracy_wer", "WER"),
    ("numeric_unit_integrity", "Numeric/Unit"),
    ("negation_modality_scope", "Negation/Modality"),
    ("temporal_scheduling_accuracy", "Temporal"),
    ("entity_factual_integrity", "Entity"),
    ("semantic_paraphrase_preservation", "Paraphrase"),
    ("acoustic_noise_robustness", "Acoustic Noise"),
]
CATEGORY_LABELS = dict(CATEGORY_COLUMNS)
ASR_LEADERBOARD_MODELS = [
    ("mlx-community/whisper-large-v3-turbo-asr-fp16", "whisper-large-v3-turbo-refresh"),
    ("mlx-community/Qwen3-ASR-1.7B-8bit", "qwen3-asr-1.7b-refresh"),
    ("mlx-community/VibeVoice-ASR-4bit", "vibevoice-asr-refresh"),
]
ASR_FALLBACK_MODELS = [
    "mlx-community/whisper-small.en-asr-4bit",
    "mlx-community/parakeet-rnnt-0.6b",
    "mlx-community/GLM-ASR-Nano-2512-4bit",
]
GEMINI_SECRET_ENV = "/Users/wangyauli/.openclaw/secrets/open-audio-judge-gemini.env"


@dataclass(frozen=True)
class ModelSummary:
    model: str
    result_count: int
    ok_count: int
    judge_samples: int
    average_score: float
    labels: Counter[str]


@dataclass(frozen=True)
class CategorySummary:
    category: str
    result_count: int
    average_score: float
    labels: Counter[str]


@dataclass(frozen=True)
class SourceResultFileSummary:
    path: Path
    report_path: Path
    models: tuple[str, ...]
    result_count: int
    ok_count: int
    judge_samples: int
    average_score: float
    labels: Counter[str]
    categories: Counter[str]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh the ASR leaderboard demo from verified result JSONL artifacts.",
    )
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--page", type=Path, default=DEFAULT_PAGE)
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="Write a machine-readable summary artifact for the hosted ASR demo.",
    )
    parser.add_argument(
        "--refresh-report-out",
        type=Path,
        default=DEFAULT_REFRESH_REPORT,
        help="Write a human-readable ASR leaderboard refresh report.",
    )
    parser.add_argument(
        "--refresh-commands-out",
        type=Path,
        default=DEFAULT_REFRESH_COMMANDS,
        help="Write a shell playbook with the generated ASR leaderboard refresh commands.",
    )
    parser.add_argument(
        "--next-runs-out",
        type=Path,
        default=DEFAULT_NEXT_RUNS,
        help="Write a machine-readable next-refresh plan for missing ASR model/category cells.",
    )
    parser.add_argument(
        "--expected-cases-per-model",
        type=int,
        default=35,
        help="Fail unless every model has this many judged results.",
    )
    args = parser.parse_args()

    results = load_results_jsonl(args.results)
    generated = render_generated_sections(
        results,
        results_path=args.results,
        expected_cases_per_model=args.expected_cases_per_model,
    )
    replace_generated_block(args.page, generated)
    write_summary_artifact(
        results,
        args.summary_out,
        results_path=args.results,
        expected_cases_per_model=args.expected_cases_per_model,
    )
    write_refresh_report(
        results,
        args.refresh_report_out,
        results_path=args.results,
        expected_cases_per_model=args.expected_cases_per_model,
    )
    write_refresh_commands_script(
        args.refresh_commands_out,
        source_result_paths=[],
    )
    write_next_run_plan_artifact(
        results,
        args.next_runs_out,
        expected_cases_per_model=args.expected_cases_per_model,
    )
    print(f"Updated {args.page} from {args.results} ({len(results)} results)")
    print(f"Summary: {args.summary_out}")
    print(f"Refresh report: {args.refresh_report_out}")
    print(f"Refresh commands: {args.refresh_commands_out}")
    print(f"Next-refresh plan: {args.next_runs_out}")


def render_generated_sections(
    results: list[EvaluationResult],
    *,
    results_path: Path,
    expected_cases_per_model: int,
    source_result_paths: list[Path] | None = None,
) -> str:
    if not results:
        raise ValueError("No ASR evaluation results were loaded.")

    model_summaries = summarize_models(results)
    validate_coverage(results, model_summaries, expected_cases_per_model=expected_cases_per_model)
    total_judge_samples = sum(summary.judge_samples for summary in model_summaries)
    categories = sorted({str(result.metadata.get("eval_category", "")) for result in results})
    category_list = ", ".join(f"<code>{html.escape(category)}</code>" for category in categories)
    category_columns = category_columns_for_results(results)
    results_label = html.escape(_repo_relative(results_path))
    report_label = html.escape(_repo_relative(results_path.with_name("report.html")))
    summary_label = html.escape(_repo_relative(DEFAULT_SUMMARY))
    refresh_report_label = html.escape(_repo_relative(DEFAULT_REFRESH_REPORT))
    refresh_commands_label = html.escape(_repo_relative(DEFAULT_REFRESH_COMMANDS))
    manifest_label = html.escape(_repo_relative(DEFAULT_RUN_MANIFEST))
    validation_label = html.escape(_repo_relative(DEFAULT_MANIFEST_VALIDATION))
    seed_validation_label = html.escape(_repo_relative(DEFAULT_SEED_MANIFEST_VALIDATION))
    next_runs_label = html.escape(_repo_relative(DEFAULT_NEXT_RUNS))
    hosted_manifest_label = html.escape(_repo_relative(DEFAULT_HOSTED_MANIFEST))
    artifact_index_label = html.escape(_repo_relative(DEFAULT_ARTIFACT_INDEX))
    workflow = _refresh_workflow([])
    workflow_commands = [
        ("Validate seed manifest", workflow["seed_manifest_validation_command"]),
        ("Materialize audio", workflow["audio_materialization_command"]),
        ("Run one MLX ASR model", workflow["model_run_template"]),
        ("Discover latest complete runs", workflow["discover_refresh_command"]),
        ("Refresh committed artifacts", workflow["manifest_refresh_command"]),
        ("Run refresh shell playbook", ["bash", workflow["refresh_commands_path"]]),
        ("Check generated page", workflow["page_validation_command"]),
        ("Sync hosted artifacts", workflow["hosted_artifact_command"]),
    ]
    output_artifacts = build_output_artifact_index(results_path=results_path)
    source_file_summaries = summarize_source_result_files(source_result_paths or [])
    source_report_rows = _render_source_report_rows(source_file_summaries)

    return "\n".join(
        [
            START_MARKER,
            "    <h2>Verified Leaderboard Results</h2>",
            (
                '    <p class="muted">Generated from <code>'
                f"{results_label}</code>. The verified matrix covers {len(results)} judged transcripts "
                f"across {len(model_summaries)} MLX ASR models and {len(categories)} research categories: "
                f"{category_list}.</p>"
            ),
            "    <table>",
            "      <thead><tr><th>Model</th><th>Cases</th><th>Gemini Samples</th><th>Average Score</th><th>Labels</th></tr></thead>",
            "      <tbody>",
            *(_render_model_row(summary) for summary in model_summaries),
            "      </tbody>",
            "    </table>",
            "",
            "    <h2>Category Breakdown</h2>",
            "    <table>",
            "      <thead><tr><th>Model</th>"
            + "".join(f"<th>{html.escape(label)}</th>" for _, label in category_columns)
            + "</tr></thead>",
            "      <tbody>",
            *(
                _render_category_row(model, results, category_columns=category_columns)
                for model in [summary.model for summary in model_summaries]
            ),
            "      </tbody>",
            "    </table>",
            (
                '    <p class="muted">Total Gemini judge samples: '
                f"{total_judge_samples}. Refresh this block with "
                '<code>.venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py</code> '
                "after rerunning the verified ASR model jobs. The combined local report is "
                f"<code>{report_label}</code> and the committed summary artifact is "
                f"<code>{summary_label}</code>. The generated refresh report is "
                f"<code>{refresh_report_label}</code>, and the generated shell playbook is "
                f"<code>{refresh_commands_label}</code>. The committed run manifest is "
                f"<code>{manifest_label}</code>, with coverage validation in "
                f"<code>{validation_label}</code> and seed-manifest validation in "
                f"<code>{seed_validation_label}</code>. The next-refresh plan is "
                f"<code>{next_runs_label}</code>, and the hosted artifact manifest is "
                f"<code>{hosted_manifest_label}</code>. The artifact bundle index is "
                f"<code>{artifact_index_label}</code>; together they include the source result files, "
                "complete model/category matrix, missing-cell guidance, hosted copy map, and reproducible refresh workflow. Pass "
                f"<code>{DEFAULT_HOSTED_DIR_ENV}</code> with "
                "<code>--hosted-dir-from-env</code> to copy the same verified artifacts into the hosted Pages checkout.</p>"
            ),
            "",
            "    <h2>Generated Refresh Workflow</h2>",
            "    <p class=\"muted\">These commands are generated from the same workflow metadata written to "
            f"<code>{summary_label}</code> and <code>{refresh_report_label}</code>.</p>",
            "    <table>",
            "      <thead><tr><th>Step</th><th>Command</th></tr></thead>",
            "      <tbody>",
            *(
                "        <tr>"
                f"<td>{html.escape(label)}</td>"
                f"<td><code>{html.escape(_shell_join(command))}</code></td>"
                "</tr>"
                for label, command in workflow_commands
            ),
            "      </tbody>",
            "    </table>",
            "",
            "    <h2>Generated Model Refresh Commands</h2>",
            "    <p class=\"muted\">Load the Gemini secret only in the local shell before running live judge calls: "
            f"<code>{html.escape(_shell_join(workflow['local_secret_env_command']))}</code>.</p>",
            "    <table>",
            "      <thead><tr><th>Model</th><th>Run Command</th></tr></thead>",
            "      <tbody>",
            *(
                "        <tr>"
                f"<td><code>{html.escape(command['model'])}</code></td>"
                f"<td><code>{html.escape(_shell_join(command['command']))}</code></td>"
                "</tr>"
                for command in workflow["model_run_commands"]
            ),
            "      </tbody>",
            "    </table>",
            (
                "    <p class=\"muted\">If a primary MLX ASR model is unsupported locally, record that "
                "blocked state in the run notes before trying the documented fallbacks: "
                + ", ".join(
                    f"<code>{html.escape(model)}</code>"
                    for model in workflow["fallback_model_ids"]
                )
                + ".</p>"
            ),
            "",
            "    <h2>Generated Artifacts</h2>",
            "    <table>",
            "      <thead><tr><th>Path</th><th>Purpose</th></tr></thead>",
            "      <tbody>",
            *(
                "        <tr>"
                f"<td><code>{html.escape(artifact['path'])}</code></td>"
                f"<td>{html.escape(artifact['purpose'])}</td>"
                "</tr>"
                for artifact in output_artifacts
            ),
            "      </tbody>",
            "    </table>",
            *source_report_rows,
            END_MARKER,
        ]
    )


def write_summary_artifact(
    results: list[EvaluationResult],
    output_path: Path,
    *,
    results_path: Path,
    expected_cases_per_model: int,
    source_result_paths: list[Path] | None = None,
) -> None:
    model_summaries = summarize_models(results)
    validate_coverage(results, model_summaries, expected_cases_per_model=expected_cases_per_model)
    category_summaries = summarize_categories(results)
    runtime_status = build_refresh_runtime_status(results)
    coverage_matrix = build_model_category_matrix(results)
    category_columns = category_columns_for_results(results)
    source_file_summaries = summarize_source_result_files(source_result_paths or [])
    output_artifacts = build_output_artifact_index(results_path=results_path)
    next_runs = build_next_run_plan(results, expected_cases_per_model=expected_cases_per_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "results_path": _repo_relative(results_path),
                "report_path": _repo_relative(results_path.with_name("report.html")),
                "source_result_paths": [
                    _repo_relative(path)
                    for path in source_result_paths or []
                ],
                "source_result_files": [
                    _source_file_summary_json(summary)
                    for summary in source_file_summaries
                ],
                "run_manifest_path": _repo_relative(DEFAULT_RUN_MANIFEST),
                "refresh_commands_path": _repo_relative(DEFAULT_REFRESH_COMMANDS),
                "manifest_validation_path": _repo_relative(DEFAULT_MANIFEST_VALIDATION),
                "seed_manifest_validation_path": _repo_relative(DEFAULT_SEED_MANIFEST_VALIDATION),
                "next_runs_path": _repo_relative(DEFAULT_NEXT_RUNS),
                "hosted_manifest_path": _repo_relative(DEFAULT_HOSTED_MANIFEST),
                "artifact_index_path": _repo_relative(DEFAULT_ARTIFACT_INDEX),
                "output_artifacts": output_artifacts,
                "refresh_workflow": _refresh_workflow(source_result_paths or []),
                "refresh_runtime_status": runtime_status,
                "next_run_plan": next_runs,
                "total_results": len(results),
                "model_count": len(model_summaries),
                "category_count": len(category_summaries),
                "expected_cases_per_model": expected_cases_per_model,
                "category_columns": [
                    {"category": category, "label": label}
                    for category, label in category_columns
                ],
                "total_gemini_judge_samples": sum(summary.judge_samples for summary in model_summaries),
                "model_category_matrix": coverage_matrix,
                "models": [
                    {
                        "model": summary.model,
                        "result_count": summary.result_count,
                        "ok_count": summary.ok_count,
                        "judge_samples": summary.judge_samples,
                        "average_score": round(summary.average_score, 3),
                        "labels": _ordered_label_counts(summary.labels),
                    }
                    for summary in model_summaries
                ],
                "categories": [
                    {
                        "category": summary.category,
                        "result_count": summary.result_count,
                        "average_score": round(summary.average_score, 3),
                        "labels": _ordered_label_counts(summary.labels),
                    }
                    for summary in category_summaries
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_refresh_report(
    results: list[EvaluationResult],
    output_path: Path,
    *,
    results_path: Path,
    expected_cases_per_model: int,
    source_result_paths: list[Path] | None = None,
) -> None:
    model_summaries = summarize_models(results)
    validate_coverage(results, model_summaries, expected_cases_per_model=expected_cases_per_model)
    category_summaries = summarize_categories(results)
    workflow = _refresh_workflow(source_result_paths or [])
    runtime_status = build_refresh_runtime_status(results)
    coverage_matrix = build_model_category_matrix(results)
    category_columns = category_columns_for_results(results)
    source_file_summaries = summarize_source_result_files(source_result_paths or [])
    output_artifacts = build_output_artifact_index(results_path=results_path)
    next_runs = build_next_run_plan(results, expected_cases_per_model=expected_cases_per_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(
            [
                "# ASR Leaderboard Refresh Report",
                "",
                "This generated report summarizes the verified ASR leaderboard artifact set.",
                "",
                "## Coverage",
                "",
                f"- Results: `{_repo_relative(results_path)}`",
                f"- Combined report: `{_repo_relative(results_path.with_name('report.html'))}`",
                f"- Summary JSON: `{_repo_relative(DEFAULT_SUMMARY)}`",
                f"- Run manifest: `{_repo_relative(DEFAULT_RUN_MANIFEST)}`",
                f"- Refresh command playbook: `{_repo_relative(DEFAULT_REFRESH_COMMANDS)}`",
                f"- Manifest validation: `{_repo_relative(DEFAULT_MANIFEST_VALIDATION)}`",
                f"- Seed manifest validation: `{_repo_relative(DEFAULT_SEED_MANIFEST_VALIDATION)}`",
                f"- Next-refresh plan: `{_repo_relative(DEFAULT_NEXT_RUNS)}`",
                f"- Hosted artifact manifest: `{_repo_relative(DEFAULT_HOSTED_MANIFEST)}`",
                f"- Artifact bundle index: `{_repo_relative(DEFAULT_ARTIFACT_INDEX)}`",
                f"- Total judged transcripts: {len(results)}",
                f"- Models: {len(model_summaries)}",
                f"- Categories: {len(category_summaries)}",
                f"- Expected cases per model: {expected_cases_per_model}",
                "",
                "## Model Scores",
                "",
                "| Model | Cases | Gemini Samples | Average Score | Labels |",
                "| --- | ---: | ---: | ---: | --- |",
                *(_model_markdown_row(summary) for summary in model_summaries),
                "",
                "## Category Scores",
                "",
                "| Category | Results | Average Score | Labels |",
                "| --- | ---: | ---: | --- |",
                *(_category_markdown_row(summary) for summary in category_summaries),
                "",
                "## Model Category Matrix",
                "",
                "| Model | " + " | ".join(label for _, label in category_columns) + " |",
                "| --- | " + " | ".join("---:" for _ in category_columns) + " |",
                *(
                    _model_category_matrix_row(row, category_columns=category_columns)
                    for row in coverage_matrix
                ),
                "",
                "## Source Result Files",
                "",
                *(
                    f"- `{_repo_relative(path)}`"
                    for path in source_result_paths or [results_path]
                ),
                "",
                "## Source Result File Coverage",
                "",
                "| Path | Report | Models | Cases | Categories | Gemini Samples | Average Score | Labels |",
                "| --- | --- | --- | ---: | --- | ---: | ---: | --- |",
                *(
                    _source_file_markdown_row(summary)
                    for summary in source_file_summaries
                ),
                "",
                "## Next Refresh Plan",
                "",
                f"- Status: {next_runs['status']}",
                f"- Missing model/category cells: {next_runs['missing_cell_count']}",
                f"- Next run commands: {next_runs['next_run_command_count']}",
                *(
                    f"- Run {command['model']} for {command['missing_case_count']} missing case(s): "
                    f"`{_shell_join(command['command'])}`"
                    for command in next_runs["next_run_commands"]
                ),
                "",
                "## Generated Artifact Index",
                "",
                "| Path | Purpose |",
                "| --- | --- |",
                *(
                    f"| `{artifact['path']}` | {artifact['purpose']} |"
                    for artifact in output_artifacts
                ),
                "",
                "## Refresh Commands",
                "",
                f"- Generated shell playbook: `{_repo_relative(DEFAULT_REFRESH_COMMANDS)}`",
                f"- Seed manifest validation: `{_shell_join(workflow['seed_manifest_validation_command'])}`",
                f"- Audio materialization: `{_shell_join(workflow['audio_materialization_command'])}`",
                f"- Load local Gemini secret before model runs: `{_shell_join(workflow['local_secret_env_command'])}`",
                *(
                    f"- Run {command['model']}: `{_shell_join(command['command'])}`"
                    for command in workflow["model_run_commands"]
                ),
                "- Fallback models if a primary model is blocked: "
                + ", ".join(f"`{model}`" for model in workflow["fallback_model_ids"]),
                f"- Fallback handling: {workflow['fallback_handling']}",
                f"- Combine and refresh committed artifacts: `{_shell_join(workflow['combine_refresh_command'])}`",
                f"- Discover latest complete runs: `{_shell_join(workflow['discover_refresh_command'])}`",
                f"- Manifest-based refresh: `{_shell_join(workflow['manifest_refresh_command'])}`",
                f"- Page validation: `{_shell_join(workflow['page_validation_command'])}`",
                f"- Hosted artifact sync: `{_shell_join(workflow['hosted_artifact_command'])}`",
                "",
                "## Runtime Status",
                "",
                f"- MLX ASR: {runtime_status['mlx_asr']}",
                f"- Gemini judge: {runtime_status['gemini_judge']}",
                f"- Live model calls during refresh: {runtime_status['live_model_calls']}",
                f"- Loaded result providers: {', '.join(runtime_status['loaded_result_providers'])}",
                f"- All loaded results ok: {runtime_status['all_loaded_results_ok']}",
                "",
                "Gemini secrets must be loaded only at runtime from the local secret file.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_next_run_plan_artifact(
    results: list[EvaluationResult],
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


def write_refresh_commands_script(
    output_path: Path,
    *,
    source_result_paths: list[Path] | None = None,
) -> None:
    workflow = _refresh_workflow(source_result_paths or [])
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated ASR leaderboard refresh playbook.",
        "# By default this refreshes committed artifacts from verified result files.",
        "# Live model runs require the local Gemini secret and are listed below as opt-in commands.",
        "",
        _shell_join(workflow["seed_manifest_validation_command"]),
        _shell_join(workflow["combine_refresh_command"]),
        _shell_join(workflow["page_validation_command"]),
        "",
        "# Optional hosted sync; set ASR_LEADERBOARD_HOSTED_DIR to the Pages checkout path first.",
        "# " + _shell_join(workflow["hosted_artifact_command"]),
        "",
        "# Optional when seed cases change: materialize local audio under ignored runs/.",
        "# " + _shell_join(workflow["audio_materialization_command"]),
        "",
        "# Optional live refresh: load the Gemini key only in your local shell before judge calls.",
        "# " + _shell_join(workflow["local_secret_env_command"]),
        "",
        "# Optional live refresh: run primary MLX ASR model jobs when the local runtime is ready.",
        *(
            "# " + _shell_join(command["command"])
            for command in workflow["model_run_commands"]
        ),
        "",
        "# If a primary model is blocked, record the unsupported state before trying fallbacks.",
        "# Fallback models: " + ", ".join(workflow["fallback_model_ids"]),
        "",
        "# Alternative: discover the newest complete primary-model runs.",
        "# " + _shell_join(workflow["discover_refresh_command"]),
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _refresh_workflow(source_result_paths: list[Path]) -> dict[str, object]:
    refresh_command = [
        ".venv/bin/python",
        "scripts/refresh_asr_leaderboard_artifacts.py",
    ]
    for path in source_result_paths:
        refresh_command.extend(["--results", _repo_relative(path)])
    if source_result_paths:
        refresh_command.append("--update-run-manifest")

    return {
        "seed_manifest_validation_command": [
            ".venv/bin/python",
            "scripts/validate_asr_seed_manifest.py",
            "--summary-out",
            _repo_relative(DEFAULT_SEED_MANIFEST_VALIDATION),
        ],
        "audio_materialization_command": [
            ".venv/bin/python",
            "scripts/synthesize_tts_cases.py",
            "--cases",
            _repo_relative(DEFAULT_SEED_CASES),
            "--out",
            "runs/asr-research-audio",
            "--discard-text-sidecars",
            "--summary-out",
            "runs/asr-research-audio/summary.json",
        ],
        "model_run_template": [
            ".venv/bin/oaj",
            "autojudge-mlx-asr",
            "--python-bin",
            ".venv/bin/python",
            "--cases",
            _repo_relative(DEFAULT_AUDIO_CASES),
            "--model",
            "<mlx-community/model-id>",
            "--judge-provider",
            "gemini",
            "--judge-samples",
            "3",
            "--out",
            "runs/asr-leaderboard/<run-name>",
        ],
        "model_run_commands": _model_run_commands(),
        "fallback_model_ids": ASR_FALLBACK_MODELS,
        "fallback_handling": (
            "If a primary MLX ASR model is blocked or unsupported, record the unsupported state "
            "explicitly before trying the fallback model list; do not substitute silently."
        ),
        "combine_refresh_command": refresh_command,
        "discover_refresh_command": [
            ".venv/bin/python",
            "scripts/refresh_asr_leaderboard_artifacts.py",
            "--discover-complete-model-runs",
            "--update-run-manifest",
        ],
        "manifest_refresh_command": [
            ".venv/bin/python",
            "scripts/refresh_asr_leaderboard_artifacts.py",
        ],
        "refresh_commands_path": _repo_relative(DEFAULT_REFRESH_COMMANDS),
        "page_validation_command": [
            ".venv/bin/python",
            "scripts/check_asr_leaderboard_page.py",
        ],
        "hosted_artifact_command": [
            ".venv/bin/python",
            "scripts/refresh_asr_leaderboard_artifacts.py",
            "--hosted-dir-from-env",
        ],
        "hosted_artifact_env_var": DEFAULT_HOSTED_DIR_ENV,
        "local_secret_env_command": [
            "source",
            GEMINI_SECRET_ENV,
        ],
        "secret_handling": (
            "Load the Gemini API key from the local secret file only at runtime; "
            "do not commit or print secrets."
        ),
    }


def _model_run_commands() -> list[dict[str, object]]:
    commands = []
    for model, run_name in ASR_LEADERBOARD_MODELS:
        commands.append(
            {
                "model": model,
                "run_name": run_name,
                "command": [
                    ".venv/bin/oaj",
                    "autojudge-mlx-asr",
                    "--python-bin",
                    ".venv/bin/python",
                    "--cases",
                    _repo_relative(DEFAULT_AUDIO_CASES),
                    "--model",
                    model,
                    "--judge-provider",
                    "gemini",
                    "--judge-samples",
                    "3",
                    "--out",
                    f"runs/asr-leaderboard/{run_name}",
                ],
            }
        )
    return commands


def _run_name_for_model(model: str) -> str:
    for configured_model, run_name in ASR_LEADERBOARD_MODELS:
        if configured_model == model:
            return run_name
    return f"{model.split('/')[-1].replace(' ', '-').replace('_', '-').lower()}-refresh"


def build_output_artifact_index(*, results_path: Path) -> list[dict[str, str]]:
    return [
        {
            "path": _repo_relative(results_path),
            "purpose": "Combined ASR judge results used by the generated page and report.",
        },
        {
            "path": _repo_relative(results_path.with_name("report.html")),
            "purpose": "Local combined HTML report with per-case judge details.",
        },
        {
            "path": _repo_relative(DEFAULT_SUMMARY),
            "purpose": "Machine-readable leaderboard summary and reproducible refresh workflow.",
        },
        {
            "path": _repo_relative(DEFAULT_REFRESH_REPORT),
            "purpose": "Human-readable coverage, score, source-file, and command report.",
        },
        {
            "path": _repo_relative(DEFAULT_REFRESH_COMMANDS),
            "purpose": "Generated shell playbook for repeatable ASR leaderboard refreshes.",
        },
        {
            "path": _repo_relative(DEFAULT_RUN_MANIFEST),
            "purpose": "Committed source result manifest for manifest-based refreshes.",
        },
        {
            "path": _repo_relative(DEFAULT_MANIFEST_VALIDATION),
            "purpose": "Coverage validation for the model/category result matrix.",
        },
        {
            "path": _repo_relative(DEFAULT_SEED_MANIFEST_VALIDATION),
            "purpose": "Seed-manifest validation proving public-safe ASR cases keep exact category coverage.",
        },
        {
            "path": _repo_relative(DEFAULT_NEXT_RUNS),
            "purpose": "Machine-readable next-refresh plan for missing ASR model/category cells.",
        },
        {
            "path": _repo_relative(DEFAULT_HOSTED_MANIFEST),
            "purpose": "Machine-readable manifest of ASR demo artifacts mirrored to the hosted Pages checkout.",
        },
        {
            "path": _repo_relative(DEFAULT_ARTIFACT_INDEX),
            "purpose": "Single machine-readable index for the ASR leaderboard artifact bundle.",
        },
    ]


def build_next_run_plan(
    results: list[EvaluationResult],
    *,
    expected_cases_per_model: int,
) -> dict[str, object]:
    categories = [summary.category for summary in summarize_categories(results)]
    models = [summary.model for summary in summarize_models(results)]
    expected_cases_per_category = (
        expected_cases_per_model // len(categories)
        if categories and expected_cases_per_model % len(categories) == 0
        else None
    )
    counts_by_model_category: Counter[tuple[str, str]] = Counter(
        (
            str(result.metadata.get("candidate_model") or ""),
            str(result.metadata.get("eval_category") or ""),
        )
        for result in results
        if result.status == "ok"
    )

    missing_cells = []
    next_run_commands = []
    for model in models:
        missing_for_model = []
        for category in categories:
            observed = counts_by_model_category.get((model, category), 0)
            expected = expected_cases_per_category or observed
            missing = max(expected - observed, 0)
            if missing:
                missing_for_model.append(
                    {
                        "category": category,
                        "observed_ok_cases": observed,
                        "expected_ok_cases": expected,
                        "missing_ok_cases": missing,
                    }
                )
        missing_cells.extend({"model": model, **cell} for cell in missing_for_model)
        if missing_for_model:
            run_name = _run_name_for_model(model)
            next_run_commands.append(
                {
                    "model": model,
                    "run_name": run_name,
                    "missing_case_count": sum(int(cell["missing_ok_cases"]) for cell in missing_for_model),
                    "categories": [str(cell["category"]) for cell in missing_for_model],
                    "command": [
                        ".venv/bin/oaj",
                        "autojudge-mlx-asr",
                        "--python-bin",
                        ".venv/bin/python",
                        "--cases",
                        _repo_relative(DEFAULT_AUDIO_CASES),
                        "--model",
                        model,
                        "--judge-provider",
                        "gemini",
                        "--judge-samples",
                        "3",
                        "--out",
                        f"runs/asr-leaderboard/{run_name}",
                    ],
                }
            )

    return {
        "status": "complete" if not missing_cells else "incomplete",
        "expected_cases_per_model": expected_cases_per_model,
        "expected_cases_per_category": expected_cases_per_category,
        "model_count": len(models),
        "category_count": len(categories),
        "missing_cell_count": len(missing_cells),
        "missing_cells": missing_cells,
        "next_run_command_count": len(next_run_commands),
        "next_run_commands": next_run_commands,
        "fallback_model_ids": ASR_FALLBACK_MODELS,
        "fallback_handling": (
            "Record unsupported primary model states explicitly before trying fallbacks; "
            "do not silently substitute models."
        ),
    }


def build_refresh_runtime_status(results: list[EvaluationResult]) -> dict[str, object]:
    providers = sorted({result.provider for result in results if result.provider})
    return {
        "mlx_asr": "not_executed_by_refresh; transcripts loaded from verified result artifacts",
        "gemini_judge": (
            "verified_from_loaded_results"
            if "gemini" in providers
            else "not_detected_in_loaded_results"
        ),
        "live_model_calls": "none",
        "loaded_result_providers": providers,
        "all_loaded_results_ok": all(result.status == "ok" for result in results),
    }


def build_model_category_matrix(results: list[EvaluationResult]) -> list[dict[str, object]]:
    model_summaries = summarize_models(results)
    category_order = [category for category, _ in category_columns_for_results(results)]
    matrix = []
    for summary in model_summaries:
        model_results = [
            result
            for result in results
            if result.metadata.get("candidate_model") == summary.model
        ]
        counts = Counter(str(result.metadata.get("eval_category") or "") for result in model_results)
        matrix.append(
            {
                "model": summary.model,
                "total_results": len(model_results),
                "category_counts": {
                    category: counts.get(category, 0)
                    for category in category_order
                },
            }
        )
    return matrix


def category_columns_for_results(results: list[EvaluationResult]) -> list[tuple[str, str]]:
    observed_categories = {
        str(result.metadata.get("eval_category") or "")
        for result in results
    }
    if "" in observed_categories:
        raise ValueError("Every result must include metadata.eval_category.")

    columns = [
        (category, label)
        for category, label in CATEGORY_COLUMNS
        if category in observed_categories or len(observed_categories) <= len(CATEGORY_COLUMNS)
    ]
    known_categories = {category for category, _ in CATEGORY_COLUMNS}
    columns.extend(
        (category, CATEGORY_LABELS.get(category, _titleize_category(category)))
        for category in sorted(observed_categories - known_categories)
    )
    return columns


def summarize_source_result_files(result_paths: list[Path]) -> list[SourceResultFileSummary]:
    summaries = []
    for path in result_paths:
        file_results = load_results_jsonl(path)
        if not file_results:
            raise ValueError(f"Source result file is empty: {path}")
        models = tuple(
            sorted(
                {
                    str(result.metadata.get("candidate_model") or "")
                    for result in file_results
                }
            )
        )
        if any(not model for model in models):
            raise ValueError(f"Missing metadata.candidate_model in {path}")
        categories = Counter(str(result.metadata.get("eval_category") or "") for result in file_results)
        if any(not category for category in categories):
            raise ValueError(f"Missing metadata.eval_category in {path}")
        summaries.append(
            SourceResultFileSummary(
                path=path,
                report_path=path.with_name("report.html"),
                models=models,
                result_count=len(file_results),
                ok_count=sum(1 for result in file_results if result.status == "ok"),
                judge_samples=sum(int(result.metadata.get("judge_sample_count") or 1) for result in file_results),
                average_score=statistics.mean(result.overall_score for result in file_results),
                labels=Counter(result.label for result in file_results),
                categories=categories,
            )
        )
    return summaries


def summarize_models(results: list[EvaluationResult]) -> list[ModelSummary]:
    by_model: dict[str, list[EvaluationResult]] = defaultdict(list)
    for result in results:
        model = str(result.metadata.get("candidate_model") or "")
        if not model:
            raise ValueError(f"Missing metadata.candidate_model for {result.case_id}")
        by_model[model].append(result)

    summaries = []
    for model, model_results in by_model.items():
        summaries.append(
            ModelSummary(
                model=model,
                result_count=len(model_results),
                ok_count=sum(1 for result in model_results if result.status == "ok"),
                judge_samples=sum(int(result.metadata.get("judge_sample_count") or 1) for result in model_results),
                average_score=statistics.mean(result.overall_score for result in model_results),
                labels=Counter(result.label for result in model_results),
            )
        )
    return sorted(summaries, key=lambda summary: (-summary.average_score, summary.model.lower()))


def summarize_categories(results: list[EvaluationResult]) -> list[CategorySummary]:
    by_category: dict[str, list[EvaluationResult]] = defaultdict(list)
    for result in results:
        category = str(result.metadata.get("eval_category") or "")
        if not category:
            raise ValueError(f"Missing metadata.eval_category for {result.case_id}")
        by_category[category].append(result)

    summaries = []
    for category, category_results in by_category.items():
        summaries.append(
            CategorySummary(
                category=category,
                result_count=len(category_results),
                average_score=statistics.mean(result.overall_score for result in category_results),
                labels=Counter(result.label for result in category_results),
            )
        )
    column_order = {category: index for index, (category, _) in enumerate(CATEGORY_COLUMNS)}
    return sorted(
        summaries,
        key=lambda summary: (column_order.get(summary.category, len(column_order)), summary.category),
    )


def validate_coverage(
    results: list[EvaluationResult],
    model_summaries: list[ModelSummary],
    *,
    expected_cases_per_model: int,
) -> None:
    category_counts = Counter(str(result.metadata.get("eval_category") or "") for result in results)
    if any(not category for category in category_counts):
        raise ValueError("Every result must include metadata.eval_category.")
    if len(set(category_counts.values())) != 1:
        raise ValueError(f"Uneven category coverage: {dict(category_counts)}")
    for summary in model_summaries:
        if summary.result_count != expected_cases_per_model:
            raise ValueError(
                f"{summary.model} has {summary.result_count} results; expected {expected_cases_per_model}."
            )
        if summary.ok_count != summary.result_count:
            raise ValueError(f"{summary.model} has {summary.result_count - summary.ok_count} non-ok results.")
        model_category_counts = Counter(
            str(result.metadata.get("eval_category") or "")
            for result in results
            if result.metadata.get("candidate_model") == summary.model
        )
        if set(model_category_counts) != set(category_counts) or len(set(model_category_counts.values())) != 1:
            raise ValueError(
                f"{summary.model} has uneven category coverage: {dict(model_category_counts)}"
            )


def replace_generated_block(page: Path, generated: str) -> None:
    html_text = page.read_text(encoding="utf-8")
    if START_MARKER not in html_text or END_MARKER not in html_text:
        raise ValueError(f"{page} must contain {START_MARKER} and {END_MARKER}.")
    before, remainder = html_text.split(START_MARKER, 1)
    _, after = remainder.split(END_MARKER, 1)
    page.write_text(before + generated + after, encoding="utf-8")


def _render_model_row(summary: ModelSummary) -> str:
    labels = ", ".join(
        f"{summary.labels[label]} {label}"
        for label in ("accurate", "needs_review", "inaccurate")
        if summary.labels[label]
    )
    return (
        "        <tr>"
        f"<td><code>{html.escape(summary.model)}</code></td>"
        f"<td>{summary.ok_count}/{summary.result_count} ok</td>"
        f"<td>{summary.judge_samples}</td>"
        f"<td>{summary.average_score:.1f}</td>"
        f"<td>{html.escape(labels)}</td>"
        "</tr>"
    )


def _ordered_label_counts(labels: Counter[str]) -> dict[str, int]:
    return {
        label: labels[label]
        for label in ("accurate", "needs_review", "inaccurate")
        if labels[label]
    }


def _model_markdown_row(summary: ModelSummary) -> str:
    labels = ", ".join(
        f"{summary.labels[label]} {label}"
        for label in ("accurate", "needs_review", "inaccurate")
        if summary.labels[label]
    )
    return (
        f"| `{summary.model}` | {summary.ok_count}/{summary.result_count} ok | "
        f"{summary.judge_samples} | {summary.average_score:.1f} | {labels} |"
    )


def _category_markdown_row(summary: CategorySummary) -> str:
    labels = ", ".join(
        f"{summary.labels[label]} {label}"
        for label in ("accurate", "needs_review", "inaccurate")
        if summary.labels[label]
    )
    return f"| `{summary.category}` | {summary.result_count} | {summary.average_score:.1f} | {labels} |"


def _model_category_matrix_row(
    row: dict[str, object],
    *,
    category_columns: list[tuple[str, str]] | None = None,
) -> str:
    counts = row["category_counts"]
    if not isinstance(counts, dict):
        raise TypeError("category_counts must be a dictionary")
    columns = category_columns or CATEGORY_COLUMNS
    cells = " | ".join(str(counts.get(category, 0)) for category, _ in columns)
    return f"| `{row['model']}` | {cells} |"


def _source_file_summary_json(summary: SourceResultFileSummary) -> dict[str, object]:
    return {
        "path": _repo_relative(summary.path),
        "report_path": _repo_relative(summary.report_path),
        "models": list(summary.models),
        "result_count": summary.result_count,
        "ok_count": summary.ok_count,
        "judge_samples": summary.judge_samples,
        "average_score": round(summary.average_score, 3),
        "labels": _ordered_label_counts(summary.labels),
        "categories": {
            category: summary.categories[category]
            for category in sorted(summary.categories)
        },
    }


def _source_file_markdown_row(summary: SourceResultFileSummary) -> str:
    models = "<br>".join(f"`{model}`" for model in summary.models)
    categories = ", ".join(
        f"`{category}`: {count}"
        for category, count in sorted(summary.categories.items())
    )
    labels = ", ".join(
        f"{summary.labels[label]} {label}"
        for label in ("accurate", "needs_review", "inaccurate")
        if summary.labels[label]
    )
    return (
        f"| `{_repo_relative(summary.path)}` | `{_repo_relative(summary.report_path)}` | {models} | "
        f"{summary.ok_count}/{summary.result_count} ok | {categories} | "
        f"{summary.judge_samples} | {summary.average_score:.1f} | {labels} |"
    )


def _render_source_report_rows(summaries: list[SourceResultFileSummary]) -> list[str]:
    if not summaries:
        return []
    rows = [
        "",
        "    <h2>Source Run Reports</h2>",
        (
            "    <p class=\"muted\">Each source run keeps its own local report alongside "
            "the JSONL file that feeds the combined 35-case leaderboard.</p>"
        ),
        "    <table>",
        "      <thead><tr><th>Result File</th><th>Local Report</th><th>Cases</th><th>Categories</th></tr></thead>",
        "      <tbody>",
    ]
    for summary in summaries:
        categories = ", ".join(
            f"{html.escape(category)}: {count}"
            for category, count in sorted(summary.categories.items())
        )
        rows.append(
            "        <tr>"
            f"<td><code>{html.escape(_repo_relative(summary.path))}</code></td>"
            f"<td><code>{html.escape(_repo_relative(summary.report_path))}</code></td>"
            f"<td>{summary.ok_count}/{summary.result_count} ok</td>"
            f"<td>{categories}</td>"
            "</tr>"
        )
    rows.extend(["      </tbody>", "    </table>"])
    return rows


def _shell_join(command: object) -> str:
    if not isinstance(command, list):
        raise TypeError(f"Expected command list, got {type(command).__name__}")
    return " ".join(str(part) for part in command)


def _render_category_row(
    model: str,
    results: list[EvaluationResult],
    *,
    category_columns: list[tuple[str, str]] | None = None,
) -> str:
    model_results = [result for result in results if result.metadata.get("candidate_model") == model]
    by_category: dict[str, list[EvaluationResult]] = defaultdict(list)
    for result in model_results:
        by_category[str(result.metadata["eval_category"])].append(result)

    cells = []
    for category, _ in category_columns or CATEGORY_COLUMNS:
        category_results = by_category[category]
        if not category_results:
            cells.append("<td>0 cases</td>")
            continue
        labels = Counter(result.label for result in category_results)
        label_summary = ", ".join(
            f"{labels[label]} {label}" for label in ("accurate", "needs_review", "inaccurate") if labels[label]
        )
        average = statistics.mean(result.overall_score for result in category_results)
        cells.append(
            f"<td>{len(category_results)} cases, {average:.1f} avg, {html.escape(label_summary)}</td>"
        )

    return (
        "        <tr>"
        f"<td><code>{html.escape(model)}</code></td>"
        + "".join(cells)
        + "</tr>"
    )


def _titleize_category(category: str) -> str:
    return " ".join(word.capitalize() for word in category.split("_") if word)


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    main()
