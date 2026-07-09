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
DEFAULT_AUDIO_CASES = ROOT / "runs" / "asr-research-audio" / "tts_audio_cases.jsonl"
DEFAULT_SEED_CASES = ROOT / "examples" / "asr_research_cases.jsonl"
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
    print(f"Updated {args.page} from {args.results} ({len(results)} results)")
    print(f"Summary: {args.summary_out}")


def render_generated_sections(
    results: list[EvaluationResult],
    *,
    results_path: Path,
    expected_cases_per_model: int,
) -> str:
    if not results:
        raise ValueError("No ASR evaluation results were loaded.")

    model_summaries = summarize_models(results)
    validate_coverage(results, model_summaries, expected_cases_per_model=expected_cases_per_model)
    total_judge_samples = sum(summary.judge_samples for summary in model_summaries)
    categories = sorted({str(result.metadata.get("eval_category", "")) for result in results})
    category_list = ", ".join(f"<code>{html.escape(category)}</code>" for category in categories)
    results_label = html.escape(_repo_relative(results_path))
    report_label = html.escape(_repo_relative(results_path.with_name("report.html")))
    summary_label = html.escape(_repo_relative(DEFAULT_SUMMARY))

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
            + "".join(f"<th>{html.escape(label)}</th>" for _, label in CATEGORY_COLUMNS)
            + "</tr></thead>",
            "      <tbody>",
            *(_render_category_row(model, results) for model in [summary.model for summary in model_summaries]),
            "      </tbody>",
            "    </table>",
            (
                '    <p class="muted">Total Gemini judge samples: '
                f"{total_judge_samples}. Refresh this block with "
                '<code>.venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py</code> '
                "after rerunning the verified ASR model jobs. The combined local report is "
                f"<code>{report_label}</code> and the committed summary artifact is "
                f"<code>{summary_label}</code>; that summary includes the source result files "
                "and reproducible refresh workflow.</p>"
            ),
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
                "refresh_workflow": _refresh_workflow(source_result_paths or []),
                "total_results": len(results),
                "model_count": len(model_summaries),
                "category_count": len(category_summaries),
                "expected_cases_per_model": expected_cases_per_model,
                "total_gemini_judge_samples": sum(summary.judge_samples for summary in model_summaries),
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


def _refresh_workflow(source_result_paths: list[Path]) -> dict[str, object]:
    refresh_command = [
        ".venv/bin/python",
        "scripts/refresh_asr_leaderboard_artifacts.py",
    ]
    for path in source_result_paths:
        refresh_command.extend(["--results", _repo_relative(path)])

    return {
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
            "oaj",
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
        "combine_refresh_command": refresh_command,
        "secret_handling": (
            "Load the Gemini API key from the local secret file only at runtime; "
            "do not commit or print secrets."
        ),
    }


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


def _render_category_row(model: str, results: list[EvaluationResult]) -> str:
    model_results = [result for result in results if result.metadata.get("candidate_model") == model]
    by_category: dict[str, list[EvaluationResult]] = defaultdict(list)
    for result in model_results:
        by_category[str(result.metadata["eval_category"])].append(result)

    cells = []
    for category, _ in CATEGORY_COLUMNS:
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


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    main()
