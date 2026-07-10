from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from open_audio_judge.json_utils import parse_judge_output
from open_audio_judge.models import EvaluationCase, EvaluationResult, JudgePrompt
from open_audio_judge.prompting import render_prompt
from open_audio_judge.providers.base import JudgeProvider
from open_audio_judge.reports import label_for_score, write_html_report


def load_cases(path: Path) -> list[EvaluationCase]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        cases = [
            EvaluationCase.model_validate(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        records = data if isinstance(data, list) else data.get("cases", [])
        cases = [EvaluationCase.model_validate(record) for record in records]
    else:
        raise ValueError(f"Unsupported case file type: {path.suffix}")

    return [_resolve_audio_path(case, path.parent) for case in cases]


def evaluate_cases(
    cases: list[EvaluationCase],
    prompt: JudgePrompt,
    provider: JudgeProvider,
    out_dir: Path,
    *,
    judge_samples: int = 1,
) -> list[EvaluationResult]:
    if judge_samples < 1:
        raise ValueError("judge_samples must be at least 1.")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = [
        evaluate_case(case, prompt, provider)
        if judge_samples == 1
        else evaluate_case_with_sampling(case, prompt, provider, judge_samples=judge_samples)
        for case in cases
    ]
    write_results_jsonl(results, out_dir / "results.jsonl")
    write_html_report(results, out_dir / "report.html")
    return results


def evaluate_case_with_sampling(
    case: EvaluationCase,
    prompt: JudgePrompt,
    provider: JudgeProvider,
    *,
    judge_samples: int,
) -> EvaluationResult:
    if judge_samples < 1:
        raise ValueError("judge_samples must be at least 1.")
    attempts = [evaluate_case(case, prompt, provider) for _ in range(judge_samples)]
    successful_attempts = [attempt for attempt in attempts if attempt.status == "ok"]
    scoring_attempts = successful_attempts or attempts
    scores = [attempt.overall_score for attempt in scoring_attempts]
    average_score = round(sum(scores) / len(scores), 2)
    final_score = max(1, min(100, int(round(average_score))))
    representative = _representative_attempt(attempts)
    failure_count = judge_samples - len(successful_attempts)
    metadata = dict(representative.metadata)
    metadata.update(
        {
            "judge_sample_count": judge_samples,
            "judge_sample_success_count": len(successful_attempts),
            "judge_sample_failure_count": failure_count,
            "judge_sample_scores": scores,
            "judge_sample_average": average_score,
            "judge_sample_statuses": [attempt.status for attempt in attempts],
        }
    )

    return representative.model_copy(
        update={
            "overall_score": final_score,
            "label": label_for_score(final_score),
            "reason": _sampled_reason(
                attempts,
                scoring_attempts=scoring_attempts,
                average_score=average_score,
            ),
            "metadata": metadata,
            "raw_response": {
                "judge_samples": [
                    _sample_attempt_record(index, attempt)
                    for index, attempt in enumerate(attempts, 1)
                ]
            },
        }
    )


def evaluate_case(
    case: EvaluationCase,
    prompt: JudgePrompt,
    provider: JudgeProvider,
) -> EvaluationResult:
    rendered = render_prompt(prompt, case)
    try:
        provider_response = provider.generate(case, rendered)
    except Exception as exc:
        judge_output = None
        status = "provider_error"
        error = str(exc)
        raw_response = _provider_error_metadata(exc)
    else:
        raw_response = provider_response.raw
        try:
            judge_output = parse_judge_output(provider_response.content, prompt.response_schema)
            status = "ok"
            error = None
        except (ValidationError, ValueError) as exc:
            judge_output = None
            status = "parse_error"
            error = str(exc)

    score = judge_output.overall_score if judge_output else 1
    reason = judge_output.reason if judge_output else f"Evaluation failed: {error}"

    return EvaluationResult(
        case_id=case.id,
        task=case.task,
        judge_id=prompt.id,
        judge_version=prompt.version,
        provider=provider.name,
        overall_score=score,
        reason=reason,
        judge_transcript=judge_output.judge_transcript if judge_output else None,
        meaning_preservation=judge_output.meaning_preservation if judge_output else None,
        semantic_error_summary=judge_output.semantic_error_summary if judge_output else None,
        key_differences=judge_output.key_differences if judge_output else [],
        error_categories=judge_output.error_categories if judge_output else [],
        researcher_notes=judge_output.researcher_notes if judge_output else [],
        label=label_for_score(score),  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        error=error,
        metadata=case.metadata,
        raw_response=raw_response,
    )


def write_results_jsonl(results: list[EvaluationResult], path: Path) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.model_dump(), ensure_ascii=False) + "\n")
    return path


def load_results_jsonl(path: Path) -> list[EvaluationResult]:
    return [
        _reconcile_sampled_result(EvaluationResult.model_validate(json.loads(line)))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _reconcile_sampled_result(result: EvaluationResult) -> EvaluationResult:
    """Repair legacy sampled scores that treated failed judge calls as a score of one."""
    statuses = result.metadata.get("judge_sample_statuses")
    scores = result.metadata.get("judge_sample_scores")
    if (
        not isinstance(statuses, list)
        or not isinstance(scores, list)
        or len(statuses) != len(scores)
        or not statuses
    ):
        return result

    successful_scores = [
        score
        for score, status in zip(scores, statuses, strict=True)
        if status == "ok" and isinstance(score, (int, float))
    ]
    if not successful_scores:
        return result

    failure_count = len(statuses) - len(successful_scores)
    average_score = round(sum(successful_scores) / len(successful_scores), 2)
    final_score = max(1, min(100, int(round(average_score))))
    metadata = dict(result.metadata)
    metadata.update(
        {
            "judge_sample_success_count": len(successful_scores),
            "judge_sample_failure_count": failure_count,
            "judge_sample_scores": successful_scores,
            "judge_sample_average": average_score,
        }
    )
    if failure_count:
        representative_reason = result.reason.split("Representative reason: ", 1)[-1]
        reason = (
            f"Average of {len(successful_scores)} successful judge samples "
            f"({failure_count} failed attempt{'s' if failure_count != 1 else ''} excluded): "
            f"{average_score:.2f} (scores: {', '.join(str(score) for score in successful_scores)}). "
            f"Representative reason: {representative_reason}"
        )
    else:
        reason = result.reason
    return result.model_copy(
        update={
            "overall_score": final_score,
            "label": label_for_score(final_score),
            "reason": reason,
            "metadata": metadata,
        }
    )


def _provider_error_metadata(exc: Exception) -> dict[str, str]:
    message = str(exc).strip().replace("\n", " ")
    if len(message) > 500:
        message = f"{message[:500]}..."
    return {
        "error_type": type(exc).__name__,
        "message": message,
    }


def _representative_attempt(attempts: list[EvaluationResult]) -> EvaluationResult:
    ok_attempts = [attempt for attempt in attempts if attempt.status == "ok"]
    candidates = ok_attempts or attempts
    return sorted(candidates, key=lambda attempt: (attempt.overall_score, attempt.case_id))[
        len(candidates) // 2
    ]


def _sampled_reason(
    attempts: list[EvaluationResult],
    *,
    scoring_attempts: list[EvaluationResult],
    average_score: float,
) -> str:
    scores = ", ".join(str(attempt.overall_score) for attempt in scoring_attempts)
    representative = _representative_attempt(attempts)
    failure_count = len(attempts) - sum(attempt.status == "ok" for attempt in attempts)
    if failure_count:
        sample_summary = (
            f"Average of {len(scoring_attempts)} successful judge samples "
            f"({failure_count} failed attempt{'s' if failure_count != 1 else ''} excluded)"
            if any(attempt.status == "ok" for attempt in attempts)
            else f"No successful judge samples; using {len(scoring_attempts)} failed attempts"
        )
    else:
        sample_summary = f"Average of {len(attempts)} judge samples"
    return (
        f"{sample_summary}: {average_score:.2f} "
        f"(scores: {scores}). Representative reason: {representative.reason}"
    )


def _sample_attempt_record(index: int, attempt: EvaluationResult) -> dict[str, object]:
    return {
        "sample_index": index,
        "status": attempt.status,
        "overall_score": attempt.overall_score,
        "label": attempt.label,
        "reason": attempt.reason,
        "error_categories": attempt.error_categories,
        "semantic_error_summary": attempt.semantic_error_summary,
        "error": attempt.error,
    }


def _resolve_audio_path(case: EvaluationCase, base_dir: Path) -> EvaluationCase:
    if not case.audio_path:
        return case
    audio_path = Path(case.audio_path)
    if audio_path.is_absolute():
        return case
    return case.model_copy(update={"audio_path": str((base_dir / audio_path).resolve())})
