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
) -> list[EvaluationResult]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results = [evaluate_case(case, prompt, provider) for case in cases]
    write_results_jsonl(results, out_dir / "results.jsonl")
    write_html_report(results, out_dir / "report.html")
    return results


def evaluate_case(
    case: EvaluationCase,
    prompt: JudgePrompt,
    provider: JudgeProvider,
) -> EvaluationResult:
    rendered = render_prompt(prompt, case)
    try:
        provider_response = provider.generate(case, rendered)
        judge_output = parse_judge_output(provider_response.content)
        status = "ok"
        error = None
        raw_response = provider_response.raw
    except ValidationError as exc:
        judge_output = None
        status = "parse_error"
        error = str(exc)
        raw_response = {}
    except ValueError as exc:
        judge_output = None
        status = "parse_error"
        error = str(exc)
        raw_response = {}
    except Exception as exc:
        judge_output = None
        status = "provider_error"
        error = str(exc)
        raw_response = {}

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


def _resolve_audio_path(case: EvaluationCase, base_dir: Path) -> EvaluationCase:
    if not case.audio_path:
        return case
    audio_path = Path(case.audio_path)
    if audio_path.is_absolute():
        return case
    return case.model_copy(update={"audio_path": str((base_dir / audio_path).resolve())})
