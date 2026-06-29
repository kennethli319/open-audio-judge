from __future__ import annotations

from fastapi import FastAPI

from open_audio_judge.models import BatchEvaluateRequest, EvaluateRequest, EvaluationResult
from open_audio_judge.prompting import load_prompt
from open_audio_judge.providers import build_provider
from open_audio_judge.runner import evaluate_case

app = FastAPI(title="Open Audio Judge", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/judges")
def judges() -> dict[str, list[str]]:
    return {"judges": ["asr_error", "tts_naturalness"]}


@app.post("/v1/evaluate", response_model=EvaluationResult)
def evaluate(request: EvaluateRequest) -> EvaluationResult:
    prompt = load_prompt(request.judge)
    provider = build_provider(request.provider)
    return evaluate_case(request.case, prompt, provider)


@app.post("/v1/evaluate/batch", response_model=list[EvaluationResult])
def evaluate_batch(request: BatchEvaluateRequest) -> list[EvaluationResult]:
    prompt = load_prompt(request.judge)
    provider = build_provider(request.provider)
    return [evaluate_case(case, prompt, provider) for case in request.cases]
