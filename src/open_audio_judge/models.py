from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    role: str
    content: str


class EvaluationCase(BaseModel):
    id: str
    task: str
    audio_path: str | None = None
    audio_url: str | None = None
    turns: list[ConversationTurn] = Field(default_factory=list)
    reference_text: str | None = None
    candidate_text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class JudgePrompt(BaseModel):
    id: str
    version: str
    task: str
    description: str | None = None
    system: str
    user: str
    response_schema: dict[str, Any] = Field(default_factory=dict)


class RenderedPrompt(BaseModel):
    judge_id: str
    judge_version: str
    system: str
    user: str
    response_schema: dict[str, Any] = Field(default_factory=dict)


class ProviderResponse(BaseModel):
    content: str
    raw: dict[str, Any] = Field(default_factory=dict)


class JudgeOutput(BaseModel):
    overall_score: int = Field(ge=1, le=100)
    reason: str
    judge_transcript: str | None = None
    meaning_preservation: str | None = None
    semantic_error_summary: str | None = None
    key_differences: list[str] = Field(default_factory=list)
    error_categories: list[str] = Field(default_factory=list)
    researcher_notes: list[str] = Field(default_factory=list)


class EvaluationResult(BaseModel):
    case_id: str
    task: str
    judge_id: str
    judge_version: str
    provider: str
    overall_score: int = Field(ge=1, le=100)
    reason: str
    judge_transcript: str | None = None
    meaning_preservation: str | None = None
    semantic_error_summary: str | None = None
    key_differences: list[str] = Field(default_factory=list)
    error_categories: list[str] = Field(default_factory=list)
    researcher_notes: list[str] = Field(default_factory=list)
    label: Literal["accurate", "needs_review", "inaccurate"]
    status: Literal["ok", "parse_error", "provider_error"] = "ok"
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    raw_response: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class EvaluateRequest(BaseModel):
    case: EvaluationCase
    judge: str = "asr_error"
    provider: str | None = None


class BatchEvaluateRequest(BaseModel):
    cases: list[EvaluationCase]
    judge: str = "asr_error"
    provider: str | None = None
