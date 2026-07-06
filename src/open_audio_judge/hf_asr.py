from __future__ import annotations

import importlib
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from open_audio_judge.models import EvaluationCase


PipelineFactory = Callable[..., Callable[[str], Any]]


@dataclass(frozen=True)
class HfAsrTranscript:
    text: str
    raw: dict[str, Any]


def transcribe_cases_with_hf_asr(
    cases: Iterable[EvaluationCase],
    *,
    model: str,
    device: str | int = "cpu",
    pipeline_factory: PipelineFactory | None = None,
) -> list[EvaluationCase]:
    transcriber = build_hf_asr_pipeline(
        model=model,
        device=device,
        pipeline_factory=pipeline_factory,
    )
    transcribed: list[EvaluationCase] = []
    for case in cases:
        transcript = transcribe_case_with_hf_asr(case, transcriber)
        metadata = dict(case.metadata)
        metadata.update(
            {
                "candidate_model": model,
                "candidate_transcriber": "huggingface-transformers-asr",
                "candidate_text_source": "hf_automatic_speech_recognition",
            }
        )
        transcribed.append(
            case.model_copy(update={"candidate_text": transcript.text, "metadata": metadata})
        )
    return transcribed


def build_hf_asr_pipeline(
    *,
    model: str,
    device: str | int = "cpu",
    pipeline_factory: PipelineFactory | None = None,
) -> Callable[[str], Any]:
    factory = pipeline_factory or _load_transformers_pipeline()
    return factory(
        task="automatic-speech-recognition",
        model=model,
        device=_normalize_device(device),
    )


def transcribe_case_with_hf_asr(
    case: EvaluationCase,
    transcriber: Callable[[str], Any],
) -> HfAsrTranscript:
    if not case.audio_path:
        raise ValueError(f"Case {case.id} requires a local audio_path for Hugging Face ASR.")
    audio_path = Path(case.audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Case {case.id} audio_path file not found: {audio_path}")

    raw = transcriber(str(audio_path))
    text = _extract_transcript_text(raw)
    if not text:
        raise ValueError(f"Hugging Face ASR returned no transcript text for case {case.id}.")
    return HfAsrTranscript(text=text, raw=_safe_raw_metadata(raw))


def write_hf_asr_cases_jsonl(cases: Iterable[EvaluationCase], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case.model_dump(exclude_none=True), ensure_ascii=False) + "\n")
    return path


def write_hf_asr_summary_json(
    cases: Iterable[EvaluationCase],
    path: Path,
    *,
    source_cases: Path,
    model: str,
) -> Path:
    case_list = list(cases)
    summary = {
        "source_cases": str(source_cases),
        "candidate_model": model,
        "candidate_transcriber": "huggingface-transformers-asr",
        "total_cases": len(case_list),
        "case_ids": [case.id for case in case_list],
        "cases_with_candidate_text": sum(1 for case in case_list if case.candidate_text),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _load_transformers_pipeline() -> PipelineFactory:
    try:
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face ASR requires the optional `transformers` package. "
            "Install it in your environment, then rerun this command."
        ) from exc
    return transformers.pipeline


def _normalize_device(device: str | int) -> str | int:
    if isinstance(device, int):
        return device
    normalized = device.strip().lower()
    if normalized in {"cpu", "mps", "cuda"}:
        return normalized
    try:
        return int(normalized)
    except ValueError:
        return device


def _extract_transcript_text(raw: Any) -> str:
    if isinstance(raw, dict):
        text = raw.get("text")
        if isinstance(text, str):
            return text.strip()
        chunks = raw.get("chunks")
        if isinstance(chunks, list):
            return " ".join(
                str(chunk.get("text") or "").strip()
                for chunk in chunks
                if isinstance(chunk, dict) and str(chunk.get("text") or "").strip()
            ).strip()
    if isinstance(raw, str):
        return raw.strip()
    return ""


def _safe_raw_metadata(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {"response_type": type(raw).__name__}
    safe: dict[str, Any] = {}
    if isinstance(raw.get("chunks"), list):
        safe["chunk_count"] = len(raw["chunks"])
    if raw.get("language"):
        safe["language"] = raw["language"]
    return safe
