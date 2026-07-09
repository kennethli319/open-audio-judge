from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from open_audio_judge.models import EvaluationCase


DEFAULT_MLX_ASR_MODELS = (
    "mlx-community/whisper-large-v3-turbo-asr-fp16",
    "mlx-community/Qwen3-ASR-1.7B-8bit",
    "mlx-community/VibeVoice-ASR-4bit",
)
DEFAULT_MLX_ASR_MODULE = "mlx_audio.stt.generate"

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class MlxAsrConfig:
    model: str
    python_bin: str = sys.executable
    module: str = DEFAULT_MLX_ASR_MODULE
    timeout_seconds: float | None = None
    extra_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class MlxAsrTranscript:
    text: str
    raw_stdout: str
    raw_stderr: str


def transcribe_cases_with_mlx_asr(
    cases: Iterable[EvaluationCase],
    *,
    config: MlxAsrConfig,
    base_dir: Path | None = None,
    runner: CommandRunner = subprocess.run,
) -> list[EvaluationCase]:
    transcribed: list[EvaluationCase] = []
    for case in cases:
        transcript = transcribe_case_with_mlx_asr(
            case,
            config=config,
            base_dir=base_dir,
            runner=runner,
        )
        metadata = dict(case.metadata)
        metadata.update(
            {
                "candidate_model": config.model,
                "candidate_transcriber": "mlx-audio-stt",
                "candidate_text_source": "mlx_audio_stt_generate",
            }
        )
        transcribed.append(
            case.model_copy(update={"candidate_text": transcript.text, "metadata": metadata})
        )
    return transcribed


def transcribe_case_with_mlx_asr(
    case: EvaluationCase,
    *,
    config: MlxAsrConfig,
    base_dir: Path | None = None,
    runner: CommandRunner = subprocess.run,
) -> MlxAsrTranscript:
    if not case.audio_path:
        raise ValueError(f"Case {case.id} requires a local audio_path for MLX ASR.")
    audio_path = _resolve_audio_path(case.audio_path, base_dir=base_dir)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Case {case.id} audio_path file not found: {audio_path}")
    if config.timeout_seconds is not None and config.timeout_seconds <= 0:
        raise ValueError("MLX ASR timeout_seconds must be greater than zero.")

    command = _mlx_asr_command(config, audio_path)
    completed = runner(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=config.timeout_seconds,
    )
    text = _extract_mlx_transcript_text(completed.stdout)
    if not text:
        raise ValueError(f"MLX ASR returned no transcript text for case {case.id}.")
    return MlxAsrTranscript(
        text=text,
        raw_stdout=completed.stdout,
        raw_stderr=completed.stderr,
    )


def write_mlx_asr_cases_jsonl(cases: Iterable[EvaluationCase], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case.model_dump(exclude_none=True), ensure_ascii=False) + "\n")
    return path


def write_mlx_asr_summary_json(
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
        "candidate_transcriber": "mlx-audio-stt",
        "total_cases": len(case_list),
        "case_ids": [case.id for case in case_list],
        "cases_with_candidate_text": sum(1 for case in case_list if case.candidate_text),
        "by_eval_category": _count_metadata(case_list, "eval_category"),
        "by_asr_slice": _count_metadata(case_list, "asr_slice"),
        "by_language": _count_language(case_list),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _mlx_asr_command(config: MlxAsrConfig, audio_path: Path) -> list[str]:
    return [
        config.python_bin,
        "-m",
        config.module,
        "--model",
        config.model,
        "--audio",
        str(audio_path),
        *config.extra_args,
    ]


def _extract_mlx_transcript_text(stdout: str) -> str:
    stripped = stdout.strip()
    if not stripped:
        return ""
    for candidate in (stripped, *reversed(stripped.splitlines())):
        parsed = _maybe_json_text(candidate)
        if parsed:
            return parsed
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[-1].removeprefix("Text:").strip()


def _maybe_json_text(text: str) -> str:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return ""
    return _find_text_field(data).strip()


def _find_text_field(data: Any) -> str:
    if isinstance(data, dict):
        for key in ("text", "transcript", "prediction"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value
        segments = data.get("segments")
        if isinstance(segments, list):
            joined = " ".join(_find_text_field(segment) for segment in segments)
            return " ".join(joined.split())
    if isinstance(data, list):
        joined = " ".join(_find_text_field(item) for item in data)
        return " ".join(joined.split())
    return ""


def _resolve_audio_path(audio_path: str, *, base_dir: Path | None) -> Path:
    path = Path(audio_path)
    if path.is_absolute() or base_dir is None:
        return path
    return base_dir / path


def _count_metadata(cases: Sequence[EvaluationCase], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        value = case.metadata.get(key)
        if isinstance(value, str) and value.strip():
            counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _count_language(cases: Sequence[EvaluationCase]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        value = case.metadata.get("language")
        if isinstance(value, str) and value.strip():
            counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))
