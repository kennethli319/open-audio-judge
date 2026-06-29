"""Fill ASR case manifests with candidate transcripts from a Whisper CLI model."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from open_audio_judge.models import EvaluationCase
from open_audio_judge.runner import load_cases


DEFAULT_OUT = Path("runs") / "whisper-tiny-asr-open"


@dataclass(frozen=True)
class WhisperTranscript:
    text: str
    language: str | None
    segments: int
    json_path: Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True, help="Input ASR case JSONL/JSON.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT / "cases.jsonl")
    parser.add_argument(
        "--transcripts-dir",
        type=Path,
        default=DEFAULT_OUT / "transcripts",
        help="Directory for per-case Whisper JSON outputs.",
    )
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_OUT / "summary.json")
    parser.add_argument("--whisper-bin", default="whisper", help="OpenAI Whisper CLI executable.")
    parser.add_argument("--model", default="tiny", help="Whisper model name, such as tiny or tiny.en.")
    parser.add_argument("--language", default="en", help="Whisper language argument.")
    parser.add_argument("--device", default="cpu", help="Whisper device argument.")
    parser.add_argument("--threads", default="4", help="Whisper threads argument.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing per-case Whisper JSON outputs when present.",
    )
    args = parser.parse_args()

    cases = load_cases(args.cases)
    if args.limit is not None:
        cases = cases[: args.limit]

    transcribed = transcribe_cases(
        cases,
        transcripts_dir=args.transcripts_dir,
        whisper_bin=args.whisper_bin,
        model=args.model,
        language=args.language,
        device=args.device,
        threads=args.threads,
        skip_existing=args.skip_existing,
    )
    write_cases_jsonl(transcribed, args.out)
    write_summary_json(transcribed, args.summary_out, source_cases=args.cases, model=args.model)
    print(f"Wrote {len(transcribed)} transcribed ASR cases to {args.out}")
    print(f"Wrote summary to {args.summary_out}")


def transcribe_cases(
    cases: Iterable[EvaluationCase],
    *,
    transcripts_dir: Path,
    whisper_bin: str,
    model: str,
    language: str,
    device: str,
    threads: str,
    skip_existing: bool = False,
) -> list[EvaluationCase]:
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    transcribed: list[EvaluationCase] = []
    for case in cases:
        if not case.audio_path:
            raise ValueError(f"Case {case.id} requires a local audio_path for Whisper transcription.")
        transcript = transcribe_case(
            case,
            transcripts_dir=transcripts_dir,
            whisper_bin=whisper_bin,
            model=model,
            language=language,
            device=device,
            threads=threads,
            skip_existing=skip_existing,
        )
        metadata = dict(case.metadata)
        metadata.update(
            {
                "candidate_model": f"whisper-{model}",
                "candidate_transcriber": "openai-whisper-cli",
                "candidate_text_source": "whisper_cli_json",
                "whisper_language": transcript.language,
                "whisper_segment_count": transcript.segments,
                "whisper_json_path": str(transcript.json_path),
            }
        )
        transcribed.append(
            case.model_copy(update={"candidate_text": transcript.text, "metadata": metadata})
        )
    return transcribed


def transcribe_case(
    case: EvaluationCase,
    *,
    transcripts_dir: Path,
    whisper_bin: str,
    model: str,
    language: str,
    device: str,
    threads: str,
    skip_existing: bool = False,
) -> WhisperTranscript:
    audio_path = Path(case.audio_path or "")
    if not audio_path.is_file():
        raise FileNotFoundError(f"Case {case.id} audio_path file not found: {audio_path}")

    case_output_dir = transcripts_dir / _safe_case_id(case.id)
    case_output_dir.mkdir(parents=True, exist_ok=True)
    expected_json = case_output_dir / f"{audio_path.stem}.json"
    if not (skip_existing and expected_json.exists()):
        command = [
            whisper_bin,
            str(audio_path),
            "--model",
            model,
            "--language",
            language,
            "--device",
            device,
            "--threads",
            threads,
            "--output_dir",
            str(case_output_dir),
            "--output_format",
            "json",
            "--verbose",
            "False",
        ]
        subprocess.run(command, check=True)

    if not expected_json.exists():
        matches = sorted(case_output_dir.glob("*.json"))
        if not matches:
            raise FileNotFoundError(f"Whisper did not write JSON output for case {case.id}.")
        expected_json = matches[0]

    return read_whisper_json(expected_json)


def read_whisper_json(path: Path) -> WhisperTranscript:
    data = json.loads(path.read_text(encoding="utf-8"))
    text = str(data.get("text") or "").strip()
    if not text:
        text = _segments_text(data.get("segments", []))
    if not text:
        raise ValueError(f"Whisper JSON output has no transcript text: {path}")
    segments = data.get("segments", [])
    return WhisperTranscript(
        text=text,
        language=str(data.get("language")) if data.get("language") else None,
        segments=len(segments) if isinstance(segments, list) else 0,
        json_path=path,
    )


def write_cases_jsonl(cases: Iterable[EvaluationCase], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case.model_dump(exclude_none=True), ensure_ascii=False) + "\n")
    return path


def write_summary_json(
    cases: Iterable[EvaluationCase],
    path: Path,
    *,
    source_cases: Path,
    model: str,
) -> Path:
    case_list = list(cases)
    summary = {
        "source_cases": str(source_cases),
        "candidate_model": f"whisper-{model}",
        "total_cases": len(case_list),
        "case_ids": [case.id for case in case_list],
        "languages": sorted(
            {
                str(case.metadata.get("whisper_language"))
                for case in case_list
                if case.metadata.get("whisper_language")
            }
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _segments_text(segments: Any) -> str:
    if not isinstance(segments, list):
        return ""
    return " ".join(
        str(segment.get("text") or "").strip()
        for segment in segments
        if isinstance(segment, dict) and str(segment.get("text") or "").strip()
    ).strip()


def _safe_case_id(case_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", case_id).strip("-") or "case"


if __name__ == "__main__":
    main()
