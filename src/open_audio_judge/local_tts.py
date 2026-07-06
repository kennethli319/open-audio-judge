from __future__ import annotations

import hashlib
import json
import subprocess
import wave
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from open_audio_judge.case_contract import require_audio_and_text
from open_audio_judge.models import EvaluationCase


DEFAULT_CHATTERBOX_BIN = Path(
    "/Users/wangyauli/.openclaw/workspace/local-asr-transcriber-mac/.venv/bin/local-tts-speak"
)
DEFAULT_CHATTERBOX_MODEL = "mlx-community/chatterbox-turbo-6bit"


@dataclass(frozen=True)
class LocalTtsConfig:
    tts_bin: Path = DEFAULT_CHATTERBOX_BIN
    model: str = DEFAULT_CHATTERBOX_MODEL
    synthesis_provider: str = "local_chatterbox"
    voice: str = "af_heart"
    lang_code: str = "en"
    audio_format: str = "wav"
    keep_text_sidecars: bool = False
    dry_run: bool = False


def synthesize_cases_with_local_tts(
    cases: Iterable[EvaluationCase],
    *,
    out_dir: Path,
    config: LocalTtsConfig,
) -> list[EvaluationCase]:
    case_list = list(cases)
    _validate_tts_input_cases(case_list)
    if not config.dry_run and not config.tts_bin.is_file():
        raise FileNotFoundError(
            f"local TTS binary not found at {config.tts_bin}; pass --tts-bin or use --dry-run."
        )

    text_dir = out_dir / "text"
    audio_dir = out_dir / "audio"
    text_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    used_stems: set[str] = set()
    synthesized: list[EvaluationCase] = []
    for case in case_list:
        target_text = (case.reference_text or "").strip()
        output_stem = _unique_stem(case.id, used_stems)
        text_path = text_dir / f"{output_stem}.txt"
        audio_path = audio_dir / f"{output_stem}.{config.audio_format}"
        if config.keep_text_sidecars or not config.dry_run:
            text_path.write_text(target_text, encoding="utf-8")

        audio_metadata: dict[str, Any] = {}
        if not config.dry_run:
            audio_path = _run_local_tts(
                config=config,
                text_path=text_path,
                audio_dir=audio_dir,
                output_stem=output_stem,
            )
            if not audio_path.exists() or audio_path.stat().st_size == 0:
                raise FileNotFoundError(f"Expected synthesized audio at {audio_path}")
            audio_metadata = _audio_metadata(audio_path)
            if not config.keep_text_sidecars:
                text_path.unlink(missing_ok=True)

        metadata = dict(case.metadata)
        metadata.update(
            {
                "sample_kind": "local_synthetic_tts",
                "synthesis_provider": config.synthesis_provider,
                "synthesis_model": config.model,
                "synthesis_voice": config.voice,
                "synthesis_lang_code": config.lang_code,
                "synthesis_audio_format": config.audio_format,
                "source_case_id": case.id,
                "reference_text_sha256": _sha256_text(target_text),
                "requires_synthesis": False,
                "text_sidecar_written": config.keep_text_sidecars,
                **audio_metadata,
            }
        )
        if config.keep_text_sidecars:
            metadata["text_sidecar_path"] = _relative_path(text_path, out_dir)

        synthesized_case = case.model_copy(
            update={
                "id": f"{case.id}-local-tts",
                "audio_path": _relative_path(audio_path, out_dir),
                "audio_url": None,
                "metadata": metadata,
            }
        )
        require_audio_and_text(synthesized_case)
        synthesized.append(synthesized_case)
    return synthesized


def write_local_tts_cases_jsonl(cases: Iterable[EvaluationCase], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case.model_dump(exclude_none=True), ensure_ascii=False) + "\n")
    return path


def write_local_tts_summary_json(
    cases: Iterable[EvaluationCase],
    path: Path,
    *,
    source_cases: Path,
    model: str,
    synthesis_provider: str = "local_chatterbox",
) -> Path:
    case_list = list(cases)
    summary = {
        "source_cases": str(source_cases),
        "candidate_model": model,
        "candidate_generator": synthesis_provider,
        "total_cases": len(case_list),
        "case_ids": [case.id for case in case_list],
        "cases_with_audio_path": sum(1 for case in case_list if case.audio_path),
        "by_synthesis_provider": _count_metadata(case_list, "synthesis_provider"),
        "by_synthesis_voice": _count_metadata(case_list, "synthesis_voice"),
        "by_synthesis_audio_format": _count_metadata(case_list, "synthesis_audio_format"),
        "by_tts_slice": _count_metadata(case_list, "tts_slice"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _validate_tts_input_cases(cases: list[EvaluationCase]) -> None:
    missing_reference = [case.id for case in cases if not (case.reference_text or "").strip()]
    if missing_reference:
        raise ValueError(
            "TTS AutoJudge cases require reference_text; missing for: "
            + ", ".join(missing_reference)
        )
    duplicate_ids = _duplicate_case_ids(case.id for case in cases)
    if duplicate_ids:
        raise ValueError(
            "TTS AutoJudge cases require unique case ids; duplicates: "
            + ", ".join(duplicate_ids)
        )


def _run_local_tts(
    *,
    config: LocalTtsConfig,
    text_path: Path,
    audio_dir: Path,
    output_stem: str,
) -> Path:
    command = [
        str(config.tts_bin),
        "--text-file",
        str(text_path),
        "--model",
        config.model,
        "--output-dir",
        str(audio_dir),
        "--file-prefix",
        output_stem,
        "--audio-format",
        config.audio_format,
        "--voice",
        config.voice,
        "--lang-code",
        config.lang_code,
        "--quiet",
        "--json",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(_format_tts_failure(exc, config.tts_bin)) from exc
    output_path = _output_path_from_stdout(completed.stdout)
    if output_path is not None:
        return output_path

    matches = sorted(audio_dir.glob(f"{output_stem}*.{config.audio_format}"))
    if matches:
        return matches[-1].resolve()
    raise ValueError("local TTS command did not report or write an output audio file.")


def _format_tts_failure(exc: subprocess.CalledProcessError, tts_bin: Path) -> str:
    details = [
        f"local TTS command failed with exit code {exc.returncode}: {tts_bin.name}",
    ]
    stderr = _tail_nonempty_lines(exc.stderr)
    stdout = _tail_nonempty_lines(exc.stdout)
    if stderr:
        details.append(f"stderr: {stderr}")
    if stdout:
        details.append(f"stdout: {stdout}")
    if len(details) == 1:
        details.append("no stdout or stderr was captured")
    return "; ".join(details)


def _tail_nonempty_lines(value: str | None, *, max_lines: int = 4) -> str:
    if not value:
        return ""
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    return " | ".join(lines[-max_lines:])


def _audio_metadata(audio_path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "audio_bytes": audio_path.stat().st_size,
        "audio_sha256": _sha256_file(audio_path),
    }
    if audio_path.suffix.lower() == ".wav":
        duration = _wav_duration_seconds(audio_path)
        if duration is not None:
            metadata["audio_duration_seconds"] = duration
    return metadata


def _output_path_from_stdout(stdout: str) -> Path | None:
    for line in reversed(stdout.splitlines()):
        candidate = line.strip()
        if not (candidate.startswith("{") and candidate.endswith("}")):
            continue
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        output = data.get("output")
        if isinstance(output, str) and output.strip():
            return Path(output).resolve()
    return None


def _wav_duration_seconds(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
    except (wave.Error, EOFError):
        return None
    if rate <= 0:
        return None
    return round(frames / rate, 3)


def _relative_path(path: Path, base_dir: Path) -> str:
    return str(path.resolve().relative_to(base_dir.resolve()))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_stem(value: str) -> str:
    stem = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value)
    return stem.strip("-") or "case"


def _unique_stem(value: str, used_stems: set[str]) -> str:
    base_stem = _safe_stem(value)
    stem = base_stem
    if stem in used_stems:
        stem = f"{base_stem}-{_sha256_text(value)[:8]}"
    while stem in used_stems:
        stem = f"{base_stem}-{_sha256_text(stem)[:8]}"
    used_stems.add(stem)
    return stem


def _duplicate_case_ids(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def _count_metadata(cases: Iterable[EvaluationCase], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        value = str(case.metadata.get(field) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))
