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
    timeout_seconds: float | None = None
    keep_text_sidecars: bool = False
    dry_run: bool = False


@dataclass(frozen=True)
class LocalTtsFailure:
    case_id: str
    error_type: str
    message: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class LocalTtsBatchResult:
    cases: list[EvaluationCase]
    failures: list[LocalTtsFailure]


def synthesize_cases_with_local_tts(
    cases: Iterable[EvaluationCase],
    *,
    out_dir: Path,
    config: LocalTtsConfig,
) -> list[EvaluationCase]:
    result = synthesize_cases_with_local_tts_batch(cases, out_dir=out_dir, config=config)
    return result.cases


def synthesize_cases_with_local_tts_batch(
    cases: Iterable[EvaluationCase],
    *,
    out_dir: Path,
    config: LocalTtsConfig,
    continue_on_error: bool = False,
) -> LocalTtsBatchResult:
    case_list = list(cases)
    _validate_tts_input_cases(case_list)
    if not config.dry_run and not config.tts_bin.is_file():
        raise FileNotFoundError(
            f"local TTS binary not found at {config.tts_bin}; pass --tts-bin or use --dry-run."
        )
    if config.timeout_seconds is not None and config.timeout_seconds <= 0:
        raise ValueError("local TTS timeout_seconds must be greater than zero.")

    text_dir = out_dir / "text"
    audio_dir = out_dir / "audio"
    text_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    used_stems: set[str] = set()
    synthesized: list[EvaluationCase] = []
    failures: list[LocalTtsFailure] = []
    for case in case_list:
        output_stem = _unique_stem(case.id, used_stems)
        try:
            synthesized.append(
                _synthesize_one_case(
                    case,
                    out_dir=out_dir,
                    text_dir=text_dir,
                    audio_dir=audio_dir,
                    output_stem=output_stem,
                    config=config,
                )
            )
        except Exception as exc:
            if not continue_on_error:
                raise
            failures.append(_failure_for_case(case, exc, config=config))
    return LocalTtsBatchResult(cases=synthesized, failures=failures)


def write_local_tts_cases_jsonl(cases: Iterable[EvaluationCase], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case.model_dump(exclude_none=True), ensure_ascii=False) + "\n")
    return path


def write_local_tts_failures_jsonl(failures: Iterable[LocalTtsFailure], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for failure in failures:
            handle.write(json.dumps(_failure_record(failure), ensure_ascii=False) + "\n")
    return path


def write_local_tts_summary_json(
    cases: Iterable[EvaluationCase],
    path: Path,
    *,
    source_cases: Path,
    model: str,
    synthesis_provider: str = "local_chatterbox",
    failures: Iterable[LocalTtsFailure] = (),
    attempted_source_cases: int | None = None,
) -> Path:
    case_list = list(cases)
    failure_list = list(failures)
    attempted_count = attempted_source_cases
    if attempted_count is None:
        attempted_count = len(case_list) + len(failure_list)
    success_rate = round(len(case_list) / attempted_count, 4) if attempted_count else 0.0
    summary = {
        "source_cases": str(source_cases),
        "candidate_model": model,
        "candidate_generator": synthesis_provider,
        "total_cases": len(case_list),
        "attempted_source_cases": attempted_count,
        "synthesized_case_count": len(case_list),
        "synthesis_success_rate": success_rate,
        "synthesis_failure_count": len(failure_list),
        "case_ids": [case.id for case in case_list],
        "cases_with_audio_path": sum(1 for case in case_list if case.audio_path),
        "total_audio_bytes": _sum_numeric_metadata(case_list, "audio_bytes", integer=True),
        "total_audio_duration_seconds": _sum_numeric_metadata(
            case_list,
            "audio_duration_seconds",
        ),
        "cases_with_audio_duration": _count_present_metadata(
            case_list,
            "audio_duration_seconds",
        ),
        "by_synthesis_provider": _count_metadata(case_list, "synthesis_provider"),
        "by_synthesis_model": _count_metadata(case_list, "synthesis_model"),
        "by_synthesis_voice": _count_metadata(case_list, "synthesis_voice"),
        "by_synthesis_lang_code": _count_metadata(case_list, "synthesis_lang_code"),
        "by_synthesis_audio_format": _count_metadata(case_list, "synthesis_audio_format"),
        "by_tts_slice": _count_metadata(case_list, "tts_slice"),
        "synthesis_failures_by_error_type": _count_failure_field(failure_list, "error_type"),
        "synthesis_failures_by_provider": _count_failure_metadata(
            failure_list,
            "synthesis_provider",
        ),
        "synthesis_failures_by_model": _count_failure_metadata(failure_list, "synthesis_model"),
        "synthesis_failures_by_voice": _count_failure_metadata(failure_list, "synthesis_voice"),
        "synthesis_failures_by_lang_code": _count_failure_metadata(
            failure_list,
            "synthesis_lang_code",
        ),
        "synthesis_failures_by_audio_format": _count_failure_metadata(
            failure_list,
            "synthesis_audio_format",
        ),
        "synthesis_failures_by_tts_slice": _count_failure_metadata(failure_list, "tts_slice"),
        "synthesis_failures_by_source_category": _count_failure_metadata(
            failure_list,
            "source_category",
        ),
        "synthesis_failures_by_sample_kind": _count_failure_metadata(failure_list, "sample_kind"),
        "synthesis_failures_by_language": _count_failure_language(failure_list),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _synthesize_one_case(
    case: EvaluationCase,
    *,
    out_dir: Path,
    text_dir: Path,
    audio_dir: Path,
    output_stem: str,
    config: LocalTtsConfig,
) -> EvaluationCase:
    target_text = (case.reference_text or "").strip()
    text_path = text_dir / f"{output_stem}.txt"
    audio_path = audio_dir / f"{output_stem}.{config.audio_format}"
    if config.keep_text_sidecars or not config.dry_run:
        text_path.write_text(target_text, encoding="utf-8")

    audio_metadata: dict[str, Any] = {}
    if not config.dry_run:
        try:
            audio_path = _run_local_tts(
                config=config,
                text_path=text_path,
                audio_dir=audio_dir,
                output_stem=output_stem,
            )
            if not audio_path.exists() or audio_path.stat().st_size == 0:
                raise FileNotFoundError(f"Expected synthesized audio at {audio_path}")
            audio_metadata = _audio_metadata(audio_path)
        finally:
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
    return synthesized_case


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
    output_pattern = f"{output_stem}*.{config.audio_format}"
    existing_audio = _audio_file_snapshots(audio_dir.glob(output_pattern))
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
            timeout=config.timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(_format_tts_timeout(exc, config.tts_bin)) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(_format_tts_failure(exc, config.tts_bin)) from exc
    output_path = _output_path_from_stdout(completed.stdout, base_dir=audio_dir)
    if output_path is not None:
        _require_output_path_in_audio_dir(output_path, audio_dir)
        _require_output_path_audio_format(output_path, config.audio_format)
        return output_path

    fallback_output = _latest_new_or_changed_audio_path(
        audio_dir.glob(output_pattern),
        previous=existing_audio,
    )
    if fallback_output is not None:
        return fallback_output
    if existing_audio:
        raise ValueError(
            "local TTS command did not report an output audio file and fallback "
            "matching files were unchanged from before synthesis."
        )
    raise ValueError("local TTS command did not report or write an output audio file.")


def _audio_file_snapshots(paths: Iterable[Path]) -> dict[Path, tuple[int, int]]:
    snapshots: dict[Path, tuple[int, int]] = {}
    for path in paths:
        resolved = path.resolve()
        try:
            stat = resolved.stat()
        except FileNotFoundError:
            continue
        snapshots[resolved] = (stat.st_mtime_ns, stat.st_size)
    return snapshots


def _latest_new_or_changed_audio_path(
    paths: Iterable[Path],
    *,
    previous: dict[Path, tuple[int, int]],
) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in paths:
        resolved = path.resolve()
        try:
            stat = resolved.stat()
        except FileNotFoundError:
            continue
        snapshot = (stat.st_mtime_ns, stat.st_size)
        if previous.get(resolved) == snapshot:
            continue
        candidates.append((stat.st_mtime_ns, resolved))
    if not candidates:
        return None
    return sorted(candidates)[-1][1]


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


def _failure_for_case(
    case: EvaluationCase,
    exc: Exception,
    *,
    config: LocalTtsConfig,
) -> LocalTtsFailure:
    target_text = (case.reference_text or "").strip()
    metadata = dict(case.metadata)
    metadata.update(
        {
            "synthesis_provider": config.synthesis_provider,
            "synthesis_model": config.model,
            "synthesis_voice": config.voice,
            "synthesis_lang_code": config.lang_code,
            "synthesis_audio_format": config.audio_format,
            "source_case_id": case.id,
            "reference_text_sha256": _sha256_text(target_text),
        }
    )
    return LocalTtsFailure(
        case_id=case.id,
        error_type=type(exc).__name__,
        message=_compact_error_message(exc),
        metadata=metadata,
    )


def _failure_record(failure: LocalTtsFailure) -> dict[str, Any]:
    return {
        "case_id": failure.case_id,
        "error_type": failure.error_type,
        "message": failure.message,
        "metadata": failure.metadata,
    }


def _compact_error_message(exc: Exception) -> str:
    message = str(exc).strip().replace("\n", " ")
    if len(message) > 1000:
        return f"{message[:1000]}..."
    return message or type(exc).__name__


def _format_tts_timeout(exc: subprocess.TimeoutExpired, tts_bin: Path) -> str:
    details = [
        f"local TTS command timed out after {exc.timeout:g} seconds: {tts_bin.name}",
    ]
    stderr = _tail_nonempty_lines(_output_text(exc.stderr))
    stdout = _tail_nonempty_lines(_output_text(exc.stdout))
    if stderr:
        details.append(f"stderr: {stderr}")
    if stdout:
        details.append(f"stdout: {stdout}")
    if len(details) == 1:
        details.append("no stdout or stderr was captured")
    return "; ".join(details)


def _output_text(value: str | bytes | None) -> str | None:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _tail_nonempty_lines(value: str | None, *, max_lines: int = 4) -> str:
    if not value:
        return ""
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    return " | ".join(lines[-max_lines:])


def _require_output_path_in_audio_dir(output_path: Path, audio_dir: Path) -> None:
    try:
        output_path.resolve().relative_to(audio_dir.resolve())
    except ValueError as exc:
        raise ValueError(
            "local TTS command reported an output audio file outside the synthesis "
            f"audio directory: {output_path}"
        ) from exc


def _require_output_path_audio_format(output_path: Path, audio_format: str) -> None:
    expected_suffix = f".{audio_format.lstrip('.').lower()}"
    if output_path.suffix.lower() != expected_suffix:
        raise ValueError(
            "local TTS command reported an output audio file with extension "
            f"{output_path.suffix or '<none>'}; expected {expected_suffix} for "
            f"--audio-format {audio_format}."
        )


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


def _output_path_from_stdout(stdout: str, *, base_dir: Path | None = None) -> Path | None:
    for line in reversed(stdout.splitlines()):
        candidate = line.strip()
        if not (candidate.startswith("{") and candidate.endswith("}")):
            continue
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        output = _first_output_path_value(data, base_dir=base_dir)
        if output is not None:
            return output
    for data in _json_values_from_stdout_tail(stdout):
        output = _first_output_path_value(data, base_dir=base_dir)
        if output is not None:
            return output
    return None


def _json_values_from_stdout_tail(stdout: str) -> list[object]:
    decoder = json.JSONDecoder()
    values: list[object] = []
    for index in range(len(stdout) - 1, -1, -1):
        if stdout[index] not in "{[":
            continue
        try:
            value, end = decoder.raw_decode(stdout[index:])
        except json.JSONDecodeError:
            continue
        if stdout[index + end :].strip():
            continue
        values.append(value)
    return values


def _first_output_path_value(data: object, *, base_dir: Path | None = None) -> Path | None:
    if isinstance(data, dict):
        for key in ("output", "output_path", "audio_path", "path"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return _normalize_output_path(value, base_dir=base_dir)
        for key in ("audio", "artifact", "artifacts", "result", "outputs", "files"):
            value = data.get(key)
            output = _first_output_path_value(value, base_dir=base_dir)
            if output is not None:
                return output
    elif isinstance(data, list):
        for item in data:
            output = _first_output_path_value(item, base_dir=base_dir)
            if output is not None:
                return output
    return None


def _normalize_output_path(value: str, *, base_dir: Path | None = None) -> Path:
    path = Path(value.strip())
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.resolve()


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


def _count_failure_field(failures: Iterable[LocalTtsFailure], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for failure in failures:
        value = str(getattr(failure, field) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _count_failure_metadata(failures: Iterable[LocalTtsFailure], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for failure in failures:
        value = str(failure.metadata.get(field) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _count_failure_language(failures: Iterable[LocalTtsFailure]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for failure in failures:
        value = str(
            failure.metadata.get("language")
            or failure.metadata.get("synthesis_lang_code")
            or "unknown"
        )
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _count_present_metadata(cases: Iterable[EvaluationCase], field: str) -> int:
    return sum(1 for case in cases if case.metadata.get(field) is not None)


def _sum_numeric_metadata(
    cases: Iterable[EvaluationCase],
    field: str,
    *,
    integer: bool = False,
) -> int | float:
    total = 0.0
    for case in cases:
        value = case.metadata.get(field)
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            total += float(value)
    if integer:
        return int(total)
    return round(total, 3)
