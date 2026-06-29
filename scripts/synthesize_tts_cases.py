"""Synthesize local audio for TTS case manifests.

This helper is intended for private/local development data. It reads TTS cases,
calls the local Chatterbox CLI for each case's ``reference_text``, and writes a
derived case manifest that points at ignored audio artifacts under ``runs/``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import subprocess
import wave
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from open_audio_judge.case_contract import require_audio_and_text
from open_audio_judge.models import EvaluationCase
from open_audio_judge.runner import load_cases


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TTS = Path(
    "/Users/wangyauli/.openclaw/workspace/local-asr-transcriber-mac/.venv/bin/local-tts-speak"
)
DEFAULT_OUT = ROOT / "runs" / "tts-synthesis"
DEFAULT_MODEL = "mlx-community/chatterbox-turbo-6bit"


@dataclass(frozen=True)
class SynthesisValidationIssue:
    case_id: str
    reason: str


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True, help="Input TTS case JSONL/JSON.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Ignored output directory.")
    parser.add_argument("--tts-bin", type=Path, default=DEFAULT_TTS, help="local-tts-speak path.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--lang-code", default="en")
    parser.add_argument("--audio-format", choices=("wav", "flac", "mp3"), default="wav")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Optional metadata-only JSON summary path.",
    )
    parser.add_argument(
        "--discard-text-sidecars",
        action="store_true",
        help="Delete per-case text sidecars after synthesis, or skip them in dry-run mode.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate an existing synthesized manifest instead of invoking TTS.",
    )
    parser.add_argument(
        "--allow-missing-audio",
        action="store_true",
        help="With --validate-only, allow relative audio_path files that do not exist yet.",
    )
    parser.add_argument(
        "--require-text-context-metadata",
        action="store_true",
        help="With --validate-only, require metadata.text_context_fields to match the case text fields.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write manifest without invoking TTS.")
    args = parser.parse_args()

    if args.validate_only:
        issues = validate_synthesized_manifest(
            cases_path=args.cases,
            require_local_audio=not args.allow_missing_audio,
            require_text_context_metadata=args.require_text_context_metadata,
        )
        summary = summarize_validation_issues(issues)
        if args.summary_out is not None:
            write_validation_summary_json(issues, args.summary_out)
        print(json.dumps(summary, sort_keys=True))
        if issues:
            raise SystemExit(1)
        return

    derived = synthesize_cases(
        cases_path=args.cases,
        out_dir=args.out,
        tts_bin=args.tts_bin,
        model=args.model,
        voice=args.voice,
        lang_code=args.lang_code,
        audio_format=args.audio_format,
        limit=args.limit,
        keep_text_sidecars=not args.discard_text_sidecars,
        dry_run=args.dry_run,
    )
    if args.summary_out is not None:
        write_synthesis_summary_json(derived, args.summary_out)
    print(f"Wrote {len(derived)} derived TTS cases to {args.out / 'tts_audio_cases.jsonl'}")


def synthesize_cases(
    *,
    cases_path: Path,
    out_dir: Path,
    tts_bin: Path,
    model: str,
    voice: str,
    lang_code: str,
    audio_format: str,
    limit: int | None = None,
    keep_text_sidecars: bool = True,
    dry_run: bool = False,
) -> list[dict]:
    cases = load_cases(cases_path)
    if limit is not None:
        cases = cases[:limit]
    missing_reference = [case.id for case in cases if not (case.reference_text or "").strip()]
    if missing_reference:
        missing = ", ".join(missing_reference)
        raise ValueError(f"TTS synthesis cases require reference_text; missing for: {missing}")
    duplicate_ids = _duplicate_case_ids(case.id for case in cases)
    if duplicate_ids:
        duplicates = ", ".join(duplicate_ids)
        raise ValueError(f"TTS synthesis cases require unique case ids; duplicates: {duplicates}")
    if not dry_run and not tts_bin.is_file():
        raise FileNotFoundError(
            f"local TTS binary not found at {tts_bin}; pass --tts-bin or use --dry-run."
        )

    text_dir = out_dir / "text"
    audio_dir = out_dir / "audio"
    text_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    derived: list[dict] = []
    used_stems: set[str] = set()
    for case in cases:
        target_text = (case.reference_text or "").strip()

        output_stem = _unique_stem(case.id, used_stems)
        text_path = text_dir / f"{output_stem}.txt"
        audio_path = audio_dir / f"{output_stem}.{audio_format}"
        if keep_text_sidecars or not dry_run:
            text_path.write_text(target_text, encoding="utf-8")

        if not dry_run:
            audio_path = _run_tts(
                tts_bin=tts_bin,
                text_path=text_path,
                audio_dir=audio_dir,
                output_stem=output_stem,
                model=model,
                voice=voice,
                lang_code=lang_code,
                audio_format=audio_format,
            )
            manifest_audio_path = _manifest_audio_path(audio_path, out_dir)
            if not audio_path.exists() or audio_path.stat().st_size == 0:
                raise FileNotFoundError(f"Expected synthesized audio at {audio_path}")
            audio_metadata = _audio_metadata(audio_path)
            if not keep_text_sidecars:
                text_path.unlink(missing_ok=True)
        else:
            manifest_audio_path = _manifest_audio_path(audio_path, out_dir)
            audio_metadata = {}

        derived_case = case.model_dump(exclude_none=True)
        derived_case["id"] = f"{case.id}-local-tts"
        derived_case.pop("audio_url", None)
        derived_case["audio_path"] = manifest_audio_path
        text_context_fields = _text_context_fields(EvaluationCase.model_validate(derived_case))
        metadata = dict(derived_case.get("metadata", {}))
        metadata.update(
            {
                "sample_kind": "local_synthetic_tts",
                "synthesis_provider": "local_chatterbox",
                "synthesis_model": model,
                "synthesis_voice": voice,
                "synthesis_lang_code": lang_code,
                "source_case_id": case.id,
                "reference_text_sha256": _sha256_text(target_text),
                "text_context_fields": text_context_fields,
                "turn_count": metadata.get("turn_count", len(case.turns)),
                "turn_roles": metadata.get("turn_roles", _turn_roles(case)),
                "text_sidecar_written": keep_text_sidecars,
                **audio_metadata,
            }
        )
        derived_case["metadata"] = metadata
        require_audio_and_text(EvaluationCase.model_validate(derived_case))
        derived.append(derived_case)

    manifest_path = out_dir / "tts_audio_cases.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in derived),
        encoding="utf-8",
    )
    return derived


def summarize_synthesized_cases(cases: Iterable[dict[str, Any]]) -> dict[str, Any]:
    case_list = list(cases)
    metadata_list = [
        case.get("metadata", {})
        for case in case_list
        if isinstance(case.get("metadata", {}), dict)
    ]
    durations = [
        float(metadata["audio_duration_seconds"])
        for metadata in metadata_list
        if isinstance(metadata.get("audio_duration_seconds"), int | float)
    ]
    byte_counts = [
        int(metadata["audio_bytes"])
        for metadata in metadata_list
        if isinstance(metadata.get("audio_bytes"), int)
    ]
    return {
        "total_cases": len(case_list),
        "by_slice": _sorted_counts(
            str(metadata.get("tts_slice") or "unknown") for metadata in metadata_list
        ),
        "by_source_category": _sorted_counts(
            str(metadata.get("source_category") or "unknown") for metadata in metadata_list
        ),
        "by_sample_kind": _sorted_counts(
            str(metadata.get("sample_kind") or "unknown") for metadata in metadata_list
        ),
        "by_text_context_fields": _sorted_counts(
            _text_context_field_key(metadata.get("text_context_fields"))
            for metadata in metadata_list
        ),
        "by_turn_role_sequence": _sorted_counts(
            _turn_role_sequence_from_metadata(metadata) for metadata in metadata_list
        ),
        "multi_turn_cases": sum(
            1 for metadata in metadata_list if _metadata_turn_count(metadata) > 1
        ),
        "audio_duration_seconds": _numeric_summary(durations),
        "audio_bytes": _numeric_summary(byte_counts),
        "with_audio_sha256": sum(1 for metadata in metadata_list if metadata.get("audio_sha256")),
    }


def write_synthesis_summary_json(cases: Iterable[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summarize_synthesized_cases(cases), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def validate_synthesized_manifest(
    *,
    cases_path: Path,
    require_local_audio: bool = True,
    require_text_context_metadata: bool = False,
) -> list[SynthesisValidationIssue]:
    issues: list[SynthesisValidationIssue] = []
    for case in load_cases(cases_path):
        try:
            require_audio_and_text(case)
        except ValueError as exc:
            issues.append(SynthesisValidationIssue(case_id=case.id, reason=str(exc)))
            continue

        if not case.audio_path:
            issues.append(
                SynthesisValidationIssue(
                    case_id=case.id,
                    reason="Synthesized TTS manifests require local audio_path.",
                )
            )
            continue

        if require_local_audio and case.audio_path:
            audio_path = Path(case.audio_path)
            if not audio_path.is_absolute():
                audio_path = cases_path.parent / audio_path
            if not audio_path.is_file():
                issues.append(
                    SynthesisValidationIssue(
                        case_id=case.id,
                        reason=f"audio_path file not found: {_display_audio_path(audio_path, cases_path)}",
                    )
                )
            elif audio_path.stat().st_size == 0:
                issues.append(
                    SynthesisValidationIssue(
                        case_id=case.id,
                        reason=f"audio_path file is empty: {_display_audio_path(audio_path, cases_path)}",
                    )
                )
            else:
                issues.extend(_validate_audio_metadata(case, audio_path))
        actual_text_context_fields = _text_context_fields(case)
        metadata_text_context_fields = case.metadata.get("text_context_fields")
        if metadata_text_context_fields is None:
            if require_text_context_metadata:
                issues.append(
                    SynthesisValidationIssue(
                        case_id=case.id,
                        reason="metadata.text_context_fields is missing.",
                    )
                )
        elif _normalize_text_context_fields(metadata_text_context_fields) != actual_text_context_fields:
            expected = "+".join(actual_text_context_fields)
            observed = _text_context_field_key(metadata_text_context_fields)
            issues.append(
                SynthesisValidationIssue(
                    case_id=case.id,
                    reason=(
                        "metadata.text_context_fields does not match case text context: "
                        f"expected {expected}, got {observed}."
                    ),
                )
            )
    return issues


def _validate_audio_metadata(
    case: EvaluationCase,
    audio_path: Path,
) -> list[SynthesisValidationIssue]:
    issues: list[SynthesisValidationIssue] = []
    metadata = case.metadata
    expected_bytes = metadata.get("audio_bytes")
    if expected_bytes is not None and expected_bytes != audio_path.stat().st_size:
        issues.append(
            SynthesisValidationIssue(
                case_id=case.id,
                reason=(
                    "metadata.audio_bytes does not match audio_path file: "
                    f"expected {audio_path.stat().st_size}, got {expected_bytes}."
                ),
            )
        )
    expected_sha256 = metadata.get("audio_sha256")
    if expected_sha256 is not None:
        actual_sha256 = _sha256_file(audio_path)
        if expected_sha256 != actual_sha256:
            issues.append(
                SynthesisValidationIssue(
                    case_id=case.id,
                    reason="metadata.audio_sha256 does not match audio_path file.",
                )
            )
    expected_duration = metadata.get("audio_duration_seconds")
    if expected_duration is not None and audio_path.suffix.lower() == ".wav":
        actual_duration = _wav_duration_seconds(audio_path)
        if actual_duration is not None and expected_duration != actual_duration:
            issues.append(
                SynthesisValidationIssue(
                    case_id=case.id,
                    reason=(
                        "metadata.audio_duration_seconds does not match audio_path file: "
                        f"expected {actual_duration}, got {expected_duration}."
                    ),
                )
            )
    return issues


def summarize_validation_issues(
    issues: Iterable[SynthesisValidationIssue],
) -> dict[str, Any]:
    issue_list = list(issues)
    return {
        "valid": not issue_list,
        "issue_count": len(issue_list),
        "by_reason": _sorted_counts(issue.reason for issue in issue_list),
        "case_ids": [issue.case_id for issue in issue_list],
    }


def write_validation_summary_json(
    issues: Iterable[SynthesisValidationIssue],
    path: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summarize_validation_issues(issues), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _run_tts(
    *,
    tts_bin: Path,
    text_path: Path,
    audio_dir: Path,
    output_stem: str,
    model: str,
    voice: str,
    lang_code: str,
    audio_format: str,
) -> Path:
    completed = subprocess.run(
        [
            str(tts_bin),
            "--text-file",
            str(text_path),
            "--model",
            model,
            "--output-dir",
            str(audio_dir),
            "--file-prefix",
            output_stem,
            "--audio-format",
            audio_format,
            "--voice",
            voice,
            "--lang-code",
            lang_code,
            "--quiet",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    return Path(result["output"]).resolve()


def _safe_stem(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value).strip("-") or "case"


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


def _text_context_fields(case: EvaluationCase) -> list[str]:
    fields: list[str] = []
    if (case.reference_text or "").strip():
        fields.append("reference_text")
    if (case.candidate_text or "").strip():
        fields.append("candidate_text")
    if any(turn.content.strip() for turn in case.turns):
        fields.append("turns")
    return fields


def _text_context_field_key(value: Any) -> str:
    if not isinstance(value, list):
        return "unknown"
    fields = sorted(str(item) for item in value if str(item).strip())
    return "+".join(fields) if fields else "none"


def _normalize_text_context_fields(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    allowed_fields = {"reference_text", "candidate_text", "turns"}
    fields = [str(item) for item in value if str(item).strip() in allowed_fields]
    if not fields:
        return []
    ordered_fields = ["reference_text", "candidate_text", "turns"]
    return [field for field in ordered_fields if field in fields]


def _turn_roles(case: EvaluationCase) -> list[str]:
    return [turn.role for turn in case.turns if turn.role.strip()]


def _turn_role_sequence_from_metadata(metadata: dict[str, Any]) -> str:
    roles = metadata.get("turn_roles")
    if isinstance(roles, list):
        normalized_roles = [str(role).strip() for role in roles if str(role).strip()]
    else:
        normalized_roles = []
    return "+".join(normalized_roles) if normalized_roles else "none"


def _metadata_turn_count(metadata: dict[str, Any]) -> int:
    turn_count = metadata.get("turn_count")
    if isinstance(turn_count, int):
        return turn_count
    if isinstance(turn_count, float):
        return int(turn_count)
    roles = metadata.get("turn_roles")
    if isinstance(roles, list):
        return len([role for role in roles if str(role).strip()])
    return 0


def _sorted_counts(values: Iterable[str]) -> dict[str, int]:
    counts = Counter(values)
    return {key: counts[key] for key in sorted(counts)}


def _numeric_summary(values: Iterable[int | float]) -> dict[str, int | float | None]:
    value_list = list(values)
    if not value_list:
        return {"min": None, "max": None, "average": None, "total": None}
    total = sum(value_list)
    average = total / len(value_list)
    return {
        "min": min(value_list),
        "max": max(value_list),
        "average": round(average, 3),
        "total": round(total, 3),
    }


def _manifest_audio_path(audio_path: Path, out_dir: Path) -> str:
    try:
        return str(audio_path.resolve().relative_to(out_dir.resolve()))
    except ValueError as exc:
        raise ValueError(
            f"Synthesized audio output must be under the output directory: {audio_path}"
        ) from exc


def _display_audio_path(audio_path: Path, cases_path: Path) -> str:
    try:
        return str(audio_path.resolve().relative_to(cases_path.parent.resolve()))
    except ValueError:
        return str(audio_path)


def _audio_metadata(audio_path: Path) -> dict[str, int | float | str]:
    metadata: dict[str, int | float | str] = {
        "audio_sha256": _sha256_file(audio_path),
        "audio_bytes": audio_path.stat().st_size,
    }
    if audio_path.suffix.lower() == ".wav":
        duration = _wav_duration_seconds(audio_path)
        if duration is not None:
            metadata["audio_duration_seconds"] = duration
    return metadata


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wav_duration_seconds(path: Path) -> float | None:
    with contextlib.suppress(wave.Error, EOFError):
        with wave.open(str(path), "rb") as handle:
            frame_rate = handle.getframerate()
            if frame_rate > 0:
                return round(handle.getnframes() / frame_rate, 3)
    return None


if __name__ == "__main__":
    main()
