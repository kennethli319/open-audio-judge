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
from collections.abc import Iterable
from pathlib import Path

from open_audio_judge.case_contract import require_audio_and_text
from open_audio_judge.models import EvaluationCase
from open_audio_judge.runner import load_cases


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TTS = Path(
    "/Users/wangyauli/.openclaw/workspace/local-asr-transcriber-mac/.venv/bin/local-tts-speak"
)
DEFAULT_OUT = ROOT / "runs" / "tts-synthesis"
DEFAULT_MODEL = "mlx-community/chatterbox-turbo-6bit"


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
    parser.add_argument("--dry-run", action="store_true", help="Write manifest without invoking TTS.")
    args = parser.parse_args()

    derived = synthesize_cases(
        cases_path=args.cases,
        out_dir=args.out,
        tts_bin=args.tts_bin,
        model=args.model,
        voice=args.voice,
        lang_code=args.lang_code,
        audio_format=args.audio_format,
        limit=args.limit,
        dry_run=args.dry_run,
    )
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
        else:
            manifest_audio_path = _manifest_audio_path(audio_path, out_dir)
            audio_metadata = {}

        derived_case = case.model_dump(exclude_none=True)
        derived_case["id"] = f"{case.id}-local-tts"
        derived_case.pop("audio_url", None)
        derived_case["audio_path"] = manifest_audio_path
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


def _manifest_audio_path(audio_path: Path, out_dir: Path) -> str:
    try:
        return str(audio_path.resolve().relative_to(out_dir.resolve()))
    except ValueError as exc:
        raise ValueError(
            f"Synthesized audio output must be under the output directory: {audio_path}"
        ) from exc


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
