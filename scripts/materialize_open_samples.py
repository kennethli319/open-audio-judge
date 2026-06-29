"""Download linked open audio samples and derive local WAV fixtures.

The repository keeps only links and JSONL manifests in Git. This helper creates
ignored local artifacts under ``runs/`` for development and live provider smoke
tests.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFESTS = (
    ROOT / "examples" / "asr_open_samples.jsonl",
    ROOT / "examples" / "tts_open_samples.jsonl",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "runs" / "open-audio-samples",
        help="Output directory for downloaded sources, WAV files, and derived cases.",
    )
    args = parser.parse_args()

    out_dir = args.out
    source_dir = out_dir / "source"
    wav_dir = out_dir / "wav"
    source_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    samples = _unique_samples(DEFAULT_MANIFESTS)
    for sample in samples:
        sample_id = _sample_slug(sample["id"])
        source_path = source_dir / f"{sample_id}{_suffix(sample['audio_url'])}"
        wav_path = wav_dir / f"{sample_id}.wav"
        _download(sample["audio_url"], source_path)
        _ffmpeg_to_wav(source_path, wav_path)

    _write_wav_manifest(
        source_manifest=ROOT / "examples" / "asr_open_samples.jsonl",
        output_manifest=out_dir / "asr_wav_cases.jsonl",
        wav_dir=wav_dir,
    )
    _write_wav_manifest(
        source_manifest=ROOT / "examples" / "tts_open_samples.jsonl",
        output_manifest=out_dir / "tts_wav_cases.jsonl",
        wav_dir=wav_dir,
    )

    print(f"Wrote {len(samples)} WAV files to {wav_dir}")
    print(f"Wrote derived case manifests to {out_dir}")


def _unique_samples(manifests: tuple[Path, ...]) -> list[dict]:
    seen: set[str] = set()
    samples: list[dict] = []
    for manifest in manifests:
        for case in _read_jsonl(manifest):
            url = case.get("audio_url")
            if not url or url in seen:
                continue
            seen.add(url)
            samples.append(case)
    return samples


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    with urllib.request.urlopen(url, timeout=30) as response:
        path.write_bytes(response.read())


def _ffmpeg_to_wav(source_path: Path, wav_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(wav_path),
        ],
        check=True,
    )


def _write_wav_manifest(source_manifest: Path, output_manifest: Path, wav_dir: Path) -> None:
    cases = []
    for case in _read_jsonl(source_manifest):
        wav_name = f"{_sample_slug(case['id'])}.wav"
        derived = dict(case)
        derived["id"] = f"{case['id']}-wav"
        derived.pop("audio_url", None)
        derived["audio_path"] = str(Path("wav") / wav_name)
        derived.setdefault("metadata", {})["derived_format"] = "wav_pcm_16khz_mono"
        cases.append(derived)

    output_manifest.write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases)
    )

    missing = [case["audio_path"] for case in cases if not (wav_dir.parent / case["audio_path"]).exists()]
    if missing:
        raise FileNotFoundError(f"Missing derived WAV files: {missing}")


def _sample_slug(case_id: str) -> str:
    slug = case_id
    for prefix in ("asr-open-", "tts-open-"):
        if slug.startswith(prefix):
            slug = slug.removeprefix(prefix)
    return slug


def _suffix(url: str) -> str:
    name = urllib.request.urlparse(url).path.rsplit("/", 1)[-1].lower()
    if "." not in name:
        return ".audio"
    return f".{name.rsplit('.', 1)[-1]}"


if __name__ == "__main__":
    main()
