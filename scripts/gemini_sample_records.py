"""Maintain committed records for Gemini sample smoke runs.

Sample records keep hosted-provider smoke tests from being repeated on every
iteration. A sample should be rerun only when its manifest data, prompt version,
provider, or model changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from open_audio_judge.runner import load_cases


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORDS = ROOT / "examples" / "gemini_sample_records.jsonl"
DEFAULT_SAMPLE_MANIFESTS = (
    ROOT / "examples" / "asr_open_samples.jsonl",
    ROOT / "examples" / "tts_open_samples.jsonl",
)
PROMPTS_BY_TASK = {
    "asr_error": ROOT / "prompts" / "asr_error_judge.yaml",
    "tts_naturalness": ROOT / "prompts" / "tts_naturalness.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    check_parser.add_argument("--provider", default="gemini")
    check_parser.add_argument("--model", default="gemini-3.5-flash")

    update_parser = subparsers.add_parser("update")
    update_parser.add_argument("--results", type=Path, action="append", required=True)
    update_parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    update_parser.add_argument("--provider", default="gemini")
    update_parser.add_argument("--model", default="gemini-3.5-flash")

    args = parser.parse_args()
    if args.command == "check":
        missing = missing_records(args.records, args.provider, args.model)
        if missing:
            for case_id in missing:
                print(f"missing_or_changed: {case_id}")
            raise SystemExit(1)
        print("All Gemini sample records are current.")
    elif args.command == "update":
        update_records(args.results, args.records, args.provider, args.model)


def missing_records(records_path: Path, provider: str, model: str) -> list[str]:
    expected = expected_fingerprints(provider, model)
    records = {
        record["base_case_id"]: record
        for record in read_jsonl(records_path)
        if record.get("provider") == provider and record.get("model") == model
    }

    missing: list[str] = []
    for case_id, fingerprint in expected.items():
        record = records.get(case_id)
        if not record or record.get("sample_fingerprint") != fingerprint or record.get("status") != "ok":
            missing.append(case_id)
    return missing


def update_records(
    result_paths: list[Path],
    records_path: Path,
    provider: str,
    model: str,
) -> None:
    expected = expected_fingerprints(provider, model)
    base_cases = sample_cases_by_id()
    existing = [
        record
        for record in read_jsonl(records_path)
        if not (
            record.get("provider") == provider
            and record.get("model") == model
            and record.get("base_case_id") in expected
        )
    ]

    new_records: list[dict[str, Any]] = []
    for result_path in result_paths:
        for result in read_jsonl(result_path):
            base_case_id = _base_case_id(result["case_id"])
            if base_case_id not in expected or result.get("status") != "ok":
                continue
            base_case = base_cases[base_case_id]
            new_records.append(
                {
                    "case_id": result["case_id"],
                    "base_case_id": base_case_id,
                    "task": result["task"],
                    "judge_id": result["judge_id"],
                    "judge_version": result["judge_version"],
                    "provider": provider,
                    "model": model,
                    "sample_fingerprint": expected[base_case_id],
                    "status": result["status"],
                    "overall_score": result["overall_score"],
                    "label": result["label"],
                    "reason": result["reason"],
                    "judge_transcript": result.get("judge_transcript"),
                    "meaning_preservation": result.get("meaning_preservation"),
                    "error_categories": result.get("error_categories", []),
                    "source_page": base_case.metadata.get("source_page"),
                    "sample_kind": base_case.metadata.get("sample_kind"),
                    "result_created_at": result["created_at"],
                    "recorded_from": _display_path(result_path),
                }
            )

    records_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in sorted(existing + new_records, key=lambda row: row["case_id"])
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(new_records)} Gemini sample records to {records_path}")


def expected_fingerprints(provider: str, model: str) -> dict[str, str]:
    prompts = {task: _read_prompt(path) for task, path in PROMPTS_BY_TASK.items()}
    fingerprints: dict[str, str] = {}
    for case_id, case in sample_cases_by_id().items():
        prompt = prompts[case.task]
        payload = {
            "case": case.model_dump(),
            "judge_id": prompt["id"],
            "judge_version": prompt["version"],
            "provider": provider,
            "model": model,
            "schema": "open-audio-judge.gemini-sample-record.v1",
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        fingerprints[case_id] = hashlib.sha256(encoded).hexdigest()
    return fingerprints


def sample_cases_by_id() -> dict[str, Any]:
    cases = {}
    for manifest in DEFAULT_SAMPLE_MANIFESTS:
        for case in load_cases(manifest):
            cases[case.id] = case
    return cases


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_prompt(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _base_case_id(case_id: str) -> str:
    return case_id.removesuffix("-wav")


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
