from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "docs" / "asr-leaderboard-report-bundle.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the generated ASR leaderboard report bundle against local artifacts.",
    )
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    args = parser.parse_args()

    summary = check_asr_report_bundle(args.bundle)
    print(
        "Validated ASR report bundle: "
        f"{summary['total_results']} results, {summary['source_report_count']} source reports."
    )


def check_asr_report_bundle(bundle_path: Path) -> dict[str, Any]:
    bundle = _read_json(bundle_path)
    issues: list[str] = []

    combined_report = _dict_value(bundle, "combined_report", issues)
    results_path = _required_repo_path(combined_report, "results_path", issues)
    report_path = _required_repo_path(combined_report, "report_path", issues)
    _check_file_digest(
        results_path,
        _string_value(combined_report, "results_sha256", issues),
        "combined_report.results_path",
        issues,
    )
    _check_file_digest(
        report_path,
        _string_value(combined_report, "report_sha256", issues),
        "combined_report.report_path",
        issues,
    )

    source_reports = bundle.get("source_reports")
    if not isinstance(source_reports, list) or not source_reports:
        issues.append("source_reports must be a non-empty list.")
        source_reports = []
    for index, source_report in enumerate(source_reports):
        if not isinstance(source_report, dict):
            issues.append(f"source_reports[{index}] must be an object.")
            continue
        label = f"source_reports[{index}]"
        source_results_path = _required_repo_path(source_report, "results_path", issues)
        _check_file_digest(
            source_results_path,
            _string_value(source_report, "results_sha256", issues),
            f"{label}.results_path",
            issues,
        )
        if source_report.get("report_exists") is True:
            source_report_path = _required_repo_path(source_report, "report_path", issues)
            _check_file_digest(
                source_report_path,
                _string_value(source_report, "report_sha256", issues),
                f"{label}.report_path",
                issues,
            )

    coverage = _dict_value(bundle, "coverage", issues)
    total_results = _int_value(coverage, "total_results", issues)
    model_count = _int_value(coverage, "model_count", issues)
    category_count = _int_value(coverage, "category_count", issues)

    status_path = _status_path_from_bundle(bundle, issues)
    if status_path is not None and status_path.exists():
        _check_status_consistency(
            status_path,
            total_results=total_results,
            model_count=model_count,
            category_count=category_count,
            source_report_count=len(source_reports),
            issues=issues,
        )
    elif status_path is not None:
        issues.append(f"refresh_provenance.bundle_status_path is missing: {_repo_relative(status_path)}")

    if issues:
        raise ValueError("ASR report bundle validation failed: " + " ".join(issues))

    return {
        "bundle_path": _repo_relative(bundle_path),
        "total_results": total_results,
        "model_count": model_count,
        "category_count": category_count,
        "source_report_count": len(source_reports),
    }


def _check_status_consistency(
    status_path: Path,
    *,
    total_results: int,
    model_count: int,
    category_count: int,
    source_report_count: int,
    issues: list[str],
) -> None:
    status = _read_json(status_path)
    expected = {
        "status": "complete",
        "total_results": total_results,
        "model_count": model_count,
        "category_count": category_count,
        "missing_artifact_count": 0,
    }
    for key, value in expected.items():
        if status.get(key) != value:
            issues.append(f"{_repo_relative(status_path)} {key}={status.get(key)!r}, expected {value!r}.")
    if status.get("source_report_count") not in (None, source_report_count):
        issues.append(
            f"{_repo_relative(status_path)} source_report_count={status.get('source_report_count')!r}, "
            f"expected {source_report_count!r}."
        )


def _status_path_from_bundle(bundle: dict[str, Any], issues: list[str]) -> Path | None:
    provenance = _dict_value(bundle, "refresh_provenance", issues)
    raw_path = provenance.get("bundle_status_path")
    if not isinstance(raw_path, str) or not raw_path:
        issues.append("refresh_provenance.bundle_status_path is required.")
        return None
    return _resolve_repo_path(raw_path)


def _check_file_digest(
    path: Path | None,
    expected_sha256: str | None,
    label: str,
    issues: list[str],
) -> None:
    if path is None:
        return
    if not path.exists():
        issues.append(f"{label} is missing: {_repo_relative(path)}.")
        return
    if expected_sha256 is None:
        return
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        issues.append(
            f"{label} sha256 mismatch for {_repo_relative(path)}: "
            f"expected {expected_sha256}, got {actual_sha256}."
        )


def _required_repo_path(data: dict[str, Any], key: str, issues: list[str]) -> Path | None:
    raw_path = data.get(key)
    if not isinstance(raw_path, str) or not raw_path:
        issues.append(f"{key} is required.")
        return None
    return _resolve_repo_path(raw_path)


def _dict_value(data: dict[str, Any], key: str, issues: list[str]) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        issues.append(f"{key} must be an object.")
        return {}
    return value


def _int_value(data: dict[str, Any], key: str, issues: list[str]) -> int:
    value = data.get(key)
    if not isinstance(value, int):
        issues.append(f"{key} must be an integer.")
        return 0
    return value


def _string_value(data: dict[str, Any], key: str, issues: list[str]) -> str | None:
    value = data.get(key)
    if value is None:
        issues.append(f"{key} is required.")
        return None
    if not isinstance(value, str) or not value:
        issues.append(f"{key} must be a non-empty string.")
        return None
    return value


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{_repo_relative(path)} must contain a JSON object.")
    return data


def _resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return ROOT / path


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
