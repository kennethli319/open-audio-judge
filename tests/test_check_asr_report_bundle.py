import hashlib
import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path("scripts/check_asr_report_bundle.py")


def load_script_module():
    spec = importlib.util.spec_from_file_location("check_asr_report_bundle", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_check_asr_report_bundle_validates_digests_and_status(tmp_path: Path) -> None:
    module = load_script_module()
    module.ROOT = tmp_path
    combined_results = tmp_path / "runs" / "combined" / "results.jsonl"
    combined_report = tmp_path / "runs" / "combined" / "report.html"
    source_results = tmp_path / "runs" / "source" / "results.jsonl"
    source_report = tmp_path / "runs" / "source" / "report.html"
    for path, text in (
        (combined_results, "{}\n"),
        (combined_report, "<html></html>\n"),
        (source_results, "{}\n"),
        (source_report, "<html>source</html>\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    status = tmp_path / "bundle-status.json"
    status.write_text(
        json.dumps(
            {
                "status": "complete",
                "total_results": 1,
                "model_count": 1,
                "category_count": 1,
                "missing_artifact_count": 0,
                "source_report_count": 1,
            }
        ),
        encoding="utf-8",
    )
    bundle = tmp_path / "bundle.json"
    bundle.write_text(
        json.dumps(
            {
                "combined_report": {
                    "results_path": str(combined_results),
                    "results_sha256": sha256_file(combined_results),
                    "hosted_results_url": (
                        "https://kennethli319.github.io/open-audio-judge/"
                        + relative_hosted_path(combined_results)
                    ),
                    "report_path": str(combined_report),
                    "report_sha256": sha256_file(combined_report),
                    "hosted_report_url": (
                        "https://kennethli319.github.io/open-audio-judge/"
                        + relative_hosted_path(combined_report)
                    ),
                },
                "coverage": {
                    "total_results": 1,
                    "model_count": 1,
                    "category_count": 1,
                },
                "refresh_provenance": {"bundle_status_path": str(status)},
                "source_reports": [
                    {
                        "results_path": str(source_results),
                        "results_sha256": sha256_file(source_results),
                        "report_exists": True,
                        "report_path": str(source_report),
                        "report_sha256": sha256_file(source_report),
                        "hosted_report_path": relative_hosted_path(source_report),
                        "hosted_report_url": (
                            "https://kennethli319.github.io/open-audio-judge/"
                            + relative_hosted_path(source_report)
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = module.check_asr_report_bundle(bundle)

    assert summary["total_results"] == 1
    assert summary["source_report_count"] == 1


def test_check_asr_report_bundle_rejects_stale_digest(tmp_path: Path) -> None:
    module = load_script_module()
    module.ROOT = tmp_path
    results = tmp_path / "runs" / "combined" / "results.jsonl"
    report = tmp_path / "runs" / "combined" / "report.html"
    status = tmp_path / "bundle-status.json"
    results.parent.mkdir(parents=True, exist_ok=True)
    results.write_text("{}\n", encoding="utf-8")
    report.write_text("<html></html>\n", encoding="utf-8")
    status.write_text(
        json.dumps(
            {
                "status": "complete",
                "total_results": 1,
                "model_count": 1,
                "category_count": 1,
                "missing_artifact_count": 0,
            }
        ),
        encoding="utf-8",
    )
    bundle = tmp_path / "bundle.json"
    bundle.write_text(
        json.dumps(
            {
                "combined_report": {
                    "results_path": str(results),
                    "results_sha256": "0" * 64,
                    "hosted_results_url": (
                        "https://kennethli319.github.io/open-audio-judge/"
                        + relative_hosted_path(results)
                    ),
                    "report_path": str(report),
                    "report_sha256": sha256_file(report),
                    "hosted_report_url": (
                        "https://kennethli319.github.io/open-audio-judge/"
                        + relative_hosted_path(report)
                    ),
                },
                "coverage": {
                    "total_results": 1,
                    "model_count": 1,
                    "category_count": 1,
                },
                "refresh_provenance": {"bundle_status_path": str(status)},
                "source_reports": [
                    {
                        "results_path": str(results),
                        "results_sha256": sha256_file(results),
                        "report_exists": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    try:
        module.check_asr_report_bundle(bundle)
    except ValueError as exc:
        assert "sha256 mismatch" in str(exc)
    else:
        raise AssertionError("Expected stale digest to fail validation.")


def test_check_asr_report_bundle_rejects_stale_hosted_report_path(tmp_path: Path) -> None:
    module = load_script_module()
    module.ROOT = tmp_path
    results = tmp_path / "runs" / "asr-leaderboard" / "combined" / "results.jsonl"
    report = tmp_path / "runs" / "asr-leaderboard" / "combined" / "report.html"
    source_results = tmp_path / "runs" / "asr-leaderboard" / "model" / "judge-report" / "results.jsonl"
    source_report = tmp_path / "runs" / "asr-leaderboard" / "model" / "judge-report" / "report.html"
    status = tmp_path / "docs" / "bundle-status.json"
    for path, text in (
        (results, "{}\n"),
        (report, "<html></html>\n"),
        (source_results, "{}\n"),
        (source_report, "<html>source</html>\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    status.parent.mkdir(parents=True, exist_ok=True)
    status.write_text(
        json.dumps(
            {
                "status": "complete",
                "total_results": 1,
                "model_count": 1,
                "category_count": 1,
                "missing_artifact_count": 0,
                "source_report_count": 1,
            }
        ),
        encoding="utf-8",
    )
    bundle = tmp_path / "docs" / "bundle.json"
    bundle.write_text(
        json.dumps(
            {
                "combined_report": {
                    "results_path": str(results),
                    "results_sha256": sha256_file(results),
                    "hosted_results_url": (
                        "https://kennethli319.github.io/open-audio-judge/"
                        + relative_hosted_path(results)
                    ),
                    "report_path": str(report),
                    "report_sha256": sha256_file(report),
                    "hosted_report_url": (
                        "https://kennethli319.github.io/open-audio-judge/"
                        + relative_hosted_path(report)
                    ),
                },
                "coverage": {
                    "total_results": 1,
                    "model_count": 1,
                    "category_count": 1,
                },
                "refresh_provenance": {"bundle_status_path": str(status)},
                "source_reports": [
                    {
                        "results_path": str(source_results),
                        "results_sha256": sha256_file(source_results),
                        "report_exists": True,
                        "report_path": str(source_report),
                        "report_sha256": sha256_file(source_report),
                        "hosted_report_path": "asr-leaderboard/stale/report.html",
                        "hosted_report_url": (
                            "https://kennethli319.github.io/open-audio-judge/"
                            "asr-leaderboard/stale/report.html"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    try:
        module.check_asr_report_bundle(bundle)
    except ValueError as exc:
        assert "hosted_report_path" in str(exc)
        assert "hosted_report_url" in str(exc)
    else:
        raise AssertionError("Expected stale hosted report fields to fail validation.")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_hosted_path(path: Path) -> str:
    parts = path.parts
    if "docs" in parts:
        return Path(*parts[parts.index("docs") + 1 :]).as_posix()
    if "runs" in parts:
        return Path(*parts[parts.index("runs") + 1 :]).as_posix()
    raise AssertionError(f"Expected docs/ or runs/ in {path}")
