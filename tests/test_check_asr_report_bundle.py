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
    combined_results = tmp_path / "combined-results.jsonl"
    combined_report = tmp_path / "combined-report.html"
    source_results = tmp_path / "source-results.jsonl"
    source_report = tmp_path / "source-report.html"
    for path, text in (
        (combined_results, "{}\n"),
        (combined_report, "<html></html>\n"),
        (source_results, "{}\n"),
        (source_report, "<html>source</html>\n"),
    ):
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
                    "report_path": str(combined_report),
                    "report_sha256": sha256_file(combined_report),
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
    results = tmp_path / "results.jsonl"
    report = tmp_path / "report.html"
    status = tmp_path / "bundle-status.json"
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
                    "report_path": str(report),
                    "report_sha256": sha256_file(report),
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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
