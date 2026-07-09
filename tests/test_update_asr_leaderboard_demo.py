import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = Path("scripts/update_asr_leaderboard_demo.py")
REFRESH_SCRIPT = Path("scripts/refresh_asr_leaderboard_artifacts.py")


def load_script_module():
    spec = importlib.util.spec_from_file_location("update_asr_leaderboard_demo", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_refresh_module():
    spec = importlib.util.spec_from_file_location("refresh_asr_leaderboard_artifacts", REFRESH_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def result_record(
    *,
    case_id: str,
    model: str,
    category: str,
    score: int,
    label: str,
    judge_sample_count: int = 3,
) -> dict:
    return {
        "case_id": case_id,
        "task": "asr_error",
        "judge_id": "asr_error",
        "judge_version": "0.2.0",
        "provider": "gemini",
        "overall_score": score,
        "reason": "Synthetic unit test result.",
        "error_categories": ["no_error"],
        "label": label,
        "status": "ok",
        "metadata": {
            "candidate_model": model,
            "eval_category": category,
            "judge_sample_count": judge_sample_count,
        },
    }


def test_render_generated_sections_summarizes_verified_asr_results(tmp_path: Path) -> None:
    module = load_script_module()
    results_path = tmp_path / "results.jsonl"
    records = [
        result_record(
            case_id="asr-a-model-a",
            model="mlx-community/model-a",
            category="transcription_accuracy_wer",
            score=100,
            label="accurate",
        ),
        result_record(
            case_id="asr-b-model-a",
            model="mlx-community/model-a",
            category="numeric_unit_integrity",
            score=80,
            label="accurate",
        ),
        result_record(
            case_id="asr-a-model-b",
            model="mlx-community/model-b",
            category="transcription_accuracy_wer",
            score=60,
            label="needs_review",
        ),
        result_record(
            case_id="asr-b-model-b",
            model="mlx-community/model-b",
            category="numeric_unit_integrity",
            score=40,
            label="inaccurate",
        ),
    ]
    results_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    results = module.load_results_jsonl(results_path)

    html = module.render_generated_sections(
        results,
        results_path=results_path,
        expected_cases_per_model=2,
    )

    assert "Verified Leaderboard Results" in html
    assert "mlx-community/model-a" in html
    assert "2/2 ok" in html
    assert "90.0" in html
    assert "Total Gemini judge samples: 12" in html
    assert "numeric_unit_integrity" in html
    assert "report.html" in html
    assert "docs/asr-leaderboard-summary.json" in html
    assert "docs/asr-leaderboard-refresh-report.md" in html
    assert "docs/asr-leaderboard-run-manifest.json" in html
    assert "docs/asr-leaderboard-manifest-validation.json" in html
    assert "docs/asr-seed-manifest-validation.json" in html
    assert "reproducible refresh workflow" in html
    assert "--hosted-dir /path/to/kennethli319.github.io/open-audio-judge" in html
    assert "Generated Refresh Workflow" in html
    assert "Generated Artifacts" in html
    assert "Validate seed manifest" in html
    assert "scripts/validate_asr_seed_manifest.py" in html
    assert "Run one MLX ASR model" in html
    assert "--model &lt;mlx-community/model-id&gt;" in html
    assert "Machine-readable leaderboard summary" in html


def test_write_summary_artifact_records_models_and_categories(tmp_path: Path) -> None:
    module = load_script_module()
    results_path = tmp_path / "results.jsonl"
    summary_path = tmp_path / "summary.json"
    records = [
        result_record(
            case_id="asr-a-model-a",
            model="mlx-community/model-a",
            category="transcription_accuracy_wer",
            score=100,
            label="accurate",
        ),
        result_record(
            case_id="asr-b-model-a",
            model="mlx-community/model-a",
            category="numeric_unit_integrity",
            score=80,
            label="accurate",
        ),
        result_record(
            case_id="asr-a-model-b",
            model="mlx-community/model-b",
            category="transcription_accuracy_wer",
            score=60,
            label="needs_review",
        ),
        result_record(
            case_id="asr-b-model-b",
            model="mlx-community/model-b",
            category="numeric_unit_integrity",
            score=40,
            label="inaccurate",
        ),
    ]
    results_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    source_results_path = tmp_path / "model-a" / "judge-report" / "results.jsonl"
    source_results_path.parent.mkdir(parents=True)
    source_results_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records[:2]),
        encoding="utf-8",
    )
    results = module.load_results_jsonl(results_path)

    module.write_summary_artifact(
        results,
        summary_path,
        results_path=results_path,
        expected_cases_per_model=2,
        source_result_paths=[source_results_path],
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["total_results"] == 4
    assert summary["model_count"] == 2
    assert summary["category_count"] == 2
    assert summary["total_gemini_judge_samples"] == 12
    assert summary["source_result_paths"] == [
        str(source_results_path)
    ]
    assert summary["source_result_files"] == [
        {
            "path": str(source_results_path),
            "models": ["mlx-community/model-a"],
            "result_count": 2,
            "ok_count": 2,
            "judge_samples": 6,
            "average_score": 90,
            "labels": {"accurate": 2},
            "categories": {
                "numeric_unit_integrity": 1,
                "transcription_accuracy_wer": 1,
            },
        }
    ]
    assert summary["run_manifest_path"] == "docs/asr-leaderboard-run-manifest.json"
    assert summary["manifest_validation_path"] == "docs/asr-leaderboard-manifest-validation.json"
    assert summary["seed_manifest_validation_path"] == "docs/asr-seed-manifest-validation.json"
    assert summary["refresh_workflow"]["seed_manifest_validation_command"] == [
        ".venv/bin/python",
        "scripts/validate_asr_seed_manifest.py",
        "--summary-out",
        "docs/asr-seed-manifest-validation.json",
    ]
    assert summary["refresh_workflow"]["audio_materialization_command"] == [
        ".venv/bin/python",
        "scripts/synthesize_tts_cases.py",
        "--cases",
        "examples/asr_research_cases.jsonl",
        "--out",
        "runs/asr-research-audio",
        "--discard-text-sidecars",
        "--summary-out",
        "runs/asr-research-audio/summary.json",
    ]
    assert summary["refresh_workflow"]["model_run_template"] == [
        "oaj",
        "autojudge-mlx-asr",
        "--python-bin",
        ".venv/bin/python",
        "--cases",
        "runs/asr-research-audio/tts_audio_cases.jsonl",
        "--model",
        "<mlx-community/model-id>",
        "--judge-provider",
        "gemini",
        "--judge-samples",
        "3",
        "--out",
        "runs/asr-leaderboard/<run-name>",
    ]
    assert summary["refresh_workflow"]["combine_refresh_command"] == [
        ".venv/bin/python",
        "scripts/refresh_asr_leaderboard_artifacts.py",
        "--results",
        str(source_results_path),
        "--update-run-manifest",
    ]
    assert summary["refresh_workflow"]["manifest_refresh_command"] == [
        ".venv/bin/python",
        "scripts/refresh_asr_leaderboard_artifacts.py",
    ]
    assert summary["refresh_workflow"]["hosted_artifact_command"] == [
        ".venv/bin/python",
        "scripts/refresh_asr_leaderboard_artifacts.py",
        "--hosted-dir",
        "/path/to/kennethli319.github.io/open-audio-judge",
    ]
    assert "secret" in summary["refresh_workflow"]["secret_handling"].lower()
    assert summary["refresh_runtime_status"] == {
        "all_loaded_results_ok": True,
        "gemini_judge": "verified_from_loaded_results",
        "live_model_calls": "none",
        "loaded_result_providers": ["gemini"],
        "mlx_asr": "not_executed_by_refresh; transcripts loaded from verified result artifacts",
    }
    assert summary["model_category_matrix"] == [
        {
            "model": "mlx-community/model-a",
            "total_results": 2,
            "category_counts": {
                "transcription_accuracy_wer": 1,
                "numeric_unit_integrity": 1,
                "negation_modality_scope": 0,
                "temporal_scheduling_accuracy": 0,
                "entity_factual_integrity": 0,
                "semantic_paraphrase_preservation": 0,
                "acoustic_noise_robustness": 0,
            },
        },
        {
            "model": "mlx-community/model-b",
            "total_results": 2,
            "category_counts": {
                "transcription_accuracy_wer": 1,
                "numeric_unit_integrity": 1,
                "negation_modality_scope": 0,
                "temporal_scheduling_accuracy": 0,
                "entity_factual_integrity": 0,
                "semantic_paraphrase_preservation": 0,
                "acoustic_noise_robustness": 0,
            },
        },
    ]
    assert summary["models"][0]["model"] == "mlx-community/model-a"
    assert summary["models"][0]["average_score"] == 90
    assert summary["models"][0]["labels"] == {"accurate": 2}
    assert summary["categories"][0]["category"] == "transcription_accuracy_wer"
    assert summary["categories"][1]["category"] == "numeric_unit_integrity"


def test_generated_artifacts_include_new_observed_category_columns(tmp_path: Path) -> None:
    module = load_script_module()
    results_path = tmp_path / "results.jsonl"
    summary_path = tmp_path / "summary.json"
    report_path = tmp_path / "refresh-report.md"
    categories = [
        "transcription_accuracy_wer",
        "numeric_unit_integrity",
        "negation_modality_scope",
        "temporal_scheduling_accuracy",
        "entity_factual_integrity",
        "semantic_paraphrase_preservation",
        "acoustic_noise_robustness",
        "speaker_attribution_consistency",
    ]
    records = []
    for model in ("mlx-community/model-a", "mlx-community/model-b"):
        records.extend(
            [
                result_record(
                    case_id=f"asr-{category}-{model[-1]}",
                    model=model,
                    category=category,
                    score=90,
                    label="accurate",
                )
                for category in categories
            ]
        )
    results_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    results = module.load_results_jsonl(results_path)

    html = module.render_generated_sections(
        results,
        results_path=results_path,
        expected_cases_per_model=8,
    )
    module.write_summary_artifact(
        results,
        summary_path,
        results_path=results_path,
        expected_cases_per_model=8,
    )
    module.write_refresh_report(
        results,
        report_path,
        results_path=results_path,
        expected_cases_per_model=8,
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    report = report_path.read_text(encoding="utf-8")
    assert "Speaker Attribution Consistency" in html
    assert summary["category_columns"] == [
        {"category": "transcription_accuracy_wer", "label": "WER"},
        {"category": "numeric_unit_integrity", "label": "Numeric/Unit"},
        {"category": "negation_modality_scope", "label": "Negation/Modality"},
        {"category": "temporal_scheduling_accuracy", "label": "Temporal"},
        {"category": "entity_factual_integrity", "label": "Entity"},
        {"category": "semantic_paraphrase_preservation", "label": "Paraphrase"},
        {"category": "acoustic_noise_robustness", "label": "Acoustic Noise"},
        {
            "category": "speaker_attribution_consistency",
            "label": "Speaker Attribution Consistency",
        },
    ]
    assert summary["model_category_matrix"][0]["category_counts"] == {
        "transcription_accuracy_wer": 1,
        "numeric_unit_integrity": 1,
        "negation_modality_scope": 1,
        "temporal_scheduling_accuracy": 1,
        "entity_factual_integrity": 1,
        "semantic_paraphrase_preservation": 1,
        "acoustic_noise_robustness": 1,
        "speaker_attribution_consistency": 1,
    }
    assert (
        "| Model | WER | Numeric/Unit | Negation/Modality | Temporal | Entity | "
        "Paraphrase | Acoustic Noise | Speaker Attribution Consistency |"
    ) in report
    assert "| `mlx-community/model-a` | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |" in report


def test_write_refresh_report_records_coverage_and_commands(tmp_path: Path) -> None:
    module = load_script_module()
    results_path = tmp_path / "results.jsonl"
    report_path = tmp_path / "refresh-report.md"
    records = [
        result_record(
            case_id="asr-a-model-a",
            model="mlx-community/model-a",
            category="transcription_accuracy_wer",
            score=100,
            label="accurate",
        ),
        result_record(
            case_id="asr-b-model-a",
            model="mlx-community/model-a",
            category="numeric_unit_integrity",
            score=80,
            label="accurate",
        ),
        result_record(
            case_id="asr-a-model-b",
            model="mlx-community/model-b",
            category="transcription_accuracy_wer",
            score=60,
            label="needs_review",
        ),
        result_record(
            case_id="asr-b-model-b",
            model="mlx-community/model-b",
            category="numeric_unit_integrity",
            score=40,
            label="inaccurate",
        ),
    ]
    results_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    source_results_path = tmp_path / "model-a" / "judge-report" / "results.jsonl"
    source_results_path.parent.mkdir(parents=True)
    source_results_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records[:2]),
        encoding="utf-8",
    )
    results = module.load_results_jsonl(results_path)

    module.write_refresh_report(
        results,
        report_path,
        results_path=results_path,
        expected_cases_per_model=2,
        source_result_paths=[source_results_path],
    )

    text = report_path.read_text(encoding="utf-8")
    assert "# ASR Leaderboard Refresh Report" in text
    assert "Total judged transcripts: 4" in text
    assert "`mlx-community/model-a`" in text
    assert "`transcription_accuracy_wer`" in text
    assert ".venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py" in text
    assert ".venv/bin/python scripts/validate_asr_seed_manifest.py" in text
    assert "Seed manifest validation: `docs/asr-seed-manifest-validation.json`" in text
    assert "--summary-out docs/asr-seed-manifest-validation.json" in text
    assert "--results " + str(source_results_path) in text
    assert "--update-run-manifest" in text
    assert "Hosted artifact sync" in text
    assert "## Runtime Status" in text
    assert "MLX ASR: not_executed_by_refresh" in text
    assert "Gemini judge: verified_from_loaded_results" in text
    assert "Live model calls during refresh: none" in text
    assert "## Model Category Matrix" in text
    assert "| Model | WER | Numeric/Unit | Negation/Modality | Temporal | Entity | Paraphrase | Acoustic Noise |" in text
    assert "| `mlx-community/model-a` | 1 | 1 | 0 | 0 | 0 | 0 | 0 |" in text
    assert "## Source Result File Coverage" in text
    assert f"| `{source_results_path}` | `mlx-community/model-a` | 2/2 ok |" in text


def test_replace_generated_block_only_updates_marked_section(tmp_path: Path) -> None:
    module = load_script_module()
    page = tmp_path / "demo.html"
    page.write_text(
        "before\n"
        f"{module.START_MARKER}\n"
        "old generated content\n"
        f"{module.END_MARKER}\n"
        "after\n",
        encoding="utf-8",
    )

    module.replace_generated_block(page, f"{module.START_MARKER}\nnew content\n{module.END_MARKER}")

    assert page.read_text(encoding="utf-8") == (
        "before\n"
        f"{module.START_MARKER}\n"
        "new content\n"
        f"{module.END_MARKER}\n"
        "after\n"
    )


def test_refresh_asr_leaderboard_artifacts_combines_report_and_page(tmp_path: Path) -> None:
    update_module = load_script_module()
    refresh_module = load_refresh_module()
    first = tmp_path / "model-a" / "judge-report" / "results.jsonl"
    second = tmp_path / "model-b" / "results.jsonl"
    out = tmp_path / "combined"
    page = tmp_path / "demo.html"
    summary = tmp_path / "summary.json"
    refresh_report = tmp_path / "refresh-report.md"
    manifest_validation = tmp_path / "manifest-validation.json"
    seed_manifest_validation = tmp_path / "seed-manifest-validation.json"
    hosted_dir = tmp_path / "hosted" / "open-audio-judge"
    records_a = [
        result_record(
            case_id="asr-a-model-a",
            model="mlx-community/model-a",
            category="transcription_accuracy_wer",
            score=100,
            label="accurate",
        ),
        result_record(
            case_id="asr-b-model-a",
            model="mlx-community/model-a",
            category="numeric_unit_integrity",
            score=80,
            label="accurate",
        ),
    ]
    records_b = [
        result_record(
            case_id="asr-a-model-b",
            model="mlx-community/model-b",
            category="transcription_accuracy_wer",
            score=60,
            label="needs_review",
        ),
        result_record(
            case_id="asr-b-model-b",
            model="mlx-community/model-b",
            category="numeric_unit_integrity",
            score=40,
            label="inaccurate",
        ),
    ]
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("".join(json.dumps(record) + "\n" for record in records_a), encoding="utf-8")
    second.write_text("".join(json.dumps(record) + "\n" for record in records_b), encoding="utf-8")
    page.write_text(
        "before\n"
        f"{update_module.START_MARKER}\n"
        "old generated content\n"
        f"{update_module.END_MARKER}\n"
        "after\n",
        encoding="utf-8",
    )

    refresh_module.refresh_asr_leaderboard_artifacts(
        [tmp_path / "model-a", second],
        out=out,
        page=page,
        summary_out=summary,
        refresh_report_out=refresh_report,
        manifest_validation_out=manifest_validation,
        run_manifest=refresh_module.DEFAULT_RUN_MANIFEST,
        seed_manifest_validation_out=seed_manifest_validation,
        hosted_dir=hosted_dir,
        expected_cases_per_model=2,
    )

    assert (out / "results.jsonl").exists()
    assert (out / "report.html").exists()
    assert len((out / "results.jsonl").read_text(encoding="utf-8").splitlines()) == 4
    html = page.read_text(encoding="utf-8")
    assert "Verified Leaderboard Results" in html
    assert "mlx-community/model-a" in html
    assert "mlx-community/model-b" in html
    written_summary = json.loads(summary.read_text(encoding="utf-8"))
    assert written_summary["total_results"] == 4
    assert written_summary["model_count"] == 2
    assert written_summary["source_result_paths"] == [
        str(first),
        str(second),
    ]
    validation = json.loads(manifest_validation.read_text(encoding="utf-8"))
    assert validation["status"] == "complete"
    assert validation["result_file_count"] == 2
    assert validation["expected_cases_per_category"] == 1
    assert validation["models"][0]["category_counts"] == {
        "numeric_unit_integrity": 1,
        "transcription_accuracy_wer": 1,
    }
    assert (hosted_dir / "demo.html").read_text(encoding="utf-8") == page.read_text(
        encoding="utf-8"
    )
    assert (hosted_dir / "summary.json").read_text(encoding="utf-8") == summary.read_text(
        encoding="utf-8"
    )
    assert (hosted_dir / "refresh-report.md").read_text(
        encoding="utf-8"
    ) == refresh_report.read_text(encoding="utf-8")
    assert (hosted_dir / "asr-leaderboard-run-manifest.json").exists()
    assert (hosted_dir / "manifest-validation.json").read_text(
        encoding="utf-8"
    ) == manifest_validation.read_text(encoding="utf-8")
    assert json.loads(seed_manifest_validation.read_text(encoding="utf-8"))["status"] == "complete"
    assert (hosted_dir / "seed-manifest-validation.json").read_text(
        encoding="utf-8"
    ) == seed_manifest_validation.read_text(encoding="utf-8")


def test_refresh_asr_leaderboard_artifacts_reads_run_manifest(tmp_path: Path) -> None:
    refresh_module = load_refresh_module()
    nested = tmp_path / "run-a" / "judge-report" / "results.jsonl"
    direct = tmp_path / "run-b" / "results.jsonl"
    manifest = tmp_path / "manifest.json"
    nested.parent.mkdir(parents=True)
    direct.parent.mkdir(parents=True)
    nested.write_text("", encoding="utf-8")
    direct.write_text("", encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "runs": [
                    {"run_name": "run-a", "results_path": str(tmp_path / "run-a")},
                    {"run_name": "run-b", "results_path": str(direct)},
                ]
            }
        ),
        encoding="utf-8",
    )

    paths = refresh_module._result_paths_from_run_manifest(manifest)

    assert paths == [nested, direct]


def test_write_run_manifest_artifact_records_verified_result_sources(tmp_path: Path) -> None:
    refresh_module = load_refresh_module()
    results_path = tmp_path / "run-a" / "judge-report" / "results.jsonl"
    manifest = tmp_path / "manifest.json"
    records = [
        result_record(
            case_id="asr-a-model-a",
            model="mlx-community/model-a",
            category="transcription_accuracy_wer",
            score=100,
            label="accurate",
        ),
        result_record(
            case_id="asr-b-model-a",
            model="mlx-community/model-a",
            category="numeric_unit_integrity",
            score=80,
            label="accurate",
        ),
    ]
    results_path.parent.mkdir(parents=True)
    results_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    refresh_module.write_run_manifest_artifact(
        [results_path],
        manifest,
        expected_cases_per_model=2,
    )

    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["version"] == 2
    assert data["expected_cases_per_model"] == 2
    assert data["result_paths"] == [str(results_path)]
    assert data["runs"] == [
        {
            "run_name": "run-a",
            "model": "mlx-community/model-a",
            "results_path": str(results_path),
            "result_count": 2,
            "ok_count": 2,
            "category_counts": {
                "numeric_unit_integrity": 1,
                "transcription_accuracy_wer": 1,
            },
        }
    ]


def test_manifest_validation_checks_declared_models(tmp_path: Path) -> None:
    refresh_module = load_refresh_module()
    results_path = tmp_path / "run-a" / "judge-report" / "results.jsonl"
    manifest = tmp_path / "manifest.json"
    records = [
        result_record(
            case_id="asr-a-model-a",
            model="mlx-community/model-a",
            category="transcription_accuracy_wer",
            score=100,
            label="accurate",
        ),
        result_record(
            case_id="asr-b-model-a",
            model="mlx-community/model-a",
            category="numeric_unit_integrity",
            score=80,
            label="accurate",
        ),
    ]
    results_path.parent.mkdir(parents=True)
    results_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "run_name": "run-a",
                        "model": "mlx-community/model-a",
                        "results_path": str(results_path),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    results = refresh_module.load_results_jsonl(results_path)

    validation = refresh_module.build_manifest_validation(
        results,
        result_paths=[results_path],
        run_manifest=manifest,
        expected_cases_per_model=2,
    )

    assert validation["status"] == "complete"
    assert validation["result_file_checks"] == [
        {
            "path": str(results_path),
            "declared_model": "mlx-community/model-a",
            "actual_models": ["mlx-community/model-a"],
            "model_match": True,
        }
    ]


def test_manifest_validation_marks_declared_model_mismatch_incomplete(tmp_path: Path) -> None:
    refresh_module = load_refresh_module()
    results_path = tmp_path / "run-a" / "judge-report" / "results.jsonl"
    manifest = tmp_path / "manifest.json"
    records = [
        result_record(
            case_id="asr-a-model-a",
            model="mlx-community/model-a",
            category="transcription_accuracy_wer",
            score=100,
            label="accurate",
        ),
        result_record(
            case_id="asr-b-model-a",
            model="mlx-community/model-a",
            category="numeric_unit_integrity",
            score=80,
            label="accurate",
        ),
    ]
    results_path.parent.mkdir(parents=True)
    results_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "run_name": "run-a",
                        "model": "mlx-community/model-b",
                        "results_path": str(results_path),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    results = refresh_module.load_results_jsonl(results_path)

    validation = refresh_module.build_manifest_validation(
        results,
        result_paths=[results_path],
        run_manifest=manifest,
        expected_cases_per_model=2,
    )

    assert validation["status"] == "incomplete"
    assert validation["result_file_checks"][0]["declared_model"] == "mlx-community/model-b"
    assert validation["result_file_checks"][0]["actual_models"] == ["mlx-community/model-a"]
    assert validation["result_file_checks"][0]["model_match"] is False


def test_validate_coverage_rejects_uneven_model_category_counts(tmp_path: Path) -> None:
    module = load_script_module()
    results = module.load_results_jsonl(
        _write_results(
            tmp_path / "results.jsonl",
            [
                result_record(
                    case_id="asr-a-model-a",
                    model="mlx-community/model-a",
                    category="transcription_accuracy_wer",
                    score=100,
                    label="accurate",
                ),
                result_record(
                    case_id="asr-b-model-a",
                    model="mlx-community/model-a",
                    category="transcription_accuracy_wer",
                    score=90,
                    label="accurate",
                ),
                result_record(
                    case_id="asr-c-model-b",
                    model="mlx-community/model-b",
                    category="numeric_unit_integrity",
                    score=80,
                    label="accurate",
                ),
                result_record(
                    case_id="asr-d-model-b",
                    model="mlx-community/model-b",
                    category="numeric_unit_integrity",
                    score=70,
                    label="accurate",
                ),
            ]
        )
    )

    with pytest.raises(ValueError, match="uneven category coverage"):
        module.render_generated_sections(
            results,
            results_path=Path("results.jsonl"),
            expected_cases_per_model=2,
        )


def _write_results(path: Path, records: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return path
