import importlib.util
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest


SCRIPT = Path("scripts/update_asr_leaderboard_demo.py")
REFRESH_SCRIPT = Path("scripts/refresh_asr_leaderboard_artifacts.py")
CHECK_SCRIPT = Path("scripts/check_asr_leaderboard_page.py")


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


def load_check_module():
    spec = importlib.util.spec_from_file_location("check_asr_leaderboard_page", CHECK_SCRIPT)
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


def run_manifest_record(
    path_records: list[tuple[str, list[dict]]],
    *,
    expected_cases_per_model: int,
) -> dict:
    runs = []
    for results_path, records in path_records:
        models = sorted(
            {
                str(record["metadata"]["candidate_model"])
                for record in records
            }
        )
        assert len(models) == 1
        category_counts = {}
        for record in records:
            category = str(record["metadata"]["eval_category"])
            category_counts[category] = category_counts.get(category, 0) + 1
        path = Path(results_path)
        digest_fields = (
            {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            if path.exists()
            else {}
        )
        runs.append(
            {
                "run_name": Path(results_path).parent.name,
                "model": models[0],
                "results_path": results_path,
                **digest_fields,
                "result_count": len(records),
                "ok_count": sum(1 for record in records if record["status"] == "ok"),
                "category_counts": dict(sorted(category_counts.items())),
            }
        )
    return {
        "description": "Unit-test ASR leaderboard source result manifest.",
        "version": 2,
        "expected_cases_per_model": expected_cases_per_model,
        "generated_audio_manifest": "runs/asr-research-audio/tts_audio_cases.jsonl",
        "result_paths": [results_path for results_path, _ in path_records],
        "runs": runs,
    }


def hosted_manifest_record(path_records: list[tuple[str, Path]]) -> dict:
    artifacts = []
    for hosted_path, source in path_records:
        artifacts.append(
            {
                "source_path": str(source),
                "hosted_paths": [hosted_path],
                "bytes": source.stat().st_size,
                "sha256": file_sha256(source),
            }
        )
    return {
        "description": "Unit-test ASR hosted artifact manifest.",
        "version": 1,
        "hosted_base_path": "open-audio-judge",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_index_record(
    raw_path: str,
    actual_path: Path,
    *,
    digest_status: str = "ok",
) -> dict:
    if digest_status == "ok":
        return {
            "path": raw_path,
            "exists": actual_path.exists(),
            "bytes": actual_path.stat().st_size,
            "sha256": file_sha256(actual_path),
            "digest_status": "ok",
        }
    return {
        "path": raw_path,
        "exists": True,
        "bytes": None,
        "sha256": None,
        "digest_status": digest_status,
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
    assert "docs/asr-leaderboard-refresh-commands.sh" in html
    assert "docs/asr-leaderboard-run-manifest.json" in html
    assert "docs/asr-leaderboard-manifest-validation.json" in html
    assert "docs/asr-seed-manifest-validation.json" in html
    assert "reproducible refresh workflow" in html
    assert "docs/asr-leaderboard-hosted-manifest.json" in html
    assert "ASR_LEADERBOARD_HOSTED_DIR" in html
    assert "--hosted-dir-from-env" in html
    assert "Generated Refresh Workflow" in html
    assert "Preflight refresh inputs" in html
    assert "--check-only" in html
    assert "Run refresh shell playbook" in html
    assert "Generated Artifacts" in html
    assert "Validate seed manifest" in html
    assert "Discover latest complete runs" in html
    assert "--discover-complete-model-runs" in html
    assert "scripts/validate_asr_seed_manifest.py" in html
    assert "Check generated page" in html
    assert "scripts/check_asr_leaderboard_page.py" in html
    assert "Verify generated artifacts are fresh" in html
    assert "--require-generated-fresh" in html
    assert "Check hosted mirror" in html
    assert ".venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py --check-only --hosted-dir-from-env" in html
    assert "Run one MLX ASR model" in html
    assert "Check MLX ASR runtime" in html
    assert "PYTHONPATH=src .venv/bin/python -m open_audio_judge.cli check-mlx-asr-runtime" in html
    assert ".venv/bin/oaj autojudge-mlx-asr" in html
    assert "--model &lt;mlx-community/model-id&gt;" in html
    assert "Generated Model Refresh Commands" in html
    assert "mlx-community/whisper-large-v3-turbo-asr-fp16" in html
    assert "qwen3-asr-1.7b-refresh" in html
    assert "source /Users/wangyauli/.openclaw/secrets/open-audio-judge-gemini.env" in html
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
            "report_path": str(source_results_path.with_name("report.html")),
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
    assert summary["refresh_commands_path"] == "docs/asr-leaderboard-refresh-commands.sh"
    assert summary["manifest_validation_path"] == "docs/asr-leaderboard-manifest-validation.json"
    assert summary["seed_manifest_validation_path"] == "docs/asr-seed-manifest-validation.json"
    assert summary["next_runs_path"] == "docs/asr-leaderboard-next-runs.json"
    assert summary["hosted_manifest_path"] == "docs/asr-leaderboard-hosted-manifest.json"
    assert summary["artifact_index_path"] == "docs/asr-leaderboard-artifacts.json"
    assert summary["runtime_status_path"] == "docs/asr-leaderboard-runtime-status.json"
    assert summary["next_run_plan"]["status"] == "complete"
    assert summary["next_run_plan"]["missing_cell_count"] == 0
    assert summary["output_artifacts"] == [
        {
            "path": str(results_path),
            "purpose": "Combined ASR judge results used by the generated page and report.",
        },
        {
            "path": str(results_path.with_name("report.html")),
            "purpose": "Local combined HTML report with per-case judge details.",
        },
        {
            "path": "docs/asr-leaderboard-summary.json",
            "purpose": "Machine-readable leaderboard summary and reproducible refresh workflow.",
        },
        {
            "path": "docs/asr-leaderboard-refresh-report.md",
            "purpose": "Human-readable coverage, score, source-file, and command report.",
        },
        {
            "path": "docs/asr-leaderboard-refresh-commands.sh",
            "purpose": "Generated shell playbook for repeatable ASR leaderboard refreshes.",
        },
        {
            "path": "docs/asr-leaderboard-run-manifest.json",
            "purpose": "Committed source result manifest for manifest-based refreshes.",
        },
        {
            "path": "docs/asr-leaderboard-manifest-validation.json",
            "purpose": "Coverage validation for the model/category result matrix.",
        },
        {
            "path": "docs/asr-seed-manifest-validation.json",
            "purpose": "Seed-manifest validation proving public-safe ASR cases keep exact category coverage.",
        },
        {
            "path": "docs/asr-leaderboard-next-runs.json",
            "purpose": "Machine-readable next-refresh plan for missing ASR model/category cells.",
        },
        {
            "path": "docs/asr-leaderboard-hosted-manifest.json",
            "purpose": (
                "Machine-readable manifest of ASR demo artifacts mirrored to the "
                "hosted Pages checkout."
            ),
        },
        {
            "path": "docs/asr-leaderboard-artifacts.json",
            "purpose": "Single machine-readable index for the ASR leaderboard artifact bundle.",
        },
        {
            "path": "docs/asr-leaderboard-runtime-status.json",
            "purpose": "Machine-readable MLX ASR and Gemini readiness status for refresh automation.",
        },
    ]
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
        ".venv/bin/oaj",
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
    assert summary["refresh_workflow"]["mlx_runtime_check_command"] == [
        "PYTHONPATH=src",
        ".venv/bin/python",
        "-m",
        "open_audio_judge.cli",
        "check-mlx-asr-runtime",
        "--python-bin",
        ".venv/bin/python",
        "--model",
        "mlx-community/whisper-large-v3-turbo-asr-fp16",
    ]
    assert summary["refresh_workflow"]["model_run_commands"] == [
        {
            "model": "mlx-community/whisper-large-v3-turbo-asr-fp16",
            "run_name": "whisper-large-v3-turbo-refresh",
            "command": [
                ".venv/bin/oaj",
                "autojudge-mlx-asr",
                "--python-bin",
                ".venv/bin/python",
                "--cases",
                "runs/asr-research-audio/tts_audio_cases.jsonl",
                "--model",
                "mlx-community/whisper-large-v3-turbo-asr-fp16",
                "--judge-provider",
                "gemini",
                "--judge-samples",
                "3",
                "--out",
                "runs/asr-leaderboard/whisper-large-v3-turbo-refresh",
            ],
        },
        {
            "model": "mlx-community/Qwen3-ASR-1.7B-8bit",
            "run_name": "qwen3-asr-1.7b-refresh",
            "command": [
                ".venv/bin/oaj",
                "autojudge-mlx-asr",
                "--python-bin",
                ".venv/bin/python",
                "--cases",
                "runs/asr-research-audio/tts_audio_cases.jsonl",
                "--model",
                "mlx-community/Qwen3-ASR-1.7B-8bit",
                "--judge-provider",
                "gemini",
                "--judge-samples",
                "3",
                "--out",
                "runs/asr-leaderboard/qwen3-asr-1.7b-refresh",
            ],
        },
        {
            "model": "mlx-community/VibeVoice-ASR-4bit",
            "run_name": "vibevoice-asr-refresh",
            "command": [
                ".venv/bin/oaj",
                "autojudge-mlx-asr",
                "--python-bin",
                ".venv/bin/python",
                "--cases",
                "runs/asr-research-audio/tts_audio_cases.jsonl",
                "--model",
                "mlx-community/VibeVoice-ASR-4bit",
                "--judge-provider",
                "gemini",
                "--judge-samples",
                "3",
                "--out",
                "runs/asr-leaderboard/vibevoice-asr-refresh",
            ],
        },
    ]
    assert summary["refresh_workflow"]["combine_refresh_command"] == [
        ".venv/bin/python",
        "scripts/refresh_asr_leaderboard_artifacts.py",
        "--results",
        str(source_results_path),
        "--update-run-manifest",
    ]
    assert summary["refresh_workflow"]["discover_refresh_command"] == [
        ".venv/bin/python",
        "scripts/refresh_asr_leaderboard_artifacts.py",
        "--discover-complete-model-runs",
        "--update-run-manifest",
    ]
    assert summary["refresh_workflow"]["refresh_check_command"] == [
        ".venv/bin/python",
        "scripts/refresh_asr_leaderboard_artifacts.py",
        "--check-only",
    ]
    assert summary["refresh_workflow"]["freshness_check_command"] == [
        ".venv/bin/python",
        "scripts/refresh_asr_leaderboard_artifacts.py",
        "--check-only",
        "--require-generated-fresh",
    ]
    assert summary["refresh_workflow"]["manifest_refresh_command"] == [
        ".venv/bin/python",
        "scripts/refresh_asr_leaderboard_artifacts.py",
    ]
    assert summary["refresh_workflow"]["refresh_commands_path"] == (
        "docs/asr-leaderboard-refresh-commands.sh"
    )
    assert summary["refresh_workflow"]["page_validation_command"] == [
        ".venv/bin/python",
        "scripts/check_asr_leaderboard_page.py",
    ]
    assert summary["refresh_workflow"]["hosted_artifact_command"] == [
        ".venv/bin/python",
        "scripts/refresh_asr_leaderboard_artifacts.py",
        "--hosted-dir-from-env",
    ]
    assert summary["refresh_workflow"]["hosted_validation_command"] == [
        ".venv/bin/python",
        "scripts/refresh_asr_leaderboard_artifacts.py",
        "--check-only",
        "--hosted-dir-from-env",
    ]
    assert summary["refresh_workflow"]["hosted_artifact_env_var"] == "ASR_LEADERBOARD_HOSTED_DIR"
    assert summary["refresh_workflow"]["local_secret_env_command"] == [
        "source",
        "/Users/wangyauli/.openclaw/secrets/open-audio-judge-gemini.env",
    ]
    assert "secret" in summary["refresh_workflow"]["secret_handling"].lower()
    assert summary["refresh_workflow"]["fallback_model_ids"] == [
        "mlx-community/whisper-small.en-asr-4bit",
        "mlx-community/parakeet-rnnt-0.6b",
        "mlx-community/GLM-ASR-Nano-2512-4bit",
    ]
    assert "do not substitute silently" in summary["refresh_workflow"]["fallback_handling"]
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
    assert "PYTHONPATH=src .venv/bin/python -m open_audio_judge.cli check-mlx-asr-runtime" in text
    assert "Seed manifest validation: `docs/asr-seed-manifest-validation.json`" in text
    assert "Hosted artifact manifest: `docs/asr-leaderboard-hosted-manifest.json`" in text
    assert "Refresh command playbook: `docs/asr-leaderboard-refresh-commands.sh`" in text
    assert "--summary-out docs/asr-seed-manifest-validation.json" in text
    assert "Load local Gemini secret before model runs" in text
    assert "Run mlx-community/whisper-large-v3-turbo-asr-fp16" in text
    assert ".venv/bin/oaj autojudge-mlx-asr" in text
    assert "runs/asr-leaderboard/vibevoice-asr-refresh" in text
    assert "Fallback models if a primary model is blocked" in text
    assert "mlx-community/parakeet-rnnt-0.6b" in text
    assert "Preflight refresh inputs" in text
    assert "--check-only" in text
    assert "Generated artifact freshness check" in text
    assert "--require-generated-fresh" in text
    assert "--results " + str(source_results_path) in text
    assert "--update-run-manifest" in text
    assert "Discover latest complete runs" in text
    assert "--discover-complete-model-runs" in text
    assert "Hosted artifact sync" in text
    assert "Hosted mirror validation" in text
    assert ".venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py --check-only --hosted-dir-from-env" in text
    assert "Page validation" in text
    assert "scripts/check_asr_leaderboard_page.py" in text
    assert "## Runtime Status" in text
    assert "MLX ASR: not_executed_by_refresh" in text
    assert "Gemini judge: verified_from_loaded_results" in text
    assert "Live model calls during refresh: none" in text
    assert "## Model Category Matrix" in text
    assert "| Model | WER | Numeric/Unit | Negation/Modality | Temporal | Entity | Paraphrase | Acoustic Noise |" in text
    assert "| `mlx-community/model-a` | 1 | 1 | 0 | 0 | 0 | 0 | 0 |" in text
    assert "## Source Result File Coverage" in text
    assert (
        f"| `{source_results_path}` | `{source_results_path.with_name('report.html')}` | "
        "`mlx-community/model-a` | 2/2 ok |"
    ) in text
    assert "## Generated Artifact Index" in text
    assert f"| `{results_path}` | Combined ASR judge results used by the generated page and report. |" in text


def test_render_generated_sections_includes_source_run_reports(tmp_path: Path) -> None:
    module = load_script_module()
    results_path = tmp_path / "combined" / "results.jsonl"
    source_results_path = tmp_path / "model-a" / "judge-report" / "results.jsonl"
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
    source_results_path.parent.mkdir(parents=True)
    serialized = "".join(json.dumps(record) + "\n" for record in records)
    results_path.write_text(serialized, encoding="utf-8")
    source_results_path.write_text(serialized, encoding="utf-8")
    results = module.load_results_jsonl(results_path)

    html = module.render_generated_sections(
        results,
        results_path=results_path,
        expected_cases_per_model=2,
        source_result_paths=[source_results_path],
    )

    assert "Source Run Reports" in html
    assert str(source_results_path) in html
    assert str(source_results_path.with_name("report.html")) in html
    assert "transcription_accuracy_wer: 1" in html
    assert "numeric_unit_integrity: 1" in html


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
    refresh_commands = tmp_path / "refresh-commands.sh"
    run_manifest = tmp_path / "run-manifest.json"
    manifest_validation = tmp_path / "manifest-validation.json"
    seed_manifest_validation = tmp_path / "seed-manifest-validation.json"
    next_runs = tmp_path / "next-runs.json"
    hosted_manifest = tmp_path / "hosted-manifest.json"
    artifact_index = tmp_path / "artifact-index.json"
    runtime_status = tmp_path / "runtime-status.json"
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
        "Open Audio Judge ASR Leaderboard\n"
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
        refresh_commands_out=refresh_commands,
        manifest_validation_out=manifest_validation,
        run_manifest=run_manifest,
        update_run_manifest=True,
        seed_manifest_validation_out=seed_manifest_validation,
        next_runs_out=next_runs,
        hosted_manifest_out=hosted_manifest,
        artifact_index_out=artifact_index,
        runtime_status_out=runtime_status,
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
    assert all(check["digest_match"] for check in validation["result_file_checks"])
    assert all(check["actual_sha256"] for check in validation["result_file_checks"])
    assert validation["models"][0]["category_counts"] == {
        "numeric_unit_integrity": 1,
        "transcription_accuracy_wer": 1,
    }
    written_manifest = json.loads(run_manifest.read_text(encoding="utf-8"))
    assert all(run["bytes"] > 0 for run in written_manifest["runs"])
    assert all(len(run["sha256"]) == 64 for run in written_manifest["runs"])
    assert (hosted_dir / "demo.html").read_text(encoding="utf-8") == page.read_text(
        encoding="utf-8"
    )
    assert (hosted_dir / "summary.json").read_text(encoding="utf-8") == summary.read_text(
        encoding="utf-8"
    )
    assert (hosted_dir / "refresh-report.md").read_text(
        encoding="utf-8"
    ) == refresh_report.read_text(encoding="utf-8")
    assert (hosted_dir / "refresh-commands.sh").read_text(
        encoding="utf-8"
    ) == refresh_commands.read_text(encoding="utf-8")
    refresh_command_text = refresh_commands.read_text(encoding="utf-8")
    assert "ASR_SYNC_HOSTED=1" in refresh_command_text
    assert (
        ": \"${ASR_LEADERBOARD_HOSTED_DIR:?Set ASR_LEADERBOARD_HOSTED_DIR "
        "to the Pages checkout open-audio-judge directory}\""
    ) in refresh_command_text
    assert "source /Users/wangyauli/.openclaw/secrets/open-audio-judge-gemini.env" in refresh_command_text
    assert (
        "PYTHONPATH=src .venv/bin/python -m open_audio_judge.cli check-mlx-asr-runtime"
        in refresh_command_text
    )
    assert (
        ".venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py --check-only --require-generated-fresh"
        in refresh_command_text
    )
    assert ".venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py --check-only --hosted-dir-from-env" in refresh_command_text
    assert (hosted_dir / "asr-leaderboard-run-manifest.json").exists()
    assert (hosted_dir / "manifest-validation.json").read_text(
        encoding="utf-8"
    ) == manifest_validation.read_text(encoding="utf-8")
    assert json.loads(seed_manifest_validation.read_text(encoding="utf-8"))["status"] == "complete"
    assert (hosted_dir / "seed-manifest-validation.json").read_text(
        encoding="utf-8"
    ) == seed_manifest_validation.read_text(encoding="utf-8")
    assert json.loads(next_runs.read_text(encoding="utf-8"))["status"] == "complete"
    assert (hosted_dir / "asr-leaderboard-next-runs.json").read_text(
        encoding="utf-8"
    ) == next_runs.read_text(encoding="utf-8")
    assert json.loads(runtime_status.read_text(encoding="utf-8"))["status"] == "complete"
    assert (hosted_dir / "asr-leaderboard-runtime-status.json").read_text(
        encoding="utf-8"
    ) == runtime_status.read_text(encoding="utf-8")
    artifact_index_data = json.loads(artifact_index.read_text(encoding="utf-8"))
    assert artifact_index_data["status"] == "complete"
    assert artifact_index_data["total_results"] == 4
    assert {artifact["path"] for artifact in artifact_index_data["artifacts"]} >= {
        str(out / "results.jsonl"),
        str(out / "report.html"),
        str(refresh_commands),
        str(artifact_index),
        "docs/asr-leaderboard-artifacts.json",
    }
    digest_statuses = {
        artifact["path"]: artifact["digest_status"]
        for artifact in artifact_index_data["artifacts"]
    }
    assert digest_statuses[str(artifact_index)] == "deferred_circular_reference"
    assert digest_statuses[str(hosted_manifest)] == "deferred_circular_reference"
    assert digest_statuses[str(out / "results.jsonl")] == "ok"
    hosted_manifest_data = json.loads(hosted_manifest.read_text(encoding="utf-8"))
    assert hosted_manifest_data["artifact_count"] == 12
    assert {
        "asr-leaderboard/full-35-combined/results.jsonl"
    } in [set(artifact["hosted_paths"]) for artifact in hosted_manifest_data["artifacts"]]
    assert (hosted_dir / "asr-leaderboard-hosted-manifest.json").read_text(
        encoding="utf-8"
    ) == hosted_manifest.read_text(encoding="utf-8")
    assert (hosted_dir / "asr-leaderboard-artifacts.json").read_text(
        encoding="utf-8"
    ) == artifact_index.read_text(encoding="utf-8")
    assert (
        hosted_dir / "asr-leaderboard" / "full-35-combined" / "results.jsonl"
    ).read_text(encoding="utf-8") == (out / "results.jsonl").read_text(encoding="utf-8")
    assert (
        hosted_dir / "asr-leaderboard" / "full-35-combined" / "report.html"
    ).read_text(encoding="utf-8") == (out / "report.html").read_text(encoding="utf-8")


def test_refresh_asr_leaderboard_artifacts_reads_hosted_dir_from_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refresh_module = load_refresh_module()
    hosted_dir = tmp_path / "pages" / "open-audio-judge"

    monkeypatch.setenv("ASR_LEADERBOARD_HOSTED_DIR", str(hosted_dir))

    assert refresh_module._hosted_dir_from_env("ASR_LEADERBOARD_HOSTED_DIR") == hosted_dir


def test_refresh_asr_leaderboard_artifacts_requires_hosted_dir_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refresh_module = load_refresh_module()

    monkeypatch.delenv("ASR_LEADERBOARD_HOSTED_DIR", raising=False)

    with pytest.raises(ValueError, match="ASR_LEADERBOARD_HOSTED_DIR"):
        refresh_module._hosted_dir_from_env("ASR_LEADERBOARD_HOSTED_DIR")


def test_check_asr_leaderboard_refresh_inputs_validates_default_artifacts() -> None:
    refresh_module = load_refresh_module()
    result_paths = refresh_module._default_result_paths(
        35,
        run_manifest=refresh_module.DEFAULT_RUN_MANIFEST,
    )

    check_summary = refresh_module.check_asr_leaderboard_refresh_inputs(
        result_paths,
        page=refresh_module.DEFAULT_PAGE,
        summary_out=refresh_module.DEFAULT_SUMMARY,
        seed_cases=refresh_module.DEFAULT_CASES,
        expected_cases_per_model=35,
    )

    assert check_summary == {
        "status": "complete",
        "result_file_count": 18,
        "total_results": 105,
        "model_count": 3,
        "category_count": 7,
        "seed_manifest_status": "complete",
        "page_status": "complete",
    }


def test_require_generated_fresh_rejects_stale_page_block(tmp_path: Path) -> None:
    update_module = load_script_module()
    refresh_module = load_refresh_module()
    results_path = tmp_path / "model-a" / "judge-report" / "results.jsonl"
    summary_path = tmp_path / "summary.json"
    page = tmp_path / "demo.html"
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
    results = update_module.load_results_jsonl(results_path)
    generated = update_module.render_generated_sections(
        results,
        results_path=tmp_path / "combined" / "results.jsonl",
        expected_cases_per_model=2,
        source_result_paths=[results_path],
    )
    page.write_text(
        "Open Audio Judge ASR Leaderboard\n" + generated + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(
            {
                "total_results": 2,
                "model_count": 1,
                "category_count": 2,
                "expected_cases_per_model": 2,
                "source_result_paths": [str(results_path)],
            }
        ),
        encoding="utf-8",
    )

    refresh_module._validate_generated_artifacts_fresh(
        results,
        result_paths=[results_path],
        page=page,
        summary_out=summary_path,
        generated=generated,
        expected_cases_per_model=2,
    )

    page.write_text(
        page.read_text(encoding="utf-8").replace("Verified Leaderboard Results", "Stale Results"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="generated ASR leaderboard block is stale"):
        refresh_module._validate_generated_artifacts_fresh(
            results,
            result_paths=[results_path],
            page=page,
            summary_out=summary_path,
            generated=generated,
            expected_cases_per_model=2,
        )


def test_discover_complete_model_result_paths_selects_newest_complete_runs(tmp_path: Path) -> None:
    refresh_module = load_refresh_module()
    runs_root = tmp_path / "runs" / "asr-leaderboard"
    older = runs_root / "model-a-older" / "judge-report" / "results.jsonl"
    newer = runs_root / "model-a-newer" / "judge-report" / "results.jsonl"
    model_b = runs_root / "model-b" / "judge-report" / "results.jsonl"
    incomplete = runs_root / "model-a-incomplete" / "judge-report" / "results.jsonl"
    for path in (older, newer, model_b, incomplete):
        path.parent.mkdir(parents=True, exist_ok=True)

    older_records = [
        result_record(
            case_id="asr-a-model-a-older",
            model="mlx-community/model-a",
            category="transcription_accuracy_wer",
            score=70,
            label="needs_review",
        ),
        result_record(
            case_id="asr-b-model-a-older",
            model="mlx-community/model-a",
            category="numeric_unit_integrity",
            score=70,
            label="needs_review",
        ),
    ]
    newer_records = [
        result_record(
            case_id="asr-a-model-a-newer",
            model="mlx-community/model-a",
            category="transcription_accuracy_wer",
            score=100,
            label="accurate",
        ),
        result_record(
            case_id="asr-b-model-a-newer",
            model="mlx-community/model-a",
            category="numeric_unit_integrity",
            score=100,
            label="accurate",
        ),
    ]
    model_b_records = [
        result_record(
            case_id="asr-a-model-b",
            model="mlx-community/model-b",
            category="transcription_accuracy_wer",
            score=90,
            label="accurate",
        ),
        result_record(
            case_id="asr-b-model-b",
            model="mlx-community/model-b",
            category="numeric_unit_integrity",
            score=90,
            label="accurate",
        ),
    ]
    incomplete_records = newer_records[:1]
    for path, records in (
        (older, older_records),
        (newer, newer_records),
        (model_b, model_b_records),
        (incomplete, incomplete_records),
    ):
        path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    os.utime(older, (1, 1))
    os.utime(model_b, (2, 2))
    os.utime(newer, (3, 3))

    assert refresh_module.discover_complete_model_result_paths(
        runs_root,
        expected_cases_per_model=2,
        model_ids=["mlx-community/model-a", "mlx-community/model-b"],
    ) == [newer, model_b]


def test_discover_or_default_result_paths_falls_back_to_run_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    refresh_module = load_refresh_module()
    runs_root = tmp_path / "runs" / "asr-leaderboard"
    runs_root.mkdir(parents=True)
    model_a = runs_root / "model-a-segmented" / "judge-report" / "results.jsonl"
    model_b = runs_root / "model-b-segmented" / "judge-report" / "results.jsonl"
    manifest = tmp_path / "run-manifest.json"
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
    for path, records in ((model_a, records_a), (model_b, records_b)):
        path.parent.mkdir(parents=True)
        path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    manifest.write_text(
        json.dumps(
            run_manifest_record(
                [
                    (str(model_a), records_a),
                    (str(model_b), records_b),
                ],
                expected_cases_per_model=2,
            )
        ),
        encoding="utf-8",
    )

    paths = refresh_module._discover_or_default_result_paths(
        runs_root,
        expected_cases_per_model=2,
        run_manifest=manifest,
    )

    assert paths == [model_a, model_b]
    assert "using the committed run manifest/segmented sources instead" in capsys.readouterr().err


def test_manifest_validation_detects_source_result_digest_drift(tmp_path: Path) -> None:
    refresh_module = load_refresh_module()
    result_path = tmp_path / "model-a" / "judge-report" / "results.jsonl"
    manifest = tmp_path / "run-manifest.json"
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
    result_path.parent.mkdir(parents=True)
    result_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    manifest.write_text(
        json.dumps(
            run_manifest_record(
                [(str(result_path), records)],
                expected_cases_per_model=2,
            )
        ),
        encoding="utf-8",
    )

    drifted_records = [
        {**records[0], "overall_score": 99},
        records[1],
    ]
    result_path.write_text(
        "".join(json.dumps(record) + "\n" for record in drifted_records),
        encoding="utf-8",
    )
    results = refresh_module.combined_results_from_paths([result_path])

    validation = refresh_module.build_manifest_validation(
        results,
        result_paths=[result_path],
        run_manifest=manifest,
        expected_cases_per_model=2,
    )

    assert validation["status"] == "incomplete"
    check = validation["result_file_checks"][0]
    assert check["path"] == str(result_path)
    assert check["declared_model"] == "mlx-community/model-a"
    assert check["actual_models"] == ["mlx-community/model-a"]
    assert check["model_match"] is True
    assert check["declared_bytes"] != check["actual_bytes"] or check["declared_sha256"] != check["actual_sha256"]
    assert check["actual_bytes"] == result_path.stat().st_size
    assert check["actual_sha256"] == file_sha256(result_path)
    assert check["digest_match"] is False


def test_build_audio_manifest_status_validates_materialized_seed_audio(tmp_path: Path) -> None:
    refresh_module = load_refresh_module()
    seed_cases = tmp_path / "asr_research_cases.jsonl"
    audio_cases = tmp_path / "asr-research-audio" / "tts_audio_cases.jsonl"
    audio_file = audio_cases.parent / "audio" / "case-a.wav"
    seed_record = {
        "id": "asr-case-a",
        "task": "asr_error",
        "reference_text": "The public seed sentence is ready.",
        "metadata": {
            "language": "en",
            "eval_category": "transcription_accuracy_wer",
            "asr_slice": "unit_test_slice",
            "source": "research-backed-asr-demo",
            "source_basis": "Unit test source basis.",
            "expected_error_focus": "Unit test focus.",
            "requires_audio_materialization": True,
        },
    }
    audio_record = {
        **seed_record,
        "id": "asr-case-a-local-tts",
        "audio_path": "audio/case-a.wav",
        "metadata": {
            **seed_record["metadata"],
            "source_case_id": "asr-case-a",
        },
    }
    audio_file.parent.mkdir(parents=True)
    audio_file.write_bytes(b"RIFF")
    seed_cases.write_text(json.dumps(seed_record) + "\n", encoding="utf-8")
    audio_cases.write_text(json.dumps(audio_record) + "\n", encoding="utf-8")

    status = refresh_module.build_audio_manifest_status(
        seed_cases_path=seed_cases,
        audio_cases_path=audio_cases,
    )

    assert status["status"] == "complete"
    assert status["seed_case_count"] == 1
    assert status["audio_case_count"] == 1
    assert status["missing_source_case_ids"] == []
    assert status["missing_audio_file_case_ids"] == []

    audio_file.unlink()
    stale = refresh_module.build_audio_manifest_status(
        seed_cases_path=seed_cases,
        audio_cases_path=audio_cases,
    )

    assert stale["status"] == "stale"
    assert stale["missing_audio_file_case_ids"] == ["asr-case-a-local-tts"]


def test_check_asr_leaderboard_page_validates_generated_artifacts(tmp_path: Path) -> None:
    update_module = load_script_module()
    check_module = load_check_module()
    page = tmp_path / "demo.html"
    summary = tmp_path / "summary.json"
    results_path = tmp_path / "results.jsonl"
    run_manifest = tmp_path / "run-manifest.json"
    manifest_validation = tmp_path / "manifest-validation.json"
    seed_manifest_validation = tmp_path / "seed-manifest-validation.json"
    next_runs = tmp_path / "next-runs.json"
    hosted_manifest = tmp_path / "hosted-manifest.json"
    artifact_index = tmp_path / "artifact-index.json"
    runtime_status = tmp_path / "runtime-status.json"
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
    results_path.with_name("report.html").write_text("<html></html>\n", encoding="utf-8")
    results = update_module.load_results_jsonl(results_path)
    page.write_text(
        "<!doctype html><html lang=\"en\"><body>\n"
        "<h1>Open Audio Judge ASR Leaderboard</h1>\n"
        f"{update_module.START_MARKER}\n"
        "old generated content\n"
        f"{update_module.END_MARKER}\n"
        "</body></html>\n",
        encoding="utf-8",
    )
    update_module.replace_generated_block(
        page,
        update_module.render_generated_sections(
            results,
            results_path=results_path,
            expected_cases_per_model=2,
        ),
    )
    update_module.write_summary_artifact(
        results,
        summary,
        results_path=results_path,
        expected_cases_per_model=2,
    )
    model_a_records = [record for record in records if record["metadata"]["candidate_model"] == "mlx-community/model-a"]
    model_b_records = [record for record in records if record["metadata"]["candidate_model"] == "mlx-community/model-b"]
    run_manifest.write_text(
        json.dumps(
            run_manifest_record(
                [
                    ("runs/asr-leaderboard/model-a/judge-report/results.jsonl", model_a_records),
                    ("runs/asr-leaderboard/model-b/judge-report/results.jsonl", model_b_records),
                ],
                expected_cases_per_model=2,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_validation.write_text(
        json.dumps(
            {
                "status": "complete",
                "total_results": 4,
                "model_count": 2,
                "category_count": 2,
                "expected_cases_per_model": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    seed_manifest_validation.write_text(
        json.dumps({"status": "complete"}) + "\n",
        encoding="utf-8",
    )
    next_runs.write_text(
        json.dumps(
            {
                "status": "complete",
                "expected_cases_per_model": 2,
                "model_count": 2,
                "category_count": 2,
                "missing_cell_count": 0,
                "next_run_command_count": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    hosted_manifest.write_text(
        json.dumps(
            hosted_manifest_record(
                [
                    ("results.jsonl", results_path),
                    ("run-manifest.json", run_manifest),
                ]
            )
        )
        + "\n",
        encoding="utf-8",
    )
    refresh_module = load_refresh_module()
    refresh_module.write_runtime_status_artifact(runtime_status, results=results)
    summary_data = json.loads(summary.read_text(encoding="utf-8"))
    summary_data["run_manifest_path"] = str(run_manifest)
    summary_data["manifest_validation_path"] = str(manifest_validation)
    summary_data["seed_manifest_validation_path"] = str(seed_manifest_validation)
    summary_data["next_runs_path"] = str(next_runs)
    summary_data["hosted_manifest_path"] = str(hosted_manifest)
    summary_data["artifact_index_path"] = str(artifact_index)
    summary_data["runtime_status_path"] = str(runtime_status)
    summary.write_text(json.dumps(summary_data), encoding="utf-8")
    refresh_report = tmp_path / "refresh-report.md"
    refresh_report.write_text("# report\n", encoding="utf-8")
    refresh_commands = tmp_path / "refresh-commands.sh"
    update_module.write_refresh_commands_script(refresh_commands)
    summary_data = json.loads(summary.read_text(encoding="utf-8"))
    summary_data["refresh_commands_path"] = str(refresh_commands)
    summary.write_text(json.dumps(summary_data), encoding="utf-8")
    refresh_module.write_artifact_index(
        artifact_index,
        results=results,
        results_path=results_path,
        report_path=results_path.with_name("report.html"),
        page=page,
        summary_out=summary,
        refresh_report_out=refresh_report,
        refresh_commands_out=refresh_commands,
        run_manifest=run_manifest,
        manifest_validation_out=manifest_validation,
        seed_manifest_validation_out=seed_manifest_validation,
        next_runs_out=next_runs,
        hosted_manifest_out=hosted_manifest,
        runtime_status_out=runtime_status,
        expected_cases_per_model=2,
    )

    validation = check_module.check_asr_leaderboard_page(page, summary_path=summary)

    assert validation["status"] == "complete"
    assert validation["total_results"] == 4
    assert validation["model_count"] == 2
    assert validation["output_artifact_count"] == len(summary_data["output_artifacts"])

    unindexed_hosted_source = tmp_path / "unindexed-hosted-source.json"
    unindexed_hosted_source.write_text("{}\n", encoding="utf-8")
    hosted_data = json.loads(hosted_manifest.read_text(encoding="utf-8"))
    hosted_data["artifacts"].append(
        {
            "source_path": str(unindexed_hosted_source),
            "hosted_paths": ["unindexed-hosted-source.json"],
            "bytes": unindexed_hosted_source.stat().st_size,
            "sha256": file_sha256(unindexed_hosted_source),
        }
    )
    hosted_data["artifact_count"] = len(hosted_data["artifacts"])
    hosted_manifest.write_text(json.dumps(hosted_data), encoding="utf-8")
    with pytest.raises(ValueError, match="source_path is missing from .*artifact-index"):
        check_module.check_asr_leaderboard_page(page, summary_path=summary)


def test_check_asr_leaderboard_page_rejects_stale_refresh_commands(tmp_path: Path) -> None:
    update_module = load_script_module()
    check_module = load_check_module()
    refresh_commands = tmp_path / "refresh-commands.sh"
    update_module.write_refresh_commands_script(refresh_commands)
    text = refresh_commands.read_text(encoding="utf-8").replace(
        ".venv/bin/python scripts/validate_asr_seed_manifest.py --summary-out docs/asr-seed-manifest-validation.json",
        ".venv/bin/python scripts/validate_asr_seed_manifest.py --stale-summary docs/asr-seed-manifest-validation.json",
        1,
    )
    refresh_commands.write_text(text, encoding="utf-8")
    summary = {
        "refresh_commands_path": str(refresh_commands),
        "refresh_workflow": update_module._refresh_workflow([]),
    }

    with pytest.raises(ValueError, match="seed_manifest_validation_command"):
        check_module._validate_refresh_commands_script(
            summary,
            summary_path=tmp_path / "summary.json",
            artifact_root=tmp_path,
            path_maps=[],
        )


def test_check_asr_leaderboard_page_validates_hosted_artifact_layout(tmp_path: Path) -> None:
    update_module = load_script_module()
    check_module = load_check_module()
    hosted = tmp_path / "hosted"
    page = hosted / "asr-leaderboard-demo.html"
    summary = hosted / "asr-leaderboard-summary.json"
    manifest_validation = hosted / "asr-leaderboard-manifest-validation.json"
    seed_manifest_validation = hosted / "asr-seed-manifest-validation.json"
    next_runs = hosted / "asr-leaderboard-next-runs.json"
    run_manifest = hosted / "asr-leaderboard-run-manifest.json"
    refresh_commands = hosted / "asr-leaderboard-refresh-commands.sh"
    hosted_manifest = hosted / "asr-leaderboard-hosted-manifest.json"
    artifact_index = hosted / "asr-leaderboard-artifacts.json"
    runtime_status = hosted / "asr-leaderboard-runtime-status.json"
    results_path = hosted / "asr-leaderboard" / "full-35-combined" / "results.jsonl"
    report_path = hosted / "asr-leaderboard" / "full-35-combined" / "report.html"

    results_path.parent.mkdir(parents=True)
    for path in (results_path, report_path):
        path.write_text("{}\n", encoding="utf-8")
    run_manifest.write_text(
        json.dumps(
            run_manifest_record(
                [
                    (
                        "runs/asr-leaderboard/model-a/judge-report/results.jsonl",
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
                                category="numeric_unit_integrity",
                                score=80,
                                label="accurate",
                            ),
                        ],
                    ),
                    (
                        "runs/asr-leaderboard/model-b/judge-report/results.jsonl",
                        [
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
                        ],
                    ),
                ],
                expected_cases_per_model=2,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_validation.write_text(
        json.dumps(
            {
                "status": "complete",
                "total_results": 4,
                "model_count": 2,
                "category_count": 2,
                "expected_cases_per_model": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    seed_manifest_validation.write_text(json.dumps({"status": "complete"}) + "\n", encoding="utf-8")
    update_module.write_refresh_commands_script(refresh_commands)
    next_runs.write_text(
        json.dumps(
            {
                "status": "complete",
                "expected_cases_per_model": 2,
                "model_count": 2,
                "category_count": 2,
                "missing_cell_count": 0,
                "next_run_command_count": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    runtime_status.write_text(
        json.dumps(
            {
                "status": "complete",
                "mlx_runtime_preflight": {"status": "not_checked"},
                "gemini_secret": {"status": "present"},
                "audio_manifest": {"status": "complete"},
                "secret_handling": "test fixture",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    hosted_manifest.write_text(
        json.dumps(
            hosted_manifest_record(
                [
                    ("asr-leaderboard/full-35-combined/results.jsonl", results_path),
                    ("asr-leaderboard/full-35-combined/report.html", report_path),
                    ("asr-leaderboard-run-manifest.json", run_manifest),
                    ("asr-leaderboard-refresh-commands.sh", refresh_commands),
                    ("asr-leaderboard-runtime-status.json", runtime_status),
                ]
            )
        )
        + "\n",
        encoding="utf-8",
    )
    artifact_index.write_text(
        json.dumps(
            {
                "status": "complete",
                "total_results": 4,
                "model_count": 2,
                "category_count": 2,
                "expected_cases_per_model": 2,
                "artifacts": [
                    artifact_index_record(
                        "runs/asr-leaderboard/full-35-combined/results.jsonl",
                        results_path,
                    ),
                    artifact_index_record(
                        "runs/asr-leaderboard/full-35-combined/report.html",
                        report_path,
                    ),
                    artifact_index_record(
                        "docs/asr-leaderboard-run-manifest.json",
                        run_manifest,
                    ),
                    artifact_index_record(
                        "docs/asr-leaderboard-manifest-validation.json",
                        manifest_validation,
                    ),
                    artifact_index_record(
                        "docs/asr-seed-manifest-validation.json",
                        seed_manifest_validation,
                    ),
                    artifact_index_record(
                        "docs/asr-leaderboard-next-runs.json",
                        next_runs,
                    ),
                    artifact_index_record(
                        "docs/asr-leaderboard-refresh-commands.sh",
                        refresh_commands,
                    ),
                    artifact_index_record(
                        "docs/asr-leaderboard-runtime-status.json",
                        runtime_status,
                    ),
                    artifact_index_record(
                        "docs/asr-leaderboard-hosted-manifest.json",
                        hosted_manifest,
                        digest_status="deferred_circular_reference",
                    ),
                    artifact_index_record(
                        "docs/asr-leaderboard-artifacts.json",
                        artifact_index,
                        digest_status="deferred_circular_reference",
                    ),
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    summary.write_text(
        json.dumps(
            {
                "results_path": "runs/asr-leaderboard/full-35-combined/results.jsonl",
                "report_path": "runs/asr-leaderboard/full-35-combined/report.html",
                "run_manifest_path": "docs/asr-leaderboard-run-manifest.json",
                "refresh_commands_path": "docs/asr-leaderboard-refresh-commands.sh",
                "manifest_validation_path": "docs/asr-leaderboard-manifest-validation.json",
                "seed_manifest_validation_path": "docs/asr-seed-manifest-validation.json",
                "next_runs_path": "docs/asr-leaderboard-next-runs.json",
                "hosted_manifest_path": "docs/asr-leaderboard-hosted-manifest.json",
                "artifact_index_path": "docs/asr-leaderboard-artifacts.json",
                "runtime_status_path": "docs/asr-leaderboard-runtime-status.json",
                "source_result_paths": [
                    "runs/asr-leaderboard/model-a/judge-report/results.jsonl",
                    "runs/asr-leaderboard/model-b/judge-report/results.jsonl",
                ],
                "output_artifacts": [
                    {
                        "path": "runs/asr-leaderboard/full-35-combined/results.jsonl",
                        "purpose": "Combined ASR judge results used by the generated page and report.",
                    },
                    {
                        "path": "runs/asr-leaderboard/full-35-combined/report.html",
                        "purpose": "Local combined HTML report with per-case judge details.",
                    },
                    {
                        "path": "docs/asr-leaderboard-summary.json",
                        "purpose": "Machine-readable leaderboard summary and reproducible refresh workflow.",
                    },
                    {
                        "path": "docs/asr-leaderboard-run-manifest.json",
                        "purpose": "Committed source result manifest for manifest-based refreshes.",
                    },
                    {
                        "path": "docs/asr-leaderboard-refresh-commands.sh",
                        "purpose": "Generated shell playbook for repeatable ASR leaderboard refreshes.",
                    },
                        {
                            "path": "docs/asr-leaderboard-next-runs.json",
                            "purpose": "Machine-readable next-refresh plan for missing ASR model/category cells.",
                        },
                        {
                            "path": "docs/asr-leaderboard-artifacts.json",
                            "purpose": "Single machine-readable index for the ASR leaderboard artifact bundle.",
                        },
                        {
                            "path": "docs/asr-leaderboard-runtime-status.json",
                            "purpose": "Machine-readable MLX ASR and Gemini readiness status for refresh automation.",
                        },
                    ],
                "refresh_workflow": update_module._refresh_workflow([]),
                "total_results": 4,
                "model_count": 2,
                "category_count": 2,
                "expected_cases_per_model": 2,
                "models": [
                    {"model": "mlx-community/model-a"},
                    {"model": "mlx-community/model-b"},
                ],
                "categories": [
                    {"category": "transcription_accuracy_wer"},
                    {"category": "numeric_unit_integrity"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    page.write_text(
        "\n".join(
            [
                "<!doctype html><html><body>",
                "Open Audio Judge ASR Leaderboard",
                "<!-- ASR_LEADERBOARD_GENERATED_START -->",
                "Verified Leaderboard Results",
                "Category Breakdown",
                "Generated Refresh Workflow",
                "Generated Artifacts",
                "4 judged transcripts",
                "mlx-community/model-a",
                "mlx-community/model-b",
                "transcription_accuracy_wer",
                "numeric_unit_integrity",
                ".venv/bin/python scripts/validate_asr_seed_manifest.py --summary-out docs/asr-seed-manifest-validation.json",
                ".venv/bin/python scripts/synthesize_tts_cases.py --cases examples/asr_research_cases.jsonl --out runs/asr-research-audio --discard-text-sidecars --summary-out runs/asr-research-audio/summary.json",
                ".venv/bin/oaj autojudge-mlx-asr --python-bin .venv/bin/python --cases runs/asr-research-audio/tts_audio_cases.jsonl --model &lt;mlx-community/model-id&gt; --judge-provider gemini --judge-samples 3 --out runs/asr-leaderboard/&lt;run-name&gt;",
                ".venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py",
                ".venv/bin/python scripts/check_asr_leaderboard_page.py",
                ".venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py --hosted-dir-from-env",
                ".venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py --check-only --hosted-dir-from-env",
                "runs/asr-leaderboard/full-35-combined/results.jsonl",
                "Combined ASR judge results used by the generated page and report.",
                "runs/asr-leaderboard/full-35-combined/report.html",
                "Local combined HTML report with per-case judge details.",
                "docs/asr-leaderboard-summary.json",
                "Machine-readable leaderboard summary and reproducible refresh workflow.",
                "docs/asr-leaderboard-run-manifest.json",
                "Committed source result manifest for manifest-based refreshes.",
                "docs/asr-leaderboard-refresh-commands.sh",
                "Generated shell playbook for repeatable ASR leaderboard refreshes.",
                "docs/asr-leaderboard-next-runs.json",
                "Machine-readable next-refresh plan for missing ASR model/category cells.",
                "docs/asr-leaderboard-artifacts.json",
                "Single machine-readable index for the ASR leaderboard artifact bundle.",
                "docs/asr-leaderboard-runtime-status.json",
                "Machine-readable MLX ASR and Gemini readiness status for refresh automation.",
                "<!-- ASR_LEADERBOARD_GENERATED_END -->",
                "</body></html>",
            ]
        ),
        encoding="utf-8",
    )

    validation = check_module.check_asr_leaderboard_page(
        page,
        summary_path=summary,
        artifact_root=hosted,
        path_maps=[
            ("docs/", ""),
            ("runs/asr-leaderboard/", "asr-leaderboard/"),
        ],
        allow_missing_source_results=True,
    )

    assert validation["status"] == "complete"


def test_check_asr_leaderboard_page_rejects_stale_run_manifest(tmp_path: Path) -> None:
    check_module = load_check_module()
    manifest = tmp_path / "run-manifest.json"
    summary_path = tmp_path / "summary.json"
    manifest.write_text(
        json.dumps(
            run_manifest_record(
                [
                    (
                        "runs/asr-leaderboard/old-run/judge-report/results.jsonl",
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
                                category="numeric_unit_integrity",
                                score=80,
                                label="accurate",
                            ),
                        ],
                    ),
                ],
                expected_cases_per_model=2,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "run_manifest_path": str(manifest),
        "source_result_paths": [
            "runs/asr-leaderboard/new-run/judge-report/results.jsonl",
        ],
        "expected_cases_per_model": 2,
        "models": [
            {
                "model": "mlx-community/model-a",
                "result_count": 2,
                "ok_count": 2,
            }
        ],
    }

    with pytest.raises(ValueError, match="result_paths do not match"):
        check_module._validate_run_manifest_artifact(
            summary,
            summary_path=summary_path,
            artifact_root=tmp_path,
            path_maps=[],
        )


def test_check_asr_leaderboard_page_rejects_incomplete_validation_artifact(
    tmp_path: Path,
) -> None:
    update_module = load_script_module()
    check_module = load_check_module()
    page = tmp_path / "demo.html"
    summary = tmp_path / "summary.json"
    results_path = tmp_path / "results.jsonl"
    run_manifest = tmp_path / "run-manifest.json"
    manifest_validation = tmp_path / "manifest-validation.json"
    seed_manifest_validation = tmp_path / "seed-manifest-validation.json"
    refresh_commands = tmp_path / "refresh-commands.sh"
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
    results_path.with_name("report.html").write_text("<html></html>\n", encoding="utf-8")
    results = update_module.load_results_jsonl(results_path)
    page.write_text(
        "<!doctype html><html lang=\"en\"><body>\n"
        "<h1>Open Audio Judge ASR Leaderboard</h1>\n"
        f"{update_module.START_MARKER}\n"
        "old generated content\n"
        f"{update_module.END_MARKER}\n"
        "</body></html>\n",
        encoding="utf-8",
    )
    update_module.replace_generated_block(
        page,
        update_module.render_generated_sections(
            results,
            results_path=results_path,
            expected_cases_per_model=2,
        ),
    )
    update_module.write_summary_artifact(
        results,
        summary,
        results_path=results_path,
        expected_cases_per_model=2,
    )
    model_a_records = [record for record in records if record["metadata"]["candidate_model"] == "mlx-community/model-a"]
    model_b_records = [record for record in records if record["metadata"]["candidate_model"] == "mlx-community/model-b"]
    run_manifest.write_text(
        json.dumps(
            run_manifest_record(
                [
                    ("runs/asr-leaderboard/model-a/judge-report/results.jsonl", model_a_records),
                    ("runs/asr-leaderboard/model-b/judge-report/results.jsonl", model_b_records),
                ],
                expected_cases_per_model=2,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_validation.write_text(
        json.dumps(
            {
                "status": "incomplete",
                "total_results": 4,
                "model_count": 2,
                "category_count": 2,
                "expected_cases_per_model": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    seed_manifest_validation.write_text(
        json.dumps({"status": "complete"}) + "\n",
        encoding="utf-8",
    )
    refresh_commands.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    summary_data = json.loads(summary.read_text(encoding="utf-8"))
    summary_data["run_manifest_path"] = str(run_manifest)
    summary_data["refresh_commands_path"] = str(refresh_commands)
    summary_data["manifest_validation_path"] = str(manifest_validation)
    summary_data["seed_manifest_validation_path"] = str(seed_manifest_validation)
    summary.write_text(json.dumps(summary_data), encoding="utf-8")

    with pytest.raises(ValueError, match="status must be complete"):
        check_module.check_asr_leaderboard_page(page, summary_path=summary)


def test_check_asr_leaderboard_page_rejects_missing_summary_artifacts(tmp_path: Path) -> None:
    update_module = load_script_module()
    check_module = load_check_module()
    page = tmp_path / "demo.html"
    summary = tmp_path / "summary.json"
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
    ]
    results_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    results = update_module.load_results_jsonl(results_path)
    page.write_text(
        "<!doctype html><html lang=\"en\"><body>\n"
        "<h1>Open Audio Judge ASR Leaderboard</h1>\n"
        f"{update_module.START_MARKER}\n"
        "old generated content\n"
        f"{update_module.END_MARKER}\n"
        "</body></html>\n",
        encoding="utf-8",
    )
    update_module.replace_generated_block(
        page,
        update_module.render_generated_sections(
            results,
            results_path=results_path,
            expected_cases_per_model=2,
        ),
    )
    update_module.write_summary_artifact(
        results,
        summary,
        results_path=results_path,
        expected_cases_per_model=2,
    )

    with pytest.raises(ValueError, match="references missing ASR artifact"):
        check_module.check_asr_leaderboard_page(page, summary_path=summary)


def test_check_asr_leaderboard_page_rejects_missing_output_artifact(tmp_path: Path) -> None:
    update_module = load_script_module()
    check_module = load_check_module()
    page = tmp_path / "demo.html"
    summary = tmp_path / "summary.json"
    results_path = tmp_path / "results.jsonl"
    run_manifest = tmp_path / "run-manifest.json"
    manifest_validation = tmp_path / "manifest-validation.json"
    seed_manifest_validation = tmp_path / "seed-manifest-validation.json"
    refresh_commands = tmp_path / "refresh-commands.sh"
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
    results_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    results_path.with_name("report.html").write_text("<html></html>\n", encoding="utf-8")
    results = update_module.load_results_jsonl(results_path)
    page.write_text(
        "<!doctype html><html lang=\"en\"><body>\n"
        "<h1>Open Audio Judge ASR Leaderboard</h1>\n"
        f"{update_module.START_MARKER}\n"
        "old generated content\n"
        f"{update_module.END_MARKER}\n"
        "</body></html>\n",
        encoding="utf-8",
    )
    update_module.replace_generated_block(
        page,
        update_module.render_generated_sections(
            results,
            results_path=results_path,
            expected_cases_per_model=2,
        ),
    )
    update_module.write_summary_artifact(
        results,
        summary,
        results_path=results_path,
        expected_cases_per_model=2,
    )
    run_manifest.write_text(
        json.dumps(
            run_manifest_record(
                [
                    ("runs/asr-leaderboard/model-a/judge-report/results.jsonl", records),
                ],
                expected_cases_per_model=2,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_validation.write_text(
        json.dumps(
            {
                "status": "complete",
                "total_results": 2,
                "model_count": 1,
                "category_count": 2,
                "expected_cases_per_model": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    seed_manifest_validation.write_text(
        json.dumps({"status": "complete"}) + "\n",
        encoding="utf-8",
    )
    refresh_commands.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    summary_data = json.loads(summary.read_text(encoding="utf-8"))
    summary_data["run_manifest_path"] = str(run_manifest)
    summary_data["refresh_commands_path"] = str(refresh_commands)
    summary_data["manifest_validation_path"] = str(manifest_validation)
    summary_data["seed_manifest_validation_path"] = str(seed_manifest_validation)
    summary_data["output_artifacts"].append(
        {
            "path": str(tmp_path / "missing-output.json"),
            "purpose": "Regression-test missing generated artifact.",
        }
    )
    summary.write_text(json.dumps(summary_data), encoding="utf-8")

    with pytest.raises(ValueError, match="output_artifacts reference missing ASR artifact"):
        check_module.check_asr_leaderboard_page(page, summary_path=summary)


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


def test_refresh_asr_leaderboard_artifacts_rejects_duplicate_run_manifest_paths(tmp_path: Path) -> None:
    refresh_module = load_refresh_module()
    results_path = tmp_path / "run-a" / "judge-report" / "results.jsonl"
    manifest = tmp_path / "manifest.json"
    results_path.parent.mkdir(parents=True)
    results_path.write_text("", encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "runs": [
                    {"run_name": "run-a", "results_path": str(tmp_path / "run-a")},
                    {"run_name": "run-a-copy", "results_path": str(results_path)},
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate result path"):
        refresh_module._result_paths_from_run_manifest(manifest)


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
    run = data["runs"][0]
    assert run["bytes"] == results_path.stat().st_size
    assert run["sha256"] == file_sha256(results_path)
    assert data["runs"] == [
        {
            "run_name": "run-a",
            "model": "mlx-community/model-a",
            "results_path": str(results_path),
            "bytes": results_path.stat().st_size,
            "sha256": file_sha256(results_path),
            "result_count": 2,
            "ok_count": 2,
            "category_counts": {
                "numeric_unit_integrity": 1,
                "transcription_accuracy_wer": 1,
            },
        }
    ]


def test_write_run_manifest_artifact_rejects_duplicate_sources(tmp_path: Path) -> None:
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
    ]
    results_path.parent.mkdir(parents=True)
    results_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate result path"):
        refresh_module.write_run_manifest_artifact(
            [tmp_path / "run-a", results_path],
            manifest,
            expected_cases_per_model=1,
        )


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
    check = validation["result_file_checks"][0]
    assert check["path"] == str(results_path)
    assert check["declared_model"] == "mlx-community/model-a"
    assert check["actual_models"] == ["mlx-community/model-a"]
    assert check["model_match"] is True
    assert check["declared_bytes"] is None
    assert check["actual_bytes"] == results_path.stat().st_size
    assert check["declared_sha256"] is None
    assert check["actual_sha256"] == file_sha256(results_path)
    assert check["digest_match"] is True


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
