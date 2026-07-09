import importlib.util
import json
import sys
from pathlib import Path


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
    assert "docs/asr-leaderboard-run-manifest.json" in html
    assert "reproducible refresh workflow" in html


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
    results = module.load_results_jsonl(results_path)

    module.write_summary_artifact(
        results,
        summary_path,
        results_path=results_path,
        expected_cases_per_model=2,
        source_result_paths=[tmp_path / "model-a" / "judge-report" / "results.jsonl"],
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["total_results"] == 4
    assert summary["model_count"] == 2
    assert summary["category_count"] == 2
    assert summary["total_gemini_judge_samples"] == 12
    assert summary["source_result_paths"] == [
        str(tmp_path / "model-a" / "judge-report" / "results.jsonl")
    ]
    assert summary["run_manifest_path"] == "docs/asr-leaderboard-run-manifest.json"
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
        str(tmp_path / "model-a" / "judge-report" / "results.jsonl"),
    ]
    assert summary["refresh_workflow"]["manifest_refresh_command"] == [
        ".venv/bin/python",
        "scripts/refresh_asr_leaderboard_artifacts.py",
    ]
    assert "secret" in summary["refresh_workflow"]["secret_handling"].lower()
    assert summary["models"][0]["model"] == "mlx-community/model-a"
    assert summary["models"][0]["average_score"] == 90
    assert summary["models"][0]["labels"] == {"accurate": 2}
    assert summary["categories"][0]["category"] == "transcription_accuracy_wer"
    assert summary["categories"][1]["category"] == "numeric_unit_integrity"


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
