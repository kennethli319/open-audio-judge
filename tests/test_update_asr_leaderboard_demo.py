import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path("scripts/update_asr_leaderboard_demo.py")


def load_script_module():
    spec = importlib.util.spec_from_file_location("update_asr_leaderboard_demo", SCRIPT)
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
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["total_results"] == 4
    assert summary["model_count"] == 2
    assert summary["category_count"] == 2
    assert summary["total_gemini_judge_samples"] == 12
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
