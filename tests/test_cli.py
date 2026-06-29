import json
from pathlib import Path

from typer.testing import CliRunner

from open_audio_judge.cli import app


def test_build_tts_cases_writes_metadata_summary(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    out = tmp_path / "cases.jsonl"
    summary = tmp_path / "summary.json"
    source.write_text(
        json.dumps(
            {
                "id": "ome_0002",
                "category": "structured_output",
                "task": "json_decision",
                "turns": [{"role": "user", "content": "Return JSON."}],
                "ideal_answer": '{"decision":"approve"}',
                "metadata": {"tags": ["json", "format"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "build-tts-cases",
            "--source",
            str(source),
            "--source-name",
            "ome",
            "--summary-out",
            str(summary),
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0
    assert out.exists()
    assert summary.exists()
    written_summary = json.loads(summary.read_text(encoding="utf-8"))
    assert written_summary["total_cases"] == 1
    assert written_summary["by_slice"] == {"code_like": 1}
    assert "approve" not in summary.read_text(encoding="utf-8")
    written_case = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert "source_task" not in written_case["metadata"]


def test_build_tts_cases_cli_can_include_source_task(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    out = tmp_path / "cases.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "ome_0002",
                "category": "structured_output",
                "task": "json_decision",
                "ideal_answer": '{"decision":"approve"}',
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "build-tts-cases",
            "--source",
            str(source),
            "--include-source-task",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0
    written_case = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert written_case["metadata"]["source_task"] == "json_decision"


def test_build_tts_cases_can_hash_source_ids(tmp_path: Path) -> None:
    private_source_id = "private-user-request-row-2026-06-29"
    source = tmp_path / "source.jsonl"
    out = tmp_path / "cases.jsonl"
    summary = tmp_path / "summary.json"
    source.write_text(
        json.dumps(
            {
                "id": private_source_id,
                "category": "structured_output",
                "task": "json_decision",
                "ideal_answer": '{"decision":"approve"}',
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "build-tts-cases",
            "--source",
            str(source),
            "--source-name",
            "ome",
            "--hash-source-ids",
            "--summary-out",
            str(summary),
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0
    assert private_source_id not in out.read_text(encoding="utf-8")
    assert private_source_id not in summary.read_text(encoding="utf-8")
    written_summary = json.loads(summary.read_text(encoding="utf-8"))
    assert written_summary["example_source_ids_by_slice"] == {}
    written = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert written["metadata"]["source_id"].startswith("source-")
    assert "source_id_sha256" in written["metadata"]


def test_build_tts_cases_cli_can_include_hashed_summary_source_examples(
    tmp_path: Path,
) -> None:
    private_source_id = "private-user-request-row-2026-06-29"
    source = tmp_path / "source.jsonl"
    out = tmp_path / "cases.jsonl"
    summary = tmp_path / "summary.json"
    source.write_text(
        json.dumps(
            {
                "id": private_source_id,
                "category": "structured_output",
                "task": "json_decision",
                "ideal_answer": '{"decision":"approve"}',
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "build-tts-cases",
            "--source",
            str(source),
            "--source-name",
            "ome",
            "--hash-source-ids",
            "--summary-source-examples",
            "--summary-out",
            str(summary),
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0
    written_summary = json.loads(summary.read_text(encoding="utf-8"))
    example_ids = written_summary["example_source_ids_by_slice"]["code_like"]
    assert example_ids[0].startswith("source-")
    assert private_source_id not in summary.read_text(encoding="utf-8")


def test_build_tts_cases_cli_can_omit_summary_source_examples(tmp_path: Path) -> None:
    private_source_id = "private-user-request-row-2026-06-29"
    source = tmp_path / "source.jsonl"
    out = tmp_path / "cases.jsonl"
    summary = tmp_path / "summary.json"
    source.write_text(
        json.dumps(
            {
                "id": private_source_id,
                "category": "structured_output",
                "task": "json_decision",
                "ideal_answer": '{"decision":"approve"}',
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "build-tts-cases",
            "--source",
            str(source),
            "--summary-out",
            str(summary),
            "--no-summary-source-examples",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0
    written_summary = json.loads(summary.read_text(encoding="utf-8"))
    assert written_summary["example_source_ids_by_slice"] == {}
    assert private_source_id not in summary.read_text(encoding="utf-8")
    assert private_source_id in out.read_text(encoding="utf-8")
