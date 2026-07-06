import json
from pathlib import Path

from typer.testing import CliRunner

import open_audio_judge.cli as cli
from open_audio_judge.cli import app


def test_autojudge_hf_asr_cli_writes_candidate_cases_and_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF")
    cases = tmp_path / "cases.jsonl"
    out = tmp_path / "out"
    cases.write_text(
        json.dumps(
            {
                "id": "amount-case",
                "task": "asr_error",
                "audio_path": str(audio),
                "reference_text": "Transfer fifteen dollars.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        cli,
        "transcribe_cases_with_hf_asr",
        lambda loaded_cases, model, device: [
            loaded_cases[0].model_copy(
                update={
                    "candidate_text": "Transfer fifty dollars.",
                    "metadata": {"candidate_model": model, "candidate_device": device},
                }
            )
        ],
    )

    result = CliRunner().invoke(
        app,
        [
            "autojudge-hf-asr",
            "--cases",
            str(cases),
            "--model",
            "openai/whisper-tiny",
            "--judge-provider",
            "mock",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0
    assert (out / "candidate_cases.jsonl").exists()
    assert (out / "model_summary.json").exists()
    assert (out / "judge-report" / "results.jsonl").exists()
    assert (out / "judge-report" / "report.html").exists()
    written_case = json.loads((out / "candidate_cases.jsonl").read_text(encoding="utf-8"))
    assert written_case["candidate_text"] == "Transfer fifty dollars."
    assert "AutoJudged 1 Hugging Face ASR cases" in result.output


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
    assert written_summary["reference_text_sha256"] == {
        "duplicate_cases": 0,
        "duplicate_hashes": 0,
        "unique": 1,
    }
    assert "approve" not in summary.read_text(encoding="utf-8")
    written_case = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert "reference_text_sha256" in written_case["metadata"]
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


def test_build_tts_cases_cli_can_prioritize_slice_coverage(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    out = tmp_path / "cases.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {
                    "id": "json-one",
                    "category": "structured_output",
                    "task": "json_decision",
                    "ideal_answer": '{"decision":"approve"}',
                },
                {
                    "id": "json-two",
                    "category": "structured_output",
                    "task": "json_decision",
                    "ideal_answer": '{"decision":"deny"}',
                },
                {
                    "id": "time-one",
                    "category": "instruction_constraints",
                    "task": "read_time",
                    "ideal_answer": "Meet at 09:45.",
                },
            ]
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
            "--limit",
            "2",
            "--prioritize-slice-coverage",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0
    written_cases = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [case["id"] for case in written_cases] == [
        "tts-evalset-json-one",
        "tts-evalset-time-one",
    ]


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
