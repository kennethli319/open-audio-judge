import json
from pathlib import Path

from typer.testing import CliRunner

import open_audio_judge.cli as cli
from open_audio_judge.cli import app
from open_audio_judge.local_tts import LocalTtsBatchResult, LocalTtsFailure


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


def test_autojudge_local_tts_cli_writes_synthesized_cases_and_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cases = tmp_path / "cases.jsonl"
    out = tmp_path / "out"
    cases.write_text(
        json.dumps(
            {
                "id": "tts-case",
                "task": "tts_naturalness",
                "reference_text": "Read this naturally.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_synthesize_cases(loaded_cases, out_dir, config, continue_on_error=False):
        assert continue_on_error is False
        assert config.timeout_seconds == 3.5
        audio_dir = out_dir / "audio"
        audio_dir.mkdir(parents=True)
        (audio_dir / "tts-case.wav").write_bytes(b"RIFF")
        return LocalTtsBatchResult(
            cases=[
                loaded_cases[0].model_copy(
                    update={
                        "id": "tts-case-local-tts",
                        "audio_path": "audio/tts-case.wav",
                        "metadata": {
                            "synthesis_provider": config.synthesis_provider,
                            "synthesis_model": config.model,
                            "synthesis_voice": config.voice,
                            "synthesis_audio_format": config.audio_format,
                        },
                    }
                )
            ],
            failures=[],
        )

    monkeypatch.setattr(cli, "synthesize_cases_with_local_tts_batch", fake_synthesize_cases)

    result = CliRunner().invoke(
        app,
        [
            "autojudge-local-tts",
            "--cases",
            str(cases),
            "--model",
            "mlx-community/chatterbox-turbo-6bit",
            "--synthesis-provider",
            "local_test_tts",
            "--tts-timeout-seconds",
            "3.5",
            "--judge-provider",
            "mock",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0
    assert (out / "synthesis" / "tts_audio_cases.jsonl").exists()
    assert (out / "model_summary.json").exists()
    assert (out / "judge-report" / "results.jsonl").exists()
    assert (out / "judge-report" / "report.html").exists()
    written_case = json.loads(
        (out / "synthesis" / "tts_audio_cases.jsonl").read_text(encoding="utf-8")
    )
    assert written_case["audio_path"] == "audio/tts-case.wav"
    assert written_case["metadata"]["synthesis_provider"] == "local_test_tts"
    summary = json.loads((out / "model_summary.json").read_text(encoding="utf-8"))
    assert summary["candidate_generator"] == "local_test_tts"
    assert "AutoJudged 1 local TTS cases" in result.output


def test_autojudge_local_tts_cli_can_build_cases_from_evalset(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source.jsonl"
    out = tmp_path / "out"
    source.write_text(
        json.dumps(
            {
                "id": "row-001",
                "category": "instruction_constraints",
                "task": "read_time",
                "ideal_answer": "Meet me at 09:45 tomorrow.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_synthesize_cases(loaded_cases, out_dir, config, continue_on_error=False):
        audio_dir = out_dir / "audio"
        audio_dir.mkdir(parents=True)
        (audio_dir / "tts-evalset-row-001.wav").write_bytes(b"RIFF")
        return LocalTtsBatchResult(
            cases=[
                loaded_cases[0].model_copy(
                    update={
                        "id": f"{loaded_cases[0].id}-local-tts",
                        "audio_path": "audio/tts-evalset-row-001.wav",
                        "metadata": {
                            **loaded_cases[0].metadata,
                            "synthesis_provider": config.synthesis_provider,
                            "synthesis_model": config.model,
                            "synthesis_voice": config.voice,
                            "synthesis_audio_format": config.audio_format,
                        },
                    }
                )
            ],
            failures=[],
        )

    monkeypatch.setattr(cli, "synthesize_cases_with_local_tts_batch", fake_synthesize_cases)

    result = CliRunner().invoke(
        app,
        [
            "autojudge-local-tts",
            "--evalset-source",
            str(source),
            "--evalset-source-name",
            "evalset",
            "--evalset-hash-source-ids",
            "--limit",
            "1",
            "--judge-provider",
            "mock",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0
    assert (out / "evalset" / "tts_cases.jsonl").exists()
    assert (out / "evalset" / "summary.json").exists()
    assert (out / "synthesis" / "tts_audio_cases.jsonl").exists()
    assert (out / "judge-report" / "results.jsonl").exists()
    evalset_case = json.loads((out / "evalset" / "tts_cases.jsonl").read_text(encoding="utf-8"))
    assert evalset_case["reference_text"] == "Meet me at 09:45 tomorrow."
    assert "source_id_sha256" in evalset_case["metadata"]
    summary = json.loads((out / "model_summary.json").read_text(encoding="utf-8"))
    assert summary["source_cases"].endswith("evalset/tts_cases.jsonl")
    assert "Evalset:" in result.output


def test_autojudge_local_tts_cli_can_skip_failed_synthesis(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cases = tmp_path / "cases.jsonl"
    out = tmp_path / "out"
    cases.write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {
                    "id": "tts-ok",
                    "task": "tts_naturalness",
                    "reference_text": "Read this naturally.",
                    "metadata": {"tts_slice": "general"},
                },
                {
                    "id": "tts-fail",
                    "task": "tts_naturalness",
                    "reference_text": "This one fails.",
                    "metadata": {
                        "language": "en-US",
                        "source_category": "instruction_constraints",
                        "tts_slice": "numbers",
                    },
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_synthesize_cases(loaded_cases, out_dir, config, continue_on_error=False):
        assert continue_on_error is True
        audio_dir = out_dir / "audio"
        audio_dir.mkdir(parents=True)
        (audio_dir / "tts-ok.wav").write_bytes(b"RIFF")
        return LocalTtsBatchResult(
            cases=[
                loaded_cases[0].model_copy(
                    update={
                        "id": "tts-ok-local-tts",
                        "audio_path": "audio/tts-ok.wav",
                        "metadata": {
                            **loaded_cases[0].metadata,
                            "synthesis_provider": config.synthesis_provider,
                            "synthesis_model": config.model,
                            "synthesis_voice": config.voice,
                            "synthesis_audio_format": config.audio_format,
                        },
                    }
                )
            ],
            failures=[
                LocalTtsFailure(
                    case_id="tts-fail",
                    error_type="command_failed",
                    message="voice unavailable",
                    metadata=loaded_cases[1].metadata,
                )
            ],
        )

    monkeypatch.setattr(cli, "synthesize_cases_with_local_tts_batch", fake_synthesize_cases)

    result = CliRunner().invoke(
        app,
        [
            "autojudge-local-tts",
            "--cases",
            str(cases),
            "--skip-failed-synthesis",
            "--judge-provider",
            "mock",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0
    assert (out / "judge-report" / "results.jsonl").exists()
    failure_record = json.loads(
        (out / "synthesis" / "synthesis_failures.jsonl").read_text(encoding="utf-8")
    )
    assert failure_record["case_id"] == "tts-fail"
    assert failure_record["message"] == "voice unavailable"
    summary = json.loads((out / "model_summary.json").read_text(encoding="utf-8"))
    assert summary["total_cases"] == 1
    assert summary["attempted_source_cases"] == 2
    assert summary["synthesized_case_count"] == 1
    assert summary["synthesis_success_rate"] == 0.5
    assert summary["synthesis_failure_count"] == 1
    assert summary["synthesis_failures_by_error_type"] == {"command_failed": 1}
    assert summary["synthesis_failures_by_tts_slice"] == {"numbers": 1}
    assert summary["synthesis_failures_by_source_category"] == {
        "instruction_constraints": 1
    }
    assert summary["synthesis_failures_by_language"] == {"en-US": 1}
    assert "Failures:" in result.output


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
