import json
import subprocess
from pathlib import Path

import pytest

from open_audio_judge.mlx_asr import (
    MlxAsrConfig,
    check_mlx_asr_runtime,
    transcribe_case_with_mlx_asr,
    transcribe_cases_with_mlx_asr,
    write_mlx_asr_summary_json,
)
from open_audio_judge.models import EvaluationCase


def test_transcribe_cases_with_mlx_asr_adds_candidate_metadata(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF")
    case = EvaluationCase(
        id="case-1",
        task="asr_error",
        audio_path="sample.wav",
        reference_text="Transfer fifteen dollars.",
    )

    def fake_runner(command, **kwargs):
        assert command[:6] == [
            "python3",
            "-m",
            "mlx_audio.stt.generate",
            "--model",
            "mlx-community/whisper-large-v3-turbo-asr-fp16",
            "--audio",
        ]
        assert Path(command[6]) == audio
        assert command[7] == "--output-path"
        assert command[9:11] == ["--format", "txt"]
        assert kwargs["capture_output"] is True
        return subprocess.CompletedProcess(command, 0, stdout='{"text":"Transfer fifty dollars."}')

    cases = transcribe_cases_with_mlx_asr(
        [case],
        config=MlxAsrConfig(
            model="mlx-community/whisper-large-v3-turbo-asr-fp16",
            python_bin="python3",
        ),
        base_dir=tmp_path,
        runner=fake_runner,
    )

    assert cases[0].candidate_text == "Transfer fifty dollars."
    assert cases[0].metadata["candidate_model"] == "mlx-community/whisper-large-v3-turbo-asr-fp16"
    assert cases[0].metadata["candidate_transcriber"] == "mlx-audio-stt"


def test_transcribe_case_with_mlx_asr_reads_output_path(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF")
    case = EvaluationCase(id="case-1", task="asr_error", audio_path=str(audio))

    def fake_runner(command, **kwargs):
        output_path = Path(command[command.index("--output-path") + 1])
        output_path.with_suffix(".txt").write_text("file transcript\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="stdout transcript", stderr="")

    transcript = transcribe_case_with_mlx_asr(
        case,
        config=MlxAsrConfig(model="model", python_bin="python3"),
        runner=fake_runner,
    )

    assert transcript.text == "file transcript"


def test_transcribe_case_with_mlx_asr_can_parse_last_text_line(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF")
    case = EvaluationCase(id="case-1", task="asr_error", audio_path=str(audio))

    transcript = transcribe_case_with_mlx_asr(
        case,
        config=MlxAsrConfig(model="model", python_bin="python3"),
        runner=lambda command, **_: subprocess.CompletedProcess(
            command,
            0,
            stdout="Loading model\nText: hello world\n",
            stderr="",
        ),
    )

    assert transcript.text == "hello world"


def test_transcribe_case_with_mlx_asr_requires_audio_path() -> None:
    case = EvaluationCase(id="case-1", task="asr_error")

    with pytest.raises(ValueError, match="requires a local audio_path"):
        transcribe_case_with_mlx_asr(
            case,
            config=MlxAsrConfig(model="model"),
        )


def test_transcribe_case_with_mlx_asr_reports_command_failure(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF")
    case = EvaluationCase(id="case-1", task="asr_error", audio_path=str(audio))

    def fake_runner(command, **kwargs):
        raise subprocess.CalledProcessError(
            1,
            command,
            stderr="ModuleNotFoundError: No module named 'mlx_audio'",
        )

    with pytest.raises(RuntimeError, match="No module named 'mlx_audio'"):
        transcribe_case_with_mlx_asr(
            case,
            config=MlxAsrConfig(model="model"),
            runner=fake_runner,
        )


def test_check_mlx_asr_runtime_imports_configured_module() -> None:
    calls = []

    def fake_runner(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    check_mlx_asr_runtime(
        MlxAsrConfig(
            model="model",
            python_bin=".venv/bin/python",
            module="mlx_audio.stt.generate",
        ),
        runner=fake_runner,
    )

    assert calls == [
        (
            [
                ".venv/bin/python",
                "-c",
                (
                    "import importlib.util, sys; "
                    "module = sys.argv[1]; "
                    "sys.exit(0 if importlib.util.find_spec(module) else 1)"
                ),
                "mlx_audio.stt.generate",
            ],
            {"check": True, "capture_output": True, "text": True},
        )
    ]


def test_check_mlx_asr_runtime_reports_missing_module() -> None:
    def fake_runner(command, **kwargs):
        raise subprocess.CalledProcessError(
            1,
            command,
            stderr="ModuleNotFoundError: No module named 'mlx_audio'",
        )

    with pytest.raises(RuntimeError, match="cannot import 'mlx_audio.stt.generate'"):
        check_mlx_asr_runtime(
            MlxAsrConfig(model="model", python_bin=".venv/bin/python"),
            runner=fake_runner,
        )


def test_write_mlx_asr_summary_json_counts_categories(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    case = EvaluationCase(
        id="case-1",
        task="asr_error",
        candidate_text="hello",
        metadata={
            "eval_category": "semantic_error_sensitivity",
            "asr_slice": "negation_scope",
            "language": "en",
        },
    )

    write_mlx_asr_summary_json(
        [case],
        summary,
        source_cases=Path("examples/asr_research_cases.jsonl"),
        model="mlx-community/whisper-large-v3-turbo-asr-fp16",
    )

    data = json.loads(summary.read_text(encoding="utf-8"))
    assert data["candidate_transcriber"] == "mlx-audio-stt"
    assert data["cases_with_candidate_text"] == 1
    assert data["by_eval_category"] == {"semantic_error_sensitivity": 1}
