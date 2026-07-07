import json
import subprocess
from pathlib import Path

import pytest

from open_audio_judge.local_tts import (
    LocalTtsConfig,
    LocalTtsFailure,
    _output_path_from_stdout,
    _require_output_path_in_audio_dir,
    synthesize_cases_with_local_tts,
    synthesize_cases_with_local_tts_batch,
    write_local_tts_failures_jsonl,
    write_local_tts_summary_json,
)
from open_audio_judge.models import EvaluationCase


def test_synthesize_cases_with_local_tts_dry_run_writes_relative_audio_metadata(
    tmp_path: Path,
) -> None:
    case = EvaluationCase(
        id="tts sample/001",
        task="tts_naturalness",
        reference_text="Call me at 09:45.",
        metadata={"tts_slice": "dates_times", "requires_synthesis": True},
    )

    synthesized = synthesize_cases_with_local_tts(
        [case],
        out_dir=tmp_path / "synthesis",
        config=LocalTtsConfig(
            tts_bin=Path("/missing/local-tts-speak"),
            model="mlx-community/chatterbox-turbo-6bit",
            synthesis_provider="local_test_tts",
            dry_run=True,
            keep_text_sidecars=True,
        ),
    )

    assert synthesized[0].id == "tts sample/001-local-tts"
    assert synthesized[0].audio_path == "audio/tts-sample-001.wav"
    assert synthesized[0].reference_text == "Call me at 09:45."
    assert synthesized[0].metadata["sample_kind"] == "local_synthetic_tts"
    assert synthesized[0].metadata["synthesis_provider"] == "local_test_tts"
    assert synthesized[0].metadata["synthesis_model"] == "mlx-community/chatterbox-turbo-6bit"
    assert synthesized[0].metadata["source_case_id"] == "tts sample/001"
    assert synthesized[0].metadata["requires_synthesis"] is False
    assert synthesized[0].metadata["text_sidecar_path"] == "text/tts-sample-001.txt"


def test_synthesize_cases_with_local_tts_requires_reference_text(tmp_path: Path) -> None:
    case = EvaluationCase(id="tts-empty", task="tts_naturalness")

    with pytest.raises(ValueError, match="require reference_text"):
        synthesize_cases_with_local_tts(
            [case],
            out_dir=tmp_path / "synthesis",
            config=LocalTtsConfig(dry_run=True),
        )


def test_synthesize_cases_with_local_tts_reports_command_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tts_bin = tmp_path / "bin" / "local-tts-speak"
    tts_bin.parent.mkdir()
    tts_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    case = EvaluationCase(
        id="tts-fail",
        task="tts_naturalness",
        reference_text="This should fail clearly.",
    )

    def fake_run(command, check, capture_output, text, timeout):
        assert timeout is None
        raise subprocess.CalledProcessError(
            7,
            command,
            output="loading model\nretrying\nfailed after warmup\n",
            stderr="missing voice af_test\n",
        )

    monkeypatch.setattr("open_audio_judge.local_tts.subprocess.run", fake_run)

    with pytest.raises(RuntimeError) as excinfo:
        synthesize_cases_with_local_tts(
            [case],
            out_dir=tmp_path / "synthesis",
            config=LocalTtsConfig(tts_bin=tts_bin, voice="af_test"),
        )

    message = str(excinfo.value)
    assert "local TTS command failed with exit code 7: local-tts-speak" in message
    assert "stderr: missing voice af_test" in message
    assert "stdout: loading model | retrying | failed after warmup" in message
    assert not (tmp_path / "synthesis" / "text" / "tts-fail.txt").exists()


def test_synthesize_cases_with_local_tts_batch_can_continue_after_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tts_bin = tmp_path / "bin" / "local-tts-speak"
    tts_bin.parent.mkdir()
    tts_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    cases = [
        EvaluationCase(
            id="tts-ok",
            task="tts_naturalness",
            reference_text="This should synthesize.",
            metadata={"tts_slice": "general"},
        ),
        EvaluationCase(
            id="tts-fail",
            task="tts_naturalness",
            reference_text="This should fail clearly.",
            metadata={"tts_slice": "numbers"},
        ),
    ]

    def fake_run(command, check, capture_output, text, timeout):
        output_dir = Path(command[command.index("--output-dir") + 1])
        file_prefix = command[command.index("--file-prefix") + 1]
        if file_prefix == "tts-fail":
            raise subprocess.CalledProcessError(
                3,
                command,
                output="loading model\n",
                stderr="voice unavailable\n",
            )
        audio_path = output_dir / f"{file_prefix}.wav"
        audio_path.write_bytes(b"RIFF")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"output_path": str(audio_path)}) + "\n",
            stderr="",
        )

    monkeypatch.setattr("open_audio_judge.local_tts.subprocess.run", fake_run)

    result = synthesize_cases_with_local_tts_batch(
        cases,
        out_dir=tmp_path / "synthesis",
        config=LocalTtsConfig(tts_bin=tts_bin),
        continue_on_error=True,
    )

    assert [case.id for case in result.cases] == ["tts-ok-local-tts"]
    assert len(result.failures) == 1
    assert result.failures[0].case_id == "tts-fail"
    assert result.failures[0].error_type == "RuntimeError"
    assert "voice unavailable" in result.failures[0].message
    assert result.failures[0].metadata["tts_slice"] == "numbers"
    assert result.failures[0].metadata["synthesis_provider"] == "local_chatterbox"
    assert result.failures[0].metadata["synthesis_model"] == "mlx-community/chatterbox-turbo-6bit"
    assert result.failures[0].metadata["synthesis_voice"] == "af_heart"
    assert result.failures[0].metadata["synthesis_lang_code"] == "en"
    assert result.failures[0].metadata["synthesis_audio_format"] == "wav"
    assert result.failures[0].metadata["source_case_id"] == "tts-fail"
    assert "reference_text_sha256" in result.failures[0].metadata


def test_synthesize_cases_with_local_tts_reports_timeout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tts_bin = tmp_path / "bin" / "local-tts-speak"
    tts_bin.parent.mkdir()
    tts_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    case = EvaluationCase(
        id="tts-timeout",
        task="tts_naturalness",
        reference_text="This should time out clearly.",
    )

    def fake_run(command, check, capture_output, text, timeout):
        raise subprocess.TimeoutExpired(
            command,
            timeout,
            output=b"loading model\nstill warming up\n",
            stderr=b"downloading weights\n",
        )

    monkeypatch.setattr("open_audio_judge.local_tts.subprocess.run", fake_run)

    with pytest.raises(RuntimeError) as excinfo:
        synthesize_cases_with_local_tts(
            [case],
            out_dir=tmp_path / "synthesis",
            config=LocalTtsConfig(tts_bin=tts_bin, timeout_seconds=1.5),
        )

    message = str(excinfo.value)
    assert "local TTS command timed out after 1.5 seconds: local-tts-speak" in message
    assert "stderr: downloading weights" in message
    assert "stdout: loading model | still warming up" in message


def test_synthesize_cases_with_local_tts_rejects_nonpositive_timeout(tmp_path: Path) -> None:
    case = EvaluationCase(
        id="tts-timeout-invalid",
        task="tts_naturalness",
        reference_text="This should fail before synthesis.",
    )

    with pytest.raises(ValueError, match="timeout_seconds must be greater than zero"):
        synthesize_cases_with_local_tts(
            [case],
            out_dir=tmp_path / "synthesis",
            config=LocalTtsConfig(dry_run=True, timeout_seconds=0),
        )


def test_synthesize_cases_with_local_tts_rejects_unchanged_fallback_audio(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tts_bin = tmp_path / "bin" / "local-tts-speak"
    tts_bin.parent.mkdir()
    tts_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    synthesis_dir = tmp_path / "synthesis"
    stale_audio = synthesis_dir / "audio" / "tts-stale.wav"
    stale_audio.parent.mkdir(parents=True)
    stale_audio.write_bytes(b"stale-audio")
    case = EvaluationCase(
        id="tts-stale",
        task="tts_naturalness",
        reference_text="This should not reuse stale audio.",
    )

    def fake_run(command, check, capture_output, text, timeout):
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("open_audio_judge.local_tts.subprocess.run", fake_run)

    with pytest.raises(ValueError, match="fallback matching files were unchanged"):
        synthesize_cases_with_local_tts(
            [case],
            out_dir=synthesis_dir,
            config=LocalTtsConfig(tts_bin=tts_bin),
        )


def test_synthesize_cases_with_local_tts_accepts_changed_fallback_audio(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tts_bin = tmp_path / "bin" / "local-tts-speak"
    tts_bin.parent.mkdir()
    tts_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    synthesis_dir = tmp_path / "synthesis"
    audio_path = synthesis_dir / "audio" / "tts-overwrite.wav"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"old")
    case = EvaluationCase(
        id="tts-overwrite",
        task="tts_naturalness",
        reference_text="This should accept an updated fallback file.",
    )

    def fake_run(command, check, capture_output, text, timeout):
        audio_path.write_bytes(b"new-audio")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("open_audio_judge.local_tts.subprocess.run", fake_run)

    synthesized = synthesize_cases_with_local_tts(
        [case],
        out_dir=synthesis_dir,
        config=LocalTtsConfig(tts_bin=tts_bin),
    )

    assert synthesized[0].audio_path == "audio/tts-overwrite.wav"
    assert synthesized[0].metadata["audio_bytes"] == len(b"new-audio")


def test_output_path_from_stdout_accepts_progress_before_json() -> None:
    output = _output_path_from_stdout(
        "\x1b[94mText:\x1b[0m hello\n"
        "S3 Token -> Mel Inference...\n"
        '{"output": "/tmp/example.wav", "synthesis_ms": 123.4}\n'
    )

    assert output is not None
    assert output.name == "example.wav"


def test_output_path_from_stdout_accepts_pretty_json_after_progress() -> None:
    output = _output_path_from_stdout(
        "loading model\n"
        "generating speech\n"
        '{\n'
        '  "result": {\n'
        '    "outputs": [\n'
        '      {"kind": "log", "path": ""},\n'
        '      {"kind": "audio", "path": "/tmp/pretty-output.wav"}\n'
        "    ]\n"
        "  }\n"
        "}\n"
    )

    assert output is not None
    assert output.name == "pretty-output.wav"


@pytest.mark.parametrize(
    "stdout",
    [
        '{"output_path": "/tmp/output-path.wav"}\n',
        '{"audio_path": "/tmp/audio-path.wav"}\n',
        '{"audio": {"path": "/tmp/audio-object.wav"}}\n',
        '{"artifacts": [{"kind": "audio", "path": "/tmp/artifact.wav"}]}\n',
        '{"outputs": [{"path": "/tmp/outputs.wav"}]}\n',
        '{"files": [{"path": "/tmp/files.wav"}]}\n',
    ],
)
def test_output_path_from_stdout_accepts_common_wrapper_json_shapes(stdout: str) -> None:
    output = _output_path_from_stdout(stdout)

    assert output is not None
    assert output.suffix == ".wav"


def test_output_path_from_stdout_resolves_relative_paths_against_audio_dir(
    tmp_path: Path,
) -> None:
    audio_dir = tmp_path / "audio"

    output = _output_path_from_stdout(
        '{"output": "tts-case.wav", "synthesis_ms": 123.4}\n',
        base_dir=audio_dir,
    )

    assert output == (audio_dir / "tts-case.wav").resolve()


def test_require_output_path_in_audio_dir_rejects_stale_outside_paths(tmp_path: Path) -> None:
    audio_dir = tmp_path / "synthesis" / "audio"
    stale_audio = tmp_path / "previous-run" / "tts-case.wav"

    with pytest.raises(ValueError, match="outside the synthesis audio directory"):
        _require_output_path_in_audio_dir(stale_audio, audio_dir)


def test_write_local_tts_summary_json(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    case = EvaluationCase(
        id="tts-1-local-tts",
        task="tts_naturalness",
        audio_path="audio/tts-1.wav",
        reference_text="Hello.",
        metadata={
            "synthesis_voice": "af_heart",
            "synthesis_provider": "local_test_tts",
            "synthesis_model": "mlx-community/chatterbox-turbo-6bit",
            "synthesis_lang_code": "en",
            "synthesis_audio_format": "wav",
            "tts_slice": "general",
            "audio_bytes": 1234,
            "audio_duration_seconds": 1.234,
        },
    )

    write_local_tts_summary_json(
        [case],
        summary,
        source_cases=Path("examples/tts_cases.jsonl"),
        model="mlx-community/chatterbox-turbo-6bit",
        synthesis_provider="local_test_tts",
    )

    data = json.loads(summary.read_text(encoding="utf-8"))
    assert data["candidate_model"] == "mlx-community/chatterbox-turbo-6bit"
    assert data["candidate_generator"] == "local_test_tts"
    assert data["cases_with_audio_path"] == 1
    assert data["attempted_source_cases"] == 1
    assert data["synthesized_case_count"] == 1
    assert data["synthesis_success_rate"] == 1.0
    assert data["total_audio_bytes"] == 1234
    assert data["total_audio_duration_seconds"] == 1.234
    assert data["cases_with_audio_duration"] == 1
    assert data["by_synthesis_provider"] == {"local_test_tts": 1}
    assert data["by_synthesis_model"] == {"mlx-community/chatterbox-turbo-6bit": 1}
    assert data["by_synthesis_voice"] == {"af_heart": 1}
    assert data["by_synthesis_lang_code"] == {"en": 1}
    assert data["synthesis_failure_count"] == 0


def test_write_local_tts_summary_json_counts_failures(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    failures = [
        LocalTtsFailure(
            case_id="tts-fail",
            error_type="RuntimeError",
            message="voice unavailable",
            metadata={
                "language": "en-US",
                "sample_kind": "local_synthetic_tts",
                "synthesis_audio_format": "wav",
                "synthesis_lang_code": "en-US",
                "synthesis_model": "test-model",
                "synthesis_provider": "local_test_tts",
                "synthesis_voice": "af_test",
                "source_category": "instruction_constraints",
                "tts_slice": "numbers",
            },
        )
    ]

    write_local_tts_summary_json(
        [],
        summary,
        source_cases=Path("examples/tts_cases.jsonl"),
        model="mlx-community/chatterbox-turbo-6bit",
        synthesis_provider="local_test_tts",
        failures=failures,
        attempted_source_cases=3,
    )

    data = json.loads(summary.read_text(encoding="utf-8"))
    assert data["total_cases"] == 0
    assert data["attempted_source_cases"] == 3
    assert data["synthesized_case_count"] == 0
    assert data["synthesis_success_rate"] == 0.0
    assert data["synthesis_failure_count"] == 1
    assert data["synthesis_failures_by_error_type"] == {"RuntimeError": 1}
    assert data["synthesis_failures_by_provider"] == {"local_test_tts": 1}
    assert data["synthesis_failures_by_model"] == {"test-model": 1}
    assert data["synthesis_failures_by_voice"] == {"af_test": 1}
    assert data["synthesis_failures_by_lang_code"] == {"en-US": 1}
    assert data["synthesis_failures_by_audio_format"] == {"wav": 1}
    assert data["synthesis_failures_by_tts_slice"] == {"numbers": 1}
    assert data["synthesis_failures_by_source_category"] == {"instruction_constraints": 1}
    assert data["synthesis_failures_by_sample_kind"] == {"local_synthetic_tts": 1}
    assert data["synthesis_failures_by_language"] == {"en-US": 1}


def test_write_local_tts_failures_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "failures.jsonl"

    write_local_tts_failures_jsonl(
        [
            LocalTtsFailure(
                case_id="tts-fail",
                error_type="RuntimeError",
                message="voice unavailable",
                metadata={"tts_slice": "numbers"},
            )
        ],
        path,
    )

    written = json.loads(path.read_text(encoding="utf-8"))
    assert written == {
        "case_id": "tts-fail",
        "error_type": "RuntimeError",
        "message": "voice unavailable",
        "metadata": {"tts_slice": "numbers"},
    }
