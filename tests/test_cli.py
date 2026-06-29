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
