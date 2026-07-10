import importlib.util
import sys
from pathlib import Path


SCRIPT = Path("scripts/verify_asr_leaderboard_commit.py")


def load_script_module():
    spec = importlib.util.spec_from_file_location("verify_asr_leaderboard_commit", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_commit_verification_requires_generated_freshness_and_audio_ready() -> None:
    module = load_script_module()

    assert module._refresh_preflight_command() == (
        ".venv/bin/python",
        "scripts/refresh_asr_leaderboard_artifacts.py",
        "--check-only",
        "--require-generated-fresh",
        "--require-audio-ready",
    )


def test_commit_verification_can_include_runtime_preflight_summary() -> None:
    module = load_script_module()

    assert module._refresh_preflight_command(
        check_mlx_runtime=True,
        check_summary_out=module.DEFAULT_CHECK_SUMMARY,
    ) == (
        ".venv/bin/python",
        "scripts/refresh_asr_leaderboard_artifacts.py",
        "--check-only",
        "--require-generated-fresh",
        "--require-audio-ready",
        "--check-mlx-runtime",
        "--check-summary-out",
        "runs/asr-leaderboard/preflight-summary.json",
    )


def test_commit_verification_can_require_runtime_ready() -> None:
    module = load_script_module()

    assert "--require-runtime-ready" in module._refresh_preflight_command(
        require_runtime_ready=True,
    )
