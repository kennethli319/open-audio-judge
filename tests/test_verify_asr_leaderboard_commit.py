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

    assert (
        ".venv/bin/python",
        "scripts/refresh_asr_leaderboard_artifacts.py",
        "--check-only",
        "--require-generated-fresh",
        "--require-audio-ready",
    ) in module.COMMANDS
