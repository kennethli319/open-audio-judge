from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


COMMANDS = (
    (".venv/bin/ruff", "check", "."),
    (".venv/bin/python", "-m", "pytest"),
    (".venv/bin/python", "scripts/check_asr_leaderboard_page.py"),
    (
        ".venv/bin/python",
        "scripts/refresh_asr_leaderboard_artifacts.py",
        "--check-only",
        "--require-generated-fresh",
    ),
    ("git", "diff", "--check"),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run non-secret verification before committing ASR leaderboard artifacts.",
    )
    parser.parse_args()

    for command in COMMANDS:
        print("$ " + " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)

    print("ASR leaderboard commit verification passed.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
