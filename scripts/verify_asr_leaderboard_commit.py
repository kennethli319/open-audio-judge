from __future__ import annotations

import argparse
import os
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
    ("bash", "-n", "docs/asr-leaderboard-refresh-commands.sh"),
    ("bash", "-n", "docs/asr-leaderboard-live-refresh.sh"),
    ("git", "diff", "--check"),
)

DEFAULT_HOSTED_DIR_ENV = "ASR_LEADERBOARD_HOSTED_DIR"


def _hosted_dir_from_env(env_var: str) -> Path:
    raw_value = os.environ.get(env_var, "").strip()
    if not raw_value:
        raise ValueError(
            f"--hosted-dir-from-env requires ${env_var} to point to the "
            "kennethli319.github.io/open-audio-judge checkout."
        )
    return Path(raw_value).expanduser()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run non-secret verification before committing ASR leaderboard artifacts.",
    )
    parser.add_argument(
        "--hosted-dir",
        type=Path,
        help="Optional kennethli319.github.io/open-audio-judge directory to verify.",
    )
    parser.add_argument(
        "--hosted-dir-from-env",
        action="store_true",
        help=(
            f"Read the hosted Pages directory from ${DEFAULT_HOSTED_DIR_ENV} and verify "
            "that mirrored ASR artifacts match the local hosted manifest."
        ),
    )
    parser.add_argument(
        "--hosted-dir-env",
        default=DEFAULT_HOSTED_DIR_ENV,
        help=f"Environment variable used with --hosted-dir-from-env (default: {DEFAULT_HOSTED_DIR_ENV}).",
    )
    args = parser.parse_args()

    for command in COMMANDS:
        print("$ " + " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)

    hosted_dir = args.hosted_dir
    if hosted_dir is None and args.hosted_dir_from_env:
        hosted_dir = _hosted_dir_from_env(args.hosted_dir_env)
    if hosted_dir is not None:
        command = (
            ".venv/bin/python",
            "scripts/refresh_asr_leaderboard_artifacts.py",
            "--check-only",
            "--hosted-dir",
            str(hosted_dir),
            "--require-hosted-current",
        )
        print("$ " + " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)

    print("ASR leaderboard commit verification passed.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
