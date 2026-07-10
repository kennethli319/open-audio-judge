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
    ("bash", "-n", "docs/asr-leaderboard-refresh-commands.sh"),
    ("bash", "-n", "docs/asr-leaderboard-live-refresh.sh"),
    ("git", "diff", "--check"),
)

DEFAULT_HOSTED_DIR_ENV = "ASR_LEADERBOARD_HOSTED_DIR"
DEFAULT_CHECK_SUMMARY = Path("runs/asr-leaderboard/preflight-summary.json")
DEFAULT_RUNTIME_STATUS = Path("runs/asr-leaderboard/preflight-runtime-status.json")
DEFAULT_REFRESH_DECISION = Path("runs/asr-leaderboard/preflight-refresh-decision.json")
DEFAULT_NEXT_ACTION = Path("runs/asr-leaderboard/preflight-next-action.md")
DEFAULT_CRON_STATUS = Path("runs/asr-leaderboard/preflight-cron-status.json")


def _refresh_preflight_command(
    *,
    check_mlx_runtime: bool = False,
    require_runtime_ready: bool = False,
    check_summary_out: Path | None = None,
) -> tuple[str, ...]:
    command = [
        ".venv/bin/python",
        "scripts/refresh_asr_leaderboard_artifacts.py",
        "--check-only",
        "--require-generated-fresh",
        "--require-audio-ready",
    ]
    if check_mlx_runtime or require_runtime_ready:
        command.append("--check-mlx-runtime")
        command.extend(("--runtime-status-out", str(DEFAULT_RUNTIME_STATUS)))
        command.extend(("--refresh-decision-out", str(DEFAULT_REFRESH_DECISION)))
        command.extend(("--next-action-out", str(DEFAULT_NEXT_ACTION)))
        command.extend(("--cron-status-out", str(DEFAULT_CRON_STATUS)))
    if require_runtime_ready:
        command.append("--require-runtime-ready")
    if check_summary_out is not None:
        command.extend(("--check-summary-out", str(check_summary_out)))
    return tuple(command)


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
    parser.add_argument(
        "--check-mlx-runtime",
        action="store_true",
        help="Include the bounded MLX ASR runtime preflight in the refresh input check.",
    )
    parser.add_argument(
        "--require-runtime-ready",
        action="store_true",
        help=(
            "Fail unless the ASR audio manifest, Gemini secret, and bounded MLX ASR "
            "runtime preflight are ready."
        ),
    )
    parser.add_argument(
        "--check-summary-out",
        type=Path,
        help="Write the ASR refresh preflight summary JSON while verifying.",
    )
    parser.add_argument(
        "--cron-preflight-summary",
        action="store_true",
        help=f"Write the preflight summary to {DEFAULT_CHECK_SUMMARY}.",
    )
    args = parser.parse_args()

    check_summary_out = args.check_summary_out
    if check_summary_out is None and args.cron_preflight_summary:
        check_summary_out = DEFAULT_CHECK_SUMMARY

    for command in COMMANDS:
        print("$ " + " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)

    refresh_command = _refresh_preflight_command(
        check_mlx_runtime=args.check_mlx_runtime,
        require_runtime_ready=args.require_runtime_ready,
        check_summary_out=check_summary_out,
    )
    print("$ " + " ".join(refresh_command), flush=True)
    subprocess.run(refresh_command, cwd=ROOT, check=True)

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
