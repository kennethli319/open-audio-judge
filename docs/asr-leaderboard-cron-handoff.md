# ASR Leaderboard Cron Handoff

Generated summary for scheduled ASR leaderboard continuation turns.

## Decision

- Action: skip_live_refresh
- Coverage complete: True
- Live refresh required: False
- Runtime ready: not_required
- Missing model/category cells: 0
- Next run commands: 0
- Artifact digest: `296bf0550080eb0d676f351eee0db1ee41045342266b16ccdd44c7256058c7df`
- Reason: The selected ASR result bundle already covers every model/category cell.

## Public Links

- Demo: `https://kennethli319.github.io/open-audio-judge/asr-leaderboard-demo.html`
- Combined report: `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/full-35-combined/report.html`
- Report index: `https://kennethli319.github.io/open-audio-judge/asr-leaderboard-report-index.md`

## Local Artifacts

- Refresh decision: `docs/asr-leaderboard-refresh-decision.json`
- Runtime status: `docs/asr-leaderboard-runtime-status.json`
- Next action: `docs/asr-leaderboard-next-action.md`
- Cron status JSON: `docs/asr-leaderboard-cron-status.json`
- Report links: `docs/asr-leaderboard-report-links.json`

## Commands

- Preflight: `.venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py --check-only --require-generated-fresh --require-audio-ready`
- Runtime preflight: `.venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py --check-only --require-generated-fresh --require-audio-ready --check-mlx-runtime --runtime-status-out docs/asr-leaderboard-runtime-status.json --refresh-decision-out docs/asr-leaderboard-refresh-decision.json --next-action-out docs/asr-leaderboard-next-action.md --cron-status-out docs/asr-leaderboard-cron-status.json --cron-handoff-out docs/asr-leaderboard-cron-handoff.md`
- Refresh artifacts: `.venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py`
- Discover latest complete runs: `.venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py --discover-complete-model-runs --update-run-manifest`
- Sync hosted artifacts: `.venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py --hosted-dir-from-env`
- Verify commit: `.venv/bin/python scripts/verify_asr_leaderboard_commit.py`

Gemini secrets must be loaded only at runtime and must not be stored in artifacts.
