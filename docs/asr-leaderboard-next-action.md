# ASR Leaderboard Next Action

- Action: skip_live_refresh.
- Coverage complete: true (0 missing cells).
- Runtime ready: not_required.
- Reason: The selected ASR result bundle already covers every model/category cell.

## Rationale

- Coverage status: complete.
- Missing model/category cells: 0.
- Candidate live-refresh commands: 0.
- Live MLX ASR/Gemini refresh is not required for the selected result bundle.

## Fallback Policy

- Fallback models: `mlx-community/whisper-small.en-asr-4bit`, `mlx-community/parakeet-rnnt-0.6b`, `mlx-community/GLM-ASR-Nano-2512-4bit`
- Record unsupported primary model states explicitly before trying fallbacks; do not silently substitute models.

## Source

- Decision JSON: `docs/asr-leaderboard-refresh-decision.json`
- Runtime status: `docs/asr-leaderboard-runtime-status.json`
- Next-run plan: `docs/asr-leaderboard-next-runs.json`

Gemini secrets must be loaded only at runtime from the local secret file.
