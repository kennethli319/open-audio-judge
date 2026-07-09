#!/usr/bin/env bash
set -euo pipefail

# Generated ASR leaderboard refresh playbook.
# By default this refreshes committed artifacts from verified result files.
# Live model runs require the local Gemini secret and are listed below as opt-in commands.

.venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py --check-only
.venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py --check-only --require-generated-fresh --check-summary-out runs/asr-leaderboard/preflight-summary.json
.venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py --check-only --require-audio-ready
.venv/bin/python scripts/validate_asr_seed_manifest.py --summary-out docs/asr-seed-manifest-validation.json
.venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py --results runs/asr-leaderboard/whisper-large-v3-turbo-smoke/judge-report/results.jsonl --results runs/asr-leaderboard/whisper-large-v3-turbo-full-gap/judge-report/results.jsonl --results runs/asr-leaderboard/whisper-large-v3-turbo-semantic-smoke/judge-report/results.jsonl --results runs/asr-leaderboard/whisper-large-v3-turbo-entity-smoke/judge-report/results.jsonl --results runs/asr-leaderboard/whisper-large-v3-turbo-paraphrase-smoke/judge-report/results.jsonl --results runs/asr-leaderboard/whisper-large-v3-turbo-noise-smoke/judge-report/results.jsonl --results runs/asr-leaderboard/qwen3-asr-1.7b-smoke/judge-report/results.jsonl --results runs/asr-leaderboard/qwen3-asr-1.7b-full-gap/judge-report/results.jsonl --results runs/asr-leaderboard/qwen3-asr-1.7b-semantic-smoke/judge-report/results.jsonl --results runs/asr-leaderboard/qwen3-asr-1.7b-entity-smoke/judge-report/results.jsonl --results runs/asr-leaderboard/qwen3-asr-1.7b-paraphrase-smoke/judge-report/results.jsonl --results runs/asr-leaderboard/qwen3-asr-1.7b-noise-smoke/judge-report/results.jsonl --results runs/asr-leaderboard/vibevoice-asr-smoke/judge-report/results.jsonl --results runs/asr-leaderboard/vibevoice-asr-full-gap/judge-report/results.jsonl --results runs/asr-leaderboard/vibevoice-asr-semantic-smoke/judge-report/results.jsonl --results runs/asr-leaderboard/vibevoice-asr-entity-smoke/judge-report/results.jsonl --results runs/asr-leaderboard/vibevoice-asr-paraphrase-smoke/judge-report/results.jsonl --results runs/asr-leaderboard/vibevoice-asr-noise-smoke/judge-report/results.jsonl --update-run-manifest
.venv/bin/python scripts/check_asr_leaderboard_page.py
.venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py --check-only --require-generated-fresh

# Optional hosted sync; export ASR_SYNC_HOSTED=1 and set ASR_LEADERBOARD_HOSTED_DIR first.
if [[ "${ASR_SYNC_HOSTED:-0}" == "1" ]]; then
  : "${ASR_LEADERBOARD_HOSTED_DIR:?Set ASR_LEADERBOARD_HOSTED_DIR to the Pages checkout open-audio-judge directory}"
  .venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py --hosted-dir-from-env
  .venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py --check-only --hosted-dir-from-env
fi

# Optional when seed cases change: materialize local audio under ignored runs/.
# .venv/bin/python scripts/synthesize_tts_cases.py --cases examples/asr_research_cases.jsonl --out runs/asr-research-audio --discard-text-sidecars --summary-out runs/asr-research-audio/summary.json

# Optional live refresh: load the Gemini key only in your local shell before judge calls.
# source /Users/wangyauli/.openclaw/secrets/open-audio-judge-gemini.env

# Optional live refresh: check the MLX ASR runtime before model jobs.
# PYTHONPATH=src .venv/bin/python -m open_audio_judge.cli check-mlx-asr-runtime --python-bin .venv/bin/python --model mlx-community/whisper-large-v3-turbo-asr-fp16
# .venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py --check-only --check-mlx-runtime
# .venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py --check-only --require-runtime-ready

# Optional live refresh: run primary MLX ASR model jobs when the local runtime is ready.
# .venv/bin/oaj autojudge-mlx-asr --python-bin .venv/bin/python --cases runs/asr-research-audio/tts_audio_cases.jsonl --model mlx-community/whisper-large-v3-turbo-asr-fp16 --judge-provider gemini --judge-samples 3 --out runs/asr-leaderboard/whisper-large-v3-turbo-refresh
# .venv/bin/oaj autojudge-mlx-asr --python-bin .venv/bin/python --cases runs/asr-research-audio/tts_audio_cases.jsonl --model mlx-community/Qwen3-ASR-1.7B-8bit --judge-provider gemini --judge-samples 3 --out runs/asr-leaderboard/qwen3-asr-1.7b-refresh
# .venv/bin/oaj autojudge-mlx-asr --python-bin .venv/bin/python --cases runs/asr-research-audio/tts_audio_cases.jsonl --model mlx-community/VibeVoice-ASR-4bit --judge-provider gemini --judge-samples 3 --out runs/asr-leaderboard/vibevoice-asr-refresh

# If a primary model is blocked, record the unsupported state before trying fallbacks.
# Fallback models: mlx-community/whisper-small.en-asr-4bit, mlx-community/parakeet-rnnt-0.6b, mlx-community/GLM-ASR-Nano-2512-4bit

# Alternative: discover the newest complete primary-model runs.
# .venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py --discover-complete-model-runs --update-run-manifest
