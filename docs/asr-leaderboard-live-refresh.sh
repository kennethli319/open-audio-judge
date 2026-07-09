#!/usr/bin/env bash
set -euo pipefail

# Generated opt-in live ASR leaderboard refresh script.
# Runs local MLX ASR jobs and Gemini judging only after runtime preflights pass.
# Gemini secrets are sourced at runtime and are never printed by this script.

.venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py --check-only
.venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py --check-only --require-audio-ready
PYTHONPATH=src .venv/bin/python -m open_audio_judge.cli check-mlx-asr-runtime --python-bin .venv/bin/python --model mlx-community/whisper-large-v3-turbo-asr-fp16
.venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py --check-only --require-runtime-ready

if [[ ! -f "/Users/wangyauli/.openclaw/secrets/open-audio-judge-gemini.env" ]]; then
  echo "Missing Gemini secret file: /Users/wangyauli/.openclaw/secrets/open-audio-judge-gemini.env" >&2
  exit 1
fi
source /Users/wangyauli/.openclaw/secrets/open-audio-judge-gemini.env

# Primary model refreshes.
.venv/bin/oaj autojudge-mlx-asr --python-bin .venv/bin/python --cases runs/asr-research-audio/tts_audio_cases.jsonl --model mlx-community/whisper-large-v3-turbo-asr-fp16 --judge-provider gemini --judge-samples 3 --out runs/asr-leaderboard/whisper-large-v3-turbo-refresh
.venv/bin/oaj autojudge-mlx-asr --python-bin .venv/bin/python --cases runs/asr-research-audio/tts_audio_cases.jsonl --model mlx-community/Qwen3-ASR-1.7B-8bit --judge-provider gemini --judge-samples 3 --out runs/asr-leaderboard/qwen3-asr-1.7b-refresh
.venv/bin/oaj autojudge-mlx-asr --python-bin .venv/bin/python --cases runs/asr-research-audio/tts_audio_cases.jsonl --model mlx-community/VibeVoice-ASR-4bit --judge-provider gemini --judge-samples 3 --out runs/asr-leaderboard/vibevoice-asr-refresh

# Rebuild committed artifacts from the newest complete primary-model runs.
.venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py --discover-complete-model-runs --update-run-manifest
.venv/bin/python scripts/check_asr_leaderboard_page.py
.venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py --check-only --require-generated-fresh

# If a primary model fails, record the unsupported/blocked state before trying fallbacks.
# Fallback models: mlx-community/whisper-small.en-asr-4bit, mlx-community/parakeet-rnnt-0.6b, mlx-community/GLM-ASR-Nano-2512-4bit
