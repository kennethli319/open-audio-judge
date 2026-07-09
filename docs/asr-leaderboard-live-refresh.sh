#!/usr/bin/env bash
set -euo pipefail

# Generated opt-in live ASR leaderboard refresh script.
# Runs local MLX ASR jobs and Gemini judging only after runtime preflights pass.
# Gemini secrets are sourced at runtime and are never printed by this script.
# Blocked primary model runs are recorded under ignored runs/ before the script exits.

.venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py --check-only
.venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py --check-only --require-audio-ready
PYTHONPATH=src .venv/bin/python -m open_audio_judge.cli check-mlx-asr-runtime --python-bin .venv/bin/python --model mlx-community/whisper-large-v3-turbo-asr-fp16
.venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py --check-only --require-runtime-ready

if [[ ! -f "/Users/wangyauli/.openclaw/secrets/open-audio-judge-gemini.env" ]]; then
  echo "Missing Gemini secret file: /Users/wangyauli/.openclaw/secrets/open-audio-judge-gemini.env" >&2
  exit 1
fi
source /Users/wangyauli/.openclaw/secrets/open-audio-judge-gemini.env

BLOCKED_MODEL_LOG="runs/asr-leaderboard/blocked-models.jsonl"
blocked_model_count=0

run_primary_model() {
  local model="$1"
  local run_name="$2"
  shift 2
  echo "Running ${model}"
  set +e
  "$@"
  local exit_code=$?
  set -e
  if [[ ${exit_code} -ne 0 ]]; then
    mkdir -p "$(dirname "${BLOCKED_MODEL_LOG}")"
    blocked_model_count=$((blocked_model_count + 1))
    printf '{"model":"%s","run_name":"%s","status":"blocked","exit_code":%s,"recorded_at_utc":"%s","fallback_policy":"record before fallback; do not silently substitute"}\n' \
      "${model}" "${run_name}" "${exit_code}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${BLOCKED_MODEL_LOG}"
    echo "Recorded blocked primary ASR model in ${BLOCKED_MODEL_LOG}: ${model}" >&2
  fi
}

# Primary model refreshes.
run_primary_model "mlx-community/whisper-large-v3-turbo-asr-fp16" "whisper-large-v3-turbo-refresh" \
  .venv/bin/oaj autojudge-mlx-asr --python-bin .venv/bin/python --cases runs/asr-research-audio/tts_audio_cases.jsonl --model mlx-community/whisper-large-v3-turbo-asr-fp16 --judge-provider gemini --judge-samples 3 --out runs/asr-leaderboard/whisper-large-v3-turbo-refresh
run_primary_model "mlx-community/Qwen3-ASR-1.7B-8bit" "qwen3-asr-1.7b-refresh" \
  .venv/bin/oaj autojudge-mlx-asr --python-bin .venv/bin/python --cases runs/asr-research-audio/tts_audio_cases.jsonl --model mlx-community/Qwen3-ASR-1.7B-8bit --judge-provider gemini --judge-samples 3 --out runs/asr-leaderboard/qwen3-asr-1.7b-refresh
run_primary_model "mlx-community/VibeVoice-ASR-4bit" "vibevoice-asr-refresh" \
  .venv/bin/oaj autojudge-mlx-asr --python-bin .venv/bin/python --cases runs/asr-research-audio/tts_audio_cases.jsonl --model mlx-community/VibeVoice-ASR-4bit --judge-provider gemini --judge-samples 3 --out runs/asr-leaderboard/vibevoice-asr-refresh

if [[ ${blocked_model_count} -ne 0 ]]; then
  echo "${blocked_model_count} primary model run(s) were blocked. Review ${BLOCKED_MODEL_LOG} before trying fallbacks." >&2
  exit 1
fi

# Rebuild committed artifacts from the newest complete primary-model runs.
.venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py --discover-complete-model-runs --update-run-manifest
.venv/bin/python scripts/check_asr_leaderboard_page.py
.venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py --check-only --require-generated-fresh

# If a primary model fails, record the unsupported/blocked state before trying fallbacks.
# Fallback models: mlx-community/whisper-small.en-asr-4bit, mlx-community/parakeet-rnnt-0.6b, mlx-community/GLM-ASR-Nano-2512-4bit
