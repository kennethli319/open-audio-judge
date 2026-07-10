# ASR Leaderboard Refresh Report

This generated report summarizes the verified ASR leaderboard artifact set.

## Coverage

- Results: `runs/asr-leaderboard/full-35-combined/results.jsonl`
- Combined report: `runs/asr-leaderboard/full-35-combined/report.html`
- Summary JSON: `docs/asr-leaderboard-summary.json`
- Run manifest: `docs/asr-leaderboard-run-manifest.json`
- Refresh command playbook: `docs/asr-leaderboard-refresh-commands.sh`
- Refresh workflow JSON: `docs/asr-leaderboard-refresh-workflow.json`
- Live model refresh script: `docs/asr-leaderboard-live-refresh.sh`
- Report index: `docs/asr-leaderboard-report-index.md`
- Report links JSON: `docs/asr-leaderboard-report-links.json`
- Manifest validation: `docs/asr-leaderboard-manifest-validation.json`
- Seed manifest validation: `docs/asr-seed-manifest-validation.json`
- Next-refresh plan: `docs/asr-leaderboard-next-runs.json`
- Hosted artifact manifest: `docs/asr-leaderboard-hosted-manifest.json`
- Hosted demo URL: `https://kennethli319.github.io/open-audio-judge/asr-leaderboard-demo.html`
- Hosted combined report URL: `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/full-35-combined/report.html`
- Artifact bundle index: `docs/asr-leaderboard-artifacts.json`
- Runtime status: `docs/asr-leaderboard-runtime-status.json`
- Total judged transcripts: 105
- Models: 3
- Categories: 7
- Expected cases per model: 35

## Model Scores

| Model | Cases | Gemini Samples | Average Score | Labels |
| --- | ---: | ---: | ---: | --- |
| `mlx-community/VibeVoice-ASR-4bit` | 35/35 ok | 105 | 96.5 | 33 accurate, 1 needs_review, 1 inaccurate |
| `mlx-community/Qwen3-ASR-1.7B-8bit` | 35/35 ok | 105 | 95.2 | 31 accurate, 3 needs_review, 1 inaccurate |
| `mlx-community/whisper-large-v3-turbo-asr-fp16` | 35/35 ok | 105 | 93.4 | 30 accurate, 3 needs_review, 2 inaccurate |

## Category Scores

| Category | Results | Average Score | Labels |
| --- | ---: | ---: | --- |
| `transcription_accuracy_wer` | 15 | 94.6 | 13 accurate, 2 needs_review |
| `numeric_unit_integrity` | 15 | 94.4 | 13 accurate, 2 needs_review |
| `negation_modality_scope` | 15 | 97.7 | 14 accurate, 1 needs_review |
| `temporal_scheduling_accuracy` | 15 | 96.5 | 14 accurate, 1 inaccurate |
| `entity_factual_integrity` | 15 | 89.3 | 12 accurate, 3 inaccurate |
| `semantic_paraphrase_preservation` | 15 | 98.7 | 15 accurate |
| `acoustic_noise_robustness` | 15 | 93.9 | 13 accurate, 2 needs_review |

## Model Category Matrix

| Model | WER | Numeric/Unit | Negation/Modality | Temporal | Entity | Paraphrase | Acoustic Noise |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mlx-community/VibeVoice-ASR-4bit` | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| `mlx-community/Qwen3-ASR-1.7B-8bit` | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| `mlx-community/whisper-large-v3-turbo-asr-fp16` | 5 | 5 | 5 | 5 | 5 | 5 | 5 |

## Source Result Files

- `runs/asr-leaderboard/whisper-large-v3-turbo-smoke/judge-report/results.jsonl`
- `runs/asr-leaderboard/whisper-large-v3-turbo-full-gap/judge-report/results.jsonl`
- `runs/asr-leaderboard/whisper-large-v3-turbo-semantic-smoke/judge-report/results.jsonl`
- `runs/asr-leaderboard/whisper-large-v3-turbo-entity-smoke/judge-report/results.jsonl`
- `runs/asr-leaderboard/whisper-large-v3-turbo-paraphrase-smoke/judge-report/results.jsonl`
- `runs/asr-leaderboard/whisper-large-v3-turbo-noise-smoke/judge-report/results.jsonl`
- `runs/asr-leaderboard/qwen3-asr-1.7b-smoke/judge-report/results.jsonl`
- `runs/asr-leaderboard/qwen3-asr-1.7b-full-gap/judge-report/results.jsonl`
- `runs/asr-leaderboard/qwen3-asr-1.7b-semantic-smoke/judge-report/results.jsonl`
- `runs/asr-leaderboard/qwen3-asr-1.7b-entity-smoke/judge-report/results.jsonl`
- `runs/asr-leaderboard/qwen3-asr-1.7b-paraphrase-smoke/judge-report/results.jsonl`
- `runs/asr-leaderboard/qwen3-asr-1.7b-noise-smoke/judge-report/results.jsonl`
- `runs/asr-leaderboard/vibevoice-asr-smoke/judge-report/results.jsonl`
- `runs/asr-leaderboard/vibevoice-asr-full-gap/judge-report/results.jsonl`
- `runs/asr-leaderboard/vibevoice-asr-semantic-smoke/judge-report/results.jsonl`
- `runs/asr-leaderboard/vibevoice-asr-entity-smoke/judge-report/results.jsonl`
- `runs/asr-leaderboard/vibevoice-asr-paraphrase-smoke/judge-report/results.jsonl`
- `runs/asr-leaderboard/vibevoice-asr-noise-smoke/judge-report/results.jsonl`

## Source Result File Coverage

| Path | Report | Models | Cases | Categories | Gemini Samples | Average Score | Labels |
| --- | --- | --- | ---: | --- | ---: | ---: | --- |
| `runs/asr-leaderboard/whisper-large-v3-turbo-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/whisper-large-v3-turbo-smoke/judge-report/report.html` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | 3/3 ok | `transcription_accuracy_wer`: 3 | 9 | 92.0 | 2 accurate, 1 needs_review |
| `runs/asr-leaderboard/whisper-large-v3-turbo-full-gap/judge-report/results.jsonl` | `runs/asr-leaderboard/whisper-large-v3-turbo-full-gap/judge-report/report.html` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | 12/12 ok | `negation_modality_scope`: 3, `numeric_unit_integrity`: 3, `temporal_scheduling_accuracy`: 4, `transcription_accuracy_wer`: 2 | 36 | 92.6 | 10 accurate, 1 needs_review, 1 inaccurate |
| `runs/asr-leaderboard/whisper-large-v3-turbo-semantic-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/whisper-large-v3-turbo-semantic-smoke/judge-report/report.html` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | 5/5 ok | `negation_modality_scope`: 2, `numeric_unit_integrity`: 2, `temporal_scheduling_accuracy`: 1 | 15 | 93.4 | 4 accurate, 1 needs_review |
| `runs/asr-leaderboard/whisper-large-v3-turbo-entity-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/whisper-large-v3-turbo-entity-smoke/judge-report/report.html` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | 5/5 ok | `entity_factual_integrity`: 5 | 15 | 88.6 | 4 accurate, 1 inaccurate |
| `runs/asr-leaderboard/whisper-large-v3-turbo-paraphrase-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/whisper-large-v3-turbo-paraphrase-smoke/judge-report/report.html` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | 5/5 ok | `semantic_paraphrase_preservation`: 5 | 15 | 97.6 | 5 accurate |
| `runs/asr-leaderboard/whisper-large-v3-turbo-noise-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/whisper-large-v3-turbo-noise-smoke/judge-report/report.html` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | 5/5 ok | `acoustic_noise_robustness`: 5 | 15 | 96.6 | 5 accurate |
| `runs/asr-leaderboard/qwen3-asr-1.7b-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/qwen3-asr-1.7b-smoke/judge-report/report.html` | `mlx-community/Qwen3-ASR-1.7B-8bit` | 3/3 ok | `transcription_accuracy_wer`: 3 | 9 | 88.0 | 2 accurate, 1 needs_review |
| `runs/asr-leaderboard/qwen3-asr-1.7b-full-gap/judge-report/results.jsonl` | `runs/asr-leaderboard/qwen3-asr-1.7b-full-gap/judge-report/report.html` | `mlx-community/Qwen3-ASR-1.7B-8bit` | 12/12 ok | `negation_modality_scope`: 3, `numeric_unit_integrity`: 3, `temporal_scheduling_accuracy`: 4, `transcription_accuracy_wer`: 2 | 36 | 96.7 | 11 accurate, 1 needs_review |
| `runs/asr-leaderboard/qwen3-asr-1.7b-semantic-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/qwen3-asr-1.7b-semantic-smoke/judge-report/report.html` | `mlx-community/Qwen3-ASR-1.7B-8bit` | 5/5 ok | `negation_modality_scope`: 2, `numeric_unit_integrity`: 2, `temporal_scheduling_accuracy`: 1 | 15 | 100.0 | 5 accurate |
| `runs/asr-leaderboard/qwen3-asr-1.7b-entity-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/qwen3-asr-1.7b-entity-smoke/judge-report/report.html` | `mlx-community/Qwen3-ASR-1.7B-8bit` | 5/5 ok | `entity_factual_integrity`: 5 | 15 | 89.0 | 4 accurate, 1 inaccurate |
| `runs/asr-leaderboard/qwen3-asr-1.7b-paraphrase-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/qwen3-asr-1.7b-paraphrase-smoke/judge-report/report.html` | `mlx-community/Qwen3-ASR-1.7B-8bit` | 5/5 ok | `semantic_paraphrase_preservation`: 5 | 15 | 100.0 | 5 accurate |
| `runs/asr-leaderboard/qwen3-asr-1.7b-noise-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/qwen3-asr-1.7b-noise-smoke/judge-report/report.html` | `mlx-community/Qwen3-ASR-1.7B-8bit` | 5/5 ok | `acoustic_noise_robustness`: 5 | 15 | 92.4 | 4 accurate, 1 needs_review |
| `runs/asr-leaderboard/vibevoice-asr-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/vibevoice-asr-smoke/judge-report/report.html` | `mlx-community/VibeVoice-ASR-4bit` | 3/3 ok | `transcription_accuracy_wer`: 3 | 9 | 94.3 | 3 accurate |
| `runs/asr-leaderboard/vibevoice-asr-full-gap/judge-report/results.jsonl` | `runs/asr-leaderboard/vibevoice-asr-full-gap/judge-report/report.html` | `mlx-community/VibeVoice-ASR-4bit` | 12/12 ok | `negation_modality_scope`: 3, `numeric_unit_integrity`: 3, `temporal_scheduling_accuracy`: 4, `transcription_accuracy_wer`: 2 | 36 | 99.3 | 12 accurate |
| `runs/asr-leaderboard/vibevoice-asr-semantic-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/vibevoice-asr-semantic-smoke/judge-report/report.html` | `mlx-community/VibeVoice-ASR-4bit` | 5/5 ok | `negation_modality_scope`: 2, `numeric_unit_integrity`: 2, `temporal_scheduling_accuracy`: 1 | 15 | 98.8 | 5 accurate |
| `runs/asr-leaderboard/vibevoice-asr-entity-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/vibevoice-asr-entity-smoke/judge-report/report.html` | `mlx-community/VibeVoice-ASR-4bit` | 5/5 ok | `entity_factual_integrity`: 5 | 15 | 90.2 | 4 accurate, 1 inaccurate |
| `runs/asr-leaderboard/vibevoice-asr-paraphrase-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/vibevoice-asr-paraphrase-smoke/judge-report/report.html` | `mlx-community/VibeVoice-ASR-4bit` | 5/5 ok | `semantic_paraphrase_preservation`: 5 | 15 | 98.4 | 5 accurate |
| `runs/asr-leaderboard/vibevoice-asr-noise-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/vibevoice-asr-noise-smoke/judge-report/report.html` | `mlx-community/VibeVoice-ASR-4bit` | 5/5 ok | `acoustic_noise_robustness`: 5 | 15 | 92.8 | 4 accurate, 1 needs_review |

## Next Refresh Plan

- Status: complete
- Missing model/category cells: 0
- Next run commands: 0

## Generated Artifact Index

| Path | Purpose |
| --- | --- |
| `runs/asr-leaderboard/full-35-combined/results.jsonl` | Combined ASR judge results used by the generated page and report. |
| `runs/asr-leaderboard/full-35-combined/report.html` | Local combined HTML report with per-case judge details. |
| `docs/asr-leaderboard-summary.json` | Machine-readable leaderboard summary and reproducible refresh workflow. |
| `docs/asr-leaderboard-refresh-report.md` | Human-readable coverage, score, source-file, and command report. |
| `docs/asr-leaderboard-report-index.md` | Human-readable index linking the demo page, combined report, and source run reports. |
| `docs/asr-leaderboard-report-links.json` | Machine-readable map linking the demo page to combined and source ASR reports. |
| `docs/asr-leaderboard-refresh-commands.sh` | Generated shell playbook for repeatable ASR leaderboard refreshes. |
| `docs/asr-leaderboard-refresh-workflow.json` | Machine-readable generated workflow for ASR refresh automation. |
| `docs/asr-leaderboard-live-refresh.sh` | Opt-in generated shell script for live MLX ASR/Gemini refreshes. |
| `docs/asr-leaderboard-run-manifest.json` | Committed source result manifest for manifest-based refreshes. |
| `docs/asr-leaderboard-manifest-validation.json` | Coverage validation for the model/category result matrix. |
| `docs/asr-seed-manifest-validation.json` | Seed-manifest validation proving public-safe ASR cases keep exact category coverage. |
| `docs/asr-leaderboard-next-runs.json` | Machine-readable next-refresh plan for missing ASR model/category cells. |
| `docs/asr-leaderboard-hosted-manifest.json` | Machine-readable manifest of ASR demo artifacts mirrored to the hosted Pages checkout. |
| `docs/asr-leaderboard-artifacts.json` | Single machine-readable index for the ASR leaderboard artifact bundle. |
| `docs/asr-leaderboard-runtime-status.json` | Machine-readable MLX ASR and Gemini readiness status for refresh automation. |
| `docs/asr-leaderboard-refresh-decision.json` | Machine-readable runtime-gated decision for the next ASR refresh action. |
| `docs/asr-leaderboard-next-action.md` | Telegram-ready Markdown note summarizing the runtime-gated next ASR action. |
| `docs/asr-leaderboard-source-selection.json` | Machine-readable record of selected ASR source result files for the last refresh. |

## Refresh Commands

- Generated shell playbook: `docs/asr-leaderboard-refresh-commands.sh`
- Generated workflow JSON: `docs/asr-leaderboard-refresh-workflow.json`
- Generated live refresh script: `docs/asr-leaderboard-live-refresh.sh`
- Seed manifest validation: `.venv/bin/python scripts/validate_asr_seed_manifest.py --summary-out docs/asr-seed-manifest-validation.json`
- Audio materialization: `.venv/bin/python scripts/synthesize_tts_cases.py --cases examples/asr_research_cases.jsonl --out runs/asr-research-audio --discard-text-sidecars --summary-out runs/asr-research-audio/summary.json`
- MLX ASR runtime check: `PYTHONPATH=src .venv/bin/python -m open_audio_judge.cli check-mlx-asr-runtime --python-bin .venv/bin/python --model mlx-community/whisper-large-v3-turbo-asr-fp16`
- Load local Gemini secret before model runs: `source /Users/wangyauli/.openclaw/secrets/open-audio-judge-gemini.env`
- Run mlx-community/whisper-large-v3-turbo-asr-fp16: `.venv/bin/oaj autojudge-mlx-asr --python-bin .venv/bin/python --cases runs/asr-research-audio/tts_audio_cases.jsonl --model mlx-community/whisper-large-v3-turbo-asr-fp16 --judge-provider gemini --judge-samples 3 --out runs/asr-leaderboard/whisper-large-v3-turbo-refresh`
- Run mlx-community/Qwen3-ASR-1.7B-8bit: `.venv/bin/oaj autojudge-mlx-asr --python-bin .venv/bin/python --cases runs/asr-research-audio/tts_audio_cases.jsonl --model mlx-community/Qwen3-ASR-1.7B-8bit --judge-provider gemini --judge-samples 3 --out runs/asr-leaderboard/qwen3-asr-1.7b-refresh`
- Run mlx-community/VibeVoice-ASR-4bit: `.venv/bin/oaj autojudge-mlx-asr --python-bin .venv/bin/python --cases runs/asr-research-audio/tts_audio_cases.jsonl --model mlx-community/VibeVoice-ASR-4bit --judge-provider gemini --judge-samples 3 --out runs/asr-leaderboard/vibevoice-asr-refresh`
- Fallback models if a primary model is blocked: `mlx-community/whisper-small.en-asr-4bit`, `mlx-community/parakeet-rnnt-0.6b`, `mlx-community/GLM-ASR-Nano-2512-4bit`
- Fallback handling: If a primary MLX ASR model is blocked or unsupported, record the unsupported state explicitly before trying the fallback model list; do not substitute silently.
- Preflight refresh inputs: `.venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py --check-only`
- Write preflight summary: `.venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py --check-only --require-generated-fresh --check-summary-out runs/asr-leaderboard/preflight-summary.json`
- Require audio manifest readiness: `.venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py --check-only --require-audio-ready`
- Refresh runtime status artifact: `.venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py --check-only --check-mlx-runtime`
- Require live runtime readiness: `.venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py --check-only --check-mlx-runtime --require-runtime-ready`
- Full refresh readiness check: `.venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py --check-only --require-generated-fresh --require-audio-ready --check-summary-out runs/asr-leaderboard/preflight-summary.json`
- Cron refresh rehearsal: `.venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py --check-only --require-generated-fresh --require-audio-ready --check-mlx-runtime --check-summary-out runs/asr-leaderboard/preflight-summary.json`
- Combine and refresh committed artifacts: `.venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py --results runs/asr-leaderboard/whisper-large-v3-turbo-smoke/judge-report/results.jsonl --results runs/asr-leaderboard/whisper-large-v3-turbo-full-gap/judge-report/results.jsonl --results runs/asr-leaderboard/whisper-large-v3-turbo-semantic-smoke/judge-report/results.jsonl --results runs/asr-leaderboard/whisper-large-v3-turbo-entity-smoke/judge-report/results.jsonl --results runs/asr-leaderboard/whisper-large-v3-turbo-paraphrase-smoke/judge-report/results.jsonl --results runs/asr-leaderboard/whisper-large-v3-turbo-noise-smoke/judge-report/results.jsonl --results runs/asr-leaderboard/qwen3-asr-1.7b-smoke/judge-report/results.jsonl --results runs/asr-leaderboard/qwen3-asr-1.7b-full-gap/judge-report/results.jsonl --results runs/asr-leaderboard/qwen3-asr-1.7b-semantic-smoke/judge-report/results.jsonl --results runs/asr-leaderboard/qwen3-asr-1.7b-entity-smoke/judge-report/results.jsonl --results runs/asr-leaderboard/qwen3-asr-1.7b-paraphrase-smoke/judge-report/results.jsonl --results runs/asr-leaderboard/qwen3-asr-1.7b-noise-smoke/judge-report/results.jsonl --results runs/asr-leaderboard/vibevoice-asr-smoke/judge-report/results.jsonl --results runs/asr-leaderboard/vibevoice-asr-full-gap/judge-report/results.jsonl --results runs/asr-leaderboard/vibevoice-asr-semantic-smoke/judge-report/results.jsonl --results runs/asr-leaderboard/vibevoice-asr-entity-smoke/judge-report/results.jsonl --results runs/asr-leaderboard/vibevoice-asr-paraphrase-smoke/judge-report/results.jsonl --results runs/asr-leaderboard/vibevoice-asr-noise-smoke/judge-report/results.jsonl --update-run-manifest --source-selection-summary-out docs/asr-leaderboard-source-selection.json`
- Discover latest complete runs: `.venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py --discover-complete-model-runs --update-run-manifest --source-selection-summary-out docs/asr-leaderboard-source-selection.json`
- Manifest-based refresh: `.venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py --source-selection-summary-out docs/asr-leaderboard-source-selection.json`
- Page validation: `.venv/bin/python scripts/check_asr_leaderboard_page.py`
- Generated artifact freshness check: `.venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py --check-only --require-generated-fresh`
- Commit verification: `.venv/bin/python scripts/verify_asr_leaderboard_commit.py`
- Commit verification with hosted mirror: `.venv/bin/python scripts/verify_asr_leaderboard_commit.py --hosted-dir-from-env`
- Hosted artifact sync: `.venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py --hosted-dir-from-env`
- Hosted mirror validation: `.venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py --check-only --hosted-dir-from-env --require-hosted-current`
- Live model refresh script: `bash docs/asr-leaderboard-live-refresh.sh`
- Review blocked model log: `tail -n 20 runs/asr-leaderboard/blocked-models.jsonl`

## Automation Stages

| Stage | Commands | Writes committed artifacts | Runs live models |
| --- | --- | --- | --- |
| `preflight` | `cron_rehearsal_command`, `runtime_ready_check_command` | False | False |
| `live_refresh` | `local_secret_env_command`, `model_run_commands` | False | True |
| `artifact_refresh` | `discover_refresh_command`, `combine_refresh_command`, `manifest_refresh_command` | True | False |
| `verification` | `page_validation_command`, `freshness_check_command`, `commit_verification_command` | False | False |
| `hosted_sync` | `hosted_artifact_command`, `hosted_validation_command`, `hosted_commit_verification_command` | False | False |

## Runtime Status

- MLX ASR: not_executed_by_refresh; transcripts loaded from verified result artifacts
- Gemini judge: verified_from_loaded_results
- Live model calls during refresh: none
- Loaded result providers: gemini
- All loaded results ok: True

Gemini secrets must be loaded only at runtime from the local secret file.
