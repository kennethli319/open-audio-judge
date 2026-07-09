# ASR Leaderboard Refresh Report

This generated report summarizes the verified ASR leaderboard artifact set.

## Coverage

- Results: `runs/asr-leaderboard/full-35-combined/results.jsonl`
- Combined report: `runs/asr-leaderboard/full-35-combined/report.html`
- Summary JSON: `docs/asr-leaderboard-summary.json`
- Run manifest: `docs/asr-leaderboard-run-manifest.json`
- Manifest validation: `docs/asr-leaderboard-manifest-validation.json`
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

| Path | Models | Cases | Categories | Gemini Samples | Average Score | Labels |
| --- | --- | ---: | --- | ---: | ---: | --- |
| `runs/asr-leaderboard/whisper-large-v3-turbo-smoke/judge-report/results.jsonl` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | 3/3 ok | `transcription_accuracy_wer`: 3 | 9 | 92.0 | 2 accurate, 1 needs_review |
| `runs/asr-leaderboard/whisper-large-v3-turbo-full-gap/judge-report/results.jsonl` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | 12/12 ok | `negation_modality_scope`: 3, `numeric_unit_integrity`: 3, `temporal_scheduling_accuracy`: 4, `transcription_accuracy_wer`: 2 | 36 | 92.6 | 10 accurate, 1 needs_review, 1 inaccurate |
| `runs/asr-leaderboard/whisper-large-v3-turbo-semantic-smoke/judge-report/results.jsonl` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | 5/5 ok | `negation_modality_scope`: 2, `numeric_unit_integrity`: 2, `temporal_scheduling_accuracy`: 1 | 15 | 93.4 | 4 accurate, 1 needs_review |
| `runs/asr-leaderboard/whisper-large-v3-turbo-entity-smoke/judge-report/results.jsonl` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | 5/5 ok | `entity_factual_integrity`: 5 | 15 | 88.6 | 4 accurate, 1 inaccurate |
| `runs/asr-leaderboard/whisper-large-v3-turbo-paraphrase-smoke/judge-report/results.jsonl` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | 5/5 ok | `semantic_paraphrase_preservation`: 5 | 15 | 97.6 | 5 accurate |
| `runs/asr-leaderboard/whisper-large-v3-turbo-noise-smoke/judge-report/results.jsonl` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | 5/5 ok | `acoustic_noise_robustness`: 5 | 15 | 96.6 | 5 accurate |
| `runs/asr-leaderboard/qwen3-asr-1.7b-smoke/judge-report/results.jsonl` | `mlx-community/Qwen3-ASR-1.7B-8bit` | 3/3 ok | `transcription_accuracy_wer`: 3 | 9 | 88.0 | 2 accurate, 1 needs_review |
| `runs/asr-leaderboard/qwen3-asr-1.7b-full-gap/judge-report/results.jsonl` | `mlx-community/Qwen3-ASR-1.7B-8bit` | 12/12 ok | `negation_modality_scope`: 3, `numeric_unit_integrity`: 3, `temporal_scheduling_accuracy`: 4, `transcription_accuracy_wer`: 2 | 36 | 96.7 | 11 accurate, 1 needs_review |
| `runs/asr-leaderboard/qwen3-asr-1.7b-semantic-smoke/judge-report/results.jsonl` | `mlx-community/Qwen3-ASR-1.7B-8bit` | 5/5 ok | `negation_modality_scope`: 2, `numeric_unit_integrity`: 2, `temporal_scheduling_accuracy`: 1 | 15 | 100.0 | 5 accurate |
| `runs/asr-leaderboard/qwen3-asr-1.7b-entity-smoke/judge-report/results.jsonl` | `mlx-community/Qwen3-ASR-1.7B-8bit` | 5/5 ok | `entity_factual_integrity`: 5 | 15 | 89.0 | 4 accurate, 1 inaccurate |
| `runs/asr-leaderboard/qwen3-asr-1.7b-paraphrase-smoke/judge-report/results.jsonl` | `mlx-community/Qwen3-ASR-1.7B-8bit` | 5/5 ok | `semantic_paraphrase_preservation`: 5 | 15 | 100.0 | 5 accurate |
| `runs/asr-leaderboard/qwen3-asr-1.7b-noise-smoke/judge-report/results.jsonl` | `mlx-community/Qwen3-ASR-1.7B-8bit` | 5/5 ok | `acoustic_noise_robustness`: 5 | 15 | 92.4 | 4 accurate, 1 needs_review |
| `runs/asr-leaderboard/vibevoice-asr-smoke/judge-report/results.jsonl` | `mlx-community/VibeVoice-ASR-4bit` | 3/3 ok | `transcription_accuracy_wer`: 3 | 9 | 94.3 | 3 accurate |
| `runs/asr-leaderboard/vibevoice-asr-full-gap/judge-report/results.jsonl` | `mlx-community/VibeVoice-ASR-4bit` | 12/12 ok | `negation_modality_scope`: 3, `numeric_unit_integrity`: 3, `temporal_scheduling_accuracy`: 4, `transcription_accuracy_wer`: 2 | 36 | 99.3 | 12 accurate |
| `runs/asr-leaderboard/vibevoice-asr-semantic-smoke/judge-report/results.jsonl` | `mlx-community/VibeVoice-ASR-4bit` | 5/5 ok | `negation_modality_scope`: 2, `numeric_unit_integrity`: 2, `temporal_scheduling_accuracy`: 1 | 15 | 98.8 | 5 accurate |
| `runs/asr-leaderboard/vibevoice-asr-entity-smoke/judge-report/results.jsonl` | `mlx-community/VibeVoice-ASR-4bit` | 5/5 ok | `entity_factual_integrity`: 5 | 15 | 90.2 | 4 accurate, 1 inaccurate |
| `runs/asr-leaderboard/vibevoice-asr-paraphrase-smoke/judge-report/results.jsonl` | `mlx-community/VibeVoice-ASR-4bit` | 5/5 ok | `semantic_paraphrase_preservation`: 5 | 15 | 98.4 | 5 accurate |
| `runs/asr-leaderboard/vibevoice-asr-noise-smoke/judge-report/results.jsonl` | `mlx-community/VibeVoice-ASR-4bit` | 5/5 ok | `acoustic_noise_robustness`: 5 | 15 | 92.8 | 4 accurate, 1 needs_review |

## Refresh Commands

- Audio materialization: `.venv/bin/python scripts/synthesize_tts_cases.py --cases examples/asr_research_cases.jsonl --out runs/asr-research-audio --discard-text-sidecars --summary-out runs/asr-research-audio/summary.json`
- Combine and refresh committed artifacts: `.venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py --results runs/asr-leaderboard/whisper-large-v3-turbo-smoke/judge-report/results.jsonl --results runs/asr-leaderboard/whisper-large-v3-turbo-full-gap/judge-report/results.jsonl --results runs/asr-leaderboard/whisper-large-v3-turbo-semantic-smoke/judge-report/results.jsonl --results runs/asr-leaderboard/whisper-large-v3-turbo-entity-smoke/judge-report/results.jsonl --results runs/asr-leaderboard/whisper-large-v3-turbo-paraphrase-smoke/judge-report/results.jsonl --results runs/asr-leaderboard/whisper-large-v3-turbo-noise-smoke/judge-report/results.jsonl --results runs/asr-leaderboard/qwen3-asr-1.7b-smoke/judge-report/results.jsonl --results runs/asr-leaderboard/qwen3-asr-1.7b-full-gap/judge-report/results.jsonl --results runs/asr-leaderboard/qwen3-asr-1.7b-semantic-smoke/judge-report/results.jsonl --results runs/asr-leaderboard/qwen3-asr-1.7b-entity-smoke/judge-report/results.jsonl --results runs/asr-leaderboard/qwen3-asr-1.7b-paraphrase-smoke/judge-report/results.jsonl --results runs/asr-leaderboard/qwen3-asr-1.7b-noise-smoke/judge-report/results.jsonl --results runs/asr-leaderboard/vibevoice-asr-smoke/judge-report/results.jsonl --results runs/asr-leaderboard/vibevoice-asr-full-gap/judge-report/results.jsonl --results runs/asr-leaderboard/vibevoice-asr-semantic-smoke/judge-report/results.jsonl --results runs/asr-leaderboard/vibevoice-asr-entity-smoke/judge-report/results.jsonl --results runs/asr-leaderboard/vibevoice-asr-paraphrase-smoke/judge-report/results.jsonl --results runs/asr-leaderboard/vibevoice-asr-noise-smoke/judge-report/results.jsonl`
- Manifest-based refresh: `.venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py`
- Hosted artifact sync: `.venv/bin/python scripts/refresh_asr_leaderboard_artifacts.py --hosted-dir /path/to/kennethli319.github.io/open-audio-judge`

## Runtime Status

- MLX ASR: not_executed_by_refresh; transcripts loaded from verified result artifacts
- Gemini judge: verified_from_loaded_results
- Live model calls during refresh: none
- Loaded result providers: gemini
- All loaded results ok: True

Gemini secrets must be loaded only at runtime from the local secret file.
