# ASR Leaderboard Report Index

This generated index maps the hosted ASR demo to the full combined report and source run reports.

## Combined Full-35 Report

- Results JSONL: `runs/asr-leaderboard/full-35-combined/results.jsonl`
- HTML report: `runs/asr-leaderboard/full-35-combined/report.html`
- Demo page: `docs/asr-leaderboard-demo.html`
- Summary JSON: `docs/asr-leaderboard-summary.json`
- Refresh report: `docs/asr-leaderboard-refresh-report.md`
- Report links JSON: `docs/asr-leaderboard-report-links.json`
- Hosted demo URL: `https://kennethli319.github.io/open-audio-judge/asr-leaderboard-demo.html`
- Hosted combined report URL: `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/full-35-combined/report.html`

## Coverage

- Total judged transcripts: 105
- Models: 3
- Expected cases per model: 35

| Model | Cases | Gemini Samples | Average Score |
| --- | ---: | ---: | ---: |
| `mlx-community/VibeVoice-ASR-4bit` | 35/35 ok | 105 | 96.5 |
| `mlx-community/Qwen3-ASR-1.7B-8bit` | 35/35 ok | 105 | 95.2 |
| `mlx-community/whisper-large-v3-turbo-asr-fp16` | 35/35 ok | 105 | 93.4 |

## Source Run Reports

| Results | Report | Model | Cases | Categories |
| --- | --- | --- | ---: | --- |
| `runs/asr-leaderboard/whisper-large-v3-turbo-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/whisper-large-v3-turbo-smoke/judge-report/report.html` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | 3/3 ok | `transcription_accuracy_wer`: 3 |
| `runs/asr-leaderboard/whisper-large-v3-turbo-full-gap/judge-report/results.jsonl` | `runs/asr-leaderboard/whisper-large-v3-turbo-full-gap/judge-report/report.html` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | 12/12 ok | `negation_modality_scope`: 3, `numeric_unit_integrity`: 3, `temporal_scheduling_accuracy`: 4, `transcription_accuracy_wer`: 2 |
| `runs/asr-leaderboard/whisper-large-v3-turbo-semantic-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/whisper-large-v3-turbo-semantic-smoke/judge-report/report.html` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | 5/5 ok | `negation_modality_scope`: 2, `numeric_unit_integrity`: 2, `temporal_scheduling_accuracy`: 1 |
| `runs/asr-leaderboard/whisper-large-v3-turbo-entity-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/whisper-large-v3-turbo-entity-smoke/judge-report/report.html` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | 5/5 ok | `entity_factual_integrity`: 5 |
| `runs/asr-leaderboard/whisper-large-v3-turbo-paraphrase-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/whisper-large-v3-turbo-paraphrase-smoke/judge-report/report.html` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | 5/5 ok | `semantic_paraphrase_preservation`: 5 |
| `runs/asr-leaderboard/whisper-large-v3-turbo-noise-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/whisper-large-v3-turbo-noise-smoke/judge-report/report.html` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | 5/5 ok | `acoustic_noise_robustness`: 5 |
| `runs/asr-leaderboard/qwen3-asr-1.7b-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/qwen3-asr-1.7b-smoke/judge-report/report.html` | `mlx-community/Qwen3-ASR-1.7B-8bit` | 3/3 ok | `transcription_accuracy_wer`: 3 |
| `runs/asr-leaderboard/qwen3-asr-1.7b-full-gap/judge-report/results.jsonl` | `runs/asr-leaderboard/qwen3-asr-1.7b-full-gap/judge-report/report.html` | `mlx-community/Qwen3-ASR-1.7B-8bit` | 12/12 ok | `negation_modality_scope`: 3, `numeric_unit_integrity`: 3, `temporal_scheduling_accuracy`: 4, `transcription_accuracy_wer`: 2 |
| `runs/asr-leaderboard/qwen3-asr-1.7b-semantic-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/qwen3-asr-1.7b-semantic-smoke/judge-report/report.html` | `mlx-community/Qwen3-ASR-1.7B-8bit` | 5/5 ok | `negation_modality_scope`: 2, `numeric_unit_integrity`: 2, `temporal_scheduling_accuracy`: 1 |
| `runs/asr-leaderboard/qwen3-asr-1.7b-entity-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/qwen3-asr-1.7b-entity-smoke/judge-report/report.html` | `mlx-community/Qwen3-ASR-1.7B-8bit` | 5/5 ok | `entity_factual_integrity`: 5 |
| `runs/asr-leaderboard/qwen3-asr-1.7b-paraphrase-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/qwen3-asr-1.7b-paraphrase-smoke/judge-report/report.html` | `mlx-community/Qwen3-ASR-1.7B-8bit` | 5/5 ok | `semantic_paraphrase_preservation`: 5 |
| `runs/asr-leaderboard/qwen3-asr-1.7b-noise-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/qwen3-asr-1.7b-noise-smoke/judge-report/report.html` | `mlx-community/Qwen3-ASR-1.7B-8bit` | 5/5 ok | `acoustic_noise_robustness`: 5 |
| `runs/asr-leaderboard/vibevoice-asr-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/vibevoice-asr-smoke/judge-report/report.html` | `mlx-community/VibeVoice-ASR-4bit` | 3/3 ok | `transcription_accuracy_wer`: 3 |
| `runs/asr-leaderboard/vibevoice-asr-full-gap/judge-report/results.jsonl` | `runs/asr-leaderboard/vibevoice-asr-full-gap/judge-report/report.html` | `mlx-community/VibeVoice-ASR-4bit` | 12/12 ok | `negation_modality_scope`: 3, `numeric_unit_integrity`: 3, `temporal_scheduling_accuracy`: 4, `transcription_accuracy_wer`: 2 |
| `runs/asr-leaderboard/vibevoice-asr-semantic-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/vibevoice-asr-semantic-smoke/judge-report/report.html` | `mlx-community/VibeVoice-ASR-4bit` | 5/5 ok | `negation_modality_scope`: 2, `numeric_unit_integrity`: 2, `temporal_scheduling_accuracy`: 1 |
| `runs/asr-leaderboard/vibevoice-asr-entity-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/vibevoice-asr-entity-smoke/judge-report/report.html` | `mlx-community/VibeVoice-ASR-4bit` | 5/5 ok | `entity_factual_integrity`: 5 |
| `runs/asr-leaderboard/vibevoice-asr-paraphrase-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/vibevoice-asr-paraphrase-smoke/judge-report/report.html` | `mlx-community/VibeVoice-ASR-4bit` | 5/5 ok | `semantic_paraphrase_preservation`: 5 |
| `runs/asr-leaderboard/vibevoice-asr-noise-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/vibevoice-asr-noise-smoke/judge-report/report.html` | `mlx-community/VibeVoice-ASR-4bit` | 5/5 ok | `acoustic_noise_robustness`: 5 |

## Hosted Layout

- The demo page and generated docs are copied to `open-audio-judge/`.
- The combined full-35 results and report are copied to `open-audio-judge/asr-leaderboard/full-35-combined/`.
- Source run reports remain local unless their run directories are explicitly published.
