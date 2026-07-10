# ASR Leaderboard Report Index

This generated index maps the hosted ASR demo to the full combined report and source run reports.

## Combined Full-35 Report

- Results JSONL: `runs/asr-leaderboard/full-35-combined/results.jsonl`
- HTML report: `runs/asr-leaderboard/full-35-combined/report.html`
- Results SHA-256: `652343f6394ee3da19096df6d08170f979d7eb0132dabdee39a92d9c3e0fbb03`
- Report SHA-256: `03f096b07f24053b4eefd096c9c5a33059d8b8140f8dba9bebeb3c9c20162139`
- Demo page: `docs/asr-leaderboard-demo.html`
- Summary JSON: `docs/asr-leaderboard-summary.json`
- Refresh report: `docs/asr-leaderboard-refresh-report.md`
- Report links JSON: `docs/asr-leaderboard-report-links.json`
- Run manifest: `docs/asr-leaderboard-run-manifest.json`
- Run manifest SHA-256: `9766f1f88ecbf277bba4f07023abced547b6ba0d9df132d7ae0100127c52c226`
- Source result files: 18
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
| `mlx-community/whisper-large-v3-turbo-asr-fp16` | 35/35 ok | 105 | 94.3 |

## Category Matrix

| Model | WER | Numeric/Unit | Negation/Modality | Temporal | Entity | Paraphrase | Acoustic Noise |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mlx-community/VibeVoice-ASR-4bit` | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| `mlx-community/Qwen3-ASR-1.7B-8bit` | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| `mlx-community/whisper-large-v3-turbo-asr-fp16` | 5 | 5 | 5 | 5 | 5 | 5 | 5 |

## Source Run Reports

| Results | Local Report | Hosted Report | Model | Cases | Score | Report Status | Categories |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| `runs/asr-leaderboard/whisper-large-v3-turbo-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/whisper-large-v3-turbo-smoke/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/whisper-large-v3-turbo-smoke/judge-report/report.html` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | 3/3 ok | 92.0 | 45145 bytes, `1447bfbe5451a54d275f097a897fe2a8dd551a522a5eec87ac0b791ffcd67766` | `transcription_accuracy_wer`: 3 |
| `runs/asr-leaderboard/whisper-large-v3-turbo-full-gap/judge-report/results.jsonl` | `runs/asr-leaderboard/whisper-large-v3-turbo-full-gap/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/whisper-large-v3-turbo-full-gap/judge-report/report.html` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | 12/12 ok | 92.6 | 101886 bytes, `c740b61f2f91f6a3b056f44024356daf9db2b7df2c0675daa7d82ca11989708b` | `negation_modality_scope`: 3, `numeric_unit_integrity`: 3, `temporal_scheduling_accuracy`: 4, `transcription_accuracy_wer`: 2 |
| `runs/asr-leaderboard/whisper-large-v3-turbo-semantic-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/whisper-large-v3-turbo-semantic-smoke/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/whisper-large-v3-turbo-semantic-smoke/judge-report/report.html` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | 5/5 ok | 100.0 | 60761 bytes, `0c45cd42b488c66510f7bf50b6e2cc115d87713fa00ae6671feb1fc9aa070009` | `negation_modality_scope`: 2, `numeric_unit_integrity`: 2, `temporal_scheduling_accuracy`: 1 |
| `runs/asr-leaderboard/whisper-large-v3-turbo-entity-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/whisper-large-v3-turbo-entity-smoke/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/whisper-large-v3-turbo-entity-smoke/judge-report/report.html` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | 5/5 ok | 88.6 | 56373 bytes, `108931082c11b55d3f1369cde392382e6b1340998b9625aeaf3f81576518db92` | `entity_factual_integrity`: 5 |
| `runs/asr-leaderboard/whisper-large-v3-turbo-paraphrase-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/whisper-large-v3-turbo-paraphrase-smoke/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/whisper-large-v3-turbo-paraphrase-smoke/judge-report/report.html` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | 5/5 ok | 97.6 | 52111 bytes, `be307e0a5e7c96483dee43d1eb07ff4d620c1358ee44eddd4672a6972f572dcf` | `semantic_paraphrase_preservation`: 5 |
| `runs/asr-leaderboard/whisper-large-v3-turbo-noise-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/whisper-large-v3-turbo-noise-smoke/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/whisper-large-v3-turbo-noise-smoke/judge-report/report.html` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | 5/5 ok | 96.6 | 51292 bytes, `a2ca275e11420d16cb777080a468f2acd7c88cbf1388bb849f9b641d2d1b9989` | `acoustic_noise_robustness`: 5 |
| `runs/asr-leaderboard/qwen3-asr-1.7b-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/qwen3-asr-1.7b-smoke/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/qwen3-asr-1.7b-smoke/judge-report/report.html` | `mlx-community/Qwen3-ASR-1.7B-8bit` | 3/3 ok | 88.0 | 45004 bytes, `3ecc8918a7bd486e992656993da27d0d7a52e86ab665d20b1ca8dab0a4cd9060` | `transcription_accuracy_wer`: 3 |
| `runs/asr-leaderboard/qwen3-asr-1.7b-full-gap/judge-report/results.jsonl` | `runs/asr-leaderboard/qwen3-asr-1.7b-full-gap/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/qwen3-asr-1.7b-full-gap/judge-report/report.html` | `mlx-community/Qwen3-ASR-1.7B-8bit` | 12/12 ok | 96.7 | 100425 bytes, `37c2cfeefda89ecc394955b17cc3764e251501ce5d63c32a5f3182ffda7cf114` | `negation_modality_scope`: 3, `numeric_unit_integrity`: 3, `temporal_scheduling_accuracy`: 4, `transcription_accuracy_wer`: 2 |
| `runs/asr-leaderboard/qwen3-asr-1.7b-semantic-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/qwen3-asr-1.7b-semantic-smoke/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/qwen3-asr-1.7b-semantic-smoke/judge-report/report.html` | `mlx-community/Qwen3-ASR-1.7B-8bit` | 5/5 ok | 100.0 | 58050 bytes, `5e911ee259d99fe9d3105049d398130bc9d8a4c14d07e82aa146ddbf837c7564` | `negation_modality_scope`: 2, `numeric_unit_integrity`: 2, `temporal_scheduling_accuracy`: 1 |
| `runs/asr-leaderboard/qwen3-asr-1.7b-entity-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/qwen3-asr-1.7b-entity-smoke/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/qwen3-asr-1.7b-entity-smoke/judge-report/report.html` | `mlx-community/Qwen3-ASR-1.7B-8bit` | 5/5 ok | 89.0 | 55981 bytes, `f3d8e36f85e9247c4778fd899ddfb2c02d39c12fde83ee7dd6a1ebc1ca2273a1` | `entity_factual_integrity`: 5 |
| `runs/asr-leaderboard/qwen3-asr-1.7b-paraphrase-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/qwen3-asr-1.7b-paraphrase-smoke/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/qwen3-asr-1.7b-paraphrase-smoke/judge-report/report.html` | `mlx-community/Qwen3-ASR-1.7B-8bit` | 5/5 ok | 100.0 | 49657 bytes, `eaf65460d2b37510c6f1caa4fc50e10325e7862ecc0527ca79c3044aa545d6c0` | `semantic_paraphrase_preservation`: 5 |
| `runs/asr-leaderboard/qwen3-asr-1.7b-noise-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/qwen3-asr-1.7b-noise-smoke/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/qwen3-asr-1.7b-noise-smoke/judge-report/report.html` | `mlx-community/Qwen3-ASR-1.7B-8bit` | 5/5 ok | 92.4 | 54041 bytes, `009888bc039b4d7388631aa413097a4d9c0bd0bee95da2c6858e465c3564a8fb` | `acoustic_noise_robustness`: 5 |
| `runs/asr-leaderboard/vibevoice-asr-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/vibevoice-asr-smoke/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/vibevoice-asr-smoke/judge-report/report.html` | `mlx-community/VibeVoice-ASR-4bit` | 3/3 ok | 94.3 | 40961 bytes, `b069a252a5c69ca08ec31bd536af261f54a4ffc58fefe2b28fa2d48f793e581f` | `transcription_accuracy_wer`: 3 |
| `runs/asr-leaderboard/vibevoice-asr-full-gap/judge-report/results.jsonl` | `runs/asr-leaderboard/vibevoice-asr-full-gap/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/vibevoice-asr-full-gap/judge-report/report.html` | `mlx-community/VibeVoice-ASR-4bit` | 12/12 ok | 99.3 | 95931 bytes, `1ab919c7476612f9f4e5eb7110a1f995a7ee5db9ea09bcde80ba39bcc31ee17b` | `negation_modality_scope`: 3, `numeric_unit_integrity`: 3, `temporal_scheduling_accuracy`: 4, `transcription_accuracy_wer`: 2 |
| `runs/asr-leaderboard/vibevoice-asr-semantic-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/vibevoice-asr-semantic-smoke/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/vibevoice-asr-semantic-smoke/judge-report/report.html` | `mlx-community/VibeVoice-ASR-4bit` | 5/5 ok | 98.8 | 59983 bytes, `6a4533b2b49d4a1206b13425d48b3242915c578af30ba5aac43014bfce578963` | `negation_modality_scope`: 2, `numeric_unit_integrity`: 2, `temporal_scheduling_accuracy`: 1 |
| `runs/asr-leaderboard/vibevoice-asr-entity-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/vibevoice-asr-entity-smoke/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/vibevoice-asr-entity-smoke/judge-report/report.html` | `mlx-community/VibeVoice-ASR-4bit` | 5/5 ok | 90.2 | 55490 bytes, `4a7888cfdb9d486df4e48e7c5c5641a363d5b1061a95adf07d4ad711ecaea0e5` | `entity_factual_integrity`: 5 |
| `runs/asr-leaderboard/vibevoice-asr-paraphrase-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/vibevoice-asr-paraphrase-smoke/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/vibevoice-asr-paraphrase-smoke/judge-report/report.html` | `mlx-community/VibeVoice-ASR-4bit` | 5/5 ok | 98.4 | 53051 bytes, `7ddbf7c82a822d251c6f5a5daf892fde8588e9a310753ae6ce07a2751bd09048` | `semantic_paraphrase_preservation`: 5 |
| `runs/asr-leaderboard/vibevoice-asr-noise-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/vibevoice-asr-noise-smoke/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/vibevoice-asr-noise-smoke/judge-report/report.html` | `mlx-community/VibeVoice-ASR-4bit` | 5/5 ok | 92.8 | 55086 bytes, `ae14250f92728086b3ba25ac0fe5340f56c3d0c01352130529fb618f226ba9ff` | `acoustic_noise_robustness`: 5 |

## Hosted Layout

- The demo page and generated docs are copied to `open-audio-judge/`.
- The combined full-35 results and report are copied to `open-audio-judge/asr-leaderboard/full-35-combined/`.
- Source run reports are copied to their matching `open-audio-judge/asr-leaderboard/.../report.html` paths when they live under `runs/asr-leaderboard/`.
