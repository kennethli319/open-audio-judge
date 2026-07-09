# ASR Leaderboard Report Index

This generated index maps the hosted ASR demo to the full combined report and source run reports.

## Combined Full-35 Report

- Results JSONL: `runs/asr-leaderboard/full-35-combined/results.jsonl`
- HTML report: `runs/asr-leaderboard/full-35-combined/report.html`
- Results SHA-256: `381ea5a536fed20456887b2bb171b64618f51ba5b22cb0b23a625189cfa61976`
- Report SHA-256: `21246cd2be336e37d0c5da76ab7edfd7944e4bb38c509a0c3ccce9dc5fd45cf7`
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

## Category Matrix

| Model | WER | Numeric/Unit | Negation/Modality | Temporal | Entity | Paraphrase | Acoustic Noise |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mlx-community/VibeVoice-ASR-4bit` | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| `mlx-community/Qwen3-ASR-1.7B-8bit` | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| `mlx-community/whisper-large-v3-turbo-asr-fp16` | 5 | 5 | 5 | 5 | 5 | 5 | 5 |

## Source Run Reports

| Results | Report | Model | Cases | Score | Report Status | Categories |
| --- | --- | --- | ---: | ---: | --- | --- |
| `runs/asr-leaderboard/whisper-large-v3-turbo-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/whisper-large-v3-turbo-smoke/judge-report/report.html` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | 3/3 ok | 92.0 | 44479 bytes, `b2c8783dc22204ce4ea1db6ea14cb1039f8e0eee36ff77f579778bdf3f39432f` | `transcription_accuracy_wer`: 3 |
| `runs/asr-leaderboard/whisper-large-v3-turbo-full-gap/judge-report/results.jsonl` | `runs/asr-leaderboard/whisper-large-v3-turbo-full-gap/judge-report/report.html` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | 12/12 ok | 92.6 | 89605 bytes, `9ac46fcbbc4e0ab2702e9e9b0033999b1065a4f2482b6f6334ddda63118cedc6` | `negation_modality_scope`: 3, `numeric_unit_integrity`: 3, `temporal_scheduling_accuracy`: 4, `transcription_accuracy_wer`: 2 |
| `runs/asr-leaderboard/whisper-large-v3-turbo-semantic-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/whisper-large-v3-turbo-semantic-smoke/judge-report/report.html` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | 5/5 ok | 93.4 | 56416 bytes, `15e588b5b78495d008231f1c8dc8b0efefc1f919f40c83c5ea0da79863dd8e84` | `negation_modality_scope`: 2, `numeric_unit_integrity`: 2, `temporal_scheduling_accuracy`: 1 |
| `runs/asr-leaderboard/whisper-large-v3-turbo-entity-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/whisper-large-v3-turbo-entity-smoke/judge-report/report.html` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | 5/5 ok | 88.6 | 54821 bytes, `a2749dd85b93f462ba9ea39a217bf49d832b23bb0629be23bce71dabcf7349a6` | `entity_factual_integrity`: 5 |
| `runs/asr-leaderboard/whisper-large-v3-turbo-paraphrase-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/whisper-large-v3-turbo-paraphrase-smoke/judge-report/report.html` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | 5/5 ok | 97.6 | 50021 bytes, `667ec7f9ef23cccda1712650a1c5d85bda7c2251920dc70d940673c958552a81` | `semantic_paraphrase_preservation`: 5 |
| `runs/asr-leaderboard/whisper-large-v3-turbo-noise-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/whisper-large-v3-turbo-noise-smoke/judge-report/report.html` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | 5/5 ok | 96.6 | 49449 bytes, `959a2192e14927b8fde329ebf3a05ee5e836f9aed50a80e22172f575614a5300` | `acoustic_noise_robustness`: 5 |
| `runs/asr-leaderboard/qwen3-asr-1.7b-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/qwen3-asr-1.7b-smoke/judge-report/report.html` | `mlx-community/Qwen3-ASR-1.7B-8bit` | 3/3 ok | 88.0 | 44401 bytes, `1d112998b833ec8ec2a33d3dad408658c34818f2c87d714d64726559fb1ac69e` | `transcription_accuracy_wer`: 3 |
| `runs/asr-leaderboard/qwen3-asr-1.7b-full-gap/judge-report/results.jsonl` | `runs/asr-leaderboard/qwen3-asr-1.7b-full-gap/judge-report/report.html` | `mlx-community/Qwen3-ASR-1.7B-8bit` | 12/12 ok | 96.7 | 88026 bytes, `9aabf21100b2de38d86619ac79e5fe0a4cb5b96ea868ed9469b22a7c5087c051` | `negation_modality_scope`: 3, `numeric_unit_integrity`: 3, `temporal_scheduling_accuracy`: 4, `transcription_accuracy_wer`: 2 |
| `runs/asr-leaderboard/qwen3-asr-1.7b-semantic-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/qwen3-asr-1.7b-semantic-smoke/judge-report/report.html` | `mlx-community/Qwen3-ASR-1.7B-8bit` | 5/5 ok | 100.0 | 53520 bytes, `13c93e81662c83445a5097354af2a9416ca39e4b3ad7ddaf535d9c8bdd00e852` | `negation_modality_scope`: 2, `numeric_unit_integrity`: 2, `temporal_scheduling_accuracy`: 1 |
| `runs/asr-leaderboard/qwen3-asr-1.7b-entity-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/qwen3-asr-1.7b-entity-smoke/judge-report/report.html` | `mlx-community/Qwen3-ASR-1.7B-8bit` | 5/5 ok | 89.0 | 53538 bytes, `8ee58dc1f78837ff66a168e9e7317504793ba3fd1fa5f956fd69c59633ad7b53` | `entity_factual_integrity`: 5 |
| `runs/asr-leaderboard/qwen3-asr-1.7b-paraphrase-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/qwen3-asr-1.7b-paraphrase-smoke/judge-report/report.html` | `mlx-community/Qwen3-ASR-1.7B-8bit` | 5/5 ok | 100.0 | 47682 bytes, `643c4825f4b554c1fe1e460d839bf65fa8a627a676d511caf7bf29864b2e1a2a` | `semantic_paraphrase_preservation`: 5 |
| `runs/asr-leaderboard/qwen3-asr-1.7b-noise-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/qwen3-asr-1.7b-noise-smoke/judge-report/report.html` | `mlx-community/Qwen3-ASR-1.7B-8bit` | 5/5 ok | 92.4 | 51834 bytes, `eed8a650d0038098bac0464db1a7d9d21bf023a91bfa03bf9f26abb33767bfd3` | `acoustic_noise_robustness`: 5 |
| `runs/asr-leaderboard/vibevoice-asr-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/vibevoice-asr-smoke/judge-report/report.html` | `mlx-community/VibeVoice-ASR-4bit` | 3/3 ok | 94.3 | 40503 bytes, `e4d426984afa425e292f08ee79719f243178d02e89ddc12e9cbcc0e6a5608907` | `transcription_accuracy_wer`: 3 |
| `runs/asr-leaderboard/vibevoice-asr-full-gap/judge-report/results.jsonl` | `runs/asr-leaderboard/vibevoice-asr-full-gap/judge-report/report.html` | `mlx-community/VibeVoice-ASR-4bit` | 12/12 ok | 99.3 | 83874 bytes, `7e9f94596faa19805ecf3eb6aaca459fdaf0e1d082b099b7f0e6356282ed0538` | `negation_modality_scope`: 3, `numeric_unit_integrity`: 3, `temporal_scheduling_accuracy`: 4, `transcription_accuracy_wer`: 2 |
| `runs/asr-leaderboard/vibevoice-asr-semantic-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/vibevoice-asr-semantic-smoke/judge-report/report.html` | `mlx-community/VibeVoice-ASR-4bit` | 5/5 ok | 98.8 | 55384 bytes, `cfd7a4dafb899d9102025096cda2b511a12cdade373ebd72b78352cea611a526` | `negation_modality_scope`: 2, `numeric_unit_integrity`: 2, `temporal_scheduling_accuracy`: 1 |
| `runs/asr-leaderboard/vibevoice-asr-entity-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/vibevoice-asr-entity-smoke/judge-report/report.html` | `mlx-community/VibeVoice-ASR-4bit` | 5/5 ok | 90.2 | 53374 bytes, `7719f0881d29f9602a68dd4c5f90923e281e6046b1001ffe8372c93197724802` | `entity_factual_integrity`: 5 |
| `runs/asr-leaderboard/vibevoice-asr-paraphrase-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/vibevoice-asr-paraphrase-smoke/judge-report/report.html` | `mlx-community/VibeVoice-ASR-4bit` | 5/5 ok | 98.4 | 50209 bytes, `9847426ce105299b8395c2d0c4d1b3382676e561f4e19395b745609876f7d139` | `semantic_paraphrase_preservation`: 5 |
| `runs/asr-leaderboard/vibevoice-asr-noise-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/vibevoice-asr-noise-smoke/judge-report/report.html` | `mlx-community/VibeVoice-ASR-4bit` | 5/5 ok | 92.8 | 52927 bytes, `3a5db2b26d710b9970feb3fb0358b660835ab65d6eb60fb053724198b677a226` | `acoustic_noise_robustness`: 5 |

## Hosted Layout

- The demo page and generated docs are copied to `open-audio-judge/`.
- The combined full-35 results and report are copied to `open-audio-judge/asr-leaderboard/full-35-combined/`.
- Source run reports remain local unless their run directories are explicitly published.
