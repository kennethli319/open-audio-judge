# ASR Leaderboard Report Index

This generated index maps the hosted ASR demo to the full combined report and source run reports.

## Combined Full-35 Report

- Results JSONL: `runs/asr-leaderboard/full-35-combined/results.jsonl`
- HTML report: `runs/asr-leaderboard/full-35-combined/report.html`
- Results SHA-256: `8bff4dff3a4b3e9bd9436e10efc28ea57649ac64ab8b82e337fda43d75992dfb`
- Report SHA-256: `051245e138659e17ced1f556e45bdeb39b249d4d4bb4b5de3ff33422844a454c`
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
| `runs/asr-leaderboard/whisper-large-v3-turbo-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/whisper-large-v3-turbo-smoke/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/whisper-large-v3-turbo-smoke/judge-report/report.html` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | 3/3 ok | 92.0 | 45929 bytes, `857670c787c395806d12d23a126c8237e58b9292fad4af59d04e9d85bd1a1251` | `transcription_accuracy_wer`: 3 |
| `runs/asr-leaderboard/whisper-large-v3-turbo-full-gap/judge-report/results.jsonl` | `runs/asr-leaderboard/whisper-large-v3-turbo-full-gap/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/whisper-large-v3-turbo-full-gap/judge-report/report.html` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | 12/12 ok | 92.6 | 102670 bytes, `e4d01278bdf722e9135e34d1bb413bc09209a547b8db03ae4bea75e913d02ab1` | `negation_modality_scope`: 3, `numeric_unit_integrity`: 3, `temporal_scheduling_accuracy`: 4, `transcription_accuracy_wer`: 2 |
| `runs/asr-leaderboard/whisper-large-v3-turbo-semantic-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/whisper-large-v3-turbo-semantic-smoke/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/whisper-large-v3-turbo-semantic-smoke/judge-report/report.html` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | 5/5 ok | 100.0 | 62466 bytes, `39135f0dd8102aa54992e630de05345bf99397f65a9f1e46c0124fd8ec40894f` | `negation_modality_scope`: 2, `numeric_unit_integrity`: 2, `temporal_scheduling_accuracy`: 1 |
| `runs/asr-leaderboard/whisper-large-v3-turbo-entity-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/whisper-large-v3-turbo-entity-smoke/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/whisper-large-v3-turbo-entity-smoke/judge-report/report.html` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | 5/5 ok | 88.6 | 57157 bytes, `6332f49567e4ad2989af9e773f11c33e45c4385c28a9ac29bd6bfca61ab73341` | `entity_factual_integrity`: 5 |
| `runs/asr-leaderboard/whisper-large-v3-turbo-paraphrase-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/whisper-large-v3-turbo-paraphrase-smoke/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/whisper-large-v3-turbo-paraphrase-smoke/judge-report/report.html` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | 5/5 ok | 97.6 | 52864 bytes, `51febd51232b756acfc1beb0b8c7236f9f4e79ba55ba9b4ed9472fb42de60ded` | `semantic_paraphrase_preservation`: 5 |
| `runs/asr-leaderboard/whisper-large-v3-turbo-noise-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/whisper-large-v3-turbo-noise-smoke/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/whisper-large-v3-turbo-noise-smoke/judge-report/report.html` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | 5/5 ok | 96.6 | 52045 bytes, `cab16af6f4053bb49e02d889377dd20b6ae4751b56dd575b3c2c435f08f1358a` | `acoustic_noise_robustness`: 5 |
| `runs/asr-leaderboard/qwen3-asr-1.7b-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/qwen3-asr-1.7b-smoke/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/qwen3-asr-1.7b-smoke/judge-report/report.html` | `mlx-community/Qwen3-ASR-1.7B-8bit` | 3/3 ok | 88.0 | 45788 bytes, `1358c92b7a1f272560f10cddf41b661b6432682b8b6bd9bbc0a1b89d16fe4eae` | `transcription_accuracy_wer`: 3 |
| `runs/asr-leaderboard/qwen3-asr-1.7b-full-gap/judge-report/results.jsonl` | `runs/asr-leaderboard/qwen3-asr-1.7b-full-gap/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/qwen3-asr-1.7b-full-gap/judge-report/report.html` | `mlx-community/Qwen3-ASR-1.7B-8bit` | 12/12 ok | 96.7 | 101209 bytes, `e325c7b69d42c67b4bc25bd2135ab79ce5f2b52b5f29b5d20c65c1cd4c433e24` | `negation_modality_scope`: 3, `numeric_unit_integrity`: 3, `temporal_scheduling_accuracy`: 4, `transcription_accuracy_wer`: 2 |
| `runs/asr-leaderboard/qwen3-asr-1.7b-semantic-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/qwen3-asr-1.7b-semantic-smoke/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/qwen3-asr-1.7b-semantic-smoke/judge-report/report.html` | `mlx-community/Qwen3-ASR-1.7B-8bit` | 5/5 ok | 100.0 | 58803 bytes, `98409f1b77c2c526618176aee43bb1bd2f34962b8020a9202b403bc14059b024` | `negation_modality_scope`: 2, `numeric_unit_integrity`: 2, `temporal_scheduling_accuracy`: 1 |
| `runs/asr-leaderboard/qwen3-asr-1.7b-entity-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/qwen3-asr-1.7b-entity-smoke/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/qwen3-asr-1.7b-entity-smoke/judge-report/report.html` | `mlx-community/Qwen3-ASR-1.7B-8bit` | 5/5 ok | 89.0 | 56765 bytes, `996801b6f86c23ba97f9bd6f7b1ebd863e1232703ab100ea5f7937b90157733d` | `entity_factual_integrity`: 5 |
| `runs/asr-leaderboard/qwen3-asr-1.7b-paraphrase-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/qwen3-asr-1.7b-paraphrase-smoke/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/qwen3-asr-1.7b-paraphrase-smoke/judge-report/report.html` | `mlx-community/Qwen3-ASR-1.7B-8bit` | 5/5 ok | 100.0 | 50410 bytes, `9c8f615fb2ace98e85f4101e90316496a5e44db12693d7912ec7cc5bd5c028c8` | `semantic_paraphrase_preservation`: 5 |
| `runs/asr-leaderboard/qwen3-asr-1.7b-noise-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/qwen3-asr-1.7b-noise-smoke/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/qwen3-asr-1.7b-noise-smoke/judge-report/report.html` | `mlx-community/Qwen3-ASR-1.7B-8bit` | 5/5 ok | 92.4 | 54825 bytes, `ed899c0cd759798abbfb9240ecbd33865944a009f298c3d232b17e25561981a0` | `acoustic_noise_robustness`: 5 |
| `runs/asr-leaderboard/vibevoice-asr-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/vibevoice-asr-smoke/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/vibevoice-asr-smoke/judge-report/report.html` | `mlx-community/VibeVoice-ASR-4bit` | 3/3 ok | 94.3 | 41714 bytes, `52c712586439b12b6b40dfb9ac2970db7786e941e3b63f2b267c4c40de827aba` | `transcription_accuracy_wer`: 3 |
| `runs/asr-leaderboard/vibevoice-asr-full-gap/judge-report/results.jsonl` | `runs/asr-leaderboard/vibevoice-asr-full-gap/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/vibevoice-asr-full-gap/judge-report/report.html` | `mlx-community/VibeVoice-ASR-4bit` | 12/12 ok | 99.3 | 96684 bytes, `84a34229f885029cccc93ee2bb6ea338fb3b7529dbb12aefd23fa90bf7064f2c` | `negation_modality_scope`: 3, `numeric_unit_integrity`: 3, `temporal_scheduling_accuracy`: 4, `transcription_accuracy_wer`: 2 |
| `runs/asr-leaderboard/vibevoice-asr-semantic-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/vibevoice-asr-semantic-smoke/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/vibevoice-asr-semantic-smoke/judge-report/report.html` | `mlx-community/VibeVoice-ASR-4bit` | 5/5 ok | 98.8 | 60736 bytes, `a4793291ac5d40bfce8be37cba19829f3073318e2d6519c42fa28446b8735495` | `negation_modality_scope`: 2, `numeric_unit_integrity`: 2, `temporal_scheduling_accuracy`: 1 |
| `runs/asr-leaderboard/vibevoice-asr-entity-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/vibevoice-asr-entity-smoke/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/vibevoice-asr-entity-smoke/judge-report/report.html` | `mlx-community/VibeVoice-ASR-4bit` | 5/5 ok | 90.2 | 56274 bytes, `476e1dee71027d9b857eaaad5004bc87499eb943693b0b0135808dd0e2b00309` | `entity_factual_integrity`: 5 |
| `runs/asr-leaderboard/vibevoice-asr-paraphrase-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/vibevoice-asr-paraphrase-smoke/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/vibevoice-asr-paraphrase-smoke/judge-report/report.html` | `mlx-community/VibeVoice-ASR-4bit` | 5/5 ok | 98.4 | 53804 bytes, `9c314c29420c5c2ac3daaa3a5e864f04bba049dd8b1d059c19a1033f991eba4a` | `semantic_paraphrase_preservation`: 5 |
| `runs/asr-leaderboard/vibevoice-asr-noise-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/vibevoice-asr-noise-smoke/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/vibevoice-asr-noise-smoke/judge-report/report.html` | `mlx-community/VibeVoice-ASR-4bit` | 5/5 ok | 92.8 | 55870 bytes, `e21dbae2a10a0c95880a03a97a7aa1742c804d2596fd14327337235c9edab457` | `acoustic_noise_robustness`: 5 |

## Hosted Layout

- The demo page and generated docs are copied to `open-audio-judge/`.
- The combined results and report are copied to `open-audio-judge/asr-leaderboard/full-35-combined/`.
- Source run reports are copied to their matching `open-audio-judge/asr-leaderboard/.../report.html` paths when they live under `runs/asr-leaderboard/`.
