# ASR Leaderboard Report Index

This generated index maps the hosted ASR demo to the full combined report and source run reports.

## Combined Full-35 Report

- Results JSONL: `runs/asr-leaderboard/full-35-combined/results.jsonl`
- HTML report: `runs/asr-leaderboard/full-35-combined/report.html`
- Results SHA-256: `8bff4dff3a4b3e9bd9436e10efc28ea57649ac64ab8b82e337fda43d75992dfb`
- Report SHA-256: `7816e7113843acd30b2f74a5b75f3e94b13c8593339e4f8e62b261583bdd14a1`
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
| `runs/asr-leaderboard/whisper-large-v3-turbo-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/whisper-large-v3-turbo-smoke/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/whisper-large-v3-turbo-smoke/judge-report/report.html` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | 3/3 ok | 92.0 | 46587 bytes, `22ceba42ac1bbde50dc0d5f1164c40ad8c8ad9dbb44ed88cafe4b60a86075d66` | `transcription_accuracy_wer`: 3 |
| `runs/asr-leaderboard/whisper-large-v3-turbo-full-gap/judge-report/results.jsonl` | `runs/asr-leaderboard/whisper-large-v3-turbo-full-gap/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/whisper-large-v3-turbo-full-gap/judge-report/report.html` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | 12/12 ok | 92.6 | 103328 bytes, `243301fb4d9cf9a007488389e933f34b70a384c33b74dedf0ddaf5bf6c674a26` | `negation_modality_scope`: 3, `numeric_unit_integrity`: 3, `temporal_scheduling_accuracy`: 4, `transcription_accuracy_wer`: 2 |
| `runs/asr-leaderboard/whisper-large-v3-turbo-semantic-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/whisper-large-v3-turbo-semantic-smoke/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/whisper-large-v3-turbo-semantic-smoke/judge-report/report.html` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | 5/5 ok | 100.0 | 63124 bytes, `5a8d7b2f475f8bb2e3e6f774a0142d226c4bba2ff5549b6099a6e4f27fed200a` | `negation_modality_scope`: 2, `numeric_unit_integrity`: 2, `temporal_scheduling_accuracy`: 1 |
| `runs/asr-leaderboard/whisper-large-v3-turbo-entity-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/whisper-large-v3-turbo-entity-smoke/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/whisper-large-v3-turbo-entity-smoke/judge-report/report.html` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | 5/5 ok | 88.6 | 57815 bytes, `f75cf61c922affb9ead8910448c2e53fb60197f9a147cef62cc472574fe7109d` | `entity_factual_integrity`: 5 |
| `runs/asr-leaderboard/whisper-large-v3-turbo-paraphrase-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/whisper-large-v3-turbo-paraphrase-smoke/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/whisper-large-v3-turbo-paraphrase-smoke/judge-report/report.html` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | 5/5 ok | 97.6 | 53522 bytes, `d09ae3ebd414526e0e450b5cdd942b6947834f35135d731d58eafcf98f9249e2` | `semantic_paraphrase_preservation`: 5 |
| `runs/asr-leaderboard/whisper-large-v3-turbo-noise-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/whisper-large-v3-turbo-noise-smoke/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/whisper-large-v3-turbo-noise-smoke/judge-report/report.html` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | 5/5 ok | 96.6 | 52703 bytes, `88397489492dfe14ff8758e6b2326fcec6c4b835990a810531fe5d1e027c9973` | `acoustic_noise_robustness`: 5 |
| `runs/asr-leaderboard/qwen3-asr-1.7b-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/qwen3-asr-1.7b-smoke/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/qwen3-asr-1.7b-smoke/judge-report/report.html` | `mlx-community/Qwen3-ASR-1.7B-8bit` | 3/3 ok | 88.0 | 46446 bytes, `ac0323e6fbe58aa53f3a6b98a1c6f8cf4db094a0786db41937a81f4dfac14929` | `transcription_accuracy_wer`: 3 |
| `runs/asr-leaderboard/qwen3-asr-1.7b-full-gap/judge-report/results.jsonl` | `runs/asr-leaderboard/qwen3-asr-1.7b-full-gap/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/qwen3-asr-1.7b-full-gap/judge-report/report.html` | `mlx-community/Qwen3-ASR-1.7B-8bit` | 12/12 ok | 96.7 | 101867 bytes, `34306e9292986aa7ce97f0bc972c9547b725b96230ec69588645d9ffe17569d3` | `negation_modality_scope`: 3, `numeric_unit_integrity`: 3, `temporal_scheduling_accuracy`: 4, `transcription_accuracy_wer`: 2 |
| `runs/asr-leaderboard/qwen3-asr-1.7b-semantic-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/qwen3-asr-1.7b-semantic-smoke/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/qwen3-asr-1.7b-semantic-smoke/judge-report/report.html` | `mlx-community/Qwen3-ASR-1.7B-8bit` | 5/5 ok | 100.0 | 59461 bytes, `942b37c19a5c873ae9e9d836f8d1df30038b896170e2cf28e1049038c76483d5` | `negation_modality_scope`: 2, `numeric_unit_integrity`: 2, `temporal_scheduling_accuracy`: 1 |
| `runs/asr-leaderboard/qwen3-asr-1.7b-entity-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/qwen3-asr-1.7b-entity-smoke/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/qwen3-asr-1.7b-entity-smoke/judge-report/report.html` | `mlx-community/Qwen3-ASR-1.7B-8bit` | 5/5 ok | 89.0 | 57423 bytes, `df3cf3cb02d08f0afa0169c59e47b10e78a329b030a7ae69a03ac55866769185` | `entity_factual_integrity`: 5 |
| `runs/asr-leaderboard/qwen3-asr-1.7b-paraphrase-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/qwen3-asr-1.7b-paraphrase-smoke/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/qwen3-asr-1.7b-paraphrase-smoke/judge-report/report.html` | `mlx-community/Qwen3-ASR-1.7B-8bit` | 5/5 ok | 100.0 | 51068 bytes, `8f48aba791d54993fc009125a65eb87e3de331f0a6c4e06608413ef81714785f` | `semantic_paraphrase_preservation`: 5 |
| `runs/asr-leaderboard/qwen3-asr-1.7b-noise-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/qwen3-asr-1.7b-noise-smoke/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/qwen3-asr-1.7b-noise-smoke/judge-report/report.html` | `mlx-community/Qwen3-ASR-1.7B-8bit` | 5/5 ok | 92.4 | 55483 bytes, `01e4ae83885b0d9e10a8da71d751765b2705bdf4bfc8ba1768c451c9404a103f` | `acoustic_noise_robustness`: 5 |
| `runs/asr-leaderboard/vibevoice-asr-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/vibevoice-asr-smoke/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/vibevoice-asr-smoke/judge-report/report.html` | `mlx-community/VibeVoice-ASR-4bit` | 3/3 ok | 94.3 | 42372 bytes, `b79c16d73e731333a8f599fdb783342d8f5149d02fcbe7ae3290fb2e23e213a5` | `transcription_accuracy_wer`: 3 |
| `runs/asr-leaderboard/vibevoice-asr-full-gap/judge-report/results.jsonl` | `runs/asr-leaderboard/vibevoice-asr-full-gap/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/vibevoice-asr-full-gap/judge-report/report.html` | `mlx-community/VibeVoice-ASR-4bit` | 12/12 ok | 99.3 | 97342 bytes, `0e7f6c0e1695eddc365db8b1c7007c242d66b7ca0648d2e9ad77e6983c8cc74a` | `negation_modality_scope`: 3, `numeric_unit_integrity`: 3, `temporal_scheduling_accuracy`: 4, `transcription_accuracy_wer`: 2 |
| `runs/asr-leaderboard/vibevoice-asr-semantic-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/vibevoice-asr-semantic-smoke/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/vibevoice-asr-semantic-smoke/judge-report/report.html` | `mlx-community/VibeVoice-ASR-4bit` | 5/5 ok | 98.8 | 61394 bytes, `6ed47f1eb78b8a233706c2fd1cfe92c9df010240d24b3962c34b7a19fb8baf12` | `negation_modality_scope`: 2, `numeric_unit_integrity`: 2, `temporal_scheduling_accuracy`: 1 |
| `runs/asr-leaderboard/vibevoice-asr-entity-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/vibevoice-asr-entity-smoke/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/vibevoice-asr-entity-smoke/judge-report/report.html` | `mlx-community/VibeVoice-ASR-4bit` | 5/5 ok | 90.2 | 56932 bytes, `328a585702ba6ce8e1af97d693fbce321bcb701e26b1c06eba8b4e401651bf18` | `entity_factual_integrity`: 5 |
| `runs/asr-leaderboard/vibevoice-asr-paraphrase-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/vibevoice-asr-paraphrase-smoke/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/vibevoice-asr-paraphrase-smoke/judge-report/report.html` | `mlx-community/VibeVoice-ASR-4bit` | 5/5 ok | 98.4 | 54462 bytes, `e0d96f509504e71a1b1a19645f740aec5e6ac1193b0113971e2f8bd3c672bc1b` | `semantic_paraphrase_preservation`: 5 |
| `runs/asr-leaderboard/vibevoice-asr-noise-smoke/judge-report/results.jsonl` | `runs/asr-leaderboard/vibevoice-asr-noise-smoke/judge-report/report.html` | `https://kennethli319.github.io/open-audio-judge/asr-leaderboard/vibevoice-asr-noise-smoke/judge-report/report.html` | `mlx-community/VibeVoice-ASR-4bit` | 5/5 ok | 92.8 | 56528 bytes, `b5a929f95474a4a85f4882e92abb68ee7c9b58e2095c1ee3e144c0b9d785b1d4` | `acoustic_noise_robustness`: 5 |

## Hosted Layout

- The demo page and generated docs are copied to `open-audio-judge/`.
- The combined results and report are copied to `open-audio-judge/asr-leaderboard/full-35-combined/`.
- Source run reports are copied to their matching `open-audio-judge/asr-leaderboard/.../report.html` paths when they live under `runs/asr-leaderboard/`.
