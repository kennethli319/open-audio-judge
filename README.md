# Open Audio Judge

Open Audio Judge is a prompt-based evaluation harness for audio LLM judges. The goal is to make ASR, TTS, VAD, diarization, speech translation, and speech-event evaluations comparable even when teams run the judge on private datasets.

**Live demo:** [TTS model leaderboard judged by Gemini](https://kennethli319.github.io/open-audio-judge/tts-leaderboard-demo.html)

The first MVP focuses on:

- A reusable judge protocol: audio sample + task context + prompt rubric + strict JSON result.
- A local Qwen/Qwen3-Omni-compatible provider through an OpenAI-style `/v1/chat/completions` endpoint.
- Two initial prompts: TTS naturalness and semantic ASR transcription quality.
- CLI and REST API entry points.
- HTML reports with score bars, labels, reasons, and per-case diagnostics instead of only raw JSON.

## Why This Shape

Recent work supports a few design choices:

- LLM-as-judge systems need explicit criteria, structured outputs, and meta-evaluation because prompts can bias results ([Li et al., 2024](https://arxiv.org/html/2412.05579v2)).
- Audio LLM benchmarks now cover speech, audio-scene, and paralinguistic understanding, so the protocol should be task-extensible rather than ASR-only ([AudioBench, NAACL 2025](https://aclanthology.org/2025.naacl-long.218/)).
- ASR evaluation should not rely on WER alone because WER misses meaning preservation and downstream impact ([Google Research, 2024](https://research.google/blog/assessing-asr-performance-with-meaning-preservation/); [Pulikodan et al., Interspeech 2025](https://www.isca-archive.org/interspeech_2025/pulikodan25_interspeech.pdf)).
- Audio-aware judges can align with humans on some speech qualities, but the evidence is task-specific, so prompts should name the exact criterion being judged ([Chiang et al., 2025](https://arxiv.org/html/2506.05984v1)).
- Speech-quality judge research is moving toward structured, explanation-rich outputs, not scalar-only MOS predictions ([SpeechLLM-as-Judges, 2025](https://arxiv.org/html/2510.14664v1)).

See [docs/research.md](docs/research.md), [docs/tts-eval-taxonomy.md](docs/tts-eval-taxonomy.md),
and [docs/plan.md](docs/plan.md) for the full research notes, TTS category taxonomy, and project
plan.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
oaj eval --provider mock --judge asr_error --cases examples/asr_cases.jsonl --out runs/asr-demo
open runs/asr-demo/report.html
```

The mock provider is deterministic and useful for testing the pipeline without a model.
For ASR cases with a reference transcript, the mock provider now uses a lightweight semantic baseline:
it still considers edit distance, but caps scores for high-impact meaning changes such as negation,
numbers, units, dates/times, and entity-like terms. Basic digit and number-word equivalents, such as `15` and `fifteen`,
are normalized before applying the number-error cap. This keeps local smoke reports aligned with the
project goal of going beyond WER before a real audio LLM judge is connected.
The bundled ASR examples include synthetic calibration cases where a small token change has high
semantic impact, such as negation, amount, unit, appointment/deadline, and entity substitutions.

## Build Local TTS Case Manifests

Use `build-tts-cases` to transform a local text evalset JSONL into a TTS naturalness case manifest.
The command preserves multi-turn context and records only source ids/categories/tags in metadata by
default; write outputs under `runs/` unless the source text is safe to publish.

```bash
oaj build-tts-cases \
  --source /path/to/evalset/seed_v0.jsonl \
  --source-name ome \
  --limit 25 \
  --hash-source-ids \
  --out runs/tts-evalset/cases.jsonl
```

Use `--hash-source-ids` for private evalsets when row ids may reveal source details. The generated
cases keep a deterministic `source_id_sha256` for local traceability without writing the raw row id.
Raw source task labels are omitted by default because private evalsets sometimes use prompt-like task
names; pass `--include-source-task` only when those labels are safe and useful for local debugging.

For a small synthesis batch with deliberate coverage, filter or cap classified slices:

```bash
oaj build-tts-cases \
  --source /path/to/evalset/seed_v0.jsonl \
  --source-name ome \
  --slices numbers,dates_times,code_like,punctuation_format \
  --per-slice-limit 2 \
  --prioritize-slice-coverage \
  --summary-out runs/tts-evalset/smoke-summary.json \
  --no-summary-source-examples \
  --out runs/tts-evalset/smoke-cases.jsonl
```

The resulting draft cases use `reference_text` as the target text to synthesize and include
`metadata.requires_synthesis=true` plus `metadata.reference_text_sha256`, so a later local TTS step
can attach ignored audio artifacts and audit target-text identity without committing private rows.
Each case also records `metadata.turn_context_source` as either `source_turns` or
`fallback_instruction`, which makes it clear whether multi-turn context came from the source eval row
or from the bridge's generic "read this aloud" fallback.
Use `--prioritize-slice-coverage` with small `--limit` values when the source evalset is ordered by
category and you want the draft manifest to cover more TTS slices before truncation.
Do not pass draft text-only cases to hosted audio judges until an `audio_path` or `audio_url` has
been attached. The optional summary is metadata-only: counts by classified TTS slice and source
category, source modality, source scoring type, turn-role sequence and turn-context-source coverage,
text-context-field combinations, multi-turn case count, text-length min/max/average,
unique/duplicate target-text hash counts, capped example source ids by slice, and the number of
cases still requiring synthesis. Pass
`--no-summary-source-examples` when even source row ids should stay out of local summary artifacts.

To synthesize a small local Chatterbox sample set from a private manifest, write artifacts under
`runs/` and keep only the derived local manifest there:

```bash
python scripts/synthesize_tts_cases.py \
  --cases runs/tts-evalset/cases.jsonl \
  --limit 5 \
  --discard-text-sidecars \
  --summary-out runs/tts-synthesis/summary.json \
  --out runs/tts-synthesis
```

This calls `local-tts-speak` with `mlx-community/chatterbox-turbo-6bit` by default, writes per-case
audio files under the ignored output directory, and emits `runs/tts-synthesis/tts_audio_cases.jsonl`
with local `audio_path` fields for provider smoke tests. With `--discard-text-sidecars`, temporary
text files are deleted after synthesis, or skipped entirely during `--dry-run`, while the manifest
keeps only the target text hash and non-secret synthesis metadata.
For model-comparison batches that use a compatible wrapper, pass `--tts-bin`, `--model`, and
`--synthesis-provider` so downstream reports can distinguish Chatterbox from Kokoro, Qwen3-TTS, or
other MLX Community candidates without assuming they share the same runtime.
When sidecars are kept for local review, the manifest records a relative `metadata.text_sidecar_path`
so validation can later confirm the sidecar still matches `reference_text`.
The optional synthesis summary is metadata-only: counts by TTS slice, source category, sample kind,
synthesis provider/model/voice/language/audio format, audio bytes/duration aggregates when audio
exists, and the number of cases with audio hashes.
Before passing a derived manifest to a hosted judge, validate that every case still satisfies the
audio-plus-text contract and that relative local audio files exist:

```bash
python scripts/synthesize_tts_cases.py \
  --cases runs/tts-synthesis/tts_audio_cases.jsonl \
  --validate-only \
  --require-relative-audio-path \
  --redact-summary-case-ids \
  --summary-out runs/tts-synthesis/validation-summary.json
```

Use `--allow-missing-audio` only for dry-run manifests that intentionally point at future audio
paths. Add `--require-text-sidecars` when validating a local review manifest that intentionally kept
sidecar text files. Validation summaries contain issue classes plus metadata-only manifest coverage
such as TTS slice, source category, sample kind, text-context fields, turn-role sequences, audio
bytes/duration aggregates, and audio-hash coverage. They do not include prompt text; pass
`--redact-summary-case-ids` when case ids may reveal private source details.

## Run With Local Qwen/Qwen3-Omni

Start a local Qwen/Qwen3-Omni server that exposes an OpenAI-compatible chat-completions endpoint. For vLLM-Omni, the upstream examples use `http://localhost:8091/v1/chat/completions` with `modalities: ["text"]` for text-only judge output.

```bash
export OAJ_PROVIDER=qwen
export OAJ_BASE_URL=http://localhost:8091/v1
export OAJ_MODEL=Qwen/Qwen3-Omni-30B-A3B-Instruct
export OAJ_API_KEY=EMPTY

oaj eval --provider qwen --judge asr_error --cases examples/asr_cases.jsonl --out runs/qwen-asr
open runs/qwen-asr/report.html
```

Provider-backed judging requires both audio and text context. Audio can be passed as `audio_url`
in the case file or as `audio_path`; textual context should be supplied through `reference_text`,
`candidate_text`, or `turns`. Local paths are encoded as data URLs for OpenAI-compatible
multimodal endpoints by default.

## Run With Gemini

Gemini hosted audio judging is available through the `gemini` provider. Provide the API key at runtime only:

```bash
export GEMINI_API_KEY=...
oaj eval --provider gemini --judge asr_error --cases examples/asr_cases.jsonl --out runs/gemini-asr
```

The default Gemini model is `gemini-3.5-flash`. See [docs/provider-gemini.md](docs/provider-gemini.md).

## Generate A Whisper Tiny ASR Report

For a local ASR model smoke test, first materialize the open audio samples if they are not already present:

```bash
python scripts/materialize_open_samples.py
```

Then transcribe the local WAV ASR cases with Whisper tiny and judge the resulting candidate transcripts:

```bash
python scripts/transcribe_asr_cases.py \
  --cases runs/open-audio-samples/asr_wav_cases.jsonl \
  --model tiny \
  --out runs/whisper-tiny-asr-open/cases.jsonl \
  --summary-out runs/whisper-tiny-asr-open/summary.json

oaj eval \
  --provider mock \
  --judge asr_error \
  --cases runs/whisper-tiny-asr-open/cases.jsonl \
  --out runs/whisper-tiny-asr-open/judge-report
```

This writes a local sample report to `runs/whisper-tiny-asr-open/judge-report/report.html`.
The mock judge is a deterministic reference-vs-candidate semantic baseline; rerun with a hosted or
local audio LLM judge when model-grade audio-aware judging is needed.

## AutoJudge A Hugging Face ASR Model

For the end-to-end ASR model workflow, `autojudge-hf-asr` loads a Hugging Face
`automatic-speech-recognition` pipeline, transcribes local-audio ASR cases, then runs the selected
Open Audio Judge provider over the generated candidate transcripts.

```bash
oaj autojudge-hf-asr \
  --model openai/whisper-tiny \
  --cases runs/open-audio-samples/asr_wav_cases.jsonl \
  --judge-provider mock \
  --out runs/hf-whisper-tiny-asr
```

The command writes:

- `candidate_cases.jsonl`: source cases plus model-generated `candidate_text`.
- `model_summary.json`: model id, source manifest, and candidate coverage.
- `judge-report/results.jsonl` and `judge-report/report.html`: aggregate scores, top issue
  categories, priority cases, and sample-by-sample breakdowns.

Install `transformers` and any model-specific audio dependencies in the runtime environment before
using this command with real Hugging Face models. Keep private or generated audio artifacts under
`runs/` unless they are explicitly safe to publish.

## AutoJudge A Local Chatterbox TTS Model

For the TTS-first open-model workflow, `autojudge-local-tts` synthesizes a TTS case manifest with a
local `local-tts-speak` compatible command, then judges the generated audio with the selected Open
Audio Judge provider.
See the committed [TTS leaderboard demo page](docs/tts-leaderboard-demo.html) for the
full command flow, expected output files, and a representative report preview.

```bash
oaj autojudge-local-tts \
  --model mlx-community/chatterbox-turbo-6bit \
  --cases examples/tts_multiturn_cases.jsonl \
  --judge-provider gemini \
  --judge-samples 3 \
  --limit 410 \
  --out runs/chatterbox-tts-autojudge
```

To run the ideal flow directly from a local text evalset, pass `--evalset-source` instead of
`--cases`. The command will first build a metadata-safe TTS manifest, then synthesize and judge it:

```bash
oaj autojudge-local-tts \
  --evalset-source /path/to/evalset/seed_v0.jsonl \
  --evalset-source-name ome \
  --evalset-slices numbers,dates_times,code_like,punctuation_format \
  --evalset-per-slice-limit 2 \
  --evalset-prioritize-slice-coverage \
  --evalset-hash-source-ids \
  --model mlx-community/chatterbox-turbo-6bit \
  --tts-timeout-seconds 180 \
  --judge-provider gemini \
  --judge-samples 3 \
  --limit 8 \
  --out runs/chatterbox-tts-evalset-autojudge
```

The command writes:

- `evalset/tts_cases.jsonl` and `evalset/summary.json` when `--evalset-source` is used.
- `synthesis/tts_audio_cases.jsonl`: generated-audio cases with local `audio_path` values.
- `synthesis/synthesis_failures.jsonl` when `--skip-failed-synthesis` records skipped local TTS
  failures.
- `model_summary.json`: model id, voice, audio format, attempted/synthesized counts, synthesis
  success rate, and failure breakdowns by error type such as `command_failed`,
  `command_timeout`, `missing_output`, `stale_fallback_output`, `invalid_output_path`, or
  `audio_format_mismatch`, plus synthesis provider/model/voice/language, audio format, TTS slice,
  source category, and sample kind.
- `judge-report/results.jsonl` and `judge-report/report.html`: aggregate naturalness scores, top
  issue categories, weakest category/slice/model segments, a ranked model-category action matrix,
  research-backed category guidance, likely fix areas, priority cases, category score breakdowns,
  and searchable/sortable sample-by-sample reasons with label, status, model, category, and slice
  filters. Use `--judge-samples` to run multiple independent judge calls per
  synthesized sample and report the averaged score plus the individual judge scores.

Use `--judge-provider mock` for a cheap pipeline check. Use an audio-capable judge such as Gemini or
a local Qwen/Qwen3-Omni endpoint for real perceptual judging.

After running the same case file through multiple local TTS wrappers, combine their
`judge-report/results.jsonl` files into one model-comparison report:

```bash
oaj report \
  --results runs/leaderboard/chatterbox/judge-report/results.jsonl \
  --results runs/leaderboard/kokoro/judge-report/results.jsonl \
  --results runs/leaderboard/qwen3-tts/judge-report/results.jsonl \
  --baseline-model mlx-community/chatterbox-turbo-6bit \
  --out runs/leaderboard/comparison-report
```

The combined report reuses the same score, category, slice, voice, language, weakest-segment, and
model-category action matrix views across all supplied result files. When multiple synthesis models
share `metadata.source_case_id`, the report also adds baseline deltas against
the model selected with `--baseline-model` (default:
`mlx-community/chatterbox-turbo-6bit`), including matched-case counts, win/tie/loss totals, average
score deltas, and the largest per-case regressions to inspect first. A companion baseline
regression-slices section groups those matched deltas by evaluation category and TTS slice, then
surfaces likely fix areas and representative regressions so engineers can prioritize the model
behaviors most responsible for low scores. Category-focused rows also show the taxonomy focus and
`source_basis` signals from the manifest, which helps route low scores to concrete work such as
numeric/date normalization, multilingual phonemization, speaker embedding stability, or
discourse-level prosody.

For non-Chatterbox wrappers that accept the same `local-tts-speak` arguments, pass `--tts-bin` and
`--synthesis-provider` so generated case metadata and report aggregates identify the actual local
backend. Use `--tts-timeout-seconds` to fail a single stuck synthesis call with captured stdout and
stderr instead of leaving the full AutoJudge run hanging.
Wrappers should emit JSON with an `output`, `output_path`, or `audio_path` field when possible. If a
wrapper only writes files, Open Audio Judge will use a matching newly created or modified audio file
and reject unchanged leftovers from an earlier run.
By default, a synthesis error fails the command before judging. Add `--skip-failed-synthesis` for
larger batches where the run should write `synthesis_failures.jsonl`, include failure counts in
`model_summary.json`, and judge the samples that did synthesize successfully. Failure records keep
the attempted synthesis provider, model, voice, language, audio format, source case id, and target
text hash for provenance without storing the target text.

## REST API

```bash
oaj serve --host 127.0.0.1 --port 8000
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/v1/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "judge": "asr_error",
    "provider": "mock",
    "case": {
      "id": "demo",
      "task": "asr_error",
      "audio_url": "https://example.test/audio.wav",
      "reference_text": "Please transfer fifteen dollars to Maya.",
      "candidate_text": "Please transfer fifty dollars to Maya."
    }
  }'
```

The judge result always includes the core score fields and may include task-specific diagnostics:

```json
{
  "overall_score": 62,
  "reason": "The transcript is mostly fluent but changes a key amount from fifteen to fifty.",
  "judge_transcript": "Please transfer fifteen dollars to Maya.",
  "meaning_preservation": "partial_loss",
  "semantic_error_summary": "The candidate preserves the request but changes the payment amount.",
  "key_differences": ["fifteen dollars was transcribed as fifty dollars"],
  "error_categories": ["number_error", "substitution"],
  "researcher_notes": ["Prioritize numeric amount robustness in payment-like utterances."]
}
```

For ASR, the judge is instructed to listen to the audio, form its own concise best-effort transcript, and compare the submitted transcript against both that audio-derived transcript and any provided reference. The score is based on semantic preservation and downstream impact, not raw word edit distance.

## Case Format

Cases are JSONL records:

```json
{"id":"asr-001","task":"asr_error","audio_path":"audio/sample.wav","reference_text":"...","candidate_text":"...","metadata":{"language":"en"}}
```

Fields:

- `id`: stable case identifier.
- `task`: prompt family, such as `asr_error` or `tts_naturalness`.
- `audio_path` or `audio_url`: required for provider-backed audio judging.
- `turns`: optional multi-turn context as `{"role": "user|assistant|agent|system", "content": "..."}` records. Use this when judging whether an ASR/TTS output matches a prior user request and agent response in context.
- `reference_text`: expected transcript or target text, when available.
- `candidate_text`: model output transcript, translation, or synthesis text.
- `metadata`: language, domain, speaker notes, expected events, or proprietary labels.

Every provider-backed judge case must include audio plus at least one text source:
`reference_text`, `candidate_text`, or `turns`. Text-only fixtures may be used for deterministic
mock calibration, but they are not valid hosted audio-judge inputs.

## Current Scope

The current version includes Qwen/OpenAI-compatible judging, Gemini hosted judging, a deterministic mock provider, ASR/TTS prompt families, semantic ASR diagnostics, TTS naturalness diagnostics, CLI/API execution, local TTS manifest tooling, open development sample manifests, and HTML reporting. Next steps are provider smoke automation, calibration-set governance, pairwise judging, bootstrap confidence intervals, and prompt version release policy.

Development audio sample manifests are listed in [docs/sample-audio.md](docs/sample-audio.md). Each current judge family has at least three linked open-source/free-license audio samples for smoke testing, plus a helper to materialize local WAV fixtures under ignored `runs/`. Gemini sample smoke results are recorded in `examples/gemini_sample_records.jsonl` and should be reused unless the sample, prompt, provider, or model changes.
