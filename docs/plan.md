# Project Plan

## North Star

Open Audio Judge should become a standard protocol for prompt-based audio LLM evaluation across private and public datasets. People should be able to bring their own audio, labels, and judge model while keeping prompt versions, scoring, result schemas, and reports comparable.

## Current Scope

The current implementation includes:

- Prompt registry with versioned YAML prompts.
- ASR error judge prompt following the requested three-stage structure, with audio-derived judge transcription and semantic diagnostics.
- TTS naturalness prompt with naturalness, artifact, text-faithfulness, and researcher-note diagnostics.
- Case schema for audio path/URL, reference text, candidate text, multi-turn context, and metadata.
- Provider-backed judge calls require audio plus text context; transcript/reference fields complement
  the audio but do not replace it.
- Local Qwen/Qwen3-Omni provider through OpenAI-compatible chat completions.
- Gemini hosted provider through the Interactions API.
- Mock provider for tests and offline demos.
- CLI batch runner.
- TTS evalset bridge and local synthesis manifest tooling.
- FastAPI REST API for single-case and batch judging.
- HTML report with average score, score distribution, accurate/needs-review/inaccurate labels, per-case reasons, and optional diagnostic fields.

## Prompt Contract

Every judge prompt should contain:

- `id`, `version`, and `task`.
- Stage 1: evaluator role and judging stance.
- Stage 2: task definition and scoring rubric.
- Stage 3: private checklist and strict output rules.
- A user prompt template rendered with the case fields.
- Stable core JSON result fields: `overall_score` and `reason`.
- Optional task-specific diagnostics such as `judge_transcript`, `meaning_preservation`, `semantic_error_summary`, `key_differences`, `error_categories`, and `researcher_notes`.

The judge may internally inspect detailed dimensions, but it should not expose hidden chain-of-thought. Diagnostic fields should be concise observations that help researchers improve models.

## Scoring Direction

All first-version prompts use `1` to `100`, where higher is better:

- ASR error judge: 100 means the candidate transcript faithfully preserves the spoken meaning, as judged from the audio plus any reference transcript.
- TTS naturalness judge: 100 means highly natural, intelligible, artifact-free synthesized speech.

Reports can map scores into labels with configurable thresholds:

- `accurate`: score >= 81
- `needs_review`: 60 <= score < 81
- `inaccurate`: score < 60

These labels are deliberately generic and should be renamed per task in later report templates.

## Provider Status

Implemented:

- `mock`: deterministic local scoring with a lightweight semantic ASR baseline for reference/candidate cases. It is non-audio and non-authoritative, but it flags high-impact number, negation, and entity-like changes so offline smoke reports do not behave like pure WER.
- `qwen`: OpenAI-compatible `/v1/chat/completions`, defaulting to `http://localhost:8091/v1`.
- `openai-compatible`: same transport with a generic model/base URL.
- `gemini`: hosted audio understanding using the Gemini Interactions API.

Near-term:

- `transformers-qwen`: direct local inference for users who do not expose a REST server.
- `ensemble`: multiple judges with median/trimmed-mean aggregation.
- `pairwise`: A/B audio comparison protocols for TTS, speech-to-speech translation, and dialogue systems, informed by AudioJudge and MTalk-Bench.

## Report Status

The runner emits an HTML report plus `results.jsonl`. Current reports summarize score distribution, semantic diagnostics, high-impact ASR errors, researcher notes, calibration mismatches, priority cases, and per-case details.

Near-term report additions:

- Score histograms by dataset slice, language, speaker, model, and domain.
- Pairwise comparison report for A/B TTS and speech translation outputs.
- Confidence intervals via bootstrap resampling.
- Calibration report comparing audio LLM judge scores to human labels or legacy metrics.
- Failure taxonomy: substitutions, deletions, insertions, entity errors, number errors, negation errors, speaker-turn errors, acoustic artifacts.
- ASR semantic slices: meaning preservation class, critical-token error type, and downstream-impact buckets.
- Judge bias checks: position bias, length/verbosity bias, and modality-specific failures.
- Multi-aspect reports that separate lexical content, speech quality, paralinguistic behavior, and ambient-sound reasoning before summarizing an overall score.

## Calibration Direction

The ASR judge should be calibrated on cases where WER-like overlap and semantic severity disagree:

- `fifteen dollars` versus `fifty dollars`: small edit distance, large downstream payment impact.
- `five milligrams` versus `five micrograms`: tiny lexical change, large dosage-unit impact.
- Deleted negation such as `do not take` versus `do take`: low word error count, reversed intent.
- Wrong location, name, medication, account, or product entity: one-token error with high task impact.
- Harmless formatting and normalization such as `twenty one` versus `21`: lexical difference with preserved meaning.

These cases should appear in both the prompt examples and report-level slice analysis so researchers can see whether an ASR model is improving on meaning preservation rather than only aggregate WER.

## Open Questions

- Whether the canonical public field should be named `overall_score` or `score`. The MVP uses `overall_score` because it matches the requested "overall score" wording and leaves room for future dimension scores.
- Whether report labels should be task-specific by default. The current generic labels keep the first implementation simple.
- How to publish prompt calibration sets without leaking proprietary user data.
- Whether a future public leaderboard should score judge prompts, judge models, or evaluated audio models separately.
