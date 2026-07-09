# ASR Evaluation Taxonomy

This page defines the first research-guided ASR categories for Open Audio Judge.
The goal is to pair traditional transcript metrics with audio-judge review for
meaning preservation and downstream task risk.

## Current Seed Set

Manifest: `examples/asr_research_cases.jsonl`

- 30 public-safe seed cases.
- 6 categories with exactly 5 cases each.
- Cases are text/reference seeds with `metadata.requires_audio_materialization=true`.
- Cron runs should materialize local audio under ignored `runs/`, transcribe it
  with MLX ASR models, judge candidate transcripts with Gemini, and publish only
  safe aggregate/demo artifacts.

## Categories

### transcription_accuracy_wer

Classic WER/edit-distance calibration: substitutions, insertions, deletions,
homophones, punctuation boundaries, and disfluency policy. This remains the
baseline metric because it is simple, reproducible, and widely reported.

### entity_factual_integrity

Names, organizations, addresses, product labels, alphanumeric IDs, and other
slots where one token can change the real-world referent. This category tracks
custom-vocabulary and named-entity recall failures that aggregate WER can hide.

### numeric_unit_integrity

Amounts, dosage units, decimal measurements, account numbers, percentages, and
minimal numeric pairs such as `ninety` versus `nineteen`. These errors often
have high user impact despite low edit distance.

### negation_modality_scope

Negation, permission, prohibition, `only`, `unless`, and modality scope. These
cases target meaning inversions such as `can` versus `cannot` and clinical
negation errors.

### temporal_scheduling_accuracy

Dates, times, durations, AM/PM, relative ordering, and deadline fields. These
cases stress scheduling and workflow transcripts where small time errors change
the intended action.

### semantic_paraphrase_preservation

Meaning preservation under paraphrase: events, causal relations, metric
tradeoffs, coreference, and conditional instructions. This category is designed
for LLM-as-judge review rather than pure token matching.

## Source Signals

- WER/edit distance remains the standard ASR baseline and should stay visible in
  reports.
- Meaning-preservation ASR research argues that WER misses semantic severity,
  especially for negation, numbers, dates, entities, and downstream task impact.
- Entity recall/custom vocabulary analysis is a practical production slice for
  assistants, medical, support, and meeting transcripts.
- Semantic transcript evaluation should separate harmless paraphrase from
  meaning-changing errors.

## First MLX Model Set

Use these three MLX Community candidates first:

- `mlx-community/whisper-large-v3-turbo-asr-fp16`
- `mlx-community/Qwen3-ASR-1.7B-8bit`
- `mlx-community/VibeVoice-ASR-4bit`

Fallbacks if a wrapper fails locally:

- `mlx-community/whisper-small.en-asr-4bit`
- `mlx-community/parakeet-rnnt-0.6b`
- `mlx-community/GLM-ASR-Nano-2512-4bit`

Record unsupported/blocked states explicitly rather than silently swapping
models.
