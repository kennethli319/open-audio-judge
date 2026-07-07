# TTS Eval Taxonomy

This taxonomy keeps the public TTS comparison manifest grounded in published evaluation practice
instead of one-off demo ideas. The committed examples are synthetic and public-safe; real production
runs can map private prompts into the same categories without copying private rows into this repo.

## Source Signals

- **Seed-TTS / Seed-TTS-Eval** emphasizes objective intelligibility and speaker consistency through
  WER/CER-style transcription checks and speaker similarity metrics.
  Sources: [Seed-TTS paper](https://arxiv.org/html/2406.02430v1),
  [seed-tts-eval](https://github.com/BytedanceSpeech/seed-tts-eval).
- **Discrete-token SLM TTS evaluation** studies speaking style, intelligibility, speaker consistency,
  prosodic variation, and spontaneous behavior.
  Source: [LREC-COLING 2024 paper](https://arxiv.org/html/2405.09768v1).
- **InstructTTSEval** focuses on complex natural-language style control, including acoustic
  parameter specification, descriptive style directives, and role play, and uses Gemini as an
  automatic judge for instruction-following TTS.
  Source: [InstructTTSEval](https://arxiv.org/html/2506.16381v1).
- **VoiceBench / VocalBench-style speech benchmarks** motivate spoken instruction following,
  emotional empathy, safety/robustness, speaker styles, environmental/content variation, and
  real-world voice-assistant behaviors.
  Sources: [VoiceBench](https://arxiv.org/html/2410.17196v2),
  [VocalBench](https://arxiv.org/html/2505.15727v2).
- **TTSDS / distribution-style TTS evaluation** motivates separate tracking of speaker identity,
  intelligibility, and prosody rather than a single scalar naturalness score.
  Source: [TTSDS paper](https://arxiv.org/abs/2407.12707).

## Public Manifest Categories

- `paralinguistics`: emotion, confidence, uncertainty, warmth, urgency, and empathy.
- `instruction_following`: exact text, pronunciation directives, word emphasis, punctuation, and
  do-not-say constraints.
- `information_tuning`: numbers, dates/times, units, procedural steps, and safety-critical wording.
- `storytelling_dialogue`: narration, role play, conversational spontaneity, character consistency,
  and scene pacing.
- `speech_steerability`: register, pace, volume, pitch, and emphasis controls.
- `robustness_intelligibility`: rare words, acronyms, code-like strings, long text, and clear
  multilingual snippets.
- `speaker_voice_consistency`: stable persona, timbre, age/role cues, and cross-sentence voice
  consistency.
- `multilingual_code_switching`: short bilingual phrases, language switches, names, loan words, and
  respectful accent handling.
- `long_form_discourse`: paragraph-level breath control, list structure, topic transitions,
  parentheticals, and end-of-passage stability.
- `text_normalization`: currency, percentages, URLs, email addresses, version strings, symbols,
  addresses, and abbreviations that must be spoken unambiguously.
- `acoustic_contexts`: phone-like support, public address projection, quiet near-field reminders,
  mobile navigation, and clean dispatch styles where delivery context can hurt intelligibility.
- `spontaneous_conversation`: hesitation, self-correction, backchannels, thinking pauses, and
  conversational restarts that should sound natural without hallucinated filler or dropped words.
- `affective_transitions`: within-utterance shifts such as concern to relief, apology to confidence,
  friendly guidance to warning, surprise to neutral delivery, and empathy to practical instruction.
- `punctuation_prosody`: quote boundaries, colon/semicolon grouping, aside phrasing, question-answer
  contours, and ellipsis pauses that should guide prosody without changing the text.
- `domain_terminology`: clinical, legal, engineering, finance, and science terms that should remain
  intelligible in realistic domain-specific registers.
- `heteronym_disambiguation`: context-dependent pronunciations such as read/read, wind/wind,
  bass/bass, minute/minute, and polish/Polish that should be resolved from sentence meaning.
- `formatting_markup_robustness`: markdown emphasis, bracketed labels, inline bullets, code-style
  identifiers, and symbolic emoticon text that should not produce literal or hallucinated speech.
- `nonverbal_paralinguistic_cues`: controlled chuckles, sighs, whisper-like delivery, amused
  restraint, and breath management that should add expression without harming intelligibility.
- `voice_conversion_similarity`: reference-style transfer, content preservation under speaker
  transfer, prosody retention, anti-generic voice leakage, and identity-preserved style shifts.
- `accent_dialect_handling`: light regional English accent cues, respectful dialect handling,
  place-name clarity, acronym preservation, and intelligibility without caricature.
- `artifact_suppression`: plosive and sibilance control, repeat-loop resistance, clean silence
  boundaries, quiet tail stability, and artifact-free intelligibility under difficult acoustics.
- `temporal_rhythm_control`: countdown spacing, relative pause duration, tempo ramps, repeated
  phrase rhythm, and short-clause timing that should stay spoken rather than sung or flattened.
- `safety_privacy_delivery`: consent notices, credential warnings, redacted identifiers, location
  choice, and irreversible-action warnings that must preserve exact safety and privacy conditions.
- `semantic_contrast_focus`: negation scope, exception boundaries, corrections, balanced
  alternatives, and threshold conditions where the model must preserve meaning through emphasis and
  prosodic grouping.
- `dialogue_turn_management`: spoken-assistant acknowledgments, clarification questions, repair
  restarts, service handoffs, and task-closing next actions where turn boundaries must stay natural
  and exact.
- `compositional_style_control`: simultaneous style constraints such as quiet urgency, friendly
  formality, upbeat slow coaching, serious reassurance, and conversational precision where one
  control should not erase another.
- `named_entity_pronunciation`: person names, place names, product names, acronyms, and program
  names where TTS should preserve intelligibility without over-spelling or caricature.
- `disfluency_repair_control`: false starts, bounded filled pauses, intentional repetition,
  restart cues, and cautious hesitation where spontaneous-speech cues must remain controlled and
  text-faithful.
- `lexical_stress_disambiguation`: part-of-speech stress shifts such as record/record,
  permit/permit, project/project, object/object, and conduct/conduct where pronunciation should
  follow sentence meaning without awkward overpronunciation.
- `pragmatic_intent_delivery`: speech-act intent such as polite refusal, gentle suggestion, firm
  deadline reminder, supportive boundary-setting, and invitation with opt-out where prosody should
  preserve intent beyond literal words.

## Manifest Contract

The public comparison manifest is intentionally balanced at 150 cases for now: 30 categories with 5
cases each. Expansion should add complete five-case category batches before deepening existing
categories.

Every committed case should preserve the following metadata contract:

- `source=research-backed-tts-demo`.
- `eval_category` must be one of the public categories above.
- `tts_slice` should be unique within the public seed set so model reports can point to a narrow
  failure mode before aggregating upward.
- `style_prompt`, `expected_style`, and `expected_instruction` should describe observable speech
  behavior, not implementation details.
- `source_basis` should name at least one grounding signal from the source list, such as
  Seed-TTS-Eval intelligibility, speaker similarity, InstructTTSEval style control, VoiceBench or
  VocalBench spoken behavior, TTSDS identity/intelligibility/prosody dimensions, or discrete-token
  SLM prosodic/style evaluation.

Each category should have up to five public-safe examples before expanding breadth further. Every
case should include `eval_category`, `tts_slice`, `style_prompt`, `expected_style`, and
`expected_instruction` metadata so reports can aggregate by model/category and engineers can map low
scores back to likely fix areas. Public demo cases should also use
`source=research-backed-tts-demo` and a short `source_basis` that names the research signal or eval
set family behind the slice.
