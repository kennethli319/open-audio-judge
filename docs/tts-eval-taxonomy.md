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
- `symbolic_math_reading`: formulas, conditional probability, polynomial terms, coordinate pairs,
  ratios, variables, signs, and units where exact symbolic structure must remain intelligible.
- `multi_speaker_attribution`: narrator/quote boundaries, labeled speaker turns, handoff ownership,
  embedded reported speech, and panel Q&A roles where the right words must stay attached to the
  right speaker or actor.
- `structured_enumeration_delivery`: ranked lists, labeled options, phase checklists, compact status
  rows, and nested plan items where spoken structure must remain scannable without visual layout.
- `phonetic_confusability`: minimal pairs, confusable letter and digit codes, similar names,
  positive/negative contrasts, and small function-word distinctions where intelligibility must
  survive similar sounds.
- `referential_cohesion`: former/latter references, pronoun antecedents, this/that contrasts,
  previous/next warnings, and same/different conditions where spoken grouping must preserve which
  entity or option is being referenced.
- `measurement_unit_disambiguation`: abbreviated and confusable measurement units such as mg/mL,
  mm/m, ms/min, lb/oz/mL, and W/Wh where TTS must preserve exact unit meaning under realistic
  clinical, engineering, logistics, and device-spec wording.
- `contextual_abbreviation_expansion`: context-dependent abbreviations such as St., No., month
  labels, business shorthand, corporate suffixes, and weekday labels where the model must expand or
  preserve the right spoken form from surrounding meaning rather than punctuation alone.
- `noise_resilience_delivery`: projected but composed delivery for transit alerts, household
  timers, warehouse headset instructions, crowded lobby announcements, and hallway safety messages
  where key entities and conditions must remain intelligible in noisy real-world settings.
- `audience_register_adaptation`: child-friendly safety, patient plain-language, executive
  briefings, engineering handoffs, and public-radio narration where style must adapt to the
  listener without losing exact conditions, entities, or reported-speech boundaries.
- `uncertainty_calibration_delivery`: probability forecasts, diagnostic caveats, confidence
  scores, escalation handoffs, and estimate ranges where prosody must preserve calibrated
  uncertainty without turning cautious statements into guarantees.
- `real_time_streaming_delivery`: first-token openings, streamed chunk boundaries, barge-in repair,
  concise alerts, and progressive guidance where low-latency delivery must stay intelligible,
  stable, and interruption-aware.
- `numeric_identifier_delivery`: support tickets, verification codes, device serials, record
  locators, and lab sample IDs where long identifiers must be chunked and transcribed exactly.
- `sentence_boundary_inference`: punctuation-light status updates, compact lists, terse agendas,
  alert cascades, and unpolished messages where prosody must infer sensible phrase boundaries
  without changing the words.
- `cross_lingual_name_pronunciation`: person names, place names, menu items, product names, and
  organization names from non-English contexts that should stay intelligible and respectful inside
  English utterances without caricature.
- `speech_mode_stability`: rhymes, countdowns, slogans, repeated phrases, and stage-like cues that
  should remain spoken rather than drifting into chant, song, loops, or exaggerated performance.
- `dialogue_act_prosody`: confirmations, clarifying questions, apologies with repair actions,
  offers, and bounded deferrals where prosody must preserve the intended speech act rather than
  flattening it into a generic assistant sentence.
- `address_wayfinding_delivery`: street addresses, apartment entry details, intersections, campus
  rooms, and emergency access points where numbers, names, alphanumeric units, landmarks, and
  access conditions must be chunked clearly enough for first-listen transcription.
- `repair_sensitive_delivery`: corrections, cancellations, replacements, reversed directions, and
  status repairs where the model must make the rejected value and active value unmistakable without
  adding words or flattening the repair cue.
- `pause_breath_control`: explicit short and long pauses, clean breath-group boundaries,
  condition-preserving phrasing, guided breathing phases, parenthetical asides, and long-update
  breath placement where prosody should aid intelligibility without adding words or breaking
  meaning-critical spans.
- `citation_reference_delivery`: academic citations, regulatory sections, table footnotes, DOI
  identifiers, and internal clause references where source labels, years, numbers, letters, and
  cross-reference hierarchy must remain intelligible without hallucinated wording.
- `digital_locator_delivery`: email aliases, URLs with query strings, file paths, meeting links,
  passcodes, social handles, and chat channels where punctuation, separators, and identifier
  boundaries must be grouped clearly enough for first-listen transcription.
- `conditional_logic_delivery`: unless, only-if, if/otherwise, nested conditions, and exception
  boundaries where prosody must preserve the controlling condition and active action without
  weakening safety-critical logic.
- `priority_escalation_delivery`: critical alerts, severity labels, triage priority, alert
  downgrades, escalation owners, and deadlines where urgency must be calibrated while preserving
  exact action, owner, and condition details.
- `contrastive_pair_delivery`: A/B options, before/after metrics, left/right visual states,
  positive/negative controls, and safe/destructive action pairs where prosody must keep both sides
  balanced, distinct, and exactly attached to their labels.
- `statistical_notation_delivery`: p-values, confidence intervals, scientific notation, effect
  sizes, percentiles, interquartile ranges, and compact metric rows where labels, signs, decimals,
  ranges, and uncertainty cues must remain attached to the right values.
- `readback_confirmation_delivery`: closed-loop confirmations, order readbacks, clinical
  dose-route-schedule repeats, dispatch acknowledgments, corrected readbacks, and queued future
  actions where prosody must preserve what was received, corrected, confirmed, or still pending.
- `focus_particle_scope_delivery`: focus particles such as only, also, even, just, and not all
  where prosody must preserve whether a condition is exclusive, additive, unexpected, limited, or
  broadly negated.
- `slot_value_pairing_delivery`: form fields, inventory rows, calendar records, lab requisitions,
  and API status payloads where spoken labels must stay attached to their exact values.
- `commitment_scope_delivery`: apologies, partial commitments, clinical support limits, security
  prerequisites, and operations promises where prosody must preserve what is committed, refused,
  conditional, or explicitly outside scope.
- `ordinal_ranking_delivery`: priority order, floor sequences, tied ranks, rubric levels, and
  version-order instructions where ordinal words, numeric labels, and relation changes must stay
  attached to the right item.
- `temporal_relation_delivery`: before/during/after windows, through/starting/until dose changes,
  local-time versus elapsed-time travel updates, service grace periods, and lab timing sequences
  where time spans and event order must remain attached to the correct action.
- `syntactic_attachment_delivery`: subordinate clauses, modifier attachment, coordination scope,
  reduced-relative descriptors, and appositive names where prosodic grouping must preserve which
  phrase attaches to which actor, object, or condition.
- `quantifier_scope_delivery`: all/except, not every, at least/at most, every/except, exactly one,
  and neither constructions where prosody must preserve set membership, exclusions, lower and upper
  bounds, and exact cardinality.
- `acronym_initialism_delivery`: spoken acronyms, letter-by-letter initialisms, medical and
  technical abbreviations, mixed acronym-number identifiers, and confusable product suffixes where
  text normalization must preserve intelligibility and exact letter/digit boundaries.
- `spatial_relation_delivery`: above/below, inside/outside, left/right, behind/in front, and
  under/over relations where prosody must keep spatial words, objects, and rejected alternatives
  attached to the correct location.
- `homograph_number_format_delivery`: versions versus decimals, room numbers versus extensions,
  slash dates versus ratios, hash versus pound symbols, and filename suffixes versus semantic
  versions where context must choose the right spoken normalization.
- `currency_financial_delivery`: currency amounts, currency codes, basis points versus percent,
  negative refunds, exchange rates, transfer caps, and near-matching balances where financial text
  normalization must preserve exact money, units, signs, and conditions.
- `medication_dosage_delivery`: medication dose, route, schedule, taper phase, look-alike drug
  names, missed-dose boundaries, and maximum daily constraints where clinical text normalization
  must preserve exact units, timing, conditions, and safety warnings.
- `morphosyntactic_marker_delivery`: plural, possessive, contraction, tense, comparative, and
  pronoun-case markers where small grammatical sounds must remain audible enough to preserve who
  did what, when, and to whom.
- `operator_precedence_delivery`: parentheses, boolean grouping, chained inequalities, exponent
  scope, and subscript boundaries where spoken math or logic must preserve which operation binds
  to which term.

## Manifest Contract

The public comparison manifest is intentionally balanced at 350 cases for now: 70 categories with 5
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
