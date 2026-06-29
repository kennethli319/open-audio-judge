# Research Notes

This snapshot was prepared on 2026-06-29 and is biased toward work that affects the Open Audio Judge protocol.

## What Is Stable Enough To Build On

LLM-as-judge is now a recognizable evaluation pattern, but it is sensitive to prompt design, model choice, position/order effects, and hidden bias. The survey by Li et al. frames judge systems around evaluation criteria, input items, optional references, explanations, and feedback, and stresses meta-evaluation of the judge itself. This supports making prompts versioned artifacts and storing raw judge responses alongside normalized scores. Source: [LLMs-as-Judges: A Comprehensive Survey on LLM-based Evaluation Methods](https://arxiv.org/html/2412.05579v2).

Audio LLM evaluation is broader than transcription. AudioBench divides evaluation into speech understanding, audio scene understanding, and voice/paralinguistic understanding across many datasets. AIR-Bench similarly evaluates generative audio comprehension over speech, natural sounds, and music. This supports a task registry rather than a single hard-coded ASR metric. Sources: [AudioBench, NAACL 2025](https://aclanthology.org/2025.naacl-long.218/) and [AIR-Bench, ACL 2024](https://arxiv.org/abs/2402.07729).

Advanced audio reasoning benchmarks are exposing gaps in current models. MMAU stresses expert-level reasoning over speech, environmental sound, and music. OmniBench extends this to tri-modal image-audio-text reasoning and reports that many baselines remain weak even with textual alternatives. This argues for storing enough context and metadata for later failure analysis rather than only reporting averages. Sources: [MMAU](https://arxiv.org/abs/2410.19168) and [OmniBench](https://openreview.net/forum?id=SSF4qgsNYE).

## Audio LLMs As Judges

The strongest directly relevant evidence is emerging but task-specific. Chiang et al. show that GPT-4o-audio and Gemini-2.5-Pro can judge speaking styles, with Gemini-human agreement comparable to human-human agreement in their setting. They explicitly caution that this does not prove all speech attributes are judgeable by current audio LLMs. This supports narrow, rubric-first judges such as `tts_naturalness` and `asr_error`, not a vague "judge quality" prompt. Source: [Audio-Aware Large Language Models as Judges for Speaking Styles](https://arxiv.org/html/2506.05984v1).

SpeechLLM-as-Judges proposes structured, explanation-based speech-quality evaluation over quality assessment, pairwise comparison, improvement suggestion, and deepfake detection. Their SQ-LLM is built around prompts and structured outputs, with intermediate quality dimensions guiding the final result. This supports asking a judge to privately inspect dimensions but emit a compact JSON object. Source: [SpeechLLM-as-Judges](https://arxiv.org/html/2510.14664v1).

GatherMOS uses an LLM as a meta-evaluator that combines acoustic descriptors and pseudo-labels from existing speech metrics. That is a useful later direction for Open Audio Judge: prompts can accept auxiliary metrics such as DNSMOS, VQScore, WER, CER, VAD segment stats, or diarization error rate when available, while keeping the final schema unchanged. Source: [Few-Shot and Pseudo-Label Guided Speech Quality Evaluation with Large Language Models](https://arxiv.org/abs/2604.13528).

TRACE, from "Hearing Between the Lines," separates speech-to-speech evaluation into content, voice quality, and paralinguistics, then fuses dimension-wise judgments into an overall rating. That pattern maps well to Open Audio Judge's future multi-dimensional reports. Source: [Hearing Between the Lines, EACL 2026](https://aclanthology.org/2026.findings-eacl.151.pdf).

## ASR Evaluation Implications

WER remains useful but incomplete. Recent ASR-evaluation papers and Google Research's meaning-preservation work emphasize that lexical edit distance can treat harmless formatting differences and meaning-changing substitutions as equally severe. For Open Audio Judge, ASR prompts should score meaning preservation, named entities, numbers, negation, omissions, insertions, speaker turns, and downstream task impact. Sources: [Google Research meaning preservation](https://research.google/blog/assessing-asr-performance-with-meaning-preservation/), [Evaluation of Automatic Speech Recognition Using Generative Large Language Models](https://arxiv.org/html/2604.21928v3), and [An Approach to Measuring ASR Performance for LLM Applications](https://www.isca-archive.org/interspeech_2025/pulikodan25_interspeech.pdf).

The ASR judge should support both reference-based and audio-direct judging. If a reference transcript is available, the judge compares the candidate to both the audio and reference. If no reference is available, the judge listens to the audio and estimates transcript fidelity directly, while clearly reducing confidence in the reason.

## Model And Serving Notes

Qwen3-Omni is a strong open local judge candidate because it supports text, image, audio, and video inputs with text output. The Qwen repo recommends vLLM or DashScope for scalable invocation and lists the Instruct, Thinking, and Captioner variants. vLLM-Omni documents OpenAI-compatible chat completions and uses `modalities: ["text"]` when only text output is wanted. Sources: [Qwen3-Omni GitHub](https://github.com/QwenLM/Qwen3-Omni), [Qwen3-Omni technical report](https://arxiv.org/abs/2509.17765), and [vLLM-Omni Qwen3-Omni docs](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/examples/online_serving/qwen3_omni/).

Gemini's official docs expose audio understanding for transcription, translation, diarization, emotion detection, and timestamped analysis. That makes it a natural hosted provider target once the MVP provider boundary is stable. Source: [Gemini audio understanding docs](https://ai.google.dev/gemini-api/docs/audio).

## Design Requirements Derived From Research

1. Version every prompt and include the prompt id/version in every result.
2. Keep the judge output schema small and stable: `overall_score` and `reason`.
3. Store the raw response for audits and parser improvements.
4. Do not expose private chain-of-thought in outputs; ask models to reason internally and summarize evidence.
5. Support auxiliary metrics later, but do not require them for the first protocol.
6. Produce human-scannable reports with distributions, thresholds, and per-case explanations.
7. Treat each judge family as calibrated separately; do not compare TTS naturalness scores directly to ASR error scores without calibration.
