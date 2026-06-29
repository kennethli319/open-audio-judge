from __future__ import annotations

import json
import re

from open_audio_judge.models import EvaluationCase, ProviderResponse, RenderedPrompt


class MockProvider:
    name = "mock"

    def generate(self, case: EvaluationCase, prompt: RenderedPrompt) -> ProviderResponse:
        if case.reference_text and case.candidate_text:
            score = _score_by_word_overlap(case.reference_text, case.candidate_text)
            reason = (
                "Mock score based on normalized word overlap between reference and candidate; "
                "use a real audio LLM provider for actual judging."
            )
        else:
            score = 75
            reason = "Mock score used because reference or candidate text was missing."

        content = json.dumps(
            {
                "overall_score": score,
                "reason": reason,
                "judge_transcript": case.reference_text,
                "meaning_preservation": "uncertain",
                "semantic_error_summary": "Mock provider does not listen to audio or infer semantics.",
                "key_differences": [],
                "error_categories": ["mock"],
                "researcher_notes": ["Use qwen or another audio LLM provider for real ASR diagnostics."],
            }
        )
        return ProviderResponse(content=content, raw={"provider": self.name, "judge": prompt.judge_id})


def _score_by_word_overlap(reference: str, candidate: str) -> int:
    ref_words = _normalize_words(reference)
    cand_words = _normalize_words(candidate)
    if not ref_words and not cand_words:
        return 100
    if not ref_words or not cand_words:
        return 1

    distance = _levenshtein(ref_words, cand_words)
    wer = distance / max(len(ref_words), 1)
    return max(1, min(100, round((1.0 - min(wer, 1.0)) * 100)))


def _normalize_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _levenshtein(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for i, left_word in enumerate(left, start=1):
        current = [i]
        for j, right_word in enumerate(right, start=1):
            substitution_cost = 0 if left_word == right_word else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]
