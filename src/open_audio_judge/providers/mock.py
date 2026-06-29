from __future__ import annotations

import json

from open_audio_judge.asr_semantics import analyze_reference_candidate
from open_audio_judge.models import EvaluationCase, ProviderResponse, RenderedPrompt


class MockProvider:
    name = "mock"

    def generate(self, case: EvaluationCase, prompt: RenderedPrompt) -> ProviderResponse:
        if case.reference_text and case.candidate_text:
            diff = analyze_reference_candidate(case.reference_text, case.candidate_text)
            reason = (
                "Mock semantic baseline based on reference-candidate diagnostics; "
                "use a real audio LLM provider for actual judging."
            )
        else:
            diff = None
            reason = "Mock score used because reference or candidate text was missing."

        content = json.dumps(
            {
                "overall_score": diff.score if diff else 75,
                "reason": reason,
                "judge_transcript": case.reference_text,
                "meaning_preservation": diff.meaning_preservation if diff else "uncertain",
                "semantic_error_summary": (
                    diff.semantic_error_summary
                    if diff
                    else "Mock provider does not listen to audio or infer semantics."
                ),
                "key_differences": diff.key_differences if diff else [],
                "error_categories": diff.error_categories if diff else ["mock"],
                "researcher_notes": (
                    diff.researcher_notes
                    if diff
                    else ["Use qwen or another audio LLM provider for real ASR diagnostics."]
                ),
            }
        )
        return ProviderResponse(content=content, raw={"provider": self.name, "judge": prompt.judge_id})
