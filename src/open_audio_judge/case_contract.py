from __future__ import annotations

from open_audio_judge.models import EvaluationCase


def require_audio_and_text(case: EvaluationCase) -> None:
    """Require the core Open Audio Judge input contract for hosted audio judges."""
    if not case.audio_url and not case.audio_path:
        raise ValueError("Audio judge cases require audio_url or audio_path.")
    has_text_context = any(
        [
            bool((case.reference_text or "").strip()),
            bool((case.candidate_text or "").strip()),
            any(turn.content.strip() for turn in case.turns),
        ]
    )
    if not has_text_context:
        raise ValueError(
            "Audio judge cases require textual context via reference_text, candidate_text, or turns."
        )
