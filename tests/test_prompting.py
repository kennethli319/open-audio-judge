from open_audio_judge.models import EvaluationCase
from open_audio_judge.prompting import load_prompt, render_prompt


def test_load_and_render_asr_prompt() -> None:
    prompt = load_prompt("asr_error")
    rendered = render_prompt(
        prompt,
        EvaluationCase(
            id="case-1",
            task="asr_error",
            reference_text="Transfer fifteen dollars.",
            candidate_text="Transfer fifty dollars.",
            metadata={"language": "en", "domain": "payments"},
        ),
    )

    assert prompt.id == "asr_error"
    assert "Transfer fifteen dollars." in rendered.user
    assert "Transfer fifty dollars." in rendered.user
    assert "Return only valid JSON" in rendered.system
