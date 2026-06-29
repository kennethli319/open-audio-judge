from open_audio_judge.models import EvaluationCase
from open_audio_judge.prompting import format_turns, load_prompt, render_prompt


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
    assert prompt.version == "0.2.0"
    assert "Transfer fifteen dollars." in rendered.user
    assert "Transfer fifty dollars." in rendered.user
    assert "your own concise best-effort transcript" in rendered.system
    assert "Return only valid JSON" in rendered.system


def test_render_multiturn_context() -> None:
    prompt = load_prompt("tts_naturalness")
    rendered = render_prompt(
        prompt,
        EvaluationCase(
            id="case-2",
            task="tts_naturalness",
            reference_text="Sure, I can summarize that.",
            turns=[
                {"role": "user", "content": "Can you summarize the launch notes?"},
                {"role": "assistant", "content": "Sure, I can summarize that."},
            ],
        ),
    )

    assert "Conversation context:" in rendered.user
    assert "USER: Can you summarize the launch notes?" in rendered.user
    assert "ASSISTANT: Sure, I can summarize that." in rendered.user


def test_format_turns() -> None:
    assert format_turns([{"role": "user", "content": "Hello"}]) == "USER: Hello"
