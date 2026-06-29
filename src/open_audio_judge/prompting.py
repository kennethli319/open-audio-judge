from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, StrictUndefined

from open_audio_judge.models import EvaluationCase, JudgePrompt, RenderedPrompt


PROMPT_FILENAMES = {
    "asr_error": "asr_error_judge.yaml",
    "tts_naturalness": "tts_naturalness.yaml",
}


def prompt_dir() -> Path:
    env_dir = os.getenv("OAJ_PROMPTS_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()

    cwd_prompts = Path.cwd() / "prompts"
    if cwd_prompts.exists():
        return cwd_prompts.resolve()

    return (Path(__file__).resolve().parents[2] / "prompts").resolve()


def resolve_prompt_path(judge_or_path: str | Path) -> Path:
    candidate = Path(judge_or_path)
    if candidate.exists():
        return candidate.resolve()

    name = str(judge_or_path)
    filename = PROMPT_FILENAMES.get(name, f"{name}.yaml")
    path = prompt_dir() / filename
    if path.exists():
        return path

    matches = sorted(prompt_dir().glob(f"{name}*.yaml"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"No prompt found for {judge_or_path!s} in {prompt_dir()}")


def load_prompt(judge_or_path: str | Path) -> JudgePrompt:
    path = resolve_prompt_path(judge_or_path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return JudgePrompt.model_validate(data)


def render_prompt(prompt: JudgePrompt, case: EvaluationCase) -> RenderedPrompt:
    env = Environment(undefined=StrictUndefined, autoescape=False, trim_blocks=True, lstrip_blocks=True)
    context: dict[str, Any] = case.model_dump()
    user_text = env.from_string(prompt.user).render(**context)
    system_text = env.from_string(prompt.system).render(**context)
    return RenderedPrompt(
        judge_id=prompt.id,
        judge_version=prompt.version,
        system=system_text,
        user=user_text,
    )
