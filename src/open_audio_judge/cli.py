from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from open_audio_judge.prompting import load_prompt
from open_audio_judge.providers import build_provider
from open_audio_judge.runner import evaluate_cases, load_cases

console = Console()
app = typer.Typer(no_args_is_help=True, help="Open Audio Judge CLI")


@app.command("eval")
def eval_command(
    cases: Annotated[Path, typer.Option("--cases", "-c", help="JSONL or JSON case file.")],
    judge: Annotated[str, typer.Option("--judge", "-j", help="Judge prompt id or path.")] = "asr_error",
    provider: Annotated[str, typer.Option("--provider", "-p", help="Provider name.")] = "qwen",
    out: Annotated[Path, typer.Option("--out", "-o", help="Output directory.")] = Path("runs/latest"),
) -> None:
    prompt = load_prompt(judge)
    judge_provider = build_provider(provider)
    loaded_cases = load_cases(cases)
    results = evaluate_cases(loaded_cases, prompt, judge_provider, out)
    ok_count = sum(1 for result in results if result.status == "ok")
    console.print(f"[bold]Evaluated {len(results)} cases[/bold] ({ok_count} ok)")
    console.print(f"Results: {out / 'results.jsonl'}")
    console.print(f"Report:  {out / 'report.html'}")


@app.command("serve")
def serve_command(
    host: Annotated[str, typer.Option("--host", help="Bind host.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Bind port.")] = 8000,
    reload: Annotated[bool, typer.Option("--reload", help="Enable development reload.")] = False,
) -> None:
    import uvicorn

    uvicorn.run("open_audio_judge.api:app", host=host, port=port, reload=reload)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
