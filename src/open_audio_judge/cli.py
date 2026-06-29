from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from open_audio_judge.evalset_bridge import build_tts_cases, load_evalset_records, write_cases_jsonl
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


@app.command("build-tts-cases")
def build_tts_cases_command(
    source: Annotated[Path, typer.Option("--source", help="Source JSONL evalset path.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output JSONL case manifest.")] = Path(
        "runs/tts-evalset/cases.jsonl"
    ),
    source_name: Annotated[
        str, typer.Option("--source-name", help="Stable source label for case metadata.")
    ] = "evalset",
    limit: Annotated[int | None, typer.Option("--limit", help="Maximum cases to write.")] = 25,
    categories: Annotated[
        str,
        typer.Option(
            "--categories",
            help="Optional comma-separated source categories to include before TTS slicing.",
        ),
    ] = "",
    slice_labels: Annotated[
        str,
        typer.Option(
            "--slices",
            help="Optional comma-separated TTS slice labels to include after classification.",
        ),
    ] = "",
    per_slice_limit: Annotated[
        int | None,
        typer.Option("--per-slice-limit", help="Maximum cases to keep for each TTS slice."),
    ] = None,
) -> None:
    category_filter = {item.strip() for item in categories.split(",") if item.strip()} or None
    slice_filter = {item.strip() for item in slice_labels.split(",") if item.strip()} or None
    records = load_evalset_records(source)
    cases = build_tts_cases(
        records,
        source_name=source_name,
        limit=limit,
        category_filter=category_filter,
        slice_filter=slice_filter,
        per_slice_limit=per_slice_limit,
    )
    write_cases_jsonl(cases, out)
    console.print(f"[bold]Wrote {len(cases)} TTS cases[/bold]")
    console.print(f"Cases: {out}")


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
