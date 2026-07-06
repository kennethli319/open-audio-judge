from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from open_audio_judge.evalset_bridge import (
    build_tts_cases,
    load_evalset_records,
    summarize_tts_cases,
    write_cases_jsonl,
    write_tts_summary_json,
)
from open_audio_judge.hf_asr import (
    transcribe_cases_with_hf_asr,
    write_hf_asr_cases_jsonl,
    write_hf_asr_summary_json,
)
from open_audio_judge.local_tts import (
    DEFAULT_CHATTERBOX_BIN,
    DEFAULT_CHATTERBOX_MODEL,
    LocalTtsConfig,
    synthesize_cases_with_local_tts,
    write_local_tts_cases_jsonl,
    write_local_tts_summary_json,
)
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


@app.command("autojudge-hf-asr")
def autojudge_hf_asr_command(
    cases: Annotated[Path, typer.Option("--cases", "-c", help="Local-audio ASR case file.")],
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Hugging Face ASR model id."),
    ] = "openai/whisper-tiny",
    judge: Annotated[str, typer.Option("--judge", "-j", help="Judge prompt id or path.")] = "asr_error",
    judge_provider: Annotated[
        str,
        typer.Option("--judge-provider", help="Provider used to judge candidate transcripts."),
    ] = "mock",
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Output directory for transcripts and judge report."),
    ] = Path("runs/hf-asr-autojudge"),
    device: Annotated[
        str,
        typer.Option("--device", help="Transformers pipeline device, such as cpu, mps, cuda, or 0."),
    ] = "cpu",
    limit: Annotated[int | None, typer.Option("--limit", help="Maximum cases to evaluate.")] = None,
) -> None:
    loaded_cases = load_cases(cases)
    if limit is not None:
        loaded_cases = loaded_cases[:limit]

    candidate_cases = transcribe_cases_with_hf_asr(
        loaded_cases,
        model=model,
        device=device,
    )
    candidate_path = out / "candidate_cases.jsonl"
    summary_path = out / "model_summary.json"
    judge_out = out / "judge-report"
    write_hf_asr_cases_jsonl(candidate_cases, candidate_path)
    write_hf_asr_summary_json(
        candidate_cases,
        summary_path,
        source_cases=cases,
        model=model,
    )

    prompt = load_prompt(judge)
    provider = build_provider(judge_provider)
    results = evaluate_cases(candidate_cases, prompt, provider, judge_out)
    ok_count = sum(1 for result in results if result.status == "ok")
    console.print(f"[bold]AutoJudged {len(results)} Hugging Face ASR cases[/bold] ({ok_count} ok)")
    console.print(f"Model:   {model}")
    console.print(f"Cases:   {candidate_path}")
    console.print(f"Summary: {summary_path}")
    console.print(f"Results: {judge_out / 'results.jsonl'}")
    console.print(f"Report:  {judge_out / 'report.html'}")


@app.command("autojudge-local-tts")
def autojudge_local_tts_command(
    cases: Annotated[Path, typer.Option("--cases", "-c", help="TTS case file with reference_text.")],
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Local TTS model id."),
    ] = DEFAULT_CHATTERBOX_MODEL,
    judge: Annotated[str, typer.Option("--judge", "-j", help="Judge prompt id or path.")] = "tts_naturalness",
    judge_provider: Annotated[
        str,
        typer.Option("--judge-provider", help="Provider used to judge synthesized audio."),
    ] = "mock",
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Output directory for audio and judge report."),
    ] = Path("runs/chatterbox-tts-autojudge"),
    tts_bin: Annotated[
        Path,
        typer.Option("--tts-bin", help="Path to local-tts-speak-compatible command."),
    ] = DEFAULT_CHATTERBOX_BIN,
    voice: Annotated[str, typer.Option("--voice", help="TTS voice id.")] = "af_heart",
    lang_code: Annotated[str, typer.Option("--lang-code", help="TTS language code.")] = "en",
    audio_format: Annotated[
        str,
        typer.Option("--audio-format", help="Generated audio format."),
    ] = "wav",
    keep_text_sidecars: Annotated[
        bool,
        typer.Option("--keep-text-sidecars", help="Keep local target-text sidecar files."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Write manifests without invoking the TTS model."),
    ] = False,
    limit: Annotated[int | None, typer.Option("--limit", help="Maximum cases to synthesize.")] = None,
) -> None:
    loaded_cases = load_cases(cases)
    if limit is not None:
        loaded_cases = loaded_cases[:limit]

    candidate_dir = out / "synthesis"
    synthesized_cases = synthesize_cases_with_local_tts(
        loaded_cases,
        out_dir=candidate_dir,
        config=LocalTtsConfig(
            tts_bin=tts_bin,
            model=model,
            voice=voice,
            lang_code=lang_code,
            audio_format=audio_format,
            keep_text_sidecars=keep_text_sidecars,
            dry_run=dry_run,
        ),
    )
    candidate_path = candidate_dir / "tts_audio_cases.jsonl"
    summary_path = out / "model_summary.json"
    judge_out = out / "judge-report"
    write_local_tts_cases_jsonl(synthesized_cases, candidate_path)
    write_local_tts_summary_json(
        synthesized_cases,
        summary_path,
        source_cases=cases,
        model=model,
    )

    prompt = load_prompt(judge)
    provider = build_provider(judge_provider)
    resolved_cases = load_cases(candidate_path)
    results = evaluate_cases(resolved_cases, prompt, provider, judge_out)
    ok_count = sum(1 for result in results if result.status == "ok")
    console.print(f"[bold]AutoJudged {len(results)} local TTS cases[/bold] ({ok_count} ok)")
    console.print(f"Model:   {model}")
    console.print(f"Cases:   {candidate_path}")
    console.print(f"Summary: {summary_path}")
    console.print(f"Results: {judge_out / 'results.jsonl'}")
    console.print(f"Report:  {judge_out / 'report.html'}")


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
    hash_source_ids: Annotated[
        bool,
        typer.Option("--hash-source-ids", help="Hash source row ids in generated case metadata."),
    ] = False,
    include_source_task: Annotated[
        bool,
        typer.Option(
            "--include-source-task",
            help="Include raw source task labels in generated case metadata.",
        ),
    ] = False,
    prioritize_slice_coverage: Annotated[
        bool,
        typer.Option(
            "--prioritize-slice-coverage",
            help="When --limit is set, interleave slices before truncating for broader smoke coverage.",
        ),
    ] = False,
    summary_out: Annotated[
        Optional[Path],
        typer.Option("--summary-out", help="Optional metadata-only JSON summary path."),
    ] = None,
    summary_source_examples: Annotated[
        Optional[bool],
        typer.Option(
            "--summary-source-examples/--no-summary-source-examples",
            help=(
                "Include capped example source ids in the optional summary. "
                "Defaults off when --hash-source-ids is used."
            ),
        ),
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
        hash_source_ids=hash_source_ids,
        include_source_task=include_source_task,
        prioritize_slice_coverage=prioritize_slice_coverage,
    )
    write_cases_jsonl(cases, out)
    include_summary_examples = (
        not hash_source_ids if summary_source_examples is None else summary_source_examples
    )
    summary = summarize_tts_cases(cases, include_example_source_ids=include_summary_examples)
    console.print(f"[bold]Wrote {len(cases)} TTS cases[/bold]")
    console.print(f"Cases: {out}")
    console.print(f"Slices: {summary.by_slice}")
    if summary_out is not None:
        write_tts_summary_json(
            cases,
            summary_out,
            include_example_source_ids=include_summary_examples,
        )
        console.print(f"Summary: {summary_out}")


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
