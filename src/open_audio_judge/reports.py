from __future__ import annotations

import html
from collections import Counter
import statistics
from pathlib import Path
from typing import Iterable

from open_audio_judge.models import EvaluationResult


def label_for_score(score: int, accurate_threshold: int = 80, review_threshold: int = 60) -> str:
    if score >= accurate_threshold:
        return "accurate"
    if score >= review_threshold:
        return "needs_review"
    return "inaccurate"


def write_html_report(results: list[EvaluationResult], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html_report(results), encoding="utf-8")
    return output_path


def render_html_report(results: list[EvaluationResult]) -> str:
    scores = [result.overall_score for result in results]
    average = statistics.mean(scores) if scores else 0
    median = statistics.median(scores) if scores else 0
    counts = {label: sum(1 for result in results if result.label == label) for label in LABELS}
    buckets = _bucket_counts(scores)
    meaning_counts = _field_counts(result.meaning_preservation for result in results)
    category_counts = _category_counts(results)
    high_impact_counts = _high_impact_category_counts(results)
    researcher_note_counts = _researcher_note_counts(results)
    priority_cases = _priority_cases(results)
    calibration_checks = _calibration_checks(results)

    rows = "\n".join(_render_row(result) for result in results)
    bucket_markup = "\n".join(
        f'<div class="bucket"><span>{html.escape(name)}</span>'
        f'<div class="track"><div class="fill" style="width:{_pct(count, len(results))}%"></div></div>'
        f"<strong>{count}</strong></div>"
        for name, count in buckets
    )
    meaning_markup = _render_count_list(meaning_counts, empty_label="No meaning diagnostics")
    category_markup = _render_count_list(category_counts, empty_label="No error categories")
    high_impact_markup = _render_count_list(
        high_impact_counts,
        empty_label="No high-impact semantic errors",
    )
    researcher_note_markup = _render_count_list(
        researcher_note_counts,
        empty_label="No researcher notes",
    )
    priority_markup = _render_priority_cases(priority_cases)
    calibration_markup = _render_calibration_checks(calibration_checks)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Open Audio Judge Report</title>
  <style>
    :root {{
      --ink: #172026;
      --muted: #5d6975;
      --line: #d8dee5;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --good: #0f8a5f;
      --warn: #b36b00;
      --bad: #c93535;
      --accent: #2864b4;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    header {{
      padding: 28px 32px 20px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
    }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    h2 {{ margin: 28px 0 12px; font-size: 18px; letter-spacing: 0; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 24px 24px 40px; }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 12px;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 16px;
    }}
    .metric span {{ display: block; color: var(--muted); font-size: 13px; }}
    .metric strong {{ display: block; margin-top: 6px; font-size: 26px; }}
    .bucket {{
      display: grid;
      grid-template-columns: 72px 1fr 42px;
      align-items: center;
      gap: 10px;
      margin: 8px 0;
      color: var(--muted);
    }}
    .track {{ height: 10px; border-radius: 999px; background: #e8edf2; overflow: hidden; }}
    .fill {{ height: 100%; background: var(--accent); }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    th, td {{ padding: 12px 14px; border-bottom: 1px solid var(--line); vertical-align: top; }}
    th {{ text-align: left; font-size: 13px; color: var(--muted); background: #fbfcfd; }}
    tr:last-child td {{ border-bottom: 0; }}
    .scorebar {{ min-width: 160px; }}
    .bar {{ height: 9px; background: #e8edf2; border-radius: 999px; overflow: hidden; margin-top: 7px; }}
    .bar > div {{ height: 100%; }}
    .accurate {{ color: var(--good); }}
    .needs_review {{ color: var(--warn); }}
    .inaccurate {{ color: var(--bad); }}
    .accurate-fill {{ background: var(--good); }}
    .needs_review-fill {{ background: var(--warn); }}
    .inaccurate-fill {{ background: var(--bad); }}
    .reason {{ color: var(--ink); line-height: 1.45; }}
    .muted {{ color: var(--muted); }}
    details {{ margin-top: 8px; }}
    summary {{ cursor: pointer; color: var(--accent); }}
    ul {{ margin: 6px 0 10px 18px; padding: 0; }}
    li {{ margin: 3px 0; }}
    .counts {{ list-style: none; margin: 10px 0 0; }}
    .counts li {{ display: flex; justify-content: space-between; gap: 14px; }}
    @media (max-width: 760px) {{
      header {{ padding: 22px 18px 16px; }}
      main {{ padding: 18px 12px 28px; }}
      table, thead, tbody, th, td, tr {{ display: block; }}
      thead {{ display: none; }}
      tr {{ border-bottom: 1px solid var(--line); padding: 10px 0; }}
      td {{ border-bottom: 0; padding: 8px 12px; }}
      td::before {{ content: attr(data-label); display: block; color: var(--muted); font-size: 12px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Open Audio Judge Report</h1>
    <div class="muted">Prompt-based audio LLM evaluation summary</div>
  </header>
  <main>
    <section class="summary">
      <div class="metric"><span>Cases</span><strong>{len(results)}</strong></div>
      <div class="metric"><span>Average</span><strong>{average:.1f}</strong></div>
      <div class="metric"><span>Median</span><strong>{median:.1f}</strong></div>
      <div class="metric"><span>Accurate</span><strong>{counts["accurate"]}</strong></div>
      <div class="metric"><span>Needs Review</span><strong>{counts["needs_review"]}</strong></div>
      <div class="metric"><span>Inaccurate</span><strong>{counts["inaccurate"]}</strong></div>
    </section>

    <h2>Score Distribution</h2>
    {bucket_markup}

    <h2>Semantic Diagnostics</h2>
    <section class="summary">
      <div class="metric"><span>Meaning Preservation</span>{meaning_markup}</div>
      <div class="metric"><span>Error Categories</span>{category_markup}</div>
      <div class="metric"><span>High-Impact Errors</span>{high_impact_markup}</div>
      <div class="metric"><span>Actionable Notes</span>{researcher_note_markup}</div>
    </section>

    <h2>Calibration Checks</h2>
    {calibration_markup}

    <h2>Priority Cases</h2>
    {priority_markup}

    <h2>Case Results</h2>
    <table>
      <thead>
        <tr>
          <th>Case</th>
          <th>Score</th>
          <th>Label</th>
          <th>Reason</th>
          <th>Diagnostics</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
  </main>
</body>
</html>"""


LABELS = ("accurate", "needs_review", "inaccurate")
HIGH_IMPACT_CATEGORIES = {
    "negation_error",
    "number_error",
    "entity_error",
    "date_time_error",
    "unit_error",
}
MEANING_SEVERITY = {
    "not_preserved": 5,
    "major_loss": 4,
    "partial_loss": 3,
    "minor_loss": 2,
    "preserved": 1,
}


def _render_row(result: EvaluationResult) -> str:
    score = result.overall_score
    label = result.label
    return f"""<tr>
  <td data-label="Case">{html.escape(result.case_id)}</td>
  <td data-label="Score" class="scorebar"><strong>{score}</strong>
    <div class="bar"><div class="{label}-fill" style="width:{score}%"></div></div>
  </td>
  <td data-label="Label" class="{label}">{html.escape(label.replace("_", " "))}</td>
  <td data-label="Reason" class="reason">{html.escape(result.reason)}</td>
  <td data-label="Diagnostics">{_render_diagnostics(result)}</td>
  <td data-label="Status">{html.escape(result.status)}</td>
</tr>"""


def _bucket_counts(scores: list[int]) -> list[tuple[str, int]]:
    ranges = [(1, 20), (21, 40), (41, 60), (61, 80), (81, 100)]
    return [(f"{low}-{high}", sum(1 for score in scores if low <= score <= high)) for low, high in ranges]


def _field_counts(values: Iterable[str | None]) -> list[tuple[str, int]]:
    counts = Counter(value for value in values if value)
    return counts.most_common()


def _category_counts(results: list[EvaluationResult]) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    for result in results:
        counts.update(result.error_categories)
    return counts.most_common()


def _high_impact_category_counts(results: list[EvaluationResult]) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    for result in results:
        counts.update(
            category
            for category in result.error_categories
            if category in HIGH_IMPACT_CATEGORIES
        )
    return counts.most_common()


def _researcher_note_counts(results: list[EvaluationResult]) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    for result in results:
        counts.update(result.researcher_notes)
    return counts.most_common(6)


def _render_count_list(items: list[tuple[str, int]], empty_label: str) -> str:
    if not items:
        return f'<strong class="muted">{html.escape(empty_label)}</strong>'
    rendered_items = "".join(
        f"<li><span>{html.escape(name.replace('_', ' '))}</span> <strong>{count}</strong></li>"
        for name, count in items
    )
    return f'<ul class="counts">{rendered_items}</ul>'


def _priority_cases(results: list[EvaluationResult], limit: int = 5) -> list[EvaluationResult]:
    candidates = [
        result
        for result in results
        if _meaning_severity(result) >= MEANING_SEVERITY["partial_loss"]
        or HIGH_IMPACT_CATEGORIES.intersection(result.error_categories)
        or result.label != "accurate"
        or result.status != "ok"
    ]
    return sorted(
        candidates,
        key=lambda result: (
            result.status == "ok",
            -_meaning_severity(result),
            -_high_impact_count(result),
            result.overall_score,
            result.case_id,
        ),
    )[:limit]


def _render_priority_cases(results: list[EvaluationResult]) -> str:
    if not results:
        return '<div class="metric"><strong class="muted">No priority cases</strong></div>'

    rows = "\n".join(
        f"""<tr>
  <td data-label="Case">{html.escape(result.case_id)}</td>
  <td data-label="Score">{result.overall_score}</td>
  <td data-label="Meaning">{html.escape((result.meaning_preservation or "unknown").replace("_", " "))}</td>
  <td data-label="Category">{html.escape(_priority_reason(result))}</td>
  <td data-label="Summary">{html.escape(result.semantic_error_summary or result.reason)}</td>
</tr>"""
        for result in results
    )
    return f"""<table>
      <thead>
        <tr>
          <th>Case</th>
          <th>Score</th>
          <th>Meaning</th>
          <th>Category</th>
          <th>Summary</th>
        </tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>"""


def _calibration_checks(results: list[EvaluationResult]) -> list[tuple[EvaluationResult, list[str]]]:
    checks: list[tuple[EvaluationResult, list[str]]] = []
    for result in results:
        mismatches: list[str] = []
        expected_meaning = result.metadata.get("expected_meaning_preservation")
        if isinstance(expected_meaning, str) and expected_meaning != result.meaning_preservation:
            mismatches.append(
                "expected meaning "
                f"{expected_meaning.replace('_', ' ')}, got "
                f"{(result.meaning_preservation or 'unknown').replace('_', ' ')}"
            )

        expected_categories = result.metadata.get("expected_error_categories")
        if isinstance(expected_categories, list):
            missing_categories = sorted(
                category
                for category in expected_categories
                if isinstance(category, str) and category not in result.error_categories
            )
            if missing_categories:
                mismatches.append(
                    "missing categories: "
                    + ", ".join(category.replace("_", " ") for category in missing_categories)
                )

        if mismatches:
            checks.append((result, mismatches))
    return checks


def _render_calibration_checks(checks: list[tuple[EvaluationResult, list[str]]]) -> str:
    if not checks:
        return '<div class="metric"><strong class="muted">All calibration expectations matched</strong></div>'

    rows = "\n".join(
        f"""<tr>
  <td data-label="Case">{html.escape(result.case_id)}</td>
  <td data-label="Focus">{html.escape(str(result.metadata.get("calibration_focus", "unspecified")).replace("_", " "))}</td>
  <td data-label="Mismatch">{_render_list("Expectation misses", mismatches)}</td>
</tr>"""
        for result, mismatches in checks
    )
    return f"""<table>
      <thead>
        <tr>
          <th>Case</th>
          <th>Focus</th>
          <th>Mismatch</th>
        </tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>"""


def _meaning_severity(result: EvaluationResult) -> int:
    return MEANING_SEVERITY.get(result.meaning_preservation or "", 0)


def _high_impact_count(result: EvaluationResult) -> int:
    return len(HIGH_IMPACT_CATEGORIES.intersection(result.error_categories))


def _priority_reason(result: EvaluationResult) -> str:
    high_impact = [
        category.replace("_", " ")
        for category in result.error_categories
        if category in HIGH_IMPACT_CATEGORIES
    ]
    if high_impact:
        return ", ".join(high_impact)
    if result.status != "ok":
        return result.status.replace("_", " ")
    if result.error_categories:
        return ", ".join(category.replace("_", " ") for category in result.error_categories)
    return result.label.replace("_", " ")


def _render_diagnostics(result: EvaluationResult) -> str:
    parts: list[str] = []
    if result.meaning_preservation:
        parts.append(
            f'<div><span class="muted">Meaning</span><br>{html.escape(result.meaning_preservation)}</div>'
        )
    if result.semantic_error_summary:
        parts.append(
            f'<div><span class="muted">Semantic impact</span><br>'
            f"{html.escape(result.semantic_error_summary)}</div>"
        )
    if result.judge_transcript:
        parts.append(
            f'<details><summary>Judge transcript</summary>'
            f"{html.escape(result.judge_transcript)}</details>"
        )
    if result.key_differences:
        parts.append(_render_list("Key differences", result.key_differences))
    if result.error_categories:
        categories = ", ".join(result.error_categories)
        parts.append(f'<div><span class="muted">Categories</span><br>{html.escape(categories)}</div>')
    if result.researcher_notes:
        parts.append(_render_list("Researcher notes", result.researcher_notes))
    return "".join(parts) if parts else '<span class="muted">None</span>'


def _render_list(title: str, items: list[str]) -> str:
    rendered_items = "".join(f"<li>{html.escape(item)}</li>" for item in items)
    return f'<div><span class="muted">{html.escape(title)}</span><ul>{rendered_items}</ul></div>'


def _pct(count: int, total: int) -> int:
    if total == 0:
        return 0
    return round((count / total) * 100)
