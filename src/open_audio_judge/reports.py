from __future__ import annotations

import html
import statistics
from pathlib import Path

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

    rows = "\n".join(_render_row(result) for result in results)
    bucket_markup = "\n".join(
        f'<div class="bucket"><span>{html.escape(name)}</span>'
        f'<div class="track"><div class="fill" style="width:{_pct(count, len(results))}%"></div></div>'
        f"<strong>{count}</strong></div>"
        for name, count in buckets
    )

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

    <h2>Case Results</h2>
    <table>
      <thead>
        <tr>
          <th>Case</th>
          <th>Score</th>
          <th>Label</th>
          <th>Reason</th>
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
  <td data-label="Status">{html.escape(result.status)}</td>
</tr>"""


def _bucket_counts(scores: list[int]) -> list[tuple[str, int]]:
    ranges = [(1, 20), (21, 40), (41, 60), (61, 80), (81, 100)]
    return [(f"{low}-{high}", sum(1 for score in scores if low <= score <= high)) for low, high in ranges]


def _pct(count: int, total: int) -> int:
    if total == 0:
        return 0
    return round((count / total) * 100)
