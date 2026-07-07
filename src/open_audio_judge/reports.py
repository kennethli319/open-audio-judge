from __future__ import annotations

import html
from collections import Counter
from dataclasses import dataclass
import statistics
from pathlib import Path
from typing import Iterable

from open_audio_judge.models import EvaluationResult


BASELINE_SYNTHESIS_MODEL = "mlx-community/chatterbox-turbo-6bit"


def label_for_score(score: int, accurate_threshold: int = 80, review_threshold: int = 60) -> str:
    if score >= accurate_threshold:
        return "accurate"
    if score >= review_threshold:
        return "needs_review"
    return "inaccurate"


def write_html_report(
    results: list[EvaluationResult],
    output_path: Path,
    *,
    baseline_model: str = BASELINE_SYNTHESIS_MODEL,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_html_report(results, baseline_model=baseline_model),
        encoding="utf-8",
    )
    return output_path


def render_html_report(
    results: list[EvaluationResult],
    *,
    baseline_model: str = BASELINE_SYNTHESIS_MODEL,
) -> str:
    scores = [result.overall_score for result in results]
    average = statistics.mean(scores) if scores else 0
    median = statistics.median(scores) if scores else 0
    counts = {label: sum(1 for result in results if result.label == label) for label in LABELS}
    buckets = _bucket_counts(scores)
    meaning_counts = _field_counts(result.meaning_preservation for result in results)
    category_counts = _category_counts(results)
    high_impact_counts = _high_impact_category_counts(results)
    researcher_note_counts = _researcher_note_counts(results)
    tts_slice_counts = _metadata_counts(results, "tts_slice")
    synthesis_model_counts = _metadata_counts(results, "synthesis_model")
    synthesis_voice_counts = _metadata_counts(results, "synthesis_voice")
    language_counts = _language_counts(results)
    evaluation_category_counts = _evaluation_category_counts(results)
    sample_kind_counts = _metadata_counts(results, "sample_kind")
    issue_by_evaluation_category_counts = _issue_counts_by_metadata(results, "evaluation_category")
    issue_by_slice_counts = _issue_counts_by_metadata(results, "tts_slice")
    issue_by_model_counts = _issue_counts_by_metadata(results, "synthesis_model")
    issue_by_voice_counts = _issue_counts_by_metadata(results, "synthesis_voice")
    issue_by_language_counts = _issue_counts_by_metadata(results, "language")
    scores_by_evaluation_category = _score_summaries_by_metadata(results, "evaluation_category")
    scores_by_slice = _score_summaries_by_metadata(results, "tts_slice")
    scores_by_model = _score_summaries_by_metadata(results, "synthesis_model")
    scores_by_voice = _score_summaries_by_metadata(results, "synthesis_voice")
    scores_by_language = _score_summaries_by_metadata(results, "language")
    status_by_evaluation_category_counts = _status_counts_by_metadata(
        results,
        "evaluation_category",
    )
    status_by_slice_counts = _status_counts_by_metadata(results, "tts_slice")
    status_by_model_counts = _status_counts_by_metadata(results, "synthesis_model")
    status_by_voice_counts = _status_counts_by_metadata(results, "synthesis_voice")
    status_by_language_counts = _status_counts_by_metadata(results, "language")
    status_by_sample_kind_counts = _status_counts_by_metadata(results, "sample_kind")
    weakest_segments = _weakest_segments(results)
    model_category_actions = _model_category_actions(results)
    baseline_deltas = _baseline_deltas(results, baseline_model=baseline_model)
    priority_cases = _priority_cases(results)
    calibration_checks = _calibration_checks(results)

    rows = "\n".join(_render_row(result) for result in results)
    case_table_controls = _render_case_table_controls(results)
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
    tts_slice_markup = _render_count_list(tts_slice_counts, empty_label="No TTS slices")
    synthesis_model_markup = _render_count_list(
        synthesis_model_counts,
        empty_label="No synthesis models",
    )
    synthesis_voice_markup = _render_count_list(
        synthesis_voice_counts,
        empty_label="No synthesis voices",
    )
    language_markup = _render_count_list(language_counts, empty_label="No languages")
    evaluation_category_markup = _render_count_list(
        evaluation_category_counts,
        empty_label="No evaluation categories",
    )
    sample_kind_markup = _render_count_list(sample_kind_counts, empty_label="No sample kinds")
    issue_by_evaluation_category_markup = _render_count_list(
        issue_by_evaluation_category_counts,
        empty_label="No category issue categories",
    )
    issue_by_slice_markup = _render_count_list(
        issue_by_slice_counts,
        empty_label="No slice issue categories",
    )
    issue_by_model_markup = _render_count_list(
        issue_by_model_counts,
        empty_label="No model issue categories",
    )
    issue_by_voice_markup = _render_count_list(
        issue_by_voice_counts,
        empty_label="No voice issue categories",
    )
    issue_by_language_markup = _render_count_list(
        issue_by_language_counts,
        empty_label="No language issue categories",
    )
    scores_by_slice_markup = _render_score_summary_list(
        scores_by_slice,
        empty_label="No slice scores",
    )
    scores_by_model_markup = _render_score_summary_list(
        scores_by_model,
        empty_label="No model scores",
    )
    scores_by_voice_markup = _render_score_summary_list(
        scores_by_voice,
        empty_label="No voice scores",
    )
    scores_by_language_markup = _render_score_summary_list(
        scores_by_language,
        empty_label="No language scores",
    )
    scores_by_evaluation_category_markup = _render_score_summary_list(
        scores_by_evaluation_category,
        empty_label="No category scores",
    )
    status_by_evaluation_category_markup = _render_count_list(
        status_by_evaluation_category_counts,
        empty_label="No category failures",
    )
    status_by_slice_markup = _render_count_list(
        status_by_slice_counts,
        empty_label="No slice failures",
    )
    status_by_model_markup = _render_count_list(
        status_by_model_counts,
        empty_label="No model failures",
    )
    status_by_voice_markup = _render_count_list(
        status_by_voice_counts,
        empty_label="No voice failures",
    )
    status_by_language_markup = _render_count_list(
        status_by_language_counts,
        empty_label="No language failures",
    )
    status_by_sample_kind_markup = _render_count_list(
        status_by_sample_kind_counts,
        empty_label="No sample-kind failures",
    )
    weakest_segments_markup = _render_weakest_segments(weakest_segments)
    model_category_actions_markup = _render_model_category_actions(model_category_actions)
    baseline_deltas_markup = _render_baseline_deltas(baseline_deltas)
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
    .severity-low {{ border-left: 4px solid var(--bad); }}
    .severity-medium {{ border-left: 4px solid var(--warn); }}
    .severity-high {{ border-left: 4px solid var(--good); }}
    .tag {{
      display: inline-block;
      margin: 3px 4px 3px 0;
      padding: 2px 7px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fbfcfd;
      font-size: 12px;
      color: var(--muted);
    }}
    .case-tools {{
      display: grid;
      grid-template-columns: minmax(220px, 1fr) repeat(5, minmax(130px, 180px));
      gap: 10px;
      margin: 0 0 12px;
      align-items: end;
    }}
    .case-tools label {{
      display: grid;
      gap: 4px;
      color: var(--muted);
      font-size: 12px;
    }}
    .case-tools input,
    .case-tools select {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      color: var(--ink);
      background: var(--panel);
      font: inherit;
    }}
    .sort-button {{
      border: 0;
      background: transparent;
      color: inherit;
      cursor: pointer;
      font: inherit;
      padding: 0;
      text-align: left;
    }}
    .sort-button::after {{
      content: "  sort";
      color: var(--muted);
      font-size: 11px;
      font-weight: 400;
    }}
    .is-hidden {{ display: none !important; }}
    @media (max-width: 760px) {{
      header {{ padding: 22px 18px 16px; }}
      main {{ padding: 18px 12px 28px; }}
      .case-tools {{ grid-template-columns: 1fr; }}
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

    <h2>Candidate Metadata</h2>
    <section class="summary">
      <div class="metric"><span>TTS Slice</span>{tts_slice_markup}</div>
      <div class="metric"><span>Synthesis Model</span>{synthesis_model_markup}</div>
      <div class="metric"><span>Synthesis Voice</span>{synthesis_voice_markup}</div>
      <div class="metric"><span>Language</span>{language_markup}</div>
      <div class="metric"><span>Evaluation Category</span>{evaluation_category_markup}</div>
      <div class="metric"><span>Sample Kind</span>{sample_kind_markup}</div>
      <div class="metric"><span>Issues By Category</span>{issue_by_evaluation_category_markup}</div>
      <div class="metric"><span>Issues By TTS Slice</span>{issue_by_slice_markup}</div>
      <div class="metric"><span>Issues By Model</span>{issue_by_model_markup}</div>
      <div class="metric"><span>Issues By Voice</span>{issue_by_voice_markup}</div>
      <div class="metric"><span>Issues By Language</span>{issue_by_language_markup}</div>
      <div class="metric"><span>Scores By Category</span>{scores_by_evaluation_category_markup}</div>
      <div class="metric"><span>Scores By TTS Slice</span>{scores_by_slice_markup}</div>
      <div class="metric"><span>Scores By Model</span>{scores_by_model_markup}</div>
      <div class="metric"><span>Scores By Voice</span>{scores_by_voice_markup}</div>
      <div class="metric"><span>Scores By Language</span>{scores_by_language_markup}</div>
      <div class="metric"><span>Failures By Category</span>{status_by_evaluation_category_markup}</div>
      <div class="metric"><span>Failures By TTS Slice</span>{status_by_slice_markup}</div>
      <div class="metric"><span>Failures By Model</span>{status_by_model_markup}</div>
      <div class="metric"><span>Failures By Voice</span>{status_by_voice_markup}</div>
      <div class="metric"><span>Failures By Language</span>{status_by_language_markup}</div>
      <div class="metric"><span>Failures By Sample Kind</span>{status_by_sample_kind_markup}</div>
    </section>

    <h2>Calibration Checks</h2>
    {calibration_markup}

    <h2>Weakest Segments</h2>
    {weakest_segments_markup}

    <h2>Model-Category Action Matrix</h2>
    {model_category_actions_markup}

    <h2>Baseline Model Deltas</h2>
    {baseline_deltas_markup}

    <h2>Priority Cases</h2>
    {priority_markup}

    <h2>Case Results</h2>
    {case_table_controls}
    <table id="case-results-table">
      <thead>
        <tr>
          <th><button class="sort-button" type="button" data-sort="case">Case</button></th>
          <th>Provenance</th>
          <th><button class="sort-button" type="button" data-sort="score">Score</button></th>
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
  <script>
    (() => {{
      const table = document.querySelector("#case-results-table");
      if (!table) return;
      const tbody = table.querySelector("tbody");
      const rows = Array.from(tbody.querySelectorAll("tr"));
      const search = document.querySelector("#case-search");
      const label = document.querySelector("#case-label-filter");
      const status = document.querySelector("#case-status-filter");
      const model = document.querySelector("#case-model-filter");
      const category = document.querySelector("#case-category-filter");
      const slice = document.querySelector("#case-slice-filter");
      const count = document.querySelector("#case-visible-count");
      const state = {{ sortKey: "", sortDir: "desc" }};

      function applyFilters() {{
        const query = (search?.value || "").trim().toLowerCase();
        const labelValue = label?.value || "";
        const statusValue = status?.value || "";
        const modelValue = model?.value || "";
        const categoryValue = category?.value || "";
        const sliceValue = slice?.value || "";
        let visible = 0;
        rows.forEach((row) => {{
          const matches =
            (!query || row.dataset.search.includes(query)) &&
            (!labelValue || row.dataset.label === labelValue) &&
            (!statusValue || row.dataset.status === statusValue) &&
            (!modelValue || row.dataset.model === modelValue) &&
            (!categoryValue || row.dataset.category === categoryValue) &&
            (!sliceValue || row.dataset.slice === sliceValue);
          row.classList.toggle("is-hidden", !matches);
          if (matches) visible += 1;
        }});
        if (count) count.textContent = `${{visible}} / ${{rows.length}} shown`;
      }}

      function sortRows(key) {{
        state.sortDir = state.sortKey === key && state.sortDir === "asc" ? "desc" : "asc";
        state.sortKey = key;
        const dir = state.sortDir === "asc" ? 1 : -1;
        const sorted = [...rows].sort((left, right) => {{
          if (key === "score") {{
            return (Number(left.dataset.score) - Number(right.dataset.score)) * dir;
          }}
          return left.dataset.case.localeCompare(right.dataset.case) * dir;
        }});
        sorted.forEach((row) => tbody.appendChild(row));
        rows.splice(0, rows.length, ...sorted);
        applyFilters();
      }}

      [search, label, status, model, category, slice].forEach((control) => {{
        control?.addEventListener("input", applyFilters);
        control?.addEventListener("change", applyFilters);
      }});
      table.querySelectorAll("[data-sort]").forEach((button) => {{
        button.addEventListener("click", () => sortRows(button.dataset.sort));
      }});
      sortRows("score");
    }})();
  </script>
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
SEGMENT_FIELDS = (
    ("Model", "synthesis_model"),
    ("Evaluation Category", "evaluation_category"),
    ("TTS Slice", "tts_slice"),
    ("Voice", "synthesis_voice"),
    ("Language", "language"),
)
ISSUE_FIX_AREAS = {
    "pronunciation_issue": "pronunciation",
    "mispronunciation": "pronunciation",
    "prosody_issue": "prosody/pacing",
    "pacing_issue": "prosody/pacing",
    "rhythm_issue": "prosody/pacing",
    "text_faithfulness_issue": "text faithfulness",
    "omission": "text faithfulness",
    "insertion": "text faithfulness",
    "number_error": "text faithfulness",
    "date_time_error": "text faithfulness",
    "unit_error": "text faithfulness",
    "instruction_following_issue": "instruction/style following",
    "style_mismatch": "instruction/style following",
    "emotion_mismatch": "instruction/style following",
    "artifact": "artifacts",
    "audio_artifact": "artifacts",
    "clipping": "artifacts",
    "intelligibility_issue": "intelligibility",
    "unclear_speech": "intelligibility",
    "voice_consistency_issue": "voice consistency",
    "speaker_drift": "voice consistency",
    "provider_error": "synthesis failures",
    "parse_error": "synthesis failures",
}
@dataclass(frozen=True)
class SegmentSummary:
    field_label: str
    name: str
    count: int
    average: float
    low: int
    high: int
    issue_counts: list[tuple[str, int]]
    status_counts: list[tuple[str, int]]
    fix_areas: list[str]
    representative_cases: list[EvaluationResult]


@dataclass(frozen=True)
class ModelCategoryAction:
    model: str
    category: str
    count: int
    average: float
    low: int
    high: int
    issue_counts: list[tuple[str, int]]
    status_counts: list[tuple[str, int]]
    fix_areas: list[str]
    representative_cases: list[EvaluationResult]


@dataclass(frozen=True)
class BaselineDeltaSummary:
    baseline_model: str
    model: str
    count: int
    average_delta: float
    wins: int
    ties: int
    losses: int
    largest_regressions: list[tuple[EvaluationResult, EvaluationResult, int]]


def _render_row(result: EvaluationResult) -> str:
    score = result.overall_score
    label = result.label
    score_detail = _render_judge_sample_scores(result)
    search_text = _result_search_text(result)
    model = _metadata_group_value(result, "synthesis_model") or ""
    category = _metadata_group_value(result, "evaluation_category") or ""
    tts_slice = _metadata_group_value(result, "tts_slice") or ""
    return f"""<tr data-case="{html.escape(result.case_id)}" data-score="{score}" data-label="{html.escape(label)}" data-status="{html.escape(result.status)}" data-model="{html.escape(model)}" data-category="{html.escape(category)}" data-slice="{html.escape(tts_slice)}" data-search="{html.escape(search_text)}">
  <td data-label="Case">{html.escape(result.case_id)}</td>
  <td data-label="Provenance">{_render_provenance(result)}</td>
  <td data-label="Score" class="scorebar"><strong>{score}</strong>
    <div class="bar"><div class="{label}-fill" style="width:{score}%"></div></div>
    {score_detail}
  </td>
  <td data-label="Label" class="{label}">{html.escape(label.replace("_", " "))}</td>
  <td data-label="Reason" class="reason">{html.escape(result.reason)}</td>
  <td data-label="Diagnostics">{_render_diagnostics(result)}</td>
  <td data-label="Status">{html.escape(result.status)}</td>
</tr>"""


def _render_case_table_controls(results: list[EvaluationResult]) -> str:
    models = sorted(
        {
            value
            for result in results
            if (value := _metadata_group_value(result, "synthesis_model")) is not None
        }
    )
    categories = sorted(
        {
            value
            for result in results
            if (value := _metadata_group_value(result, "evaluation_category")) is not None
        }
    )
    slices = sorted(
        {
            value
            for result in results
            if (value := _metadata_group_value(result, "tts_slice")) is not None
        }
    )
    model_options = "".join(
        f'<option value="{html.escape(model)}">{html.escape(model)}</option>' for model in models
    )
    category_options = "".join(
        f'<option value="{html.escape(category)}">{html.escape(category.replace("_", " "))}</option>'
        for category in categories
    )
    slice_options = "".join(
        f'<option value="{html.escape(tts_slice)}">{html.escape(tts_slice.replace("_", " "))}</option>'
        for tts_slice in slices
    )
    return f"""<section class="case-tools" aria-label="Case result filters">
      <label>Search
        <input id="case-search" type="search" placeholder="case, model, category, reason, issue">
      </label>
      <label>Label
        <select id="case-label-filter">
          <option value="">All labels</option>
          <option value="accurate">Accurate</option>
          <option value="needs_review">Needs review</option>
          <option value="inaccurate">Inaccurate</option>
        </select>
      </label>
      <label>Status
        <select id="case-status-filter">
          <option value="">All statuses</option>
          <option value="ok">OK</option>
          <option value="provider_error">Provider error</option>
          <option value="parse_error">Parse error</option>
        </select>
      </label>
      <label>Model
        <select id="case-model-filter">
          <option value="">All models</option>
          {model_options}
        </select>
      </label>
      <label>Category
        <select id="case-category-filter">
          <option value="">All categories</option>
          {category_options}
        </select>
      </label>
      <label>TTS Slice
        <select id="case-slice-filter">
          <option value="">All slices</option>
          {slice_options}
        </select>
      </label>
      <div class="muted" id="case-visible-count">{len(results)} / {len(results)} shown</div>
    </section>"""


def _result_search_text(result: EvaluationResult) -> str:
    metadata_values = [
        value
        for key in (
            "eval_category",
            "evaluation_category",
            "source_category",
            "tts_slice",
            "synthesis_model",
            "synthesis_voice",
            "language",
            "synthesis_lang_code",
            "sample_kind",
            "source_case_id",
        )
        if isinstance((value := result.metadata.get(key)), str)
    ]
    values = [
        result.case_id,
        result.label,
        result.status,
        result.reason,
        result.semantic_error_summary or "",
        result.meaning_preservation or "",
        *result.error_categories,
        *result.researcher_notes,
        *metadata_values,
    ]
    return " ".join(value for value in values if value).lower()


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


def _metadata_counts(results: list[EvaluationResult], field: str) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    for result in results:
        value = result.metadata.get(field)
        if isinstance(value, str) and value.strip():
            counts[value.strip()] += 1
    return counts.most_common()


def _language_counts(results: list[EvaluationResult]) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    for result in results:
        language = _language_value(result)
        if language is not None:
            counts[language] += 1
    return counts.most_common()


def _evaluation_category_counts(results: list[EvaluationResult]) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    for result in results:
        category = _evaluation_category_value(result)
        if category is not None:
            counts[category] += 1
    return counts.most_common()


def _issue_counts_by_metadata(
    results: list[EvaluationResult],
    field: str,
) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    for result in results:
        value = _metadata_group_value(result, field)
        if value is None:
            continue
        for category in result.error_categories:
            if category == "no_error":
                continue
            counts[f"{value} / {category}"] += 1
    return counts.most_common(8)


def _score_summaries_by_metadata(
    results: list[EvaluationResult],
    field: str,
) -> list[tuple[str, int, float, int, int]]:
    groups: dict[str, list[int]] = {}
    for result in results:
        if result.status != "ok":
            continue
        value = _metadata_group_value(result, field)
        if value is None:
            continue
        groups.setdefault(value, []).append(result.overall_score)
    summaries = [
        (
            name,
            len(scores),
            statistics.mean(scores),
            min(scores),
            max(scores),
        )
        for name, scores in groups.items()
    ]
    return sorted(summaries, key=lambda item: (item[2], item[0]))[:8]


def _status_counts_by_metadata(
    results: list[EvaluationResult],
    field: str,
) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    for result in results:
        if result.status == "ok":
            continue
        value = _metadata_group_value(result, field)
        if value is None:
            continue
        counts[f"{value} / {result.status}"] += 1
    return counts.most_common(8)


def _weakest_segments(
    results: list[EvaluationResult],
    *,
    per_field_limit: int = 3,
    representative_limit: int = 3,
) -> list[SegmentSummary]:
    summaries: list[SegmentSummary] = []
    for field_label, field in SEGMENT_FIELDS:
        groups: dict[str, list[EvaluationResult]] = {}
        for result in results:
            value = _metadata_group_value(result, field)
            if value is None:
                continue
            groups.setdefault(value, []).append(result)

        field_summaries: list[SegmentSummary] = []
        for name, group_results in groups.items():
            scores = [result.overall_score for result in group_results]
            issue_counts = _segment_issue_counts(group_results)
            status_counts = _segment_status_counts(group_results)
            representative_cases = sorted(
                group_results,
                key=lambda result: (
                    result.status == "ok",
                    result.overall_score,
                    result.case_id,
                ),
            )[:representative_limit]
            field_summaries.append(
                SegmentSummary(
                    field_label=field_label,
                    name=name,
                    count=len(group_results),
                    average=statistics.mean(scores),
                    low=min(scores),
                    high=max(scores),
                    issue_counts=issue_counts,
                    status_counts=status_counts,
                    fix_areas=_fix_areas_for_segment(issue_counts, status_counts),
                    representative_cases=representative_cases,
                )
            )
        summaries.extend(
            sorted(field_summaries, key=lambda item: (item.average, item.name))[:per_field_limit]
        )
    return summaries


def _segment_issue_counts(results: list[EvaluationResult]) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    for result in results:
        counts.update(category for category in result.error_categories if category != "no_error")
    return counts.most_common(4)


def _segment_status_counts(results: list[EvaluationResult]) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter(result.status for result in results if result.status != "ok")
    return counts.most_common(3)


def _fix_areas_for_segment(
    issue_counts: list[tuple[str, int]],
    status_counts: list[tuple[str, int]],
) -> list[str]:
    areas: list[str] = []
    for issue, _count in [*issue_counts, *status_counts]:
        mapped = ISSUE_FIX_AREAS.get(issue, issue.replace("_", " "))
        if mapped not in areas:
            areas.append(mapped)
    if not areas:
        areas.append("inspect low-score audio and judge rationale")
    return areas[:4]


def _render_weakest_segments(items: list[SegmentSummary]) -> str:
    if not items:
        return '<div class="metric"><strong class="muted">No segment metadata available</strong></div>'

    cards = "\n".join(_render_weakest_segment_card(item) for item in items)
    return f'<section class="summary">{cards}</section>'


def _render_weakest_segment_card(item: SegmentSummary) -> str:
    severity_class = _segment_severity_class(item.average)
    issue_markup = _render_inline_tags(
        [f"{name.replace('_', ' ')} x{count}" for name, count in item.issue_counts],
        empty_label="No judge issue categories",
    )
    status_markup = _render_inline_tags(
        [f"{name.replace('_', ' ')} x{count}" for name, count in item.status_counts],
        empty_label="No failed evaluations",
    )
    fix_markup = _render_inline_tags(item.fix_areas, empty_label="Inspect judge rationale")
    case_markup = "".join(
        "<li>"
        f"<span>{html.escape(result.case_id)}</span> "
        f"<strong>{result.overall_score} / {html.escape(result.label.replace('_', ' '))}</strong>"
        "</li>"
        for result in item.representative_cases
    )
    return f"""<div class="metric {severity_class}">
      <span>{html.escape(item.field_label)}</span>
      <strong>{html.escape(item.name.replace("_", " "))}</strong>
      <div class="muted">avg {item.average:.1f} / n {item.count} / range {item.low}-{item.high}</div>
      <div><span class="muted">Likely fix areas</span><br>{fix_markup}</div>
      <div><span class="muted">Issue categories</span><br>{issue_markup}</div>
      <div><span class="muted">Evaluation failures</span><br>{status_markup}</div>
      <div><span class="muted">Representative cases</span><ul class="counts">{case_markup}</ul></div>
    </div>"""


def _segment_severity_class(average: float) -> str:
    if average < 60:
        return "severity-low"
    if average < 80:
        return "severity-medium"
    return "severity-high"


def _render_inline_tags(items: list[str], empty_label: str) -> str:
    if not items:
        return f'<span class="muted">{html.escape(empty_label)}</span>'
    return "".join(f'<span class="tag">{html.escape(item)}</span>' for item in items)


def _model_category_actions(
    results: list[EvaluationResult],
    *,
    limit: int = 8,
    representative_limit: int = 2,
) -> list[ModelCategoryAction]:
    groups: dict[tuple[str, str], list[EvaluationResult]] = {}
    for result in results:
        model = _metadata_group_value(result, "synthesis_model")
        category = _metadata_group_value(result, "evaluation_category")
        if model is None or category is None:
            continue
        groups.setdefault((model, category), []).append(result)

    actions: list[ModelCategoryAction] = []
    for (model, category), group_results in groups.items():
        scores = [result.overall_score for result in group_results]
        issue_counts = _segment_issue_counts(group_results)
        status_counts = _segment_status_counts(group_results)
        representative_cases = sorted(
            group_results,
            key=lambda result: (
                result.status == "ok",
                result.overall_score,
                result.case_id,
            ),
        )[:representative_limit]
        actions.append(
            ModelCategoryAction(
                model=model,
                category=category,
                count=len(group_results),
                average=statistics.mean(scores),
                low=min(scores),
                high=max(scores),
                issue_counts=issue_counts,
                status_counts=status_counts,
                fix_areas=_fix_areas_for_segment(issue_counts, status_counts),
                representative_cases=representative_cases,
            )
        )
    return sorted(actions, key=lambda item: (item.average, item.model, item.category))[:limit]


def _render_model_category_actions(items: list[ModelCategoryAction]) -> str:
    if not items:
        return '<div class="metric"><strong class="muted">No model/category pairs available</strong></div>'

    rows = "\n".join(_render_model_category_action_row(item) for item in items)
    return f"""<table>
      <thead>
        <tr>
          <th>Model</th>
          <th>Category</th>
          <th>Score</th>
          <th>Likely Fix Areas</th>
          <th>Evidence</th>
          <th>Representative Cases</th>
        </tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>"""


def _render_model_category_action_row(item: ModelCategoryAction) -> str:
    issue_markup = _render_inline_tags(
        [f"{name.replace('_', ' ')} x{count}" for name, count in item.issue_counts],
        empty_label="No judge issue categories",
    )
    status_markup = _render_inline_tags(
        [f"{name.replace('_', ' ')} x{count}" for name, count in item.status_counts],
        empty_label="No failed evaluations",
    )
    fix_markup = _render_inline_tags(item.fix_areas, empty_label="Inspect judge rationale")
    case_markup = "".join(
        "<li>"
        f"<span>{html.escape(result.case_id)}</span> "
        f"<strong>{result.overall_score} / {html.escape(result.label.replace('_', ' '))}</strong>"
        "</li>"
        for result in item.representative_cases
    )
    return f"""<tr class="{_segment_severity_class(item.average)}">
  <td data-label="Model">{html.escape(item.model)}</td>
  <td data-label="Category">{html.escape(item.category.replace("_", " "))}</td>
  <td data-label="Score"><strong>avg {item.average:.1f}</strong><br><span class="muted">n {item.count} / range {item.low}-{item.high}</span></td>
  <td data-label="Likely Fix Areas">{fix_markup}</td>
  <td data-label="Evidence"><span class="muted">Issues</span><br>{issue_markup}<br><span class="muted">Failures</span><br>{status_markup}</td>
  <td data-label="Representative Cases"><ul class="counts">{case_markup}</ul></td>
</tr>"""


def _baseline_deltas(
    results: list[EvaluationResult],
    *,
    baseline_model: str = BASELINE_SYNTHESIS_MODEL,
    regression_limit: int = 3,
) -> list[BaselineDeltaSummary]:
    baseline_by_case: dict[str, EvaluationResult] = {}
    by_model_and_case: dict[tuple[str, str], EvaluationResult] = {}
    for result in results:
        model = _metadata_group_value(result, "synthesis_model")
        source_case_id = _source_case_key(result)
        if model is None or source_case_id is None:
            continue
        if model == baseline_model:
            existing = baseline_by_case.get(source_case_id)
            if existing is None or result.status == "ok":
                baseline_by_case[source_case_id] = result
        else:
            by_model_and_case[(model, source_case_id)] = result

    grouped_pairs: dict[str, list[tuple[EvaluationResult, EvaluationResult, int]]] = {}
    for (model, source_case_id), result in by_model_and_case.items():
        baseline = baseline_by_case.get(source_case_id)
        if baseline is None:
            continue
        delta = result.overall_score - baseline.overall_score
        grouped_pairs.setdefault(model, []).append((result, baseline, delta))

    summaries: list[BaselineDeltaSummary] = []
    for model, pairs in grouped_pairs.items():
        deltas = [delta for _result, _baseline, delta in pairs]
        regressions = [pair for pair in pairs if pair[2] < 0]
        summaries.append(
            BaselineDeltaSummary(
                baseline_model=baseline_model,
                model=model,
                count=len(pairs),
                average_delta=statistics.mean(deltas),
                wins=sum(1 for delta in deltas if delta > 0),
                ties=sum(1 for delta in deltas if delta == 0),
                losses=sum(1 for delta in deltas if delta < 0),
                largest_regressions=sorted(
                    regressions,
                    key=lambda item: (item[2], item[0].case_id),
                )[:regression_limit],
            )
        )
    return sorted(summaries, key=lambda item: (item.average_delta, item.model))


def _render_baseline_deltas(items: list[BaselineDeltaSummary]) -> str:
    if not items:
        return (
            '<div class="metric"><strong class="muted">'
            "No matched baseline model comparisons available"
            "</strong></div>"
        )

    rows = "\n".join(_render_baseline_delta_row(item) for item in items)
    return f"""<table>
      <thead>
        <tr>
          <th>Compared Model</th>
          <th>Baseline</th>
          <th>Matched Cases</th>
          <th>Avg Delta</th>
          <th>Wins / Ties / Losses</th>
          <th>Largest Regressions</th>
        </tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>"""


def _render_baseline_delta_row(item: BaselineDeltaSummary) -> str:
    regression_markup = "".join(
        "<li>"
        f"<span>{html.escape(_source_case_key(result) or result.case_id)}</span> "
        f"<strong>{delta:+d}</strong>"
        f"<br><span class=\"muted\">{html.escape(result.case_id)}: "
        f"{result.overall_score} vs baseline {baseline.overall_score}</span>"
        "</li>"
        for result, baseline, delta in item.largest_regressions
    )
    if not regression_markup:
        regression_markup = '<li><span class="muted">No regressions</span></li>'
    return f"""<tr class="{_delta_severity_class(item.average_delta)}">
  <td data-label="Compared Model">{html.escape(item.model)}</td>
  <td data-label="Baseline">{html.escape(item.baseline_model)}</td>
  <td data-label="Matched Cases">{item.count}</td>
  <td data-label="Avg Delta"><strong>{item.average_delta:+.1f}</strong></td>
  <td data-label="Wins / Ties / Losses">{item.wins} / {item.ties} / {item.losses}</td>
  <td data-label="Largest Regressions"><ul class="counts">{regression_markup}</ul></td>
</tr>"""


def _delta_severity_class(delta: float) -> str:
    if delta <= -10:
        return "severity-low"
    if delta < 0:
        return "severity-medium"
    return "severity-high"


def _source_case_key(result: EvaluationResult) -> str | None:
    source_case_id = result.metadata.get("source_case_id")
    if isinstance(source_case_id, str) and source_case_id.strip():
        return source_case_id.strip()
    case_id = result.case_id
    suffix = "-local-tts"
    if case_id.endswith(suffix):
        return case_id[: -len(suffix)]
    return None


def _metadata_group_value(result: EvaluationResult, field: str) -> str | None:
    if field == "language":
        return _language_value(result)
    if field == "evaluation_category":
        return _evaluation_category_value(result)
    value = result.metadata.get(field)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _language_value(result: EvaluationResult) -> str | None:
    for field in ("language", "synthesis_lang_code"):
        value = result.metadata.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _evaluation_category_value(result: EvaluationResult) -> str | None:
    for field in ("eval_category", "evaluation_category", "source_category"):
        value = result.metadata.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _render_count_list(items: list[tuple[str, int]], empty_label: str) -> str:
    if not items:
        return f'<strong class="muted">{html.escape(empty_label)}</strong>'
    rendered_items = "".join(
        f"<li><span>{html.escape(name.replace('_', ' '))}</span> <strong>{count}</strong></li>"
        for name, count in items
    )
    return f'<ul class="counts">{rendered_items}</ul>'


def _render_score_summary_list(
    items: list[tuple[str, int, float, int, int]],
    empty_label: str,
) -> str:
    if not items:
        return f'<strong class="muted">{html.escape(empty_label)}</strong>'
    rendered_items = "".join(
        "<li>"
        f"<span>{html.escape(name.replace('_', ' '))}</span> "
        f"<strong>avg {average:.1f} / n {count} / {low}-{high}</strong>"
        "</li>"
        for name, count, average, low, high in items
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


def _render_provenance(result: EvaluationResult) -> str:
    fields = [
        ("Category", "eval_category"),
        ("Slice", "tts_slice"),
        ("Model", "synthesis_model"),
        ("Voice", "synthesis_voice"),
        ("Language", "language"),
        ("Lang code", "synthesis_lang_code"),
        ("Sample", "sample_kind"),
        ("Source case", "source_case_id"),
        ("Text SHA-256", "reference_text_sha256"),
        ("Duration", "audio_duration_seconds"),
        ("Bytes", "audio_bytes"),
    ]
    items: list[str] = []
    for label, field in fields:
        value = result.metadata.get(field)
        if not _has_display_value(value):
            continue
        items.append(
            f"<li><span>{html.escape(label)}</span> "
            f"<strong>{html.escape(_format_provenance_value(value, field))}</strong></li>"
        )
    if not items:
        return '<span class="muted">None</span>'
    return f'<ul class="counts provenance">{"".join(items)}</ul>'


def _render_judge_sample_scores(result: EvaluationResult) -> str:
    scores = result.metadata.get("judge_sample_scores")
    average = result.metadata.get("judge_sample_average")
    if not isinstance(scores, list) or len(scores) <= 1:
        return ""
    score_text = ", ".join(str(score) for score in scores)
    average_text = f"{average:.2f}" if isinstance(average, (int, float)) else str(result.overall_score)
    return (
        '<div class="muted" style="margin-top:6px">'
        f"judge samples: {html.escape(score_text)}; avg {html.escape(average_text)}"
        "</div>"
    )


def _has_display_value(value: object) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    return isinstance(value, (int, float))


def _format_provenance_value(value: object, field: str) -> str:
    if field == "audio_duration_seconds" and isinstance(value, (int, float)):
        return f"{value:.2f}s"
    if field == "audio_bytes" and isinstance(value, (int, float)):
        return str(int(value))
    return str(value).replace("_", " ")


def _render_list(title: str, items: list[str]) -> str:
    rendered_items = "".join(f"<li>{html.escape(item)}</li>" for item in items)
    return f'<div><span class="muted">{html.escape(title)}</span><ul>{rendered_items}</ul></div>'


def _pct(count: int, total: int) -> int:
    if total == 0:
        return 0
    return round((count / total) * 100)
