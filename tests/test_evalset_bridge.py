import json
from pathlib import Path

from open_audio_judge.evalset_bridge import (
    build_tts_cases,
    classify_tts_slice,
    load_evalset_records,
    summarize_tts_cases,
    write_cases_jsonl,
    write_tts_summary_json,
)


def test_build_tts_cases_preserves_multiturn_context_and_metadata() -> None:
    records = [
        {
            "id": "ome_0001",
            "version": "0.1.0",
            "category": "instruction_constraints",
            "task": "format_keywords",
            "turns": [{"role": "user", "content": "Reply with exactly three bullet lines."}],
            "ideal_answer": "- cobalt\n- ledger\n- orbit",
            "metadata": {"tags": ["format", "constraints"]},
        },
        {
            "id": "ome_ignored",
            "category": "short_factuality",
            "task": "capital_city",
            "turns": [{"role": "user", "content": "What is the capital of France?"}],
            "ideal_answer": "Paris",
            "metadata": {"tags": ["factuality"]},
        },
    ]

    cases = build_tts_cases(records, source_name="ome", limit=None)

    assert len(cases) == 1
    case = cases[0]
    assert case.id == "tts-ome-ome-0001"
    assert case.task == "tts_naturalness"
    assert case.reference_text == "- cobalt\n- ledger\n- orbit"
    assert case.turns[0].role == "user"
    assert case.metadata["source"] == "ome"
    assert case.metadata["source_id"] == "ome_0001"
    assert case.metadata["source_version"] == "0.1.0"
    assert case.metadata["source_category"] == "instruction_constraints"
    assert case.metadata["source_tags"] == ["format", "constraints"]
    assert case.metadata["tts_slice"] == "punctuation_format"
    assert case.metadata["requires_synthesis"] is True


def test_build_tts_cases_supports_category_filter_and_limit() -> None:
    records = [
        {
            "id": "one",
            "category": "structured_output",
            "task": "json_decision",
            "ideal_answer": '{"decision":"approve"}',
        },
        {
            "id": "two",
            "category": "multi_turn_state",
            "task": "remember_code_phrase",
            "ideal_answer": "blue lantern",
        },
    ]

    cases = build_tts_cases(
        records,
        source_name="fixture",
        limit=1,
        category_filter={"structured_output", "multi_turn_state"},
    )

    assert [case.id for case in cases] == ["tts-fixture-one"]


def test_build_tts_cases_supports_slice_filter() -> None:
    records = [
        {
            "id": "json",
            "category": "structured_output",
            "task": "json_decision",
            "ideal_answer": '{"decision":"approve"}',
        },
        {
            "id": "time",
            "category": "instruction_constraints",
            "task": "read_time",
            "ideal_answer": "Meet at 09:45.",
        },
    ]

    cases = build_tts_cases(records, source_name="fixture", slice_filter={"dates_times"})

    assert [case.id for case in cases] == ["tts-fixture-time"]
    assert cases[0].metadata["tts_slice"] == "dates_times"


def test_build_tts_cases_supports_per_slice_limit() -> None:
    records = [
        {
            "id": "json-one",
            "category": "structured_output",
            "task": "json_decision",
            "ideal_answer": '{"decision":"approve"}',
        },
        {
            "id": "json-two",
            "category": "structured_output",
            "task": "json_decision",
            "ideal_answer": '{"decision":"deny"}',
        },
        {
            "id": "time-one",
            "category": "instruction_constraints",
            "task": "read_time",
            "ideal_answer": "Meet at 09:45.",
        },
        {
            "id": "time-two",
            "category": "instruction_constraints",
            "task": "read_time",
            "ideal_answer": "Meet at 10:15.",
        },
    ]

    cases = build_tts_cases(records, source_name="fixture", per_slice_limit=1)

    assert [case.id for case in cases] == ["tts-fixture-json-one", "tts-fixture-time-one"]
    assert [case.metadata["tts_slice"] for case in cases] == ["code_like", "dates_times"]


def test_classify_tts_slice() -> None:
    assert classify_tts_slice({"category": "structured_output", "task": "json_decision"}, "{}") == "code_like"
    assert classify_tts_slice({"category": "calendar", "task": "time"}, "Meet at 09:45") == "dates_times"
    assert classify_tts_slice({"metadata": {"tags": ["privacy"]}}, "Please redact the email.") == "safety_privacy"
    assert classify_tts_slice({"category": "cross_lingual_transfer"}, "tomorrow at 3 PM") == "multilingual"


def test_build_tts_cases_includes_cross_lingual_and_numeric_evalset_rows() -> None:
    records = [
        {
            "id": "spanish-date",
            "category": "multilingual_understanding",
            "task": "spanish_date",
            "turns": [{"role": "user", "content": "Translate the weekday from Spanish."}],
            "ideal_answer": "Thursday",
            "metadata": {"tags": ["spanish", "translation"]},
        },
        {
            "id": "linear-equation",
            "category": "quantitative_math",
            "task": "linear_equation",
            "turns": [{"role": "user", "content": "Solve for x."}],
            "ideal_answer": "6",
            "metadata": {"tags": ["math", "algebra"]},
        },
        {
            "id": "factual",
            "category": "short_factuality",
            "task": "capital_city",
            "turns": [{"role": "user", "content": "What is the capital of France?"}],
            "ideal_answer": "Paris",
            "metadata": {"tags": ["factuality"]},
        },
    ]

    cases = build_tts_cases(records, source_name="fixture")

    assert [case.id for case in cases] == [
        "tts-fixture-spanish-date",
        "tts-fixture-linear-equation",
    ]
    assert [case.metadata["tts_slice"] for case in cases] == ["multilingual", "numbers"]


def test_load_and_write_cases_jsonl(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "ome_0002",
                "category": "structured_output",
                "task": "json_decision",
                "turns": [{"role": "user", "content": "Return JSON."}],
                "ideal_answer": '{"decision":"approve"}',
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_evalset_records(source)
    cases = build_tts_cases(records, source_name="ome")
    out = write_cases_jsonl(cases, tmp_path / "cases.jsonl")

    written = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert written[0]["id"] == "tts-ome-ome-0002"
    assert written[0]["metadata"]["tts_slice"] == "code_like"


def test_summarize_tts_cases_is_metadata_only(tmp_path: Path) -> None:
    records = [
        {
            "id": "json-one",
            "category": "structured_output",
            "task": "json_decision",
            "ideal_answer": '{"decision":"approve"}',
        },
        {
            "id": "time-one",
            "category": "instruction_constraints",
            "task": "read_time",
            "ideal_answer": "Meet at 09:45.",
        },
    ]

    cases = build_tts_cases(records, source_name="fixture")
    summary = summarize_tts_cases(cases)
    out = write_tts_summary_json(cases, tmp_path / "summary.json")
    written = json.loads(out.read_text(encoding="utf-8"))

    assert summary.as_dict() == written
    assert written == {
        "total_cases": 2,
        "by_slice": {"code_like": 1, "dates_times": 1},
        "by_source_category": {
            "instruction_constraints": 1,
            "structured_output": 1,
        },
        "example_source_ids_by_slice": {
            "code_like": ["json-one"],
            "dates_times": ["time-one"],
        },
        "requires_synthesis": 2,
        "text_length": {"average": 18, "max": 22, "min": 14},
    }
    assert "approve" not in out.read_text(encoding="utf-8")
    assert "09:45" not in out.read_text(encoding="utf-8")


def test_summarize_tts_cases_caps_example_source_ids_per_slice() -> None:
    records = [
        {
            "id": f"json-{index}",
            "category": "structured_output",
            "task": "json_decision",
            "ideal_answer": f'{{"decision":"{index}"}}',
        }
        for index in range(5)
    ]

    summary = summarize_tts_cases(build_tts_cases(records, source_name="fixture"))

    assert summary.example_source_ids_by_slice == {
        "code_like": ["json-0", "json-1", "json-2"]
    }
