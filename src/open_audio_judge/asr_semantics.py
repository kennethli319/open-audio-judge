from __future__ import annotations

import re
from dataclasses import dataclass, field


NEGATION_TERMS = {
    "no",
    "not",
    "never",
    "none",
    "without",
    "dont",
    "don't",
    "cannot",
    "can't",
    "cant",
    "wont",
    "won't",
    "isnt",
    "isn't",
    "arent",
    "aren't",
}

NUMBER_WORDS = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
    "hundred",
    "thousand",
    "million",
    "billion",
}


@dataclass(frozen=True)
class SemanticAsrDiff:
    score: int
    meaning_preservation: str
    semantic_error_summary: str
    key_differences: list[str] = field(default_factory=list)
    error_categories: list[str] = field(default_factory=list)
    researcher_notes: list[str] = field(default_factory=list)


def analyze_reference_candidate(reference: str, candidate: str) -> SemanticAsrDiff:
    ref_words = normalize_words(reference)
    cand_words = normalize_words(candidate)
    ref_set = set(ref_words)
    cand_set = set(cand_words)

    if not ref_words and not cand_words:
        return SemanticAsrDiff(
            score=100,
            meaning_preservation="preserved",
            semantic_error_summary="Both reference and candidate are empty.",
            error_categories=["no_error"],
            researcher_notes=["No ASR content to diagnose."],
        )
    if not cand_words:
        return SemanticAsrDiff(
            score=5,
            meaning_preservation="not_preserved",
            semantic_error_summary="Candidate transcript is empty while reference contains speech.",
            key_differences=["candidate transcript is empty"],
            error_categories=["deletion", "truncation"],
            researcher_notes=["Investigate endpointing, VAD, and decoder failure on this case."],
        )

    missing = ordered_difference(ref_words, cand_set)
    inserted = ordered_difference(cand_words, ref_set)
    distance = levenshtein(ref_words, cand_words)
    overlap_score = max(1, min(100, round((1.0 - min(distance / max(len(ref_words), 1), 1.0)) * 100)))

    categories: list[str] = []
    differences: list[str] = []
    notes: list[str] = []
    score_caps: list[int] = []

    if missing:
        categories.append("deletion")
        differences.append(f"missing from candidate: {', '.join(missing[:8])}")
    if inserted:
        categories.append("insertion")
        differences.append(f"extra in candidate: {', '.join(inserted[:8])}")
    if distance and missing and inserted:
        categories.append("substitution")

    if token_family_changed(ref_words, cand_words, NEGATION_TERMS):
        categories.append("negation_error")
        differences.append("negation or polarity changed")
        notes.append("Prioritize negation and polarity robustness; these errors often reverse user intent.")
        score_caps.append(40)

    if token_family_changed(ref_words, cand_words, NUMBER_WORDS) or digit_tokens(ref_words) != digit_tokens(cand_words):
        categories.append("number_error")
        differences.append("numeric content changed")
        notes.append("Add targeted evaluation and augmentation for numbers, quantities, dates, and currencies.")
        score_caps.append(55)

    ref_entities = entity_like_tokens(reference)
    cand_entities = entity_like_tokens(candidate)
    if ref_entities != cand_entities:
        categories.append("entity_error")
        missing_entities = [token for token in ref_entities if token not in cand_entities]
        extra_entities = [token for token in cand_entities if token not in ref_entities]
        if missing_entities:
            differences.append(f"missing entity-like terms: {', '.join(missing_entities[:6])}")
        if extra_entities:
            differences.append(f"extra entity-like terms: {', '.join(extra_entities[:6])}")
        notes.append("Inspect named-entity recall and domain vocabulary coverage for this slice.")
        score_caps.append(65)

    if not categories:
        categories = ["no_error"]
        differences = ["no meaningful reference-candidate difference detected by the heuristic"]
        notes = ["No action needed from the heuristic baseline."]

    score = min([overlap_score, *score_caps]) if score_caps else overlap_score
    meaning = meaning_preservation_for_score(score)

    return SemanticAsrDiff(
        score=score,
        meaning_preservation=meaning,
        semantic_error_summary=semantic_summary(meaning, categories),
        key_differences=dedupe(differences),
        error_categories=dedupe(categories),
        researcher_notes=dedupe(notes),
    )


def normalize_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def ordered_difference(words: list[str], other_set: set[str]) -> list[str]:
    return dedupe([word for word in words if word not in other_set])


def token_family_changed(left: list[str], right: list[str], family: set[str]) -> bool:
    return [token for token in left if token in family] != [token for token in right if token in family]


def digit_tokens(words: list[str]) -> list[str]:
    return [word for word in words if any(char.isdigit() for char in word)]


def entity_like_tokens(text: str) -> list[str]:
    tokens = re.findall(r"\b[A-Z][A-Za-z0-9'-]*\b", text)
    return dedupe(tokens[1:] if tokens else [])


def meaning_preservation_for_score(score: int) -> str:
    if score >= 81:
        return "preserved"
    if score >= 61:
        return "minor_loss"
    if score >= 41:
        return "partial_loss"
    if score >= 21:
        return "major_loss"
    return "not_preserved"


def semantic_summary(meaning: str, categories: list[str]) -> str:
    if categories == ["no_error"]:
        return "No semantic ASR error was detected by the heuristic baseline."
    readable = ", ".join(category.replace("_", " ") for category in categories)
    return f"{meaning.replace('_', ' ').title()} caused by {readable}."


def levenshtein(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for i, left_word in enumerate(left, start=1):
        current = [i]
        for j, right_word in enumerate(right, start=1):
            substitution_cost = 0 if left_word == right_word else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped
