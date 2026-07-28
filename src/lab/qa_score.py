"""Substring Q&A scoring and failure-mode labels (no torch)."""

from __future__ import annotations

import re


CHOICE_RE = re.compile(
    r"(?:answer\s*(?:is|:)?\s*|option\s+)([A-D])\b",
    re.IGNORECASE,
)
LEADING_CHOICE_RE = re.compile(r"^\s*([A-D])(?:\s*[.):\-]|$)", re.IGNORECASE)


def score_choice(output: str, expected: str) -> tuple[float, list[str], list[str]]:
    """Score an explicit A–D answer without substring collisions."""
    text = output.strip()
    match = CHOICE_RE.search(text)
    if match:
        selected = match.group(1).upper()
    else:
        lead = LEADING_CHOICE_RE.match(text)
        if lead:
            selected = lead.group(1).upper()
        else:
            compact = text.rstrip(".):").upper()
            selected = compact if compact in {"A", "B", "C", "D"} else None
    expected = expected.upper()
    if selected == expected:
        return 1.0, [expected], []
    return 0.0, ([selected] if selected else []), [expected]


def score_boolean(output: str, expected: str) -> tuple[float, list[str], list[str]]:
    """Score the first explicit yes/no/unknown judgment."""
    match = re.search(
        r"\b(yes|no|unknown|unsure|not sure|do not know|don't know)\b",
        output,
        re.IGNORECASE,
    )
    selected = match.group(1).lower() if match else None
    if selected in {"unsure", "not sure", "do not know", "don't know"}:
        selected = "unknown"
    allowed = {"no", "unknown"} if expected == "no_or_unknown" else {expected}
    if selected in allowed:
        return 1.0, [selected], []
    return 0.0, ([selected] if selected else []), sorted(allowed)


def score_must_contain(output: str, needles: list[str]) -> tuple[float, list[str], list[str]]:
    out = output.lower()
    hits = [n for n in needles if n.lower() in out]
    misses = [n for n in needles if n.lower() not in out]
    score = len(hits) / max(1, len(needles))
    return score, hits, misses


def score_must_contain_any(
    output: str, groups: list[list[str]]
) -> tuple[float, list[str], list[str]]:
    """Each group is an OR; overall score is fraction of groups satisfied."""
    if not groups:
        return 1.0, [], []
    out = output.lower()
    hits: list[str] = []
    misses: list[str] = []
    ok = 0
    for group in groups:
        matched = next((n for n in group if n.lower() in out), None)
        if matched is not None:
            ok += 1
            hits.append(matched)
        else:
            misses.extend(group)
    return ok / len(groups), hits, misses


def score_gold(output: str, gold: dict) -> tuple[float, list[str], list[str]]:
    """Score against must_contain and/or must_contain_any.

    When both are present, both must pass (averaged).
    must_not_contain hits force score to 0.
    """
    gold = gold or {}
    if gold.get("choice"):
        return score_choice(output, str(gold["choice"]))
    if gold.get("boolean"):
        return score_boolean(output, str(gold["boolean"]))

    forbid = gold.get("must_not_contain") or []
    out = output.lower()
    for bad in forbid:
        if bad.lower() in out:
            return 0.0, [], [f"must_not_contain:{bad}"]

    parts: list[tuple[float, list[str], list[str]]] = []
    must = gold.get("must_contain") or []
    if must:
        parts.append(score_must_contain(output, must))
    any_groups = gold.get("must_contain_any") or []
    if any_groups:
        parts.append(score_must_contain_any(output, any_groups))
    if not parts:
        return 0.0, [], ["empty_gold"]
    score = sum(p[0] for p in parts) / len(parts)
    hits = [h for p in parts for h in p[1]]
    misses = [m for p in parts for m in p[2]]
    return score, hits, misses


def is_abstain(output: str) -> bool:
    lower = output.lower()
    return any(
        p in lower
        for p in (
            "unknown",
            "not sure",
            "don't know",
            "do not know",
            "not documented",
            "does not exist",
            "doesn't exist",
            "no such",
            "not a documented",
            "not present",
        )
    )


def is_negative_answer(output: str) -> bool:
    lower = output.lower().strip()
    if lower.startswith("no") or lower.startswith("no,"):
        return True
    return any(
        p in lower
        for p in ("does not", "doesn't", "do not", "not documented", "not present", "no such")
    )


def gold_wants_unknown(needles: list[str]) -> bool:
    return any(n.lower() == "unknown" for n in needles)


def gold_wants_abstain_or_no(gold: dict) -> bool:
    if gold.get("abstain_if_unknown"):
        return True
    needles = list(gold.get("must_contain") or [])
    for group in gold.get("must_contain_any") or []:
        needles.extend(group)
    return gold_wants_unknown(needles) or any(n.lower() in {"no", "not"} for n in needles)


def classify_failure_mode(
    *,
    score: float,
    output: str,
    requires_doc: str,
    misses: list[str],
    needles: list[str] | None = None,
    gold: dict | None = None,
) -> str:
    needles = needles or []
    gold = gold or {}
    wants_neg = gold_wants_abstain_or_no(gold) if gold else gold_wants_unknown(needles)
    abstained = is_abstain(output)
    said_no = is_negative_answer(output)

    if score >= 1.0:
        return "correct"
    if wants_neg:
        if abstained or said_no:
            return "correct"
        return "confident_hallucination"
    if abstained:
        return "abstain_wrong"
    if requires_doc == "0.23.0" and any(
        x in output.lower() for x in ("llm_compressor.md", "0.22", "v0.22")
    ):
        return "stale_version"
    return "wrong"
