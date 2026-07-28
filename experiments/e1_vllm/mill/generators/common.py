"""Shared helpers for train-class generators."""

from __future__ import annotations

import re

_EXPLAIN = re.compile(
    r"\b(set|sets|control|controls|enable|enables|specify|specifies|"
    r"limit|limits|number of|fraction|parallel|memory|size|rank|degree)\b",
    re.I,
)
_CODEY = re.compile(
    r"(^\s*vllm\s)|(^\s*CUDA_)|(^\s*\$\w+)|(^\s*apiVersion:)|(^\s*cat\s)|"
    r"(^\s*kubectl\s)|(\\\s*$)",
    re.M,
)


def unique_flag_names(symbols: list[dict]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for sym in symbols:
        if sym.get("kind") != "flag":
            continue
        name = sym["name"]
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def unique_flags(symbols: list[dict]) -> list[dict]:
    """First symbol row per flag name (legacy helper for T5)."""
    seen: set[str] = set()
    out: list[dict] = []
    for sym in symbols:
        if sym.get("kind") != "flag":
            continue
        name = sym["name"]
        if name in seen:
            continue
        seen.add(name)
        out.append(sym)
    return out


def _strip_fences(text: str) -> str:
    return re.sub(r"```[\s\S]*?```", " ", text)


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def _inline_comment(line: str, flag: str) -> str | None:
    """`--flag 1  # explains this` → explanatory clause."""
    if flag not in line or "#" not in line:
        return None
    after = line.split("#", 1)[1].strip()
    if len(after) < 8:
        return None
    return after


def _score_answer(answer: str, flag: str) -> float:
    if flag not in answer:
        return -1e9
    score = 0.0
    low = answer.lower()
    if _EXPLAIN.search(answer):
        score += 4.0
    if answer.startswith(f"`{flag}`:"):
        score += 5.0  # comment rewrite
    if _CODEY.search(answer):
        score -= 5.0
    if "```" in answer or "\\," in answer or answer.count("\\") >= 1:
        score -= 6.0
    if any(tok in low for tok in ("bash", "python", "samplingparams", "kubectl", "apiVersion")):
        score -= 6.0
    if answer.count("--") > 2:
        score -= 4.0
    n = len(answer)
    if 40 <= n <= 320:
        score += 2.0
    elif n < 25:
        score -= 2.0
    elif n > 400:
        score -= 4.0
    return score


def _candidates_from_chunk(text: str, flag: str) -> list[str]:
    out: list[str] = []
    plain = _strip_fences(text)
    for sent in _sentences(plain):
        if flag in sent:
            out.append(re.sub(r"\s+", " ", sent).strip())
    for line in text.splitlines():
        comment = _inline_comment(line, flag)
        if comment:
            out.append(f"`{flag}`: {comment}")
    # paragraph containing flag
    for para in re.split(r"\n\s*\n", plain):
        if flag in para:
            cleaned = re.sub(r"\s+", " ", para).strip()
            if 20 < len(cleaned) < 500:
                out.append(cleaned)
    return out


def answers_for_flags(
    flags: list[str], chunks: dict[str, str]
) -> dict[str, tuple[str, str]]:
    """One pass over chunks → best (answer, chunk_id) per flag."""
    best: dict[str, tuple[float, str, str]] = {}
    flag_set = set(flags)
    for chunk_id, text in chunks.items():
        present = [f for f in flag_set if f in text]
        if not present:
            continue
        for flag in present:
            for cand in _candidates_from_chunk(text, flag):
                sc = _score_answer(cand, flag)
                prev = best.get(flag)
                if prev is None or sc > prev[0]:
                    best[flag] = (sc, cand, chunk_id)

    out: dict[str, tuple[str, str]] = {}
    for flag, (sc, answer, chunk_id) in best.items():
        if sc < 2.0:
            continue
        if not answer.startswith(f"`{flag}`:") and not answer.endswith((".", "!", "?")):
            answer = answer.rstrip(" ,;") + "."
        out[flag] = (answer, chunk_id)
    return out


def answer_for_flag(flag: str, chunks: dict[str, str]) -> tuple[str, str] | None:
    return answers_for_flags([flag], chunks).get(flag)
