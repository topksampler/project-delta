"""Build a 50-item human audit pack for dense profile grader trust."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path


FORMS = (
    "free_recall",
    "meaning_recognition",
    "meaning_true",
    "meaning_false",
)


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_audit(
    *,
    claims_path: Path,
    probes_path: Path,
    samples_path: Path,
    out_jsonl: Path,
    out_md: Path,
    n: int = 50,
    seed: int = 20260718,
) -> dict:
    claims = {c["claim_id"]: c for c in load_jsonl(claims_path)}
    probes = {p["id"]: p for p in load_jsonl(probes_path)}
    samples = [
        s
        for s in load_jsonl(samples_path)
        if s.get("claim_id") in claims and s.get("probe_form") in FORMS
    ]

    by_bucket: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for sample in samples:
        key = (
            sample["probe_form"],
            "correct" if sample.get("failure_mode") == "correct" else "incorrect",
        )
        by_bucket[key].append(sample)

    rng = random.Random(seed)
    # Aim for ~equal form × outcome; top up from remaining.
    per = max(1, n // (len(FORMS) * 2))
    chosen: list[dict] = []
    used: set[str] = set()
    for form in FORMS:
        for outcome in ("correct", "incorrect"):
            pool = list(by_bucket.get((form, outcome), []))
            rng.shuffle(pool)
            for sample in pool:
                if sample["id"] in used:
                    continue
                chosen.append(sample)
                used.add(sample["id"])
                if sum(1 for c in chosen if c["probe_form"] == form and (
                    (c.get("failure_mode") == "correct") == (outcome == "correct")
                )) >= per:
                    break

    if len(chosen) < n:
        rest = [s for s in samples if s["id"] not in used]
        rng.shuffle(rest)
        for sample in rest:
            chosen.append(sample)
            used.add(sample["id"])
            if len(chosen) >= n:
                break
    chosen = chosen[:n]
    rng.shuffle(chosen)

    rows: list[dict] = []
    for idx, sample in enumerate(chosen, start=1):
        claim = claims[sample["claim_id"]]
        probe = probes[sample["id"]]
        gold = probe.get("gold") or {}
        rows.append(
            {
                "audit_id": f"A{idx:02d}",
                "probe_id": sample["id"],
                "claim_id": sample["claim_id"],
                "entity": claim["entity"],
                "probe_form": sample["probe_form"],
                "question": sample["question"],
                "model_output": sample["output"],
                "auto_score": sample.get("content_score"),
                "auto_failure_mode": sample.get("failure_mode"),
                "gold": gold,
                "reference_answer": claim["truth"]["reference_answer"],
                "evidence_span": claim["evidence"].get("span"),
                "evidence_paths": claim["evidence"].get("source_paths"),
                "human_agree_with_auto": None,
                "human_correct_label": None,
                "human_notes": "",
            }
        )

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    lines = [
        "# Dense profile grader audit (50 probes)",
        "",
        "Serves: trust the automatic scorer before the data wheel uses it.",
        "",
        "For each item, fill in `audit.jsonl`:",
        "- `human_agree_with_auto`: true/false",
        "- `human_correct_label`: correct | wrong | abstain_ok | unclear",
        "- `human_notes`: optional",
        "",
        f"Source run: `{Path(samples_path).parent.name}`",
        f"Claims: `{claims_path}`",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['audit_id']} — `{row['entity']}` · {row['probe_form']}",
                "",
                f"Auto: **{row['auto_failure_mode']}** (score={row['auto_score']})",
                "",
                "Question:",
                "```",
                row["question"][:1200],
                "```",
                "",
                "Model output:",
                "```",
                row["model_output"][:800],
                "```",
                "",
                f"Gold: `{json.dumps(row['gold'], ensure_ascii=False)}`",
                "",
                "Docs evidence:",
                "```",
                str(row["evidence_span"])[:500],
                "```",
                "",
                f"Path: {row['evidence_paths']}",
                "",
                "Your call: agree with auto? ____  label: ____",
                "",
            ]
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[f"{row['probe_form']}:{row['auto_failure_mode']}"] += 1
    return {
        "n": len(rows),
        "out_jsonl": str(out_jsonl),
        "out_md": str(out_md),
        "by_form_mode": dict(sorted(counts.items())),
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    summary = build_audit(
        claims_path=root
        / "data/experiments/e1_vllm/profile/dense_v1/claims_verified.jsonl",
        probes_path=root
        / "data/experiments/e1_vllm/profile/dense_v1/probes_verified.jsonl",
        samples_path=root
        / "runs/e1-vllm-profile-dense-v1-qwen35-08b-modal/samples.jsonl",
        out_jsonl=root
        / "artifacts/reports/dense_v1_grader_audit_50.jsonl",
        out_md=root / "artifacts/reports/dense_v1_grader_audit_50.md",
    )
    print(json.dumps(summary, indent=2))
