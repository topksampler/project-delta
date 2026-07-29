# Eval factory v1 — 4B collapse audit (vs 0.8B)

Paired closed-book runs on frozen `e1_eval_factory_v1` (623 probes).

## invariant

```text
A size comparison is invalid until format-scorer artifacts are separated
from knowledge / calibration failures.
Boolean gold requires an explicit yes/no token (`score_boolean`).
```

## what happened

| | 0.8B | 4B |
|--|-----:|---:|
| exact accuracy | 0.512 | 0.148 |
| 08✓ → 4B✗ | — | **265** |
| both ✓ | — | 54 |
| 4B-only ✓ | — | 38 |

## taxonomy of the 265 collapses

| bucket | n | share | reading |
|--------|--:|------:|---------|
| invents change on stable delta | 106 | 40% | knowledge/hallucinated churn |
| boolean answer lacks yes/no token | 80 | 30% | mixed; only ~18 (~7% of all collapses) are polarity-correct without the token |
| wrong yes/no polarity | 31 | 12% | knowledge |
| abstain / “docs don’t say” theater | 35 | 13% | calibration (closed-book) |
| other | 13 | 5% | mostly stable-claim needle miss |

**Verdict:** primary failure is **not** “4B can’t say yes.” Dominant modes are inventing churn on unchanged entities and wrong/abstaining judgments. Format is a real minority confounder (~7% cleanly recoverable), not the story.

Evidence that format alone can’t rescue the headline: soft-rescoring boolean affirmations without `yes`/`no` lifts 4B exact only **0.148 → 0.178**.

## example (format confounder — minority)

Q: does `CompilationConfig.cudagraph_capture_sizes` exist as typed field? gold=`yes`
0.8B starts with `Yes, …` → score 1
4B: “is defined as a `List[int]`, which is a typed configuration parameter” → score 0 (`misses: ['yes']`)

## example (knowledge — majority pattern)

Stable delta probes: 0.8B says unchanged; 4B invents rename/deprecate/remove → partial credit on entity name, miss on `unchanged|stable|both`.

## paths

- audit rows: `artifacts/reports/eval_factory_v1_4b_collapse_audit.jsonl`
- summary: `artifacts/reports/eval_factory_v1_4b_collapse_summary.json`
- runs: `e1-vllm-eval-factory-v1-base-qwen35-08b-modal`, `…-4b-modal`

## what can die

- soft-rescoring heuristics used only for this audit

## what must survive

- paired run_ids + this taxonomy
- scorer contract: boolean = explicit token (document it; don’t silently loosen mid-study)

## command

```bash
# regenerate taxonomy from local samples
python3 - <<'PY'
# see artifacts/reports/eval_factory_v1_4b_collapse_summary.json
print(open('artifacts/reports/eval_factory_v1_4b_collapse_summary.json').read())
PY
```
