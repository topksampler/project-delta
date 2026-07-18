# eval schema (v3 target)

v2 uses `type` (topic) + `requires_doc`. v3 adds `eval_class` (A–D) + richer gold for failure-mode scoring.

## invariant

```text
Every row must pass validate-eval against the corpus for requires_doc.
eval_class drives analysis; type is optional topic tag.
```

## row shape

```json
{
  "id": "delta_01",
  "eval_class": "B",
  "type": "quantization",
  "question": "How did LLM Compressor documentation path change between vLLM 0.22.0 and 0.23.0?",
  "gold": {
    "must_contain": ["llm_compressor"],
    "must_contain_any": [["llm_compressor.md"], ["llm_compressor/README"]],
    "must_not_contain": [],
    "expected_version": "0.23.0",
    "abstain_if_unknown": false
  },
  "requires_doc": "0.23.0",
  "source_hint": "docs/features/quantization/"
}
```

### fields

| field | required | purpose |
|-------|----------|---------|
| `id` | yes | unique |
| `eval_class` | yes | `A` stable, `B` delta, `C` procedural, `D` version-binary |
| `question` | yes | shown to model |
| `gold.must_contain` | yes | all required (case-insensitive) |
| `gold.must_contain_any` | no | OR groups for acceptable phrasings |
| `gold.must_not_contain` | no | stale-version trap (e.g. old path on delta Q) |
| `gold.expected_version` | no | for attribution scoring |
| `gold.abstain_if_unknown` | no | correct answer is "unknown" / abstain |
| `requires_doc` | yes | `0.22.0` or `0.23.0` — truth snapshot |
| `source_hint` | no | human audit trail |
| `type` | no | topic bucket |

## eval task

**Short free-form answer** (1–3 sentences). Scorer applies rubric + assigns `failure_mode`.

Not multiple choice. Code generation only for explicit `eval_class: C` + `"output": "code"` extension later.

## mapping v2 → classes (approximate)

| eval_class | v2 count | notes |
|------------|----------|-------|
| A stable | ~31 | `requires_doc 0.22.0`, gold also in doc_8 |
| B delta | ~5 | drift_* — need “what changed” phrasing |
| C procedural | ~3 | api_05, install_* partial — need “how do I” |
| D version-binary | 0 | **gap** — add “does X exist in v0.22?” |

Type **E** = run c3/c2 on existing rows, not a separate `eval_class`.

## v3 growth targets

| class | target count |
|-------|----------------|
| A | 20 |
| B | 10 |
| C | 8 |
| D | 8 |
| **total** | ~46 |

## failure_mode (scorer output, not input)

```text
correct | wrong | stale_version | retrieval_miss | abstain_correct | abstain_wrong | confident_hallucination
```

See [e1_vllm_thesis.md](../../../docs/experiments/e1_vllm_thesis.md).

## what goes where

- schema contract: this file;
- frozen rows: `fixtures/eval_vN.jsonl`;
- corpus validation logic: `../inspect_data.py`;
- per-run outputs and assigned failure modes: B2 `runs/{run_id}/samples.jsonl`.

## what can die

Rows rejected before an eval version is frozen.

## what must survive

The schema version, frozen rows, corpus provenance, and scorer version used for
every reported result.

## command

```bash
python experiments/e1_vllm/inspect_data.py validate-eval eval_v2.jsonl
```
