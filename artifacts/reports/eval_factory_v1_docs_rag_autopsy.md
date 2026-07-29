# Why docs BM25 hurts factory eval

Compared: c0 no-hint vs c1 fresh docs BM25 (no boolhint), sealed `e1_eval_factory_v1`.

## invariant

```text
Factory gold is executable code contracts (AST).
Docs BM25 indexes documentation markdown, not those contracts.
Retrieving the wrong modality is not an intervention test — it is corpus mismatch.
```

## evidence

| measure | value |
|---------|------:|
| probe evidence paths that are `.py` | **39/39 unique (100%)** |
| overlap of evidence paths with docs corpus paths | **0** |
| retrieved chunks that are `.py` | **0** |
| probes where display_entity appears in retrieved docs text | **69/623 (11%)** |
| stable-delta invent-churn (rename/deprecate/…) c0 → c1 | **36 → 92** |
| c0✓ → c1✗ | **278** |
| exact accuracy c0 / c1 | **0.512 / 0.080** |

## mechanism

Factory probes ask about CLI/config/env/export contracts grounded in
`vllm/**/*.py`. The c1 index is `corpus_doc_8` (`docs/`, `examples/`, README).
BM25 always returns nonempty markdown (hit_rate=1.0) that almost never contains
the entity, so the model gets fluent wrong-era prose and invents churn.

This is why factory RAG ranking inverts eval_v3 (hand QA over doc language).

## paths

- this report; paired runs `…-base-…` / `…-c1-rag-fresh-nohint-…`

## what must survive

- modality mismatch numbers above
- do not conclude “RAG fails on DELTA” from docs-BM25 on a code-grounded bank
