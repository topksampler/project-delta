# Code retrieval ladder on factory eval

Bank: sealed `e1_eval_factory_v1` (623) unless noted · model Qwen3.5-0.8B

## invariant

```text
Modality match is necessary but not sufficient.
Oracle evidence spans ≠ BM25 over the same code tree.
Era (fresh vs stale snapshot) only bites when retrieval is on-contract.
```

## A — code-BM25 vs evidence oracle (v0.22→v0.23)

| condition | exact | changed | note |
|-----------|------:|--------:|------|
| closed-book | 0.512 | 0.000 | |
| docs BM25 | 0.080 | 0.000 | wrong modality |
| **code BM25 fresh (v0.23)** | **0.096** | 0.125 | right modality, weak ranking |
| code BM25 stale (v0.22) | 0.103 | 0.000 | ≈ fresh |
| **evidence oracle fresh** | **0.724** | **0.375** | probe-bound spans |

Naive code-BM25 ≈ docs-BM25. The win is **span selection**, not “any code in context.”

## B — evidence RAG on frozen v0.21→v0.22

| | exact | avg | changed | n |
|--|------:|----:|--------:|--:|
| evidence fresh | **0.642** | 0.746 | **0.667** | 674 |

Method transfers. Hit_rate 0.963 (649/674 mapped).

## C — era ladder (evidence + code-BM25)

| | fresh | stale |
|--|------:|------:|
| evidence exact | **0.724** | **0.244** |
| evidence changed | 0.375 | 0.000 |
| code-BM25 exact | 0.096 | 0.103 |

Evidence is era-sensitive (fresh ≫ stale). Code-BM25 is not — both eras are equally weak.

## paths

- JSON: `artifacts/reports/eval_factory_code_rag_ladder.json`
- code corpora: `corpus_code_{0,8}.jsonl` + `indexes/code_{0,8}_bm25.json`
- evidence maps: `factory_v1_evidence_map_{v022_v023,stale}.json`, `factory_t2122_evidence_map.json`

## what must survive

- oracle ≫ code-BM25 ≫ docs-BM25 ordering on this bank
- evidence fresh ≫ stale
- t2122 evidence numbers
