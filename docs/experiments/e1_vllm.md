# e1_vllm

First vertical slice of [Project DELTA](../project-delta.md).

This charter owns the experiment matrix and operational state. The
[thesis](./e1_vllm_thesis.md) owns the research question; the
[data guide](./e1_vllm_data.md) owns corpus and eval facts.

---

## invariant

```text
Corpus / stale index  →  doc_0 (v0.22.0)
Fresh index / eval truth for new facts  →  doc_8 (v0.23.0)
Eval task  →  short free-form Q&A + rubric + failure_mode labels
Primary output  →  failure taxonomy per method, not single accuracy number
```

The authoritative model and source revisions live in
`experiments/e1_vllm/snapshots.yaml`.

---

## snapshots

`experiments/e1_vllm/snapshots.yaml`

| tag | version | role |
|-----|---------|------|
| doc_0 | 0.22.0 | LoRA train, stale RAG |
| doc_8 | 0.23.0 | fresh RAG, doc_8-only truth |

---

## conditions

| id | retrieval | train | what we learn |
|----|-----------|-------|---------------|
| c0_base | — | — | implemented config; baseline behavior |
| c1_rag_fresh | doc_8 | — | implemented config; BM25 fresh retrieval |
| c2_rag_stale | doc_0 | — | implemented config; stale-index penalty |
| c3_ft | — | LoRA doc_0 | implemented config; stale memorized knowledge |
| c4_ft_rag | doc_8 | LoRA doc_0 | implemented config; combination attribution |
| c6_shuffle | shuffled | — | implemented config; retrieval control |

**Headline run:** c3 on class **D** and **B** items → `stale_version` rate.

---

## eval

| file | status |
|------|--------|
| `fixtures/eval_v2.jsonl` | 40 items, corpus-validated, topic-typed |
| `fixtures/EVAL_SCHEMA.md` | v3 schema (eval_class A–D) |
| `fixtures/eval_v3.jsonl` | 46 items (A20/B10/C8/D8), corpus-validated |

**Eval task:** short answer Q&A. See thesis doc — not MC, not code-first.

**Data guide:** [e1_vllm_data.md](./e1_vllm_data.md)

```bash
python experiments/e1_vllm/inspect_data.py validate-eval eval_v2.jsonl
```

---

## operational status

```text
implemented  snapshots, corpora, inspect/validate tooling, eval_v2, eval_v3,
             c0–c6 Modal configs, BM25 index builder, LoRA train/reload path,
             mill (T1–T7), comparison report,
             dense TopicKnowledgeProfile + profile wheel (v5) + seq v7 LoRA,
             base 0.8B + 4B instrument checks on factory eval,
             factory condition matrix @ 0.8B; fair RAG ladder (no hint);
             four-transition structural census (v0.20…v0.24);
             frozen EvalEnvironment v0.21→v0.22 (62 deltas / 674 probes);
             evidence-span RAG on factory eval (exact 0.724; changed 0.375);
             code-BM25 A/B + evidence era ladder; evidence RAG on v0.21→v0.22;
             symbol and hybrid-diff retrievers; factory LoRA/delta-diet arc;
             remember/forget A–D negative result; deterministic reconcile
partial      short-answer evaluator; Phase B DriftEvent v1
             (behavioral + attached factory corpus_signal);
             Phase D sealed-transition exit
planned      executable procedural probes; sealed unseen-transition reconcile;
             better within-file evidence-span ranking
deferred     full-weight FT; LoRA past v7 profile wheel (still paused);
             factory LoRA after remember/forget closeout;
             docs-BM25 / naive code-BM25; boolean_answer_hint default
```

This work completed the Phase A exit and Phase C intervention matrix for the
`e1_vllm` slice; Phase B has DriftEvent v1 + baseline. Remaining gates live in
[the DELTA roadmap](../delta-roadmap.md).

---

## B2

```text
datasets/experiments/e1_vllm/corpus_doc_0.jsonl   # uploaded
datasets/experiments/e1_vllm/corpus_doc_8.jsonl   # uploaded
datasets/experiments/e1_vllm/eval_v3.jsonl        # uploaded
artifacts/reports/e1_vllm_failure_modes.md       # uploaded
```

---

## what goes where

- hypothesis and metrics: `docs/experiments/e1_vllm_thesis.md`;
- corpus/eval facts: `docs/experiments/e1_vllm_data.md`;
- harness and fixtures: `experiments/e1_vllm/`;
- run configs: `configs/experiments/e1_vllm/`;
- durable datasets and outputs: B2 paths above.

---

## commands

```bash
python experiments/e1_vllm/build_corpus.py --upload
cp experiments/e1_vllm/fixtures/eval_v2.jsonl data/experiments/e1_vllm/
python experiments/e1_vllm/inspect_data.py stats
python experiments/e1_vllm/inspect_data.py validate-eval eval_v2.jsonl
```

---

## what can die

- local source clones, candidate fixtures, and failed smoke outputs;
- planned conditions that do not produce useful evidence;
- provisional c3 conclusions from the current summarization dataset.

## what must survive

- pinned source revisions and corpus hashes
- frozen eval plus its validation manifest
- per-sample outputs and failure labels in run artifacts
- doc_0/doc_8 corpora on B2
