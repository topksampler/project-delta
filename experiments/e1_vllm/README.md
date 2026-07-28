# e1_vllm

First BUILD/SENSE vertical slice of [Project DELTA](../../docs/project-delta.md):
version fidelity on moving technical docs.

## invariant

```text
Train / stale index  →  vLLM v0.22.0 docs
Eval / fresh index   →  vLLM v0.23.0 docs
Primary metric       →  failure_mode taxonomy, not accuracy alone
Eval task            →  short free-form Q&A (see fixtures/EVAL_SCHEMA.md)
```

The source/model pins live in `snapshots.yaml`; the
[experiment charter](../../docs/experiments/e1_vllm.md) owns condition status.

## command

```bash
python experiments/e1_vllm/build_corpus.py
python experiments/e1_vllm/inspect_data.py stats
python experiments/e1_vllm/inspect_data.py validate-eval eval_v3.jsonl
cp experiments/e1_vllm/fixtures/eval_v3.jsonl data/experiments/e1_vllm/

# BM25 index (doc_8 for c1; same tool works for doc_0 / c2 later)
python experiments/e1_vllm/build_index.py \
  --corpus data/experiments/e1_vllm/corpus_doc_8.jsonl \
  --out data/experiments/e1_vllm/indexes/doc_8_bm25.json
```

## implemented

```text
build_corpus.py    pinned vLLM tags → chunked corpus JSONL
inspect_data.py    corpus stats, search, and eval validation
build_index.py     corpus JSONL → BM25 index JSON
bm25.py            tokenize + BM25Okapi fit/query
build_train.py     provisional doc_0 summarization SFT dataset
mill/              pin → extract → generate → gate → emit → build (T1–T7)
compare.py         metrics JSON → failure-mode comparison report
sense_drift.py     c0 vs probe samples/metrics → DriftEvent + baseline
knowledge_profile/ claim bank → probes → TopicKnowledgeProfile aggregate
fixtures/eval_v3   46 items A–D short-answer
```

## knowledge profile

Closed-book `TopicKnowledgeProfile` (claim bank → probes → aggregate):

```bash
python experiments/e1_vllm/knowledge_profile/cli.py build-claims
python experiments/e1_vllm/knowledge_profile/cli.py build-probes \
  --claims data/experiments/e1_vllm/profile/claims_v0.22.0.jsonl
```

See `knowledge_profile/README.md`. Pilot run:
`e1-vllm-profile-v022-qwen35-08b-modal` →
`artifacts/reports/topic_knowledge_profile_v0.22.0.json`.

## missing

```text
corpus structural signal on DriftEvent
trusted / executable verifier
boundary-pass semantic entropy on profile unknowns
```

## what goes where

- scripts and fixture schemas: this directory;
- experiment claims and status: `docs/experiments/e1_vllm*.md`;
- run configurations: `configs/experiments/e1_vllm/`;
- generated corpora/train data: local `data/` cache and durable B2 datasets.

## what can die

- `data/experiments/e1_vllm/repos/` clones;
- rejected candidate rows and pre-freeze fixture drafts;
- local run caches after durable outputs are on B2.

## what must survive

- `snapshots.yaml`, fixture schemas, and dataset recipes in git;
- frozen corpus/eval manifests on B2;
- per-run config, samples, metrics, and ledger.
