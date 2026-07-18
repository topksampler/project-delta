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
python experiments/e1_vllm/inspect_data.py validate-eval eval_v2.jsonl
cp experiments/e1_vllm/fixtures/eval_v2.jsonl data/experiments/e1_vllm/
```

## implemented

```text
build_corpus.py    pinned vLLM tags → chunked corpus JSONL
inspect_data.py    corpus stats, search, and eval validation
build_train.py     provisional doc_0 summarization SFT dataset
fixtures/eval_v2   40 corpus-validated short-answer items
```

`build_train.py` does not yet match the Q&A eval task. Its outputs are provisional,
not the final DELTA training dataset.

## missing

```text
eval_v3.jsonl      A-D class coverage and harder gold
dataset factory    repo/tag → typed train and eval variants
build_index.py     corpus → retrieval index
compare.py         per-class failure and cost report
trusted verifier  complete failure labels, then executable tests
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
