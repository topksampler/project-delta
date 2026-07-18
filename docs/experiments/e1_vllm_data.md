# e1_vllm — understanding the data

Read this before writing eval questions or building an index.

This document owns corpus statistics and dataset validation for the
[`e1_vllm` vertical slice](./e1_vllm.md). It does not define Project DELTA's
system architecture or delivery phase.

## invariant

```text
Corpus = chunked vLLM git docs at a pinned tag (not live docs.vllm.ai crawl).
Eval gold must be verifiable against the corpus snapshot you claim (requires_doc).
RAG can only retrieve what is in the index; FT can only memorize what is in train JSONL.
```

## current artifacts

| artifact | bytes (approx) | meaning |
|----------|----------------|---------|
| `corpus_doc_0.jsonl` | 2.5 MB, 2737 chunks | vLLM **v0.22.0** `docs/`, `examples/`, `README.md` |
| `corpus_doc_8.jsonl` | 2.5 MB, 2793 chunks | vLLM **v0.23.0** same layout |

Each line:

```json
{
  "chunk_id": "doc_8:docs/getting_started/quickstart.md:3",
  "text": "...",
  "source_path": "docs/getting_started/quickstart.md",
  "vllm_version": "0.23.0",
  "git_tag": "v0.23.0",
  "docs_base": "https://docs.vllm.ai/en/v0.23.0/",
  "chunk_index": 3,
  "chunk_total": 42
}
```

Chunking: section splits (headers / rules), then paragraph packing ~1800 chars, 200 overlap.

## corpus shape

```text
~94%  docs/**     (.md, .rst)
~6%   examples/**
8     README.md chunks
```

Heavy regions (where most tokens live):

| section | chunks | eval opportunity |
|---------|--------|------------------|
| `docs/models/` | ~500+ | supported models, pooling, trust flags |
| `docs/features/quantization/` | ~300 | FP8, GPTQ, llm_compressor paths |
| `docs/getting_started/installation/` | ~280 | Python versions, pip/uv |
| `docs/serving/` | ~200 | parallelism flags, OpenAI server |
| `docs/design/` | ~200 | architecture, KV cache, V1 engine |
| `docs/benchmarking/cli.md` | ~96 | CLI flags in benchmark context |

## drift between v0.22.0 and v0.23.0

**Not** 8 weeks of your lab — **one release** of upstream vLLM.

| drift type | count | example |
|------------|-------|---------|
| shared paths | 237 files | same `source_path` in both corpora |
| only in v0.23.0 | 6 paths | `docs/features/quantization/llm_compressor/README.md` (+ fp8/int4/int8 splits) |
| only in v0.22.0 | 4 paths | flat `docs/features/quantization/llm_compressor.md` |

Most content is **byte-stable** across versions (same paths, similar chunk counts). Drift is concentrated in:

- quantization doc reorg (`llm_compressor.md` → `llm_compressor/` folder)
- new connector docs (`moriio_connector_usage.md` — search **MoRIIO** in corpus)
- benchmarking/cli.md grew (+4 chunks)

**Important:** GitHub **release notes** (Model Runner V2, Rust frontend bullets) are **not** in the cloned docs tree. Questions sourced only from blog/release PRs will **fail RAG** unless you add a `corpus_release_notes.jsonl` later.

## eval design rules

### 1. Every row needs a corpus anchor

Before adding a question, run:

```bash
python experiments/e1_vllm/inspect_data.py search PagedAttention --corpus doc_8
python experiments/e1_vllm/inspect_data.py validate-eval eval_v2.jsonl --against 0.23.0 --verbose
```

`must_contain` strings must appear in **some chunk** of the corpus matching `requires_doc`.

### 2. `requires_doc` semantics

| value | score against | RAG index for c1 | expected c2 (stale) |
|-------|---------------|------------------|---------------------|
| `0.22.0` | either corpus OK for gold check | doc_8 fresh still works | should still work |
| `0.23.0` | must be in doc_8 corpus | doc_8 | **fails** if index is doc_0 only |

Use `0.23.0` for drift-canary questions (files only in v0.23.0).

### 3. Question types

| type | tests | example |
|------|-------|---------|
| `concept` | stable ideas in README/design | PagedAttention, prefix caching |
| `api` | CLI flags | `--tensor-parallel-size` |
| `config` | serve arguments | `--max-model-len` |
| `install` | getting started | `pip install vllm`, Python 3.12 |
| `feature` | feature docs | speculative decoding, tool calling |
| `architecture` | design/ | data parallel engine cores |
| `drift` | v0.23-only paths | `moriio_connector_usage.md` |

### 4. `source_hint` (optional, for you)

Points to the file you used to write gold. Not used by scoring — documentation for humans.

### 5. What goes where

| artifact | git? | B2? |
|----------|------|-----|
| `fixtures/eval_v2.jsonl` | yes | copy on freeze |
| `data/experiments/e1_vllm/eval_v2.jsonl` | no | `datasets/experiments/e1_vllm/eval_v2.jsonl` |
| corpora JSONL | no | already uploaded |

## inspect commands

```bash
python experiments/e1_vllm/inspect_data.py stats
python experiments/e1_vllm/inspect_data.py search tensor-parallel-size --corpus doc_8
python experiments/e1_vllm/inspect_data.py validate-eval eval_v2.jsonl --against 0.23.0 -v
```

## growing eval (workflow)

1. Pick a doc section you care about (`inspect_data.py stats` → choose path).
2. `search` for a phrase you want the model to know.
3. Write question + `must_contain` from **that chunk** (not from memory).
4. Set `requires_doc` and `source_hint`.
5. `validate-eval` until 40/40 OK.
6. Copy to `data/` and upload to B2 when frozen.

## what can die

- eval rows that fail `validate-eval`
- questions about release-note trivia not in corpus

## what must survive

- frozen `eval_v2.jsonl` in git with 100% corpus validation
- corpora on B2 at pinned tags

## next

Build `eval_v3`, align training rows with the Q&A task, then harden the evaluator.
Current status and order live in [e1_vllm.md](./e1_vllm.md) and the
[DELTA roadmap](../delta-roadmap.md).
