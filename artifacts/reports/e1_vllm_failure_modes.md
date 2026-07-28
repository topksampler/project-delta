# e1_vllm — failure modes & intervention comparison

## invariant

```text
Same eval_v3 (46 items, A20/B10/C8/D8), same model Qwen/Qwen3.5-0.8B.
Primary signal = by_eval_class + failure_mode, not headline accuracy alone.
Index freshness and retrieval relevance are separate levers from LoRA@doc_0.
```

## legend

### snapshots

| name | meaning |
|---|---|
| `doc_0` | vLLM docs @ **v0.22.0** — train corpus and stale RAG index |
| `doc_8` | vLLM docs @ **v0.23.0** — fresh RAG index; truth for new facts |

### conditions

| id | recipe |
|---|---|
| `c0_base` | base model only — no LoRA, no retrieval |
| `c1_rag_fresh` | base + BM25 top-k from **doc_8** |
| `c2_rag_stale` | base + BM25 top-k from **doc_0** |
| `c3_ft` / `c3_mill_v*` | LoRA trained on doc_0 mill data; no RAG at eval |
| `c4_ft_rag` | c3 LoRA + doc_8 BM25 at eval |
| `c6_shuffle` | base + **random** doc_8 chunks (retrieval placebo) |

No `c5` in this matrix.

### eval classes (eval_v3 columns A–D)

| class | tests |
|---|---|
| **A** | stable facts (true in both versions) |
| **B** | what changed / only-in-0.23 |
| **C** | how-to / procedural |
| **D** | version-binary (“does X exist in v0.22?”) |

### mill train classes (LoRA diet)

| id | skill |
|---|---|
| T1 | declarative flag Q→A |
| T2 | procedural how-to |
| T3 | version-conditioned (“In v0.22, …”) |
| T4 | abstain / unknown |
| T5 | contrast (“not this flag”) |
| T6 | answer + cite source path |
| T7 | multi-hop (two chunks) |
| T8 | cross-version delta — not used in c3@doc_0 |

`mill v3` / `mill v4` = successive train-set builds (v4 adds T2/T6/T7).

### other

| term | meaning |
|---|---|
| `e1_vllm` | this experiment (first DELTA vertical slice) |
| BM25 | lexical retrieval over chunk text (no embeddings) |
| `eval_v3` | frozen 46-item Q&A set used for all conditions here |
| `run_id` | one Modal job; evidence under B2 `runs/{run_id}/` |

## what goes where

- this report: `artifacts/reports/e1_vllm_failure_modes.md` (also B2)
- regenerator: `experiments/e1_vllm/compare.py`
- per-run evidence: B2 `runs/{run_id}/metrics.json` + `samples.jsonl`
- condition definitions: `docs/experiments/e1_vllm.md`

## what can die

- superseded draft tables in chat logs
- local `/tmp` metric caches used to rebuild this file

## what must survive

- frozen `eval_v3.jsonl` + run_ids listed below
- this report next to its generator command
- B2 copies of metrics/samples for each run_id

## scoreboard (eval_v3)

| condition | acc | A | B | C | D | role |
|---|---:|---:|---:|---:|---:|---|
| `c6_shuffle` | 0.17 | 0.20 | 0.00 | 0.38 | 0.50 | random doc_8 chunks |
| `c0_base` | 0.24 | 0.20 | 0.10 | 0.44 | 0.50 | base, no retrieval |
| `c3_mill_v3` | 0.30 | 0.42 | 0.10 | 0.50 | 0.38 | LoRA@doc_0 mill v3 |
| `c4_ft_rag` | 0.30 | 0.40 | 0.10 | 0.50 | 0.12 | mill-v3 LoRA + doc_8 BM25 |
| `c3_mill_v4` | 0.33 | 0.45 | 0.10 | 0.56 | 0.25 | LoRA@doc_0 mill v4 T1–T7 |
| `c2_rag_stale` | 0.39 | 0.40 | 0.20 | 0.62 | 0.38 | BM25 doc_0 |
| `c1_rag_fresh` | 0.46 | 0.45 | 0.30 | 0.62 | 0.50 | BM25 doc_8 |

## failure modes (counts / 46)

| condition | correct | wrong | stale_version | abstain_wrong |
|---|---:|---:|---:|---:|
| `c6_shuffle` | 8 | 31 | 3 | 4 |
| `c0_base` | 11 | 31 | 4 | 0 |
| `c3_mill_v3` | 14 | 27 | 3 | 2 |
| `c4_ft_rag` | 14 | 20 | 1 | 11 |
| `c3_mill_v4` | 15 | 28 | 3 | 0 |
| `c2_rag_stale` | 18 | 22 | 5 | 1 |
| `c1_rag_fresh` | 21 | 21 | 3 | 1 |

## findings (locked to this matrix)

```text
1. Fresh BM25 (c1) wins overall and on B. Index freshness > doc_0 LoRA for drift.
2. Shuffle (c6) < base → c1 lift is relevance, not “any context”.
3. Stale BM25 (c2) beats base but hurts D vs c0 (wrong-era contamination).
4. LoRA@doc_0 (c3) helps A/C; does not recover B; D weak/worse on v4.
5. c4 (LoRA + fresh RAG) interferes: abstain_wrong spikes; D collapses.
6. c1 D flat vs c0: many D items are negative existence; fresh chunks bias YES.
```

## cost proxy

```text
Eval conditions: Modal A10G, ~100–140s compute each (46 generations).
No LoRA train cost for c0/c1/c2/c6.
c3/c4 require a prior H100 LoRA train (~200 steps; mill v4 ~460s train_runtime).
Dollar cost not frozen here — GPU class + wall seconds are the durable proxy.
```

| train run | gpu | note |
|---|---|---|
| `c3_train_v3` (`e1-vllm-c3-ft-mill-v3-qwen35-08b-modal`) | H100 | ~200 steps |
| `c3_train_v4` (`e1-vllm-c3-ft-mill-v4-qwen35-08b-modal`) | H100 | ~200 steps / ~460s train_runtime |

## run_ids

| condition | run_id |
|---|---|
| `c0_base` | `e1-vllm-c0-base-eval-v3-qwen35-08b-modal` |
| `c6_shuffle` | `e1-vllm-c6-shuffle-eval-v3-qwen35-08b-modal` |
| `c3_mill_v3` | `e1-vllm-c3-ft-mill-v3-eval-v3-qwen35-08b-modal` |
| `c3_mill_v4` | `e1-vllm-c3-ft-mill-v4-eval-v3-qwen35-08b-modal` |
| `c4_ft_rag` | `e1-vllm-c4-ft-rag-fresh-eval-v3-qwen35-08b-modal` |
| `c2_rag_stale` | `e1-vllm-c2-rag-stale-eval-v3-qwen35-08b-modal` |
| `c1_rag_fresh` | `e1-vllm-c1-rag-fresh-eval-v3-qwen35-08b-modal` |

## command

```bash
# after downloading metrics into a dir as {run_id}.json
python experiments/e1_vllm/compare.py --metrics-dir /tmp/e1_compare \
  --out artifacts/reports/e1_vllm_failure_modes.md
```

