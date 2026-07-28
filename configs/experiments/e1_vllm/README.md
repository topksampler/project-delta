# e1_vllm run configs

One YAML = one `./scripts/lab run`.

The experiment contract is
[`docs/experiments/e1_vllm.md`](../../../docs/experiments/e1_vllm.md). The
authoritative model ID is in
[`experiments/e1_vllm/snapshots.yaml`](../../../experiments/e1_vllm/snapshots.yaml).

## invariant

```text
One immutable YAML config → one globally unique run_id → one B2 run prefix.
```

## naming

```text
run_id: e1-vllm-{condition}-{model_slug}-{target}
```

## current configs

| file | role |
|---|---|
| `c0_base_eval_*` | evaluate frozen base model |
| `c1_rag_fresh_eval_v3_*` | base + BM25 top-k from doc_8 |
| `c2_rag_stale_eval_v3_*` | base + BM25 top-k from doc_0 |
| `c3_ft_*` / `c3_ft_eval_*` | train / reload LoRA on doc_0 |
| `c4_ft_rag_fresh_eval_v3_*` | mill-v3 LoRA + BM25 doc_8 |
| `c6_shuffle_eval_v3_*` | base + random chunks from doc_8 |
| `c3_ft_mill_v4_*` | train / eval LoRA on mill T1–T7 |
| `profile_v022_*` | closed-book TopicKnowledgeProfile pilot |
| `truthsource_*` | TruthSource D / C_distill / C_full profile arms |

## retrieval fields (c1)

```yaml
retrieval:
  enabled: true
  corpus_path: data/experiments/e1_vllm/corpus_doc_8.jsonl
  index_path: data/experiments/e1_vllm/indexes/doc_8_bm25.json
  top_k: 4
  max_chars: 3500
```

Absent `retrieval` → no context injection (c0/c3).

## what goes where

- condition definitions: experiment charter;
- source/model pins: `snapshots.yaml`;
- dispatch parameters: YAML files in this directory;
- immutable config snapshots: B2 `runs/{run_id}/config.yaml`.

## what can die

Superseded local config drafts and failed smoke output directories.

## what must survive

Every dispatched config snapshot and its immutable `run_id`; c3 eval must retain
the `adapter_run_id` of the c3 training run.

## command

```bash
./scripts/lab run --target modal \
  --config configs/experiments/e1_vllm/c0_base_eval_qwen35_08b_modal.yaml

# mill T1 LoRA (after mill build)
./scripts/lab run --target modal \
  --config configs/experiments/e1_vllm/c3_ft_mill_t1_qwen35_08b_modal.yaml
```
