# TopicKnowledgeProfile (e1_vllm)

Closed-book behavioral profile of what a fixed model checkpoint knows about a
pinned subject revision.

## invariant

```text
Unit of truth = source-grounded atomic claim, not a hand-written quiz item.
First profile is closed-book. RAG is an intervention, not the baseline.
eval_v3 stays the Phase C comparison set — do not grow it into this profile.
```

## what goes where

| path | role |
|------|------|
| `knowledge_profile/` | claim extract, probe compile, aggregate |
| `data/.../profile/claims_*.jsonl` | claim bank |
| `data/.../profile/probes_*.jsonl` | probe surfaces (`eval.path`) |
| `artifacts/reports/topic_knowledge_profile_*.json` | durable profile |
| `configs/.../profile_*_modal.yaml` | Modal closed-book run |

## what can die

- local regenerations of probes with a new seed
- intermediate sample caches after profile JSON is written

## what must survive

- claim IDs + evidence hashes for a frozen revision
- profile JSON tied to model + protocol + claim_bank_hash
- the exact probe JSONL used for a published run

## command

```bash
# 1) claim bank from mill extract artifacts
python experiments/e1_vllm/knowledge_profile/cli.py build-claims

# 2) stratified pilot probes (~150 claims × paraphrases × forms)
python experiments/e1_vllm/knowledge_profile/cli.py build-probes \
  --claims data/experiments/e1_vllm/profile/claims_v0.22.0.jsonl

# 3) upload probes, then closed-book Modal eval (no adapter)
set -a && source .env && set +a
s5cmd --endpoint-url "${S3_ENDPOINT_URL%%[[:space:]]}" cp \
  data/experiments/e1_vllm/profile/probes_v0.22.0_pilot.jsonl \
  "s3://${S3_BUCKET}/datasets/experiments/e1_vllm/profile/probes_v0.22.0_pilot.jsonl"

./scripts/lab run --target modal --gpu A10G \
  --config configs/experiments/e1_vllm/profile_v022_qwen35_08b_modal.yaml

# 4) aggregate samples → TopicKnowledgeProfile
python experiments/e1_vllm/knowledge_profile/cli.py aggregate \
  --claims data/experiments/e1_vllm/profile/claims_v0.22.0_pilot.jsonl \
  --samples runs/<run_id>/samples.jsonl \
  --metrics runs/<run_id>/metrics.json
```

## claim types (v0)

| type | source | probes |
|------|--------|--------|
| `flag_exists` | structure_index flags | recall, recognition, negation |
| `env_exists` | structure_index env | recall, negation |
| `path_exists` | docs tree paths | recall, negation |
| `flag_meaning` | corpus span for flag | recall |

## status labels

```text
known | partial | unknown | stable_misconception | unstable
```

`misattributed` is reserved for version-conditioned probes (later).
