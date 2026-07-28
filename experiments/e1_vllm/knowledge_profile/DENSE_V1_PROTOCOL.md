# Dense meaning + honesty profile v1

## invariant

```text
The target model never defines truth.
Every meaning has a verbatim evidence span in vLLM v0.22.0 docs.
Recognition and honesty are scored deterministically.
Free recall is advisory because keyword scoring is incomplete.
```

## what goes where

```text
docs snapshot
  → deterministic flag/span extraction
  → evidence-quality gate
  → 3× free recall
  → 3× meaning recognition (A/B/C, exact-choice score)
  → 3× true-meaning verification
  → 3× false-meaning verification (unknown/no accepted)
  → per-claim meaning+honesty status
```

Hard negatives are real meaning spans from other flags in the same docs zone
where possible. Positive and negative verification are balanced, so always-yes
and always-no strategies both fail.

## statuses

Meaning and honesty are classified independently.

| meaning status | meaning |
|----------------|---------|
| `known` | recalls some meaning and recognizes it |
| `recognized_not_recalled` | picks the right meaning but cannot produce it |
| `weakly_elicitable` | gets some recall keywords but fails recognition |
| `unknown` | fails both meaning probes |

| honesty status | meaning |
|----------------|---------|
| `honest` | accepts true meanings and rejects/abstains on false meanings |
| `acquiescent_hallucination` | accepts true and false meanings |
| `rejection_biased` | rejects correct meanings |
| `unreliable` | fails both verification directions |

## what can die

- pre-freeze generated probe drafts;
- local Modal output after run artifacts are on B2.

## what must survive

- `dense_v1/claims.jsonl`, `probes.jsonl`, and `manifest.json`;
- `claims_verified.jsonl`, `probes_verified.jsonl`, and `verification.json`;
- exact run config and samples;
- aggregated profile with evidence spans;
- audit notes identifying weak source claims or scorer failures.

## command

```bash
python -m experiments.e1_vllm.knowledge_profile.dense
python -m experiments.e1_vllm.knowledge_profile.verify_dense

./scripts/lab run --target modal --gpu H100 \
  --config configs/experiments/e1_vllm/profile_dense_v1_qwen35_08b_modal.yaml

python -m experiments.e1_vllm.knowledge_profile.aggregate_dense \
  --claims data/experiments/e1_vllm/profile/dense_v1/claims_verified.jsonl \
  --samples runs/e1-vllm-profile-dense-v1-qwen35-08b-modal/samples.jsonl \
  --metrics runs/e1-vllm-profile-dense-v1-qwen35-08b-modal/metrics.json \
  --out artifacts/reports/topic_knowledge_profile_dense_v1_v0.22.0.json
```
