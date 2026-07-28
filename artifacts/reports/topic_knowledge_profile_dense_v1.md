# Dense vLLM v0.22 knowledge profile — meanings + honesty

Run: `e1-vllm-profile-dense-v1-qwen35-08b-modal`  
Model: `Qwen/Qwen3.5-0.8B`  
Mode: closed-book  
Cost: $0.38 estimated; 348 GPU-seconds on H100

## invariant

```text
The target model does not define truth.
Meaning answers come from evidence spans in pinned vLLM v0.22.0 docs.
Code verifies that the named CLI flag exists; it does not generate the meaning.
Meaning and honesty are separate axes.
```

## what was measured

The deterministic docs extractor proposed 130 meaning claims. A code existence
check removed 45 polluted or out-of-scope flags (`--file`, `--with`, Docker
flags, launcher flags, and similar). The audited profile contains:

```text
85 verified flag meanings
996 evaluated probes
3 paraphrases per probe family
```

Per meaning:

1. free recall (advisory keyword score);
2. three-choice meaning recognition;
3. verification of the true meaning;
4. rejection or abstention on a wrong meaning borrowed from another flag.

## result

| behavior | accuracy | claim-cluster bootstrap 95% CI |
|----------|---------:|--------------------------------:|
| free recall (advisory) | 76.3% | 67.9–84.3% |
| meaning recognition (3 choices) | 34.5% | 27.3–42.2% |
| accepts true meanings | 95.6% | 92.0–98.8% |
| rejects/abstains on false meanings | 23.7% | 17.3–30.5% |

Random performance on the recognition task is 33.3%. The observed 34.5% is not
evidence that the model can reliably identify meanings.

The true/false asymmetry is the strongest result:

```text
true meaning presented  → says yes 95.6%
wrong meaning presented → says no/unknown only 23.7%
```

The model usually accepts a plausible vLLM-sounding meaning, even when that
meaning belongs to another flag.

This does not contradict the earlier ~99% negation result. That test asked the
model to reject obviously invented flag names. This profile asks it to reject a
real, plausible vLLM meaning attached to the wrong real flag. Name-boundary
honesty is strong; semantic-boundary honesty is weak.

## interpretation

The earlier substring “recall” score was optimistic. One matching keyword can
score a fluent guess as correct. The harder recognition test falls at chance.

For this 0.8B model:

```text
surface familiarity with vLLM: yes
reliable mapping from flag → meaning: not demonstrated
ability to reject plausible wrong meanings: poor
```

The data wheel must therefore target both:

1. correct flag-to-meaning associations;
2. contrastive negatives and abstention, so adaptation does not merely increase
   confident fluency.

## validity limits

- The 85 claims are broader than the thin pilot, but not the whole vLLM subject.
- Code verifies flag existence, not semantic completeness of each docs span.
- Free recall remains advisory until semantic grading is independently audited.
- A human audit of at least 50 meaning probes must check the automatic grader
  before recall scores are used as a training target.
- Three-choice recognition needs a larger-model control to show that the
  instrument is solvable rather than merely difficult for 0.8B.
- “Honesty” here means rejecting a mismatched documented meaning; it is not a
  general hallucination score.

## what goes where

| artifact | role |
|----------|------|
| `data/experiments/e1_vllm/profile/dense_v1/claims.jsonl` | raw docs candidates |
| `claims_verified.jsonl` | code-existence-audited 85-claim bank |
| `probes.jsonl` | exact 1,536-probe run input |
| `probes_verified.jsonl` | audited 996-probe subset |
| `verification.json` | 45 dropped entities + hashes |
| `artifacts/reports/topic_knowledge_profile_dense_v1_verified_v0.22.0.json` | item-level profile |
| `artifacts/reports/topic_knowledge_profile_dense_v1_summary.json` | aggregate + confidence intervals |

## what can die

- local copies of Modal outputs after B2 durability;
- pre-freeze candidate drafts.

## what must survive

- verified claim/probe hashes;
- exact run samples;
- item-level meaning and honesty statuses;
- the 45-item rejection audit;
- this interpretation and its validity limits.

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
  --out artifacts/reports/topic_knowledge_profile_dense_v1_verified_v0.22.0.json
```
