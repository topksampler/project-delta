# DELTA v2 d7 result and claim-coverage diagnosis

Date: 2026-07-30

Decision: reject d7; change the stability-data contract

## d7 result

d7 changed only peak learning rate from `1e-4` to `5e-5` relative to d5.
The 240-unit schedule was byte-identical after canonical serialization.

Held-out stability:

- Choice parseable: 15/15; correct: 6/15.
- Boolean true: 11/15.
- Boolean false: 8/15.
- Recall: 1/15, advisory.

The unchanged gates required at least 12/15 true and 14/15 false. d7 failed
both and did not advance.

Evidence:

- Training receipt:
  `4248f3917f217a03f970322df02af9c9392cb8a24c1e8ef38dd81966b701832b`
- Training metrics:
  `9205fc73c1d9c5e06112c246072ac9ae140b329d78e55c84611000ac8e59f2c5`
- Stability receipt:
  `1fd15c8642c3feaa32ac76833bef241a0f6988ef4c7fccac4fa9cd6560752c24`
- Stability metrics:
  `1bd8b43cf8e363ce6d6ec0a35db9610f1429820a7307b7ebcc1e7b3ebc43ebb5`

## Diagnosis

The stability Boolean probes form true/false pairs for the same 15 exact
source facts. The untouched base usually answered `no` to both members. The
d4-d7 adapters often answered `yes` to both. The candidates were moving a
global response bias, not recovering fact-specific knowledge.

Those 15 facts had zero source overlap with every training set. Because the
facts are exact vLLM v0.26.0 source contracts, source-disjoint replay cannot
teach them. The old gate therefore mixed two questions:

1. Can the model retain ingested stable facts?
2. Can the model infer unseen exact source code facts?

Only the first is required by the DELTA CI/CD loop.

## d8 correction

d8 trains on all 15 stable claims while retaining zero exact prompt overlap
with evaluation. Evaluation uses new wording, multiple choice, new false
corruptions, and exact recall for those ingested claims.

This supports generalization to new wording and corruptions on ingested
claims. It does not support generalization to unseen source facts. QLoRA,
full-weight tuning, reinforcement learning, full evaluation, and promotion
remain unauthorized.
