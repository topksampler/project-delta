# delta_v2 ADAPT handoff

Date: 2026-07-30

Repository: `/Users/lalithnarayanc/lalith-ai-lab`

Branch: `main`

Latest experiment commit at handoff: `da21aea`

## Start here

Read completely before acting:

1. `AGENTS.md`
2. `docs/project-delta.md`
3. `docs/delta-roadmap.md`
4. `docs/developer-workflow.md`
5. `docs/experiments/delta_v2.md`
6. `experiments/delta_v2/README.md`
7. `experiments/delta_v2/DATA_MODEL.md`
8. this handoff

Then inspect `git status`.

Preserve the existing user-owned uncommitted documentation. At handoff it is:

```text
 M docs/delta-roadmap.md
 M docs/experiments/README.md
 M experiments/README.md
?? artifacts/reports/delta_v2/README.md
?? configs/experiments/delta_v2/README.md
?? docs/experiments/delta_v2.md
?? experiments/delta_v2/DATA_MODEL.md
?? experiments/delta_v2/README.md
```

Do not discard, overwrite, stage, commit, or copy these files without first
reviewing their current contents and agreeing on any overlap. Keep blogs out of
the work. Do not touch or copy the frozen `e1_vllm` harness. Do not push.

## Loop position

```text
SENSE implemented
→ BUILD implemented
→ DECIDE partial
→ ADAPT partial and active
→ VERIFY implemented per candidate
→ MEMORY implemented per candidate
```

SENSE and BUILD are complete for this vertical slice, not for the full DELTA
program. The source pipeline has exact snapshots, a complete file inventory,
preregistered fact families, independently extracted facts, verified features,
executable behavior probes, split facts/features, and frozen evaluation
environments.

No candidate model has been promoted. The pinned base model remains the active
state.

## Completed no-weight controls

The first no-training ADAPT matrix is recorded in
`artifacts/reports/delta_v2/adapt_no_training_matrix.md`.

- c0: frozen base
- c1: task calibration
- c2: oracle source context
- c3: source comparison

These are diagnostic context conditions, not a production RAG system.
Production RAG remains unimplemented.

## Knowledge adaptation bank

Frozen environment: `delta-v2-vllm-knowledge-adaptation-v1`

- 21 verified v0.26-added facts used for same-claim acquisition
- 21 source-disjoint stable facts used for retention
- one source-disjoint verified feature behavior
- 63 immutable SFT rows: 21 recall, 21 verified true, 21 deterministic false
- 169 frozen evaluation probes, each run twice
- deterministic gold and exact scorers own the verdict
- recall is advisory
- no pooled overall score
- no held-out-fact generalization claim

Base model:

- repository: `Qwen/Qwen3.5-0.8B`
- revision: `2fc06364715b967f1860aea9cf38778875588b17`

## LoRA results

### c5 — balanced 1:1

Training run:
`delta-v2-c5-knowledge-lora-qwen35-08b-modal-v2`

Adapter SHA-256:
`83fb04ae52502432df6b17d2aaa12cfeb12f457df6a77c83f28fed433d9966fa`

Frozen evaluation:

- acquisition choice: 10/21
- acquisition true: 0/21
- acquisition false: 21/21
- acquisition recall: 10/21
- retention choice: 7/21
- retention true: 1/21
- retention false: 21/21
- retention recall: 3/21
- feature: 0/1

The candidate failed its gates. Exact training-surface replay then showed:

- verified true: 0/21
- deterministic false: 21/21
- recall: 9/21

c5 learned an always-`no` policy rather than the truth-conditioned distinction.

### c6 — true-weighted 4:1

Training run:
`delta-v2-c6-knowledge-lora-true-weighted-qwen35-08b-modal-v1`

Adapter SHA-256:
`8dae0e5cf7692542efc34f4fc836b1c96279bae680747edc368f646fdf938e36`

The deterministic virtual sampler consumed:

- 163 verified-true examples
- 37 deterministic-false examples
- 40 recall examples

Its conditional training-surface gate showed:

- verified true: 21/21
- deterministic false: 0/21
- recall: 8/21

c6 learned an always-`yes` policy. The gate blocked the full 169-item
evaluation. c5 and c6 bracket the answer-prior boundary between 1:1 and 4:1.

The next minimal LoRA candidate is fresh-base c7 at a 2:1 true:false sampler
ratio, with every other parameter, source row, and verification gate unchanged.
c7 has not been implemented or run.

## New ADAPT architecture requested by the user

The user wants ADAPT to explicitly contain:

1. LoRA
2. QLoRA
3. full-weight fine-tuning
4. verifier-grounded reinforcement learning

RAG remains a no-weight context control. The research goal is persistent model
weight change as the environment changes, not merely changing context.

The rigorous design has two axes:

| learning objective | adapter weights | full model weights |
|---|---|---|
| supervised | LoRA / QLoRA | full SFT or continued training |
| reinforcement | adapter-policy RL for harness validation | full-policy RL |

The canonical `docs/project-delta.md` contract currently says context, RAG,
LoRA, or later post-training. It has not yet been amended to encode this
two-axis ADAPT design. Update the canonical contract once, then keep detailed
condition matrices in the `delta_v2` charter; do not duplicate ownership in the
roadmap or harness README.

## Intended RL contract

RL means weight-changing, verifier-grounded post-training:

```text
verified development task
→ policy rollout: answer, source action, tool call, or executable test
→ deterministic code / AST / documentation / behavior verifier
→ reward vector
→ policy optimization
→ changed policy weights
→ frozen VERIFY
```

Reward should keep correctness, provenance, honesty, regression, and cost
separate. A language-model judge may review or help phrase candidates, but
must not define gold truth or override deterministic verification.

The unseen acceptance transition must never become the RL training
environment. RL may interact with verified development truth. Acceptance
remains sealed for VERIFY.

Before any RL job, freeze:

- environment state and allowed actions
- observation boundary
- rollout schema
- deterministic reward components and aggregation
- reward-hacking tests
- reference/KL or other stability policy
- retention and false-accept controls
- optimizer and update scope
- checkpoint, rollback, and promotion gates

Validate this environment first with adapter-policy RL. Full-policy RL is the
intended weight-changing research lane after the reward harness survives that
audit.

## Recommended continuation

1. Amend the canonical ADAPT contract and standalone Mermaid flow so the four
   requested lanes and two axes are explicit.
2. Preserve c5 and c6 as negative results.
3. Preregister and run c7 LoRA at 2:1.
4. Apply the cheap exact training-surface gate before a full frozen evaluation.
5. If c7 passes, run the unchanged 169-item evaluation.
6. Freeze a LoRA-versus-QLoRA parity contract using identical source rows,
   objective, adapter topology where possible, and evaluation.
7. Add full-weight supervised fine-tuning as a separately budgeted candidate.
8. Design and audit the verifier-grounded RL environment before launching RL.

Do not silently expand to multiple new training variants. Change one
preregistered factor at a time and preserve every failed run under its immutable
run ID.

## Verification and commits

At handoff:

- delta_v2 experiment tests: 211 passing
- shared tests: 34 passing
- `git diff --check`: passing

Recent checkpoints:

```text
da21aea Record delta_v2 c6 weighted LoRA result
efc5799 Preregister c6 training surface gate
40b64aa Preregister delta_v2 true-weighted LoRA
c00bc30 Record delta_v2 training surface diagnosis
9bdb8d7 Preregister delta_v2 training surface diagnostic
8406231 Record delta_v2 LoRA evaluation
1cd946c Record delta_v2 LoRA training
231ea97 Lock delta_v2 knowledge comparison gates
f4e9a61 Preregister delta_v2 LoRA evaluation
81ffe6a Amend delta_v2 LoRA launch
bfa54b6 Fix generic Modal training routing
```

All run data and model artifacts are under immutable B2
`runs/<run_id>/` prefixes. Local `runs/` content is a disposable cache.
