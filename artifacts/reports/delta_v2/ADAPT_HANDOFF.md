# delta_v2 ADAPT durable handoff

Date: 2026-07-31

Repository: `/Users/lalithnarayanc/lalith-ai-lab`

Current Codex worktree:
`/Users/lalithnarayanc/.codex/worktrees/fbf4/lalith-ai-lab`

Remote branch: `origin/main`

Evidence HEAD before this handoff: `68e2c75`

Pause state: no Modal or Lambda job is running. No candidate is promoted.

## Start here

Read this handoff completely, then read these files completely before acting:

1. `AGENTS.md`
2. `docs/project-delta.md`
3. `docs/delta-roadmap.md`
4. `docs/developer-workflow.md`
5. `docs/experiments/delta_v2.md`
6. `experiments/delta_v2/README.md`
7. `experiments/delta_v2/DATA_MODEL.md`
8. `artifacts/reports/delta_v2/full_language_sft_oracle_modal_v1.md`
9. `artifacts/reports/delta_v2/stability_lora_d8_d19_stop_decision.md`

Then inspect `git status` before changing anything. Report the exact loop
position and explain the two-axis ADAPT design in plain language before
proposing work.

## Preserve user-owned work

At pause, these documentation changes belong to the user:

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

Do not discard, overwrite, stage, commit, or mechanically copy these files.
Review their current contents before any overlapping documentation change.

Do not touch the frozen `e1_vllm` experiment, blogs, or unrelated work. Do not
launch compute or push merely because this handoff describes a next step; wait
for the returning user's instruction.

## Exact loop position

```text
SENSE implemented for the delta_v2 vertical slice
→ BUILD implemented for the frozen source/eval slice
→ DECIDE active
→ rejected ADAPT CandidateStates preserved
→ candidate VERIFY complete
→ rollback retained the pinned base
```

DELTA is at `DECIDE`, before a new `InterventionPlan`. The full-language F1
candidate completed ADAPT and failed VERIFY. Its exact training-surface
diagnostic also failed. Acquisition-margin, full frozen evaluation, promotion,
and active-pointer mutation were therefore blocked.

The active model remains the pinned base:

- repository: `Qwen/Qwen3.5-0.8B`
- revision: `2fc06364715b967f1860aea9cf38778875588b17`

SENSE and BUILD are implemented for this research slice, not for the entire
DELTA program. No closed production loop should be claimed.

## ADAPT architecture now frozen in the canonical contract

`docs/project-delta.md` owns the DELTA invariant and module contracts. It now
defines no-weight controls separately from persistent weight adaptation and
organizes weight-changing ADAPT along two independent axes:

| learning objective | adapter weights | full model weights |
|---|---|---|
| supervised | LoRA; QLoRA with a quantized frozen base | full SFT or continued training |
| verifier-grounded reinforcement | adapter-policy RL for reward-harness validation | full-policy RL after reward-harness audit |

Plain language:

- the first axis is **how the model learns**: supervised examples or verified
  reward;
- the second axis is **how much of the model may change**: a small adapter or
  the full language model;
- RAG and context injection are controls that change what the model can read,
  not its weights;
- QLoRA is an adapter method with a quantized frozen base. It is not a presumed
  quality improvement over LoRA;
- verifier-grounded RL remains ineligible until its environment, deterministic
  rewards, anti-hacking tests, stability controls, and rollback gates are
  separately frozen.

The research goal remains persistent weight adaptation as versioned sources
change. RAG is a necessary no-weight comparator, not the target mechanism.

## What has been completed

### No-weight controls

The c0-c3 matrix is recorded in
`artifacts/reports/delta_v2/adapt_no_training_matrix.md`:

- c0: frozen base;
- c1: task calibration;
- c2: oracle source context;
- c3: source comparison.

These are diagnostic controls, not a production RAG system.

### Supervised LoRA study

The original c5/c7/c6 true:false ratio sweep bracketed a response-prior
boundary rather than finding truth-conditioned learning:

- c5, 1:1, learned an always-`no` tendency;
- c7, 2:1, learned an always-`yes` tendency;
- c6, 4:1, strengthened the always-`yes` tendency.

Repaired source-grounded negative data later showed that LoRA can learn real
truth-conditioned and choice signal, but the candidates still failed the
unchanged held-out stability/false-rejection gates. The d8-d19 multisurface
recipe lineage was stopped rather than scaled further.

The zero-update d20 token-local diagnostic rejected the claim that LoRA is
useless. Representative adapters contained choice-relevant content signal, but
the implemented objectives coupled content, first-token behavior, and
termination behavior incorrectly. See
`artifacts/reports/delta_v2/stability_lora_d8_d19_stop_decision.md`.

Do not restart d8-d19, increase its steps/rank, or launch QLoRA as a quality
fix. Those are closed continuations of the rejected recipe.

### Full-language supervised feasibility oracle

The user explicitly chose to move down the mechanism-complexity ladder and run
plain full-language SFT before adding QLoRA or RL.

Training run:
`delta-v2-f1-full-language-sft-qwen35-08b-modal-v1`

- fresh pinned base;
- explicit PyTorch training loop, not `SFTTrainer`;
- plain assistant-token cross-entropy;
- all language-model and LM-head weights trainable; vision frozen;
- 752,393,024 / 852,985,920 parameters trainable (88.21%);
- 60 optimizer steps;
- 240 schedule units: 120 acquisition and 120 replay;
- 180 Boolean-pair units and 60 recall units;
- initial/final dev loss: 2.394504 / 0.292759;
- H100 compute: 217.61 seconds;
- estimated dispatch cost: USD 0.365;
- checkpoint SHA-256:
  `0037e2e25837c0ea7e36766f68c487b80853b6051c0ae382b399a59701e2171a`;
- training metrics SHA-256:
  `f53ed0cfe52aae73411bbb2b33a9789bc936f413f1e872c4e07a84d9381cfeaa`;
- training receipt SHA-256:
  `d2de9cf7eeafac0ba54fad077dcaa8a8cb16422e4d98450d447bdeea17f263c1`.

The full checkpoint is preserved in immutable B2 run storage. It is a rejected
candidate, not an active or promoted model.

## VERIFY result

Held-out stability run:
`delta-v2-f1-full-language-stability-eval-modal-v1`

The frozen bank contains 15 source claims and four same-claim/new-wording probe
surfaces, with two deterministic repeats per probe. Exact prompt overlap with
training is zero.

| held-out cell | pinned base | full-language candidate | required | verdict |
|---|---:|---:|---:|---|
| choice correct | 0/15 | 11/15 | diagnostic | improved |
| choice parseable | 0/15 | 15/15 | at least 14/15 | pass |
| Boolean true | 1/15 | 13/15 | at least 12/15 | pass |
| Boolean false | 14/15 | 10/15 | at least 14/15 | **fail** |
| exact recall | 0/15 | 0/15 | advisory | no gain |

Evidence:

- metrics SHA-256:
  `cfb54dbb3f7010c5ca6ab4be55142a27deef31e8df2324a6e7a64a0512f25d69`;
- samples SHA-256:
  `212e5fefeeabba78398306447ec6b92639ee5c5c836212da275ce752a3370f13`;
- evaluation compute: 109.03 seconds on A10;
- estimated dispatch cost: USD 0.048.

Raw-sample inspection confirmed model errors, not parser or scorer errors. Five
false statements received deterministic `yes` answers. The conjunctive
false-rejection gate correctly blocked the candidate.

## Exact training-surface diagnostic

Run:
`delta-v2-f2-full-language-train-surface-diagnostic-modal-v1`

This zero-update run replayed all 282 distinct rows actually referenced by the
training schedule, twice each. It did not mutate the checkpoint.

| exact training cell | observed | minimum | verdict |
|---|---:|---:|---|
| exact recall | 20/36, 55.56% | 80% | **fail** |
| verified true | 110/123, 89.43% | 95% | **fail** |
| verified false | 119/123, 96.75% | 95% | pass |

Evidence:

- metrics SHA-256:
  `b3aa9ba397115d777436b5fa12a987e9bc1a21f9a578c4514a0a738e459e1424`;
- samples SHA-256:
  `3a8d378ef4287926f105f43ab3b0db741cb64f5abced36bcf96415b4f9cdb4e7`;
- receipt SHA-256:
  `f4fa6ef8bce18aa57d3becc980391f5e97c0cc55783cf837a229ff5cdbc5f6d7`;
- compute: 275.70 seconds on A10G;
- estimated successful-run cost: USD 0.1015.

The first F2 dispatch stopped in preflight before model loading because six
frozen d20 evidence files were missing from the worker input declaration. It
performed zero model invocations, cost about USD 0.013, and produced no
scientific result. Commit `415cfe8` added the exact hashed evidence bindings;
the retry above completed.

## Honest conclusion

Full fine-tuning is an update mechanism, not a complete solution. Unlocking 88%
of the model produced real knowledge signal: held-out choice improved from
0/15 to 11/15 and held-out true claims from 1/15 to 13/15. Persistent weight
adaptation is therefore not disproven at this scale.

The candidate nevertheless exposed two independent failures:

1. it did not fully fit its own exact recall and verified-true training
   surfaces;
2. false rejection was 119/123 on familiar training wording but only 10/15 on
   held-out wording, a severe wording-transfer/calibration gap.

This is not evidence that LoRA is useless, and it is not evidence that full
fine-tuning works. The current narrow prompt bank plus plain generative SFT is
insufficient even when update capacity is broad. Merely repeating the identical
schedule longer is not justified.

The complete audit is
`artifacts/reports/delta_v2/full_language_sft_oracle_modal_v1.md`.

## Next controlled intervention when the user returns

Stay at the simplest mechanism: fresh-base full-language supervised training.
Change supervised coverage before adding adapter or RL complexity.

The next candidate should be a new, separately preregistered condition that:

1. uses the same 36 verified source claims and pinned base revision;
2. trains each claim on multiple independently worded recall, choice, true,
   and false surfaces;
3. keeps true and false examples balanced per claim;
4. reserves disjoint wording templates and corruption families for VERIFY;
5. has zero exact prompt overlap between train and held-out banks;
6. freezes an early-stop rule that first requires exact training-surface fit;
7. retains the same held-out Boolean-false minimum of 90%;
8. reports recall, choice, true, and false separately with no pooled score;
9. starts from the pinned base, never from rejected F1;
10. remains candidate-only with rollback to the pinned base.

Recommended order:

1. freeze and audit the multiwording data manifest without model calls;
2. freeze the fresh-base full-language SFT contract and resource budget;
3. freeze exact training-surface and held-out stability gates;
4. commit only non-overlapping files;
5. launch one training candidate;
6. run the exact training-surface gate;
7. only if it passes, run held-out stability VERIFY;
8. only if stability passes, authorize acquisition-margin and full evaluation.

Do not launch LoRA, QLoRA, full-model vision training, or verifier-grounded RL
as part of this continuation. Each remains blocked until its own contract and
safety/verification gates are frozen. Do not train on the held-out F1 stability
prompts; that bank remains evidence, not remediation data.

## Durable artifacts and platform notes

All run evidence and model artifacts live under immutable B2
`runs/<run_id>/` prefixes. Local `runs/` files are a disposable cache. The F1
checkpoint is large and need not be downloaded locally; Modal can pull it
directly from B2 with exact SHA-256 bindings.

This Codex worktree has no `.env`. For authorized dispatch or B2 operations,
the credential file used in this session was:

```text
/Users/lalithnarayanc/lalith-ai-lab/.env
```

Export it before child processes:

```bash
set -a
source /Users/lalithnarayanc/lalith-ai-lab/.env
set +a
```

Keep the B2 endpoint trimmed. A missing export causes `s5cmd` to hang during
credential/region discovery.

The worktree is detached. Successful checkpoints were pushed explicitly with:

```bash
git push origin HEAD:main
```

Never stage the user-owned documentation listed above. Check the remote before
pushing because another task may have advanced `main` during the pause.

## Recent checkpoints

```text
68e2c75 Record full-language VERIFY result
415cfe8 Declare diagnostic evidence inputs
e379c1d Preregister full-language training-surface diagnostic
398781c Preregister full-language stability VERIFY
154dd00 Preregister DELTA full-language SFT oracle
ca11bec Record DELTA token-local diagnostic result
28adf9a Preregister DELTA token-local choice diagnostic
42a15f1 Stop DELTA stability LoRA recipe lineage
```

At pause:

- no compute job is running;
- no model is promoted;
- the pinned base is the only active state;
- F1 and all LoRA candidates are preserved negative evidence;
- the next action is data/contract design, not an automatic compute launch.
