# delta_v2 first no-training ADAPT matrix

Status: `implemented`

This report is the MEMORY handoff for four completed Modal runs against the
same frozen `delta-v2-vllm-source-build-v2` acceptance environment. It compares
only no-weight-update interventions. Fine-tuning remains deferred.

## conditions

| condition | change from c0 | role |
|---|---|---|
| c0 base | no examples or source context | frozen target baseline |
| c1 task calibration | balanced development-train label examples and one feature-format example | test instruction/label calibration |
| c2 oracle source context | exact fact routing with presence and canonical hashes; verified feature booleans | source-evidence upper bound |
| c3 source comparison | exact fact routing with presence and canonical-equality booleans | representation upper bound |

Neither oracle condition makes a production retrieval claim. No acceptance
gold label, fact status field, scorer, or verifier was exposed to the model.

## verified readout

| condition | added facts | stable facts | feature | fact response distribution |
|---|---:|---:|---:|---|
| c0 base | 15/21 | 0/21 | 0/1 | 33 added, 9 changed |
| c1 task calibration | 5/21 | 0/21 | 1/1 | 14 added, 26 changed, 2 removed |
| c2 oracle source context | 0/21 | 0/21 | 1/1 | 42 changed |
| c3 source comparison | 11/21 | 0/21 | 1/1 | 11 added, 31 changed |

All completed conditions used the same exact model revision, one NVIDIA A10,
greedy generation, two repeats, and deterministic offline scoring. Every repeat
was byte-identical. There is deliberately no pooled overall score.

## what the matrix says

The feature result is the positive control. A single development feature
example fixed c1's response form, and verified source booleans produced the
exact feature object in c2 and c3. The model can use a compact truth-rooted
behavior representation for this feature task.

The fact result is a negative control with a stable failure:

1. c0 exposed a strong `added` output bias.
2. c1 changed the bias but never produced `stable`.
3. c2 removed missing knowledge by oracle-routing frozen source facts; all
   facts became `changed`.
4. c3 removed long-hash comparison as a difficulty; stable remained 0/21.

Therefore the first fact task does not measure source-change understanding for
this target model. Its capability gate fails under every tested
representation. An added-fact success or failure from these conditions must
not be interpreted as evidence that the model does or does not know the new
source.

## DELTA loop handoff

| stage | state for this slice | evidence |
|---|---|---|
| SENSE | `implemented` | exact source snapshots, full file inventory, preregistered fact families, independently extracted facts, PR provenance |
| BUILD | `implemented` | promoted verified feature, executable old/new probes, split acceptance items, frozen environment |
| DECIDE | `partial` | one target/model revision evaluated; capability gate failed, so there is no promotion decision |
| ADAPT | `partial` | task calibration and two oracle controls completed; no training branch opened |
| VERIFY | `implemented` for these runs | immutable requests, exact runtime locks, two-repeat determinism, offline exact scorers, run receipts |
| MEMORY | `implemented` for these runs | immutable B2 prefixes and committed audit reports preserve successes, failures, and negative results |

SENSE and BUILD are complete for this vertical slice, not for the whole DELTA
program. DECIDE and ADAPT remain partial because this matrix covers one small
target model and intentionally stops before weight updates or promotion.

## next rational contract

The source pipeline should continue to compute atomic-fact status
deterministically. The next model-facing fact contract should ask about the
verified value, constraint, meaning, or intended use and score it against
frozen code/documentation evidence. An LLM may propose wording or perform a
truth-anchored semantic review, but it must not manufacture the gold fact or
override a deterministic verifier.

A production retrieval condition and a stronger target/judge comparison are
future preregistered branches. They should use the redesigned fact contract,
not repeat this failed status-label task.

## execution accounting

- completed model runs: 4
- failed pre-model attempts: 2
- estimated completed-run cost: USD 0.0978
- estimated total cost including failed attempts: USD 0.1675

The costs are timing-times-list-rate estimates from each run receipt. No
fine-tuning, training, promotion, or push occurred.
