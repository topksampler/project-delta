# Version-delta eval factory pilot

Protocol ID: `e1_eval_factory_v1`
Decision date: 2026-07-28
Primary transition: vLLM `v0.22.0` → `v0.23.0`

## invariant

```text
The target model never defines truth.
Every scored claim is bound to a pinned source revision.
Executable claims must pass a deterministic verifier in both revisions.
Generated wording may vary; claim identity and gold may not.
Train/eval separation is enforced before artifacts are emitted.
No intervention run starts until this pilot passes its freeze gates.
```

This pilot tests the evaluation factory, not Qwen3.5-0.8B and not an
intervention. A model score cannot rescue a bad claim bank.

## bet and falsifier

**Bet:** a deterministic extraction and executable-verification layer,
followed by independently generated probe surfaces, can replace the 46-item
hand eval as DELTA's primary version-delta instrument without lowering truth
quality.

The bet fails for this transition if any freeze gate fails:

1. fewer than 100 executable claims survive across the two revisions;
2. fewer than 40 verified changed/added/removed claims survive;
3. executable verifier reruns disagree on any claim;
4. any eval question has exact or token-Jaccard `>= 0.85` overlap with train;
5. a 50-item stratified human audit finds less than 90% agreement on
   claim truth, version label, and answerability;
6. more than 10% of generated probes need manual gold repair.

If changed claims are naturally fewer than 40, the gate may be revised only
by recording the observed structural population before seeing model results.

## unit of truth

The factory emits claims before probes:

```json
{
  "claim_id": "vllm:cli_flag:serve:--max-model-len",
  "entity_type": "cli_flag",
  "entity": "--max-model-len",
  "scope": "serve",
  "source_before": "v0.22.0",
  "source_after": "v0.23.0",
  "status": "stable|added|removed|changed",
  "value_before": {"exists": true, "default": null, "help": "..."},
  "value_after": {"exists": true, "default": null, "help": "..."},
  "evidence_before": [{"path": "...", "line_start": 1, "line_end": 2}],
  "evidence_after": [{"path": "...", "line_start": 1, "line_end": 2}],
  "verifier": {"id": "python_ast_argparse_v1", "result": "pass"}
}
```

Claim IDs are semantic and stable across probe wording. Source evidence uses
repository-relative paths plus line ranges and revision hashes.

## claim families

Pilot priority:

1. CLI flag existence, default, choices, and help meaning;
2. environment-variable existence and documented meaning;
3. importable API symbol and signature presence where import is safe;
4. documented config field presence and default where a typed schema exists;
5. documentation path addition/removal/move.

### executable-family inclusion rules (amendment A, before multi-family census)

```text
cli_flag      literal --name in an add_argument call anywhere under vllm/
env_var       VLLM_* key in vllm.envs.environment_variables runtime registry
public_export literal key→target in vllm.MODULE_ATTRS
config_field  public annotated field on a @config class under vllm/config/
```

`ClassVar`, private fields, unresolved non-lambda environment registrations,
and symbols outside these registries are excluded with reason codes. A contract
is compared after AST normalization and excludes file path and line number:

- CLI: scope, default, choices, required, action, help;
- environment: resolver expression and statically visible default;
- public export: lazy import target;
- config field: containing class, annotation, default, and `Field`/`field`
  constraints. Adjacent description is retained as evidence but does not change
  executable status by itself.

Each family is reported separately by `stable|added|removed|changed`. The
40-delta freeze gate applies to their union, but no family may be pooled away
in reports. This amendment broadens the preregistered executable population;
it does not lower the gate after observing the CLI-only count of 10.

Executable verification means AST/static inspection or isolated import/CLI
introspection against the pinned checkout. Regex-only extraction may propose a
claim but may not mark it executable.

Free-form architectural or conceptual claims may be emitted as
`corpus_grounded`, but they are not part of the primary executable score.

## split before surface generation

Claims, not questions, are assigned to splits using a stable hash. Stable
claims use:

```text
60% train-candidate claims
20% development claims
20% frozen evaluation claims
```

Scarce `added|removed|changed` claims are stratified by family and status at
40% train / 20% development / 40% evaluation, with a one-item stratum assigned
to evaluation. This prevents the version-delta eval from becoming a
stable-knowledge benchmark while retaining delta train candidates.

All probes for one `claim_id` remain in one split. Split assignment is
independent of model failures. This amendment was made after the pre-model
split audit found only 6/42 delta claims in eval under a global 60/20/20 hash.
The old `eval_v3` remains an external historical comparison and is never used
to select the new eval claims.

## generated probes

Each claim receives three deterministic seeds:

1. binary existence in the before revision;
2. binary existence in the after revision;
3. one neutral cross-version delta question.

For `changed` claims, the delta seed asks for the actual executable difference
(choice addition/removal, default, type annotation, or validation constraint)
and gold encodes the before/after contract. A bare answer such as “it changed”
does not pass. Added/removed claims remain version-boundary existence tests.

Frozen eval wording replaces those seeds with independently generated surfaces
where the surface gate passes. A neutral eval-only template may replace a
rejected delta surface; it never receives gold beyond the structured claim.
Unresolved existence rewrites are omitted and counted against the 90% coverage
gate.

Generators may only transform a structured claim into wording. They may not
invent gold. Every emitted row retains `claim_id`, source revisions, evidence,
generator ID, prompt hash, and seed.

Primary task remains 1–3 sentence free-form generation. Choice and boolean
forms are auxiliary diagnostic probes, not substitutes for the generative
eval.

## scorer hierarchy

1. deterministic executable/value match where output is structured;
2. exact choice/boolean parsing for diagnostic forms;
3. rubric scoring against claim-derived required and forbidden facts;
4. independent semantic grader only when deterministic scoring is
   insufficient;
5. human audit as the final calibration layer.

The semantic grader receives the structured claim and evidence, not the
target model's rationale. Grader model, prompt, temperature, seed, and raw
judgment must survive.

## controls

- duplicate probe wording with shuffled entity names to detect template bias;
- stable claims, changed claims, added claims, and removed claims reported
  separately;
- positive and negative/boundary probes balanced where feasible;
- old `eval_v3` reported separately and never pooled into the new score;
- base-model run repeated with three generation seeds only after the eval
  artifact freezes;
- one larger-model instrument check, not treated as ground truth.

## primary outputs

```text
claims.jsonl              all proposed claims
claims_verified.jsonl     claims passing deterministic verification
claims_rejected.jsonl     failures with reason codes
probes_dev.jsonl          development surfaces
probes_eval.jsonl         frozen generative + diagnostic eval
manifest.json             revisions, hashes, generators, split seed, counts
audit_50.jsonl            stratified human audit sample and judgments
validation.json           reproducibility, leakage, balance, audit results
```

Headline metrics after freeze:

- knowledge recovery by drift type;
- version-attribution accuracy;
- semantic-boundary honesty (accept true / reject false separately);
- calibration or abstention quality;
- regression on stable claims;
- measured inference and intervention cost.

No single pooled accuracy is sufficient for promotion.

## what goes where

- protocol and factory code:
  `experiments/e1_vllm/eval_factory/`;
- authoritative source/model pins:
  `experiments/e1_vllm/snapshots.yaml`;
- candidate local artifacts:
  `data/experiments/e1_vllm/eval_factory/{transition}/{protocol_id}/`;
- frozen datasets:
  B2 `datasets/experiments/e1_vllm/eval_factory/{transition}/{protocol_id}/`;
- experiment hypothesis and condition matrix:
  `docs/experiments/e1_vllm.md`;
- phase status:
  `docs/delta-roadmap.md`;
- run evidence:
  B2 `runs/{run_id}/`.

## what can die

- unverified claims;
- generated surfaces rejected by grounding or leakage gates;
- local pinned checkouts and import environments;
- teacher/grader candidates before their prompts and raw outputs are frozen;
- the 100/40 target if the pre-model structural census proves it impossible
  and the protocol records why.

## what must survive

- both source revisions and content hashes;
- claim IDs, executable verifier version, and evidence lineages;
- rejected claims with reason codes;
- split assignment, generator prompts, seeds, and hashes;
- frozen eval plus contamination report;
- audit judgments and disagreements;
- every later intervention's per-sample outputs and promotion decision.

## pilot command

Target interface (implemented by this pilot):

```bash
python -m experiments.e1_vllm.eval_factory.cli build \
  --snapshots experiments/e1_vllm/snapshots.yaml \
  --before doc_0 \
  --after doc_8 \
  --protocol e1_eval_factory_v1

python -m experiments.e1_vllm.eval_factory.cli validate \
  --artifact-dir \
  data/experiments/e1_vllm/eval_factory/v0.22.0_to_v0.23.0/e1_eval_factory_v1
```

The artifact freezes only when `validation.json` reports every mandatory gate
as passed. Model evaluation is downstream of that event.
