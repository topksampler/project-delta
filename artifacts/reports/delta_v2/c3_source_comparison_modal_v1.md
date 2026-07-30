# delta_v2 c3 source comparison — Modal v1 audit

Status: `implemented`

This report records the final no-weight-update representation control in the
first `delta_v2` ADAPT matrix. It is an oracle upper bound, not a production
retrieval result.

## run identity

- run: `delta-v2-c3-source-comparison-qwen35-08b-modal-v1`
- condition: `c3_source_comparison`
- model: `Qwen/Qwen3.5-0.8B`
- model revision: `2fc06364715b967f1860aea9cf38778875588b17`
- harness commit: `fb2ea68`
- config SHA-256:
  `76e4923f64569acefb5a3d185d659e2d5f3c44150d340e35f39a04de0f2a88a6`
- protocol SHA-256:
  `90a35c59997c3866dc89d7880ee650f6dc383231a5f1fbcdbde0a1c64edc4cae`
- request SHA-256:
  `080337920a74694131463385915ddc4237ff25071ac13eff2deb9fe746dbcab1`
- sample SHA-256:
  `d5cbe48c9fb6812d5947f33a6bed775260e144a27bc3b01aebf31fa2779b718d`
- metrics SHA-256:
  `5b67fa5b3487827dca25f2befc47422f33ba14977e9bdae46be9ed7bd38b4138`
- B2 prefix:
  `runs/delta-v2-c3-source-comparison-qwen35-08b-modal-v1/`

Each fact prompt received only three deterministic observations:

```text
before_present
after_present
canonical_equal
```

The exact `EvalItem.source_id` selected the frozen fact record. Canonical
values and hashes, semantic keys, gold status, provenance, verifier, and scorer
were excluded. The feature projection was unchanged from c2: eight verified
`observed_after` booleans and no expected/gold matrix.

## readout

| stratum | correct | items | exact accuracy |
|---|---:|---:|---:|
| added atomic facts | 11 | 21 | 0.5238 |
| stable atomic facts | 0 | 21 | 0.0000 |
| added verified feature | 1 | 1 | 1.0000 |

All 43 items were parseable, and all two-repeat outputs were byte-identical.
There is no pooled overall score by protocol.

The first-repeat fact distribution was:

```text
changed 31
added   11
stable   0
removed  0
```

The preregistered interpretation remains:

```text
baseline-incapable-of-localizing-drift
```

The stable-control test cannot reject chance (`p=1.0`), so the apparent
added-versus-stable difference has no drift-localization authority.

## decision

Reducing source evidence from long hashes to explicit comparison booleans did
not restore the stable capability control. This rules out missing source
retrieval and hash-comparison difficulty as sufficient explanations for the
fact-label failure in this target model and protocol.

The feature positive control remained exact. The model can render a compact
verified behavior observation into the requested user-facing feature schema,
but it is not a reliable judge of atomic-fact change status.

No additional prompt-only fact-status condition is justified. Change status is
already a deterministic consequence of the verified before/after records and
belongs in BUILD. A later model-facing fact task should test meaning or use
against that frozen truth, rather than asking the model to recreate version
control classification.

No training or fine-tuning was run or is justified by this result.
