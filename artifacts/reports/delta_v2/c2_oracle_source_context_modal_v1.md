# delta_v2 c2 oracle source context — Modal v1 audit

Status: `implemented`

This report records an oracle-routed source-context upper bound. It is not a
production retrieval result.

## run identity

- run: `delta-v2-c2-oracle-source-context-qwen35-08b-modal-v1`
- condition: `c2_oracle_source_context`
- harness commit: `42e7107`
- config SHA-256:
  `4d8be7bccfb42a3ce1c760a1acde0daf336636e37fa22c995d3a4348fc9e2596`
- protocol SHA-256:
  `c16136937d87bb6b51269ae39f6f4ab5f4e37fa6ead48aa49a0d9239294c322a`
- request SHA-256:
  `b4b39f0e325b502d3cf8fb7d5e7fb9329b92ce1301d26e4f2758619c91a98023`
- sample SHA-256:
  `b292ac14f618f29bf073067cb5405d8cbaf338104fa3d8d230c304ebecb740e3`
- B2 prefix:
  `runs/delta-v2-c2-oracle-source-context-qwen35-08b-modal-v1/`

Each fact prompt received only:

```text
semantic_key
before_present
before_canonical_sha256
after_present
after_canonical_sha256
```

The exact `EvalItem.source_id` selected the frozen fact record. The gold
status, raw canonical values, provenance, verifier, and scorer were excluded.
This exact routing is why the condition is an oracle upper bound.

The feature prompt received eight verified `observed_after` booleans from the
frozen executable probe and no expected/gold matrix.

## readout

| stratum | correct | items |
|---|---:|---:|
| added atomic facts | 0 | 21 |
| stable atomic facts | 0 | 21 |
| added verified feature | 1 | 1 |

All 42 fact responses were parseable and all were exactly:

```text
changed
```

Repeats remained byte-identical. The feature remained exact.

## decision

The model can consume the compact verified feature booleans, but it does not
reliably compare two 64-character canonical hashes. Even identical before/after
hashes were rendered as `changed`.

One final representation control is justified: keep the same frozen source
records and oracle routing, but deterministically reduce fact evidence to
`before_present`, `after_present`, and `canonical_equal`. This tests the label
mapping without asking a 0.8B model to compare long hashes.

No training or fine-tuning is justified by c2.
