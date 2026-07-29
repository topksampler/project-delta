# delta_v2 c0 base — Modal v3 audit

Status: `implemented`

This report records the first completed target-model run against the frozen
`delta-v2-vllm-source-build-v2` acceptance environment. It is evidence for the
DELTA `DECIDE` stage, not a promotion decision.

## run identity

- run: `delta-v2-c0-base-qwen35-08b-modal-v3`
- condition: `c0_base`
- model: `Qwen/Qwen3.5-0.8B`
- model revision: `2fc06364715b967f1860aea9cf38778875588b17`
- harness commit: `3fb5548`
- config SHA-256:
  `a1545dfa36db1f42689d3644a2997a793b040e963048fe64d96ab7fdfdd77975`
- protocol SHA-256:
  `e26985e78b94365123cb8f4c0d1cf34d377d6b488f70f7ca618946336eb9b98d`
- request SHA-256:
  `3e81ecae7fa8298431b50274fa32e417e96640293a401458f91c083f56d52c18`
- sample SHA-256:
  `605482c56c7482171670db72ea5d04975106d7575cb0fc4863444dab7e2e8696`
- B2 prefix:
  `runs/delta-v2-c0-base-qwen35-08b-modal-v3/`

The pinned runtime passed before model access: Python 3.11.9, Torch
2.10.0+cu128, Transformers 5.14.1, one NVIDIA A10, bfloat16, eager attention,
and deterministic algorithms enabled.

## preregistered readout

| stratum | correct | items | exact accuracy | parseable |
|---|---:|---:|---:|---:|
| added atomic facts | 15 | 21 | 0.7143 | 21/21 |
| stable atomic facts | 0 | 21 | 0.0000 | 21/21 |
| added verified feature | 0 | 1 | 0.0000 | 0/1 |

The two greedy repeats were byte-identical for all 43 items. There is no pooled
overall accuracy by protocol.

The preregistered interpretation is:

```text
baseline-incapable-of-localizing-drift
```

The stable-control test cannot reject chance (`p=1.0`), so the added-versus-
stable comparison has no drift-localization authority.

## diagnostic read

For the 42 fact items, the model returned:

```text
added   33
changed  9
stable   0
removed  0
```

This is a severe output-label bias. The 15/21 added score therefore does not
show that the model understands new v0.26 facts; many answers are explained by
always preferring `added`. Likewise, an incorrect new-fact answer cannot yet be
called a source-knowledge gap because the model failed the stable capability
control.

The feature response used Markdown fences and assigned `not_loaded` to every
key. It was correctly treated as unparseable and incorrect without repair.

## failed pre-model attempts

- `...-modal-v1` stopped during image construction because Transformers 5.14.1
  requires `safetensors>=0.8.0`; no model invocation occurred.
- `...-modal-v2` stopped during worker import because the experiment app had
  not mounted shared worker plumbing; no model download or invocation occurred.

Both run IDs remain immutable failed attempts. Their timing and cost receipts
remain under their own B2 prefixes.

## next DELTA action

The evidence calls for task calibration before knowledge adaptation. The next
condition is a no-weight-update, development-only in-context calibration:
balanced train examples for the four fact labels and one train feature example
for plain-JSON formatting. Acceptance truth and provenance remain hidden.
