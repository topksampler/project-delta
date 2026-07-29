# delta_v2 c1 task calibration — Modal v1 audit

Status: `implemented`

This report records the first no-weight-update `ADAPT` condition after the c0
baseline failed its stable-control capability gate.

## run identity

- run: `delta-v2-c1-task-calibration-qwen35-08b-modal-v1`
- condition: `c1_task_calibration`
- model revision: `2fc06364715b967f1860aea9cf38778875588b17`
- harness commit: `93f92b3`
- config SHA-256:
  `b80df1ba823853291a8ef5dd2cc33c9ffa146e4cbf9c7ab9e170234643f88e52`
- protocol SHA-256:
  `6981b16d259b6b425d1516532a80ec629581dc87b0cb59f3f0f000d22d22eedd`
- request SHA-256:
  `96e209a30ca9b723f8e28e2af9c233f2a48b277c913bbb5da5bc4e7019ffbe77`
- sample SHA-256:
  `6b02d5342e57626d70e8a81dfe20adc8547c5ef4f28db3b3e693bac0005de58d`
- B2 prefix:
  `runs/delta-v2-c1-task-calibration-qwen35-08b-modal-v1/`

The model saw eight development-`train` fact demonstrations—two per change
label—and one development-`train` feature demonstration. It saw no acceptance
gold, provenance, scorer, source evidence, or development-`dev` item.

## readout

| stratum | c0 | c1 | change |
|---|---:|---:|---:|
| added atomic facts | 15/21 | 5/21 | -10 |
| stable atomic facts | 0/21 | 0/21 | 0 |
| added verified feature | 0/1 | 1/1 | +1 |

All repeats were byte-identical and all 42 fact responses were parseable.

The c1 fact response distribution was:

```text
changed 26
added   14
removed  2
stable   0
```

Balanced demonstrations shifted the bias from `added` toward `changed`, but did
not restore the stable capability control. The preregistered interpretation
therefore remains:

```text
baseline-incapable-of-localizing-drift
```

The feature result is a real positive control: the same model changed from a
Markdown-fenced, incorrect matrix to the exact six-key verified JSON object.
This shows that in-context response-format and behavior-example adaptation can
work even though fact change-status calibration did not.

## next DELTA action

Do not train on this result. Run an oracle-routed source-context upper bound
using the frozen before/after fact observations and verified feature probe,
while excluding the gold status field. This will separate “cannot infer the
source truth unaided” from “cannot use source truth when supplied.”
