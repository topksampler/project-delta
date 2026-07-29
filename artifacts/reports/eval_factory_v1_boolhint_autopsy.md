# Boolhint autopsy — Yes→No polarity flip

Runs compared:
- no hint: `e1-vllm-eval-factory-v1-base-qwen35-08b-modal`
- hint: `e1-vllm-eval-factory-v1-base-boolhint-qwen35-08b-modal`

## invariant

```text
The boolean scorer only reads an explicit yes/no token.
If the model output flips, that is generation bias — not a scoring bug.
```

## what happened

On **404** boolean probes:

| transition (first yes/no token) | n |
|--------------------------------|--:|
| no → no | 212 |
| **yes → no** | **154** |
| yes → yes | 37 |
| no → yes | 1 |

All 154 flips were on `expected=yes`. Every flipped no-hint answer started with `Yes`; every flipped hint answer started with `No`.

For `expected=yes` (n=402):

| | no hint | + `Answer yes or no.` |
|--|--------:|----------------------:|
| says yes | 47.5% | **9.5%** |
| says no | — | **90.5%** |
| exact | 0.475 | 0.095 |

Hint helped **1** probe.

## mechanism (best supported)

Not leakage, not gold error. The eval-time suffix ends with the word **`no`**. On Qwen3.5-0.8B this bank, that is enough to flip affirmative answers to leading `No`. Compatible with last-option / negation priming. Scorer then correctly marks them wrong.

## what that implies

- Do **not** enable `boolean_answer_hint` as default for 0.8B on this bank.
- If a format hint is needed later, try forms that do **not** end in `no` (e.g. `Reply with Yes or No as the first word.` / `Start with Yes.` for existence-true items) and A/B them — do not assume “yes or no” is neutral.
- 4B’s earlier format tax (missing token) is a different failure mode; this autopsy is 0.8B-specific.

## paths

- `artifacts/reports/eval_factory_v1_boolhint_autopsy.json`
- this file

## what can die

- `boolean_answer_hint: true` on factory 0.8B configs

## what must survive

- the yes→no flip counts and the paired run_ids
