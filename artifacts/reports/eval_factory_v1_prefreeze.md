# Eval factory v1 — pre-freeze result

Transition: vLLM v0.22.0 → v0.23.0
Protocol: `e1_eval_factory_v1`
Status: **pre-freeze; do not use for model evaluation**

## invariant

```text
The target model never defines truth.
Failed gates block model runs.
Surface-generation failures remain visible; gates are not relaxed to pass.
```

## executable census

AST inspection of every literal `add_argument("--...")` under each pinned
`vllm/` checkout produced:

| measure | count |
|---------|------:|
| verified CLI claims | 378 |
| stable | 368 |
| added in v0.23.0 | 9 |
| changed contract | 1 |
| removed | 0 |
| generated seed probes | 1,134 |
| frozen-split candidate probes | 255 |

The preregistered expectation of at least 40 changed CLI claims is false for
this transition. The observed executable CLI delta population is 10:

```text
added    --chat-template-kwargs
added    --custom-ensure-client-side-data
changed  --dataset-name
added    --self-timed
added    --timed-trace-chunk-hash-size
added    --timed-trace-label-hash-ids
added    --timed-trace-label-input-length
added    --timed-trace-label-output-length
added    --timed-trace-label-timestamp
added    --timed-trace-sec-multiplier
```

The gate remains failed. The next build must broaden executable claim families
or accumulate delta claims across the four-transition matrix; it may not
manufacture more v0.22→v0.23 CLI drift.

## surface-generation experiments

### v1 — lexical pass, semantic fail

Run: `e1-vllm-eval-factory-v1-surface-gen-qwen35-4b-modal`

```text
teacher       Qwen/Qwen3.5-4B
seed          255
accepted      231 (90.6%)
cost          $0.0509
```

Lexical contamination improved: maximum train/eval token Jaccard fell from
0.9444 to 0.5789. Manual inspection found stable delta questions rewritten as
“How did X change?”, which presupposes drift. v1 is rejected despite passing
the lexical gate.

### v2 — neutral semantic gate

Run: `e1-vllm-eval-factory-v1-surface-gen-v2-qwen35-4b-modal`

```text
teacher accepted                146
presupposing delta rejected      85
other malformed rejected         24
neutral deterministic fallback   85
final surfaces                   231 / 255 (90.6%)
cost                              $0.0416
```

The fallback asks neutrally whether the flag stayed the same or differed; it
does not receive or change gold. Final maximum train/eval Jaccard is 0.4583.

## gate status

Passing:

- 378 executable claims (minimum 100);
- identical verifier output on immediate rerun;
- artifact hashes, claim uniqueness, and claim-level split isolation;
- zero exact train/eval overlap;
- zero train/eval pairs at Jaccard `>= 0.85`;
- independent surfaces present with 90.6% coverage.

Blocking:

- changed executable claims: 10 / preregistered 40;
- human audit: 0 / 50 completed.

`freeze_ready=false`.

## amendment A — multi-family executable environment

The preregistered expansion added three families without lowering the 40-delta
gate:

| family | claims | stable | added | changed | removed |
|--------|-------:|-------:|------:|--------:|--------:|
| CLI flags | 378 | 368 | 9 | 1 | 0 |
| typed config fields | 442 | 422 | 0 | 18 | 2 |
| environment variables | 228 | 216 | 7 | 0 | 5 |
| public exports | 22 | 22 | 0 | 0 | 0 |
| **total** | **1,070** | **1,028** | **16** | **19** | **7** |

Config status excludes adjacent-description-only edits. Pydantic `Field`
wrappers are normalized into default plus constraints; newly added constraints
count as executable contract changes. The resulting delta population is 42,
passing the preregistered minimum by two.

The first global split put only 6/42 deltas in eval. Before any target-model
run, the protocol was amended to stratify delta claims per family/status at
40% train / 20% dev / 40% eval. Final delta split:

```text
train 17
dev    8
eval  17
```

The typed manifest is now `delta.eval_environment.v1`.

Changed-contract probes require the actual difference, not the word
“changed”: CLI choice additions/removals, config defaults, type annotations,
or validation constraints. The final eval contains 17 delta claims; 17 remain
available for train candidates and 8 for development.

### multi-family surfaces

Run:
`e1-vllm-eval-factory-v1-surface-gen-v3-multifamily-qwen35-4b-modal`

```text
seed surfaces                 657
teacher accepted directly     158
recovered by deterministic
  cleanup/revalidation        438
neutral delta fallback         29
final surfaces                623 (94.8%)
unresolved                     34
max train/eval Jaccard        0.55
cost                          $0.0638
```

The raw teacher acceptance of 25.3% was a validator failure: valid questions
with a trailing “answer yes/no” instruction and neutral “did X change, and if
so how?” wording were rejected. Raw candidates were preserved; deterministic
cleanup and the corrected neutral gate recovered them without another teacher
run or any gold changes.

All automatic freeze gates pass. The only remaining blocker is the
preregistered 50-claim independent/human audit.

## what goes where

- protocol/code: `experiments/e1_vllm/eval_factory/`;
- candidate local data:
  `data/experiments/e1_vllm/eval_factory/v0.22.0_to_v0.23.0/e1_eval_factory_v1/`;
- teacher evidence: B2 `runs/e1-vllm-eval-factory-v1-surface-gen-*/`;
- durable pre-freeze interpretation: this report.

## what can die

- v1 generated surfaces;
- rejected v2 candidates;
- local copies of pre-freeze JSONL after the next deterministic rebuild.

## what must survive

- protocol and code revision;
- 378/10 structural census;
- both teacher run IDs, raw rejects, metrics, and costs;
- the fact that semantic inspection rejected v1;
- final validation output and the unfinished audit packet.

## command

```bash
python -m experiments.e1_vllm.eval_factory.cli build \
  --before doc_0 --after doc_8

python -m experiments.e1_vllm.eval_factory.cli validate \
  --artifact-dir \
  data/experiments/e1_vllm/eval_factory/v0.22.0_to_v0.23.0/e1_eval_factory_v1
```
