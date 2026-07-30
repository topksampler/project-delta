# delta_v2 c7 2:1 LoRA — Modal v1 audit

Status: `implemented`; training-surface gate failed, full evaluation blocked.

c7 tested the preregistered midpoint between the c5 1:1 and c6 4:1
verified-true:false virtual sampler weights. It changed only the verified-true
weight from c6's `4` to `2`. Training restarted from the pinned fresh base; the
63 source rows, LoRA topology, optimizer, seed, runtime, and verification gates
were unchanged. No evaluation wording was added to training.

## training

- run:
  `delta-v2-c7-knowledge-lora-2to1-qwen35-08b-modal-v1`
- launch commit: `1c7458d`
- adapter SHA-256:
  `89f79fe8763e3184ef3d701ac42afa657769f7bf0d5a09df920b9d32d3544549`
- protocol SHA-256:
  `549550c953b78b23537308cec0132cb4590e34bfb1a09ce2ca76f1d2b7b57a7f`
- config SHA-256:
  `5e74c48ab422a4865c29850ebd600bf37ad26b948e175b26970a95428710dfd1`
- metrics SHA-256:
  `033bf5024879b421cb1a929aa7320f9f3d6f18603c044dded888f91a88ac646f`
- realized exposures: 120 true, 62 false, 58 recall
- initial/final recall-dev loss: 2.394504 / 0.150161
- model work: 144.34 seconds on NVIDIA H100
- estimated dispatcher cost: USD 0.1839
- B2 prefix:
  `runs/delta-v2-c7-knowledge-lora-2to1-qwen35-08b-modal-v1/`

The adapter, metrics, config, protocol, train, and dev hashes were recomputed
after pulling the uploaded artifacts and matched the run receipt.

## conditional training-surface gate

- run:
  `delta-v2-c7-knowledge-lora-2to1-train-surface-modal-v1`
- launch commit: `27c97c3`
- protocol SHA-256:
  `02a7addc93be91a072ad0afed73381fb181a8b85bf45fa7d482d3ee614400696`
- config SHA-256:
  `8fca75336a4920edb4156dce8e7c24c11902fa69ab074de1e9b1d743029a596f`
- samples SHA-256:
  `1cc412a49178a78e72022034a7a93d326d270df17a58379b4fd1b3333810c23c`
- metrics SHA-256:
  `c93c99c105f88ccf7990e3c75899a3f6f673939380ab078d0ff10e918890f3c1`
- deterministic repeats: `true`
- model work: 155.43 seconds on NVIDIA A10
- estimated dispatcher cost: USD 0.0526
- B2 prefix:
  `runs/delta-v2-c7-knowledge-lora-2to1-train-surface-modal-v1/`

| immutable training surface | c5, 1:1 | c7, 2:1 | c6, 4:1 |
|---|---:|---:|---:|
| verified true | 0/21 | 21/21 | 21/21 |
| deterministic false | 21/21 | 1/21 | 0/21 |
| source-card recall | 9/21 | 6/21 | 8/21 |

The frozen boolean gate required at least 17/21 on both verified true and
deterministic false. c7 missed the false gate by 16 items. It therefore did not
qualify for the unchanged 169-item evaluation.

## interpretation and decision

Under this fixed base, data, optimizer, and seed, the answer-prior transition
occurs between the tested 1:1 and 2:1 ratios. c5 learned an always-`no` policy;
c7 learned a near-always-`yes` policy; c6 learned an always-`yes` policy. None
learned the truth-conditioned boolean distinction, so sampler weighting alone
has not produced a viable candidate.

The training loss and recall-dev loss are not substitutes for the failed
surface-stratified generation gate. c7 is a negative result. The pinned base
remains active, promotion is unauthorized, and no full frozen evaluation was
run. Any further LoRA objective, QLoRA, full-weight, or reinforcement candidate
requires a separate preregistered contract rather than an automatic ratio or
step sweep.

Total estimated dispatcher cost for c7 training plus its gate was USD 0.2365.

## local dispatch preflight note

Before the successful training dispatch, two cockpit attempts stopped before
remote job creation: the delegated worktree first lacked the Modal CLI, then
its new ignored client environment lacked PyYAML. Before each retry, the B2
prefix was checked and contained no worker receipt, adapter, or model output.
The identical committed config and run ID were retried only after the exact
Modal app imported locally. These were platform-start failures with zero model
or optimizer work, not additional candidates.
