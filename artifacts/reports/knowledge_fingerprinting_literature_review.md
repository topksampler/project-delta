# Topic knowledge profiling for Project DELTA

Literature review and protocol recommendation. Researched 2026-07-18.

## invariant

```text
We cannot read a model's weights and enumerate what it "knows."
We can estimate a knowledge boundary behaviorally, for a fixed model checkpoint,
prompt protocol, source revision, and claim bank.

The unit is an atomic, source-grounded claim—not a question and not a document.
A claim is "known" only when the model is correct, consistent across equivalent
probes, appropriately calibrated, and attached to the right version.
```

## nomenclature correction

`doc_0` and `doc_8` are opaque legacy machine aliases:

```text
doc_0 = vLLM docs at git tag v0.22.0 (before / stale snapshot)
doc_8 = vLLM docs at git tag v0.23.0 (after / fresh snapshot)
```

The repository only records that these names replaced earlier `sha_0` / `sha_8`
aliases; it does not document a meaningful reason for the number 8. They should
remain stable where existing run IDs and B2 paths require it, but all new
human-facing artifacts should say:

```text
source_before: vLLM v0.22.0 (legacy alias: doc_0)
source_after:  vLLM v0.23.0 (legacy alias: doc_8)
```

## executive conclusion

There is no mature, standard method called *topic knowledge fingerprinting* that
directly solves DELTA's problem. The closest mature literatures are:

1. factual knowledge probing;
2. knowledge-boundary and uncertainty estimation;
3. temporal / dynamic knowledge evaluation;
4. atomic factuality verification;
5. model editing and specificity evaluation;
6. executable repository benchmarks.

Their combined lesson is:

```text
source snapshot
  → typed atomic claim bank
  → multiple semantically equivalent probe families per claim
  → closed-book model responses + probabilities / samples
  → deterministic or executable verifier
  → per-claim posterior: known / partial / unknown / misattributed / unstable
  → aggregate TopicKnowledgeProfile
```

A single prompt and substring match is not evidence that a fact is known.
A consistent wrong answer is not detected by semantic entropy alone.
Self-reported confidence is useful but not sufficient.
Internal localization is research instrumentation, not the primary fingerprint.

## terminology: use `TopicKnowledgeProfile`, not `fingerprint`

“Behavioral fingerprinting” is increasingly used for model identification,
alignment-style comparison, and provenance/IP tracking. That is a different
question: *which model is this?* DELTA asks: *what does this fixed model reliably
know about this versioned subject?*

Recommended durable object:

```yaml
topic_knowledge_profile:
  schema: delta.topic_knowledge_profile.v1
  model_state: ...
  subject: vllm
  source_revision: v0.22.0
  protocol_version: ...
  claim_bank_hash: ...
  aggregate: ...
  claims:
    - claim_id: cli.tensor_parallel_size.meaning
      truth: ...
      valid_from: ...
      valid_to: ...
      evidence: ...
      status: known | partial | unknown | misattributed | unstable
```

“Fingerprint” may remain an informal visualization of this profile.

## what the literature establishes

### 1. Factual probing: one wording is not enough

**LAMA** introduced zero-shot cloze probes for relational facts and showed that
pretrained language models can recall some knowledge without task fine-tuning.
Its limitations are central for DELTA: cloze format, single-token objects,
template sensitivity, and entity-name shortcuts.

**ParaRel** tested meaning-preserving paraphrases and found poor factual
consistency across prompt formulations. Therefore, one successful phrasing is
only a lower-bound observation; a knowledge profile needs paraphrase families.

**PopQA / EntityQuestions** demonstrated a strong popularity/long-tail effect:
models know popular entities disproportionately well, while retrieval helps the
tail. Topic coverage must therefore be stratified by source frequency or
importance; random document sampling will give a misleading profile.

**FACT-BENCH** broadens factual recall across domains and properties. It also
reports that instruction tuning can reduce factual recall and that
counterfactual demonstrations can override known facts. DELTA must profile the
actual deployed/instruction-tuned checkpoint, under the exact system prompt—not
infer its knowledge from the base checkpoint.

Design implication:

```text
Each claim gets ≥3 controlled paraphrases and ≥2 probe forms.
Profile the deployed model state, not merely its base family.
Stratify claims by entity/type/importance, not only corpus frequency.
```

### 2. “Knowing what it knows”: confidence is a separate measurement

**Language Models (Mostly) Know What They Know** introduced `P(True)` for a
proposed answer and `P(IK)` for query-level self-knowledge. Larger models showed
encouraging calibration in suitable formats, but `P(IK)` generalized poorly to
new tasks. Self-evaluation is a feature, not ground truth.

**Semantic entropy** clusters sampled responses by meaning and measures
uncertainty over semantic answers. It predicts confabulation better than lexical
variation because multiple wordings may express the same answer. The Nature
paper explicitly warns that semantic entropy does not catch systematic,
confidently wrong beliefs.

**SelfCheckGPT** similarly uses disagreement among stochastic samples as a
black-box hallucination signal. It is useful when logits are unavailable, but
stable misconceptions remain a blind spot.

**SimpleQA** separates correct, incorrect, and not-attempted answers and measures
calibration. Its narrow short-answer/single-answer design is useful for atomic
claims, not sufficient for procedural or version-compositional knowledge.

The 2025 **Knowledge Boundary Survey** organizes methods into uncertainty
estimation, confidence calibration, and internal-state probing, while noting
that a single failed question does not establish a knowledge boundary.

Design implication:

```text
Correctness and uncertainty are separate axes.

correct + low entropy     → likely known
correct + high entropy    → partial / unstable
wrong + high entropy      → unknown / confabulation
wrong + low entropy       → stable misconception (high priority)
```

For Qwen3.5-0.8B, use token probabilities when the answer space is structured,
and semantic entropy from sampled free-form answers for the boundary subset.

### 3. Temporal knowledge: every claim needs scope

**SituatedQA** shows that answers depend on temporal/geographic context and that
models struggle even when updated evidence is available. Questions must carry
explicit context rather than silently assuming “now.”

**TempLAMA** represents facts with time scope. **DynamicTempLAMA** dynamically
constructs temporal probes and splits facts into unchanged, updated, new, and
deleted categories—the closest published design to DELTA's source-version
matrix.

**FreshQA** tests never-changing, slow-changing, fast-changing, and
false-premise questions. Its false-premise category is especially relevant:
negative existence must be represented as a structured Boolean/debunking target,
not the magic string `unknown`.

Design implication:

```text
Every claim has:
  valid_from / valid_to (or source revision range)
  change_type: unchanged | added | removed | changed | moved
  positive truth and explicit negation

Every version-sensitive probe names the revision.
```

### 4. Atomic claims and source-grounded verification

**FActScore** decomposes long outputs into atomic facts and verifies each against
a designated source. Its most important conceptual contribution for DELTA is
that factuality is relative to an explicit knowledge source—not a global,
unversioned truth.

For technical repositories, source authority is typed:

```text
file/path existence       → git tree
CLI flag/default          → parser/config schema + --help
API signature             → source/AST/import reflection
runtime behavior          → executable test
recommended workflow      → versioned documentation
```

An LLM judge can normalize wording or decompose outputs, but it must not create
the truth label.

### 5. Dynamic benchmarks reduce contamination but introduce variance

Static public benchmarks can leak into pretraining or later fine-tuning.
Contamination surveys recommend dynamic/private test construction, but dynamic
benchmarks need reproducibility, collision controls, stable coverage, and
versioned generators.

DELTA has a natural advantage: probes can be generated from a newly pinned
repository revision after the model checkpoint was frozen. The generator,
templates, source hashes, and held-out claims must all survive.

Design implication:

```text
Public generator, private/fresh generated instances.
Freeze claim IDs and source evidence, not exact natural-language prompts.
Regenerate prompt surfaces with low collision across runs.
```

### 6. Mechanistic localization is optional—not the fingerprint

**Knowledge Neurons**, **ROME**, and **MEMIT** attempt to localize or edit factual
associations inside transformer MLPs. These methods are white-box and
GPU-intensive; they can be valuable for explaining a small set of surprising
claims.

But **Does Localization Inform Editing?** found causal-tracing localization
largely uncorrelated with the best layer to edit. **CounterFact+** exposed
specificity failures missed by earlier model-editing evaluations.

Design implication:

```text
Behavioral profile first.
Mechanistic analysis only for selected:
  stable misconceptions,
  adaptation interference,
  or claims whose behavior changes unexpectedly.
```

Do not spend the primary GPU budget searching for “the neuron that knows vLLM.”

### 7. Executable verification is the strongest technical-domain evidence

**SWE-bench** demonstrates a durable pattern: pin a repository state, construct
an execution environment, and use fail-to-pass tests as the primary signal.
DELTA does not need to copy the issue-resolution task, but should copy this
verification principle.

**CodeUpdateArena** and related API-evolution benchmarks show how to test whether
a model incorporates *changed* semantics while retaining unrelated behavior.
**LiveCodeBench** shows continuous post-cutoff evaluation with executable scoring.
These matter for DELTA because versioned technical knowledge is closer to API
evolution than to TriviaQA.

For vLLM topic knowledge:

```text
closed-book QA   → parametric recall
structured probe → recognition / version attribution
executable probe → ability to apply knowledge
```

These are different capabilities and must remain separate in the profile.

### 8. Keep claim history; never overwrite old truth

A source-grounded profile is a claim ledger, not a mutable quiz:

```text
claim_id
subject / predicate / object
polarity
valid_from_revision / valid_to_revision
supersedes / contradicts
evidence hash + span
authority class
verifier + result
```

Retain both before and after claims. Drift cannot be measured if the previous
truth is deleted. Authority should be explicit: executable behavior outranks
tests; tests outrank normative docs; docs outrank examples and changelog prose
unless the repository declares another contract.

Probe generation must be independent from claim extraction and, where possible,
blinded to gold implementation details. Prefer oracles in this order:

```text
1. executable tests in a pinned container
2. compiler / type / schema validation
3. deterministic structured comparison
4. evidence-span entailment with constrained answers
5. human adjudication
6. LLM judge only as advisory
```

## proposed DELTA protocol

### Step 0 — freeze the measurement context

Record:

- exact model and adapter hashes;
- tokenizer and chat template;
- system prompt;
- decoding parameters;
- source repository + revision;
- claim-bank and probe-generator versions;
- whether the run is closed-book or context-assisted.

The first profile is **closed-book**. RAG is an intervention and must not leak
into the parametric baseline.

### Step 1 — build a typed claim bank from the entire subject snapshot

Use deterministic extractors first:

| claim type | extraction | verifier |
|---|---|---|
| path exists | git tree | exact Boolean |
| symbol exists | AST / reflection | exact Boolean |
| flag exists | parser / docs | `--help` / AST |
| default value | AST / config schema | typed equality |
| API signature | AST / inspect | normalized signature |
| relation / definition | docs span | source-entailment |
| procedure | docs + examples | executable sequence if possible |

LLMs may propose candidate atomic claims from prose. They must include evidence
spans and survive deterministic/source-entailment gates.

### Step 2 — sample for importance, not convenience

Create strata:

- source centrality: public API / CLI / config / internals;
- operational impact: safety, correctness, performance, convenience;
- claim type;
- popularity/exposure in docs;
- difficulty and answer cardinality;
- stability: unchanged / added / removed / changed / moved.

The broad baseline should describe the subject at `source_before`; the delta
slice should be a separate projection onto changed claims.

### Step 3 — generate a probe bundle per claim

Minimum viable bundle:

```text
P1 free recall
P2 constrained recognition with hard negatives
P3 explicit negation / false premise
P4 version attribution
P5 application or executable probe (where meaningful)
```

For each linguistic probe, create at least three semantically equivalent
paraphrases. Do not score all paraphrases as independent facts.

Hard negatives should be generated from nearby real entities/paths, not absurd
fake names. Positive and negative wording must be balanced to avoid yes/no bias.

### Step 4 — run a two-pass GPU measurement

**Pass 1: broad scan**

- deterministic generation (`temperature=0`);
- all claims × paraphrases;
- structured outputs where possible;
- record token log-probabilities for candidate answers;
- deterministic/executable grading.

**Pass 2: boundary scan**

Run only on incorrect, low-margin, inconsistent, or high-impact claims:

- 5–10 stochastic samples;
- semantic clustering / entropy;
- `P(True)` after answer generation;
- contradiction checks across paraphrases;
- optional hidden-state probe as an experiment, not a production dependency.

This allocates GPU to ambiguous claims rather than resampling obvious ones.

### Step 5 — classify per claim

Recommended dimensions:

```yaml
correctness:       0..1
paraphrase_consistency: 0..1
semantic_entropy:  >=0
calibrated_confidence: 0..1
abstention_quality: 0..1
version_attribution: 0..1
application_score: 0..1 | null
```

Status rule (thresholds calibrated on a held-out claim set):

```text
known:
  correct across probe forms + low instability + right version

partial:
  recognition succeeds but recall/application fails, or uncertainty is high

unknown:
  incorrect with appropriate uncertainty/abstention

misattributed:
  fact content correct but revision wrong

stable_misconception:
  consistently wrong with high confidence / low entropy

unstable:
  materially changes across paraphrases or samples
```

Do not collapse `unknown` and `stable_misconception`; they imply different
interventions and risks.

### Step 6 — aggregate without hiding the boundary

Produce:

- coverage-weighted subject score;
- scores by claim type and source zone;
- known/partial/unknown/misattributed/unstable proportions;
- calibration curve and selective accuracy;
- error rate at fixed coverage;
- long-tail versus central claims;
- version confusion matrix;
- application/executable pass rate.

Retain every per-claim result. The aggregate is an index, not the artifact.

### Step 7 — connect profile to source drift

```text
TopicKnowledgeProfile(model, source_before)
                 +
ClaimDiff(source_before, source_after)
                 ↓
KnowledgeDriftProjection
```

This separates:

```text
pre-existing ignorance:
  unknown at source_before; unchanged claim

obsolete knowledge:
  known at source_before; claim changed/removed

novel gap:
  claim added at source_after

stable regression:
  unchanged claim was known before adaptation, fails after
```

Only after this projection should DECIDE compare no-op, RAG refresh, targeted
LoRA, or a larger intervention.

## proposed pilot for `e1_vllm`

### Scope

```text
model: Qwen/Qwen3.5-0.8B, exact deployed chat checkpoint
subject: vLLM v0.22.0
mode: closed-book
claim bank: 500–1,000 typed claims
probe surfaces: 3 paraphrases × 4 behavioral forms
deep samples: 8 generations for boundary/high-impact claims only
```

Claim allocation:

| zone | target claims |
|---|---:|
| CLI flags/defaults | 200 |
| public APIs/config | 150 |
| core concepts/architecture | 150 |
| procedures/examples | 150 |
| file/features existence | 100 |
| held-out verifier/calibration claims | 100 |

The exact totals should follow extracted inventory and deduplication; they are
budget targets, not quotas.

### GPU plan

1. A10G broad closed-book scan.
2. A10G boundary resampling + semantic entropy.
3. CPU deterministic/executable verification where possible.
4. Optional H100 mechanistic pilot on at most 20 stable misconceptions.

No adapter is trained in the profiling run. Training changes the object being
measured.

### Experimental checks before trusting the profile

- repeat run reproducibility;
- paraphrase consistency;
- positive/negative balance;
- exact versus semantic grader agreement;
- 100-item human audit;
- calibration on held-out claims;
- answer leakage check from prompt templates;
- profile sensitivity to system prompt/chat template;
- confidence method AUROC / AURC for known-vs-unknown claims.

### Success criterion

The pilot succeeds if it can reliably predict, on held-out claims:

1. whether the model will answer correctly;
2. whether it will abstain when wrong;
3. whether it attaches facts to the correct revision;
4. which previously known stable claims regress after an intervention.

It does **not** succeed merely by producing a large QA dataset.

## what this means for the current e1 benchmark

`eval_v3` is a small intervention comparison set, not a knowledge profile:

- 46 hand-authored questions are too narrow;
- change-heavy B/D selection is not representative of subject knowledge;
- one prompt per claim cannot establish robustness;
- `must_contain` allows contradictions and rejects valid “no” answers;
- retrieval and parametric knowledge have been mixed in interpretation;
- no calibrated uncertainty or selective prediction exists;
- procedural C lacks executable verification.

Keep it as historical Phase C evidence. Do not grow it into the new profile.
Build the claim bank and profile as a separate artifact.

## literature map

### Core factual probing

- Petroni et al. (2019), **Language Models as Knowledge Bases?**  
  https://aclanthology.org/D19-1250/
- Jiang et al. (2020), **How Can We Know What Language Models Know? (LPAQA)**  
  https://aclanthology.org/2020.tacl-1.28/
- Elazar et al. (2021), **Measuring and Improving Consistency in Pretrained
  Language Models (ParaRel)**  
  https://aclanthology.org/2021.tacl-1.60/
- Mallen et al. (2023), **When Not to Trust Language Models: Parametric and
  Non-Parametric Memories (PopQA / EntityQuestions)**  
  https://arxiv.org/abs/2212.10511
- Kandpal et al. (2023), **Large Language Models Struggle to Learn Long-Tail
  Knowledge**  
  https://proceedings.mlr.press/v202/kandpal23a.html
- Neeman et al. (2023), **DisentQA**  
  https://aclanthology.org/2023.acl-long.559/
- Youssef et al. (2023), **Give Me the Facts! A Survey on Factual Knowledge
  Probing**  
  https://aclanthology.org/2023.findings-emnlp.1043/
- Muhlgay et al. (2024), **FACTOR**  
  https://aclanthology.org/2024.eacl-long.4/
- Maekawa et al. (2024), **BELIEF / MyriadLAMA**  
  https://aclanthology.org/2024.findings-emnlp.771/
- Yuan et al. (2024), **Towards a Holistic Evaluation of LLMs on Factual
  Knowledge Recall (FACT-BENCH)**  
  https://arxiv.org/abs/2404.16164
- Wei et al. (2024), **Measuring Short-Form Factuality in Large Language Models
  (SimpleQA)**  
  https://arxiv.org/abs/2411.04368

### Uncertainty and knowledge boundaries

- Kadavath et al. (2022), **Language Models (Mostly) Know What They Know**  
  https://arxiv.org/abs/2207.05221
- Tian et al. (2023), **Just Ask for Calibration**  
  https://aclanthology.org/2023.emnlp-main.330/
- Kuhn et al. (2023), **Semantic Uncertainty: Linguistic Invariances for
  Uncertainty Estimation**  
  https://arxiv.org/abs/2302.09664
- Manakul et al. (2023), **SelfCheckGPT**  
  https://aclanthology.org/2023.emnlp-main.557/
- Farquhar et al. (2024), **Detecting Hallucinations in LLMs Using Semantic
  Entropy**  
  https://www.nature.com/articles/s41586-024-07421-0
- Xiong et al. (2024), **Can LLMs Express Their Uncertainty?**  
  https://openreview.net/forum?id=gjeQKFxFpZ
- Ren et al. (2025), **Knowledge Boundary of Large Language Models: A Survey**  
  https://aclanthology.org/2025.acl-long.256/

### Temporal and dynamic knowledge

- Zhang & Choi (2021), **SituatedQA**  
  https://aclanthology.org/2021.emnlp-main.586/
- Dhingra et al. (2022), **Time-Aware Language Models as Temporal Knowledge
  Bases (TempLAMA)**  
  https://aclanthology.org/2022.tacl-1.15/
- Margatina et al. (2023), **Dynamic Benchmarking on Temporal Concept Drift
  (DynamicTempLAMA)**  
  https://aclanthology.org/2023.eacl-main.211/
- Vu et al. (2024), **FreshLLMs / FreshQA**  
  https://aclanthology.org/2024.findings-acl.813/

### Factual verification, executable, and contamination

- Min et al. (2023), **FActScore**  
  https://aclanthology.org/2023.emnlp-main.741/
- Jimenez et al. (2024), **SWE-bench**  
  https://openreview.net/forum?id=VTF8yNQM66
- Jain et al. (2024), **LiveCodeBench**  
  https://arxiv.org/abs/2403.07974
- CodeUpdateArena (2024)  
  https://arxiv.org/abs/2407.06249
- Sainz et al. (2024), **Unveiling the Spectrum of Data Contamination in
  Language Models**  
  https://aclanthology.org/2024.findings-acl.951/

### Mechanistic localization and edit evaluation

- Dai et al. (2022), **Knowledge Neurons in Pretrained Transformers**  
  https://aclanthology.org/2022.acl-long.581/
- Meng et al. (2022), **Locating and Editing Factual Associations in GPT
  (ROME)**  
  https://proceedings.neurips.cc/paper/2022/hash/6f1d43d5a82a37e89b0665b33bf3a182-Abstract-Conference.html
- Meng et al. (2023), **Mass-Editing Memory in a Transformer (MEMIT)**  
  https://arxiv.org/abs/2210.07229
- Hase et al. (2023), **Does Localization Inform Editing?**  
  https://proceedings.neurips.cc/paper_files/paper/2023/hash/3927bbdcf0e8d1fa8aa23c26f358a281-Abstract-Conference.html
- Hoelscher-Obermaier et al. (2023), **Detecting Edit Failures in LLMs:
  CounterFact+**  
  https://aclanthology.org/2023.findings-acl.733/
- Zhong et al. (2023), **MQuAKE**  
  https://aclanthology.org/2023.emnlp-main.971/

### Research threads used

Parallel literature digests that contributed to this synthesis:

- [Factual probing literature](52b8e961-3973-4316-81b0-171122835fec)
- [Calibration and unknowns](5e6d4765-04ea-41dd-84db-a691ca3c1f6e)
- [Mechanistic knowledge methods](7941665b-41c3-4f69-a165-ad676ab2ea58)
- [Source grounded benchmark design](d598750c-1f65-40e2-8b4e-e07ec9252d87)

## what goes where

- this review: `artifacts/reports/knowledge_fingerprinting_literature_review.md`;
- target DELTA contract: `docs/project-delta.md`;
- phase status only: `docs/delta-roadmap.md`;
- proposed implementation (after protocol approval):
  `experiments/e1_vllm/knowledge_profile/`;
- durable generated profile: B2 datasets/artifacts with source/model hashes.

## what can die

- the `doc_0` / `doc_8` names in human-facing prose;
- exact natural-language prompt surfaces;
- local generation caches and semantic-clustering intermediates;
- mechanistic traces that do not improve profile prediction.

## what must survive

- source revisions and claim evidence;
- claim IDs and typed truth;
- model/prompt/decoding hashes;
- profile protocol and verifier versions;
- per-claim responses, probabilities, labels, and status;
- held-out calibration and human-audit decisions.

## command

Research artifact only; no profiling run exists yet.

```bash
# proposed future entrypoint
python experiments/e1_vllm/knowledge_profile/cli.py build-claims \
  --repo https://github.com/vllm-project/vllm.git \
  --revision v0.22.0

./scripts/lab run --target modal \
  --config configs/experiments/e1_vllm/knowledge_profile_v0.22.0.yaml \
  --gpu A10G
```

Do not train an adapter before freezing the closed-book profile.
