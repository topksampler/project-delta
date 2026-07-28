# TruthSource protocol (pre-registered)

Experiment ID: `e1_truthsource_v0`  
Frozen: 2026-07-18  
Model under test: `Qwen/Qwen3.5-0.8B`  
Subject pin: vLLM `v0.22.0` (legacy alias `doc_0`)  
Mode: closed-book profile only (no LoRA in this study)

## Decision question

Does changing the truth source docs-only → distilled-code → thorough-code
change the knowledge-profile gap set enough that the data wheel would differ?

## Arms

| Arm | Truth source |
|-----|----------------|
| D | Docs corpus + structure_index only |
| C_distill | Docs + distilled code (rule below) |
| C_full | Docs + thorough code extract (rule below) |

### C_distill inclusion rule (complete)

```text
INCLUDE a code-backed claim iff at least one holds:
  1. CLI flag from add_argument(--…) in vllm/engine/arg_utils.py
  2. Public export name in vllm/__init__.py MODULE_ATTRS
  3. Environment variable key defined in vllm/envs.py
EXCLUDE: tests/, benchmarks/, examples/, docs/ as code truth
```

If a candidate fails this rule, it is not distill — it belongs in C_full or nowhere.

### C_full inclusion rule

```text
INCLUDE:
  - all distill items, plus
  - FunctionDef / ClassDef under vllm/**/*.py (skip tests), module depth ≤ 4
  - add_argument(--…) anywhere under vllm/
CAP: 600 private (non-bridge) symbol-exists claims for the first run
```

## Bridge set

Entities that are **flag names** present in both docs `structure_index` and
C_distill argparse extract. Bridge claim IDs are identical across arms:

```text
bridge.flag.exists.<name>
bridge.flag.meaning.<name>   # only if docs meaning extractable
```

Arm-private claims use prefixes `D.`, `C_distill.`, `C_full.` and are
reported separately (not in primary Jaccard).

## Probe protocol (fixed)

- Behaviors: recall exists, recognition (flags), negation (refuse-fake), meaning recall
- 3 paraphrases where applicable
- temperature 0, max_new_tokens 128
- Same system prompt as e1 eval

## Pre-registered decision margins (primary)

On the **bridge** claim set only:

1. **Gap-set Jaccard** for `S = {claims with status in {unknown, partial, unstable}}`  
   - Declare arms “same map” if Jaccard(D, C_distill) ≥ 0.80 **and** Jaccard(C_distill, C_full) ≥ 0.80  
   - Else maps differ.

2. **Status flip rate** (bridge): fraction of claims whose status differs between arms.  
   - Material if ≥ 0.15 between an adjacent pair.

3. **Honesty** (negation probes on bridge exists claims): accuracy by arm.  
   - Material if |acc_a − acc_b| ≥ 0.10

## Decision table (locked)

| Result | Decision |
|--------|----------|
| D ≈ C_distill ≈ C_full | Don’t add code yet; exquisite docs+honesty first |
| D ≠ C_distill ≈ C_full | Distill enough; thorough wasted |
| D ≈ C_distill ≠ C_full | Distill failed; need thorough or better rule |
| D ≠ C_distill ≠ C_full | Code + coverage both matter |
| High docs↔code gold conflict on bridge | Docs-only environment is unsafe |

≈ means both Jaccard thresholds pass and flip rate < 0.15.

## Secondary (reported, not deciding alone)

- Human audit deferred  
- Docs↔code gold conflicts count on bridge  
- $/arm wall time  

## What can die

Local regenerations of private claim samples with a new seed.

## What must survive

This protocol file, arm claim/probe hashes, three run_ids, compare JSON.

## Post-hoc validity (2026-07-18)

Primary D vs C_distill Jaccard/flips are **confounded**: recognition hard-negatives differ by arm. Freeze shared `probes_bridge.jsonl` and re-run before trusting `distill_enough` for docs vs code. C_distill ≈ C_full on bridge is expected (identical bridge probes).
