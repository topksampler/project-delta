# delta_v2 prefix-cache retention feature trace audit

Status: **partial**

Decision: **keep the FeatureCandidate pending**

This audit follows one development-transition candidate from the complete
FileDelta inventory through atomic facts, merged-PR provenance, and an
executable old/new interface probe. It does not access the sealed acceptance
transition.

## Candidate

- Candidate:
  `feature-candidate:vllm-prefix-cache-retention`
- Proposed status: `added`
- Development transition: vLLM `v0.22.0` to `v0.23.0`
- Proposed capability: selectively retain prefix-cache checkpoints for
  sliding-window KV caches through a configurable retention interval
- Verification status: `pending`

The capability statement is broader than the interface behavior verified
below. The candidate therefore cannot yet become a FeatureDelta.

## Evidence-chain audit

The deterministic feature-trace audit passed:

- one linked added AtomicFactDelta:
  `VLLM_PREFIX_CACHE_RETENTION_INTERVAL`;
- seven linked modified FileDelta records;
- no unresolved evidence IDs;
- no PR paths missing from the linked FileDelta set;
- no extra candidate paths outside the PR patch.

The seven paths are:

1. `tests/v1/core/test_kv_cache_utils.py`
2. `tests/v1/core/test_prefix_caching.py`
3. `vllm/envs.py`
4. `vllm/v1/core/block_pool.py`
5. `vllm/v1/core/kv_cache_coordinator.py`
6. `vllm/v1/core/kv_cache_utils.py`
7. `vllm/v1/core/single_type_kv_cache_manager.py`

## Pull-request provenance

The candidate links merged vLLM PR
[#43447](https://github.com/vllm-project/vllm/pull/43447), whose merge commit
is `a6183563b6f604ef7b481ce8ce7af359c6dc1b74`.

Local Git ancestry verification against the pinned development commits passed:

- the development-before commit does not contain the merge;
- the development-after commit does contain the merge.

The PR title, body, and author claims are discovery and grouping evidence only.
They are not used as gold truth.

## Executable interface probe

Probe:
`behavior-probe:vllm-prefix-cache-retention-env-interface-v1`

The probe extracts the exact `vllm/envs.py` blob independently from each pinned
development commit, executes that source in a fresh module namespace for every
case, and observes attribute access.

| Environment value | v0.22.0 observation | v0.23.0 observation |
|---|---|---|
| unset | `AttributeError` | `None` |
| `0` | `AttributeError` | integer `0` |
| `64` | `AttributeError` | integer `64` |
| `-32` | `AttributeError` | integer `-32` |
| malformed text | `AttributeError` | `ValueError` |

All ten revision-case assertions passed. A second execution produced a
byte-identical result.

- Result:
  `data/experiments/delta_v2/probes/prefix_cache_retention_env.result.json`
- Result SHA-256:
  `2838adddfc48ef334a948c9d9e2d55a34abf37a07be59cdf1cce270a1fcea884`
- Before source blob:
  `b2c5f22567fa34c6bce5ab1b4912c435dc902818`
- After source blob:
  `8f4e18d2235d98863b94fd076d490c3e224cb312`

This proves an added optional-integer configuration interface. It does not
prove that the value changes prefix-cache retention.

## Broader runtime probe status

The upstream after-revision tests contain executable checks for:

- aligned interval retention;
- rejection of negative or misaligned intervals;
- retained checkpoints surviving block recycling;
- zero meaning latest replay boundary only;
- pure sliding-window sparse, latest-only, and dense-default behavior.

A bounded local execution attempt was made in a disposable virtual environment.
The source checkout is not an installed vLLM runtime, and test collection
traversed unrelated package imports before reaching KV-cache logic. After
installing several lightweight missing packages, collection remained blocked
at another runtime dependency (`cloudpickle`). No vLLM package or native
extension was built, and no test was weakened with module stubs.

This outcome is `unavailable`, not a model failure, source failure, feature
failure, or verification pass.

## Promotion decision

Do not promote the candidate.

The next promotion gate should be a reproducible CPU environment plus one
external old/new behavior contract that demonstrates:

1. unset retains the old dense behavior;
2. interval `64` changes the new revision to sparse checkpoints while the old
   revision remains dense or does not support the setting;
3. interval `0` retains only the latest replay boundary in the new revision;
4. negative and misaligned intervals are rejected by the new coordinator.

Only after those outcomes execute against both pinned development snapshots
should the candidate be reconsidered as a verified FeatureDelta.
