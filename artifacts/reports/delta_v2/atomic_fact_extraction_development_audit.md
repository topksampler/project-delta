# delta_v2 development AtomicFactDelta extraction audit

Status: implemented and awaiting human inspection.

This report covers only the development fixture `v0.22.0 → v0.23.0`.
Acceptance remains sealed. No feature was verified or scored, no evaluation
item was generated, and no model, training, B2, Modal, or Lambda job ran.

## Algorithm

Each complete Git tree is processed independently:

```text
development_before tree → before observations + before rejections
development_after tree  → after observations  + after rejections

(family_id, semantic_key) outer join
  absent  → present  = added
  present → absent   = removed
  equal   → equal    = stable
  unequal → unequal  = changed
```

Every observation contains the exact snapshot role, path, line span, Git blob
SHA-1, source SHA-256, extractor version, and a location-free canonical AST
value. Rejected candidates remain in separate logs and never become facts.

## Source and recipe pins

- Repository: `vllm-project/vllm`
- Before commit:
  `0b3ba88f165976e77ca5e6a7a3f5bba4562b80af`
- After commit:
  `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`
- Extractor: `delta-v2-python-ast-v1`
- Parser contract: `cpython.ast`, feature version `3.11`
- Fact-family manifest SHA-256:
  `aaf99951fe7da6da4aa5bacb1b63187f4c46131703d30fab1666be4d02a3d210`

The extractor parsed 1,752 Python files from the before tree and 1,805 from
the after tree. Files were selected from each complete snapshot tree rather
than from the FileDelta inventory.

## Observation and delta counts

| Family | Before observations | After observations | Stable | Added | Removed | Changed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CLI option | 245 | 254 | 244 | 9 | 0 | 1 |
| Configuration field | 450 | 448 | 430 | 0 | 2 | 18 |
| Environment variable | 266 | 268 | 261 | 7 | 5 | 0 |
| Literal domain | 46 | 46 | 45 | 0 | 0 | 1 |
| **Total** | **1,007** | **1,016** | **980** | **16** | **7** | **20** |

The joined inventory contains 1,023 facts. Forty-three are non-stable source
facts and 980 are stable controls.

## Rejection audit

| Family | Before accepted | Before rejected | After accepted | After rejected |
| --- | ---: | ---: | ---: | ---: |
| CLI option | 245 | 196 | 254 | 196 |
| Configuration field | 450 | 5 | 448 | 5 |
| Environment variable | 266 | 0 | 268 | 0 |
| Literal domain | 46 | 6 | 46 | 6 |

The 196 CLI rejections on each side comprise 190 calls with expanded keyword
arguments, three expanded positional calls, and three non-option positional
arguments. The expanded calls are dominated by engine flags whose effective
arguments come from `get_kwargs(ConfigClass)`. Their underlying configuration
fields are captured, but the CLI calls remain rejected because resolving their
effective defaults and constraints would require a separately preregistered
cross-module algorithm.

The five configuration rejections are `ClassVar` declarations. The six
literal-domain rejections contain computed or unsupported values. These
exclusions are explicit and identical on both development sides.

During the first development run, the selector treated `@config(...)` as
ambiguous and excluded four core classes, including `ModelConfig`. Inspection
showed that these are calls to the exact `config` decorator rather than aliases.
Before freezing, the registered selector was clarified to accept both
`@config` and `@config(...)` while continuing to reject attributed aliases.
This raised configuration coverage by 104 observations on each side without
changing the 43 non-stable deltas.

## Determinism and integrity

- Before observation reconstruction: pass
- After observation reconstruction: pass
- Unique stable fact IDs: pass
- Canonical ordering: pass
- Git object SHA verification for every parsed blob: pass
- Duplicate semantic-key fail-closed behavior: pass
- Second full extraction into a separate directory: byte-for-byte identical
- Cross-check against FileDelta: zero non-stable facts associated with a stable
  or missing source file
- Focused delta_v2 tests: 26 pass
- Existing repository tests: 28 pass

Output hashes:

| Output | Rows | SHA-256 |
| --- | ---: | --- |
| Before observations | 1,007 | `514dc8428b1853e1c875bdcea72780b95a62d12beb0666a79159a2d15e4a5900` |
| After observations | 1,016 | `09a4214d6661064b358b118c6ced48a7af2b7249c45fd2263e04b7ae67070c7d` |
| Before rejections | 207 | `c3fa47cc5a216b8f987b5b6fe5691aadc7b2ccec84e3ecef3159f0d136b23b72` |
| After rejections | 207 | `dc12bb5588546454dee74c827ec41d22f3de46b65e0789f9a020a6700d7b67ed` |
| Atomic fact deltas | 1,023 | `a5827d0a66289552bc707f75e20ce4ecd74a1e22dba6b3144a87ab8f27241017` |

The generated JSONL remains an ignored local build artifact. These hashes
record the current development recipe; they do not freeze the dataset.

## Next manual feature trace

The strongest next development candidate is
`VLLM_PREFIX_CACHE_RETENTION_INTERVAL`:

- it is absent from the old tree and is an added environment-variable fact in
  the new tree;
- the new tree consumes it in `vllm/v1/core/kv_cache_coordinator.py`;
- the new tree contains nine test references in
  `tests/v1/core/test_prefix_caching.py`;
- no separate documentation evidence has been located yet.

This is still only a proposed `FeatureCandidate`. It must be traced through the
relevant source, tests, and later pull-request provenance, then exercised
against both pinned revisions before it can become a verified `FeatureDelta`.
