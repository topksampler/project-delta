# delta_v2 development FileDelta inventory audit

Status: implemented and awaiting human inspection.

This report covers only the development fixture. The acceptance transition
remains sealed, and no atomic facts, feature candidates, behavior probes,
evaluation items, models, training jobs, or remote-compute jobs were run.

## Source snapshots

Repository: `vllm-project/vllm`

| Role | Revision | Commit SHA | Git tree SHA |
| --- | --- | --- | --- |
| before | `v0.22.0` | `0b3ba88f165976e77ca5e6a7a3f5bba4562b80af` | `92701b6c49f4969297d5d11a2cac91c180690482` |
| after | `v0.23.0` | `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` | `e18d76eec620cdfc6559d7152f9911b7e91f84f8` |

The inventory command verifies both commit and tree SHAs before reading either
tree.

## Inventory result

Output:
`data/experiments/delta_v2/file_deltas_development.jsonl`

Audit summary:
`data/experiments/delta_v2/file_deltas_development.summary.json`

| Status | Rows |
| --- | ---: |
| stable | 4,268 |
| added | 219 |
| removed | 75 |
| modified | 929 |
| renamed | 24 |
| **total** | **5,515** |

- Before tree entries: 5,296
- After tree entries: 5,440
- Inventory SHA-256:
  `129831f4b6fa4b0604fe2c1f31854947df66df3dd4bf9101700e2386cb95c6b9`
- Before-tree reconstruction: pass
- After-tree reconstruction: pass
- Unique file IDs: pass
- Canonical ordering: pass
- Rename policy: `exact-object-unique-v1`

The output is a local, ignored build artifact. This report records its digest
without treating the generated JSONL as frozen evidence.

## Rename policy cross-check

An independent Git comparison using `git diff-tree -r --find-renames=100%`
reported:

| Git status | Paths |
| --- | ---: |
| added | 209 |
| deleted | 65 |
| modified | 929 |
| renamed | 34 |

Git paired ten additional exact-content moves. The delta_v2 policy pairs a
rename only when one deleted entry and one added entry uniquely share the same
mode, object type, and object ID. Ambiguous identical-content moves remain ten
adds plus ten removals, so the delta_v2 inventory has ten fewer renames and ten
additional rows overall. Modified counts match exactly.

This conservative choice avoids inventing path identity where the source trees
do not determine a unique pairing.

## Reproducibility and tests

- Re-running against the original temporary repository produced a byte-for-byte
  identical JSONL.
- Fetching only the two development tags into a fresh bare repository and
  regenerating the inventory also produced a byte-for-byte identical JSONL and
  the same SHA-256.
- Ten delta_v2 unit and integration tests pass. They cover manifest validation,
  the acceptance seal, every file status, ambiguous renames, rename-plus-edit,
  modes, symlinks, gitlinks, unusual paths, deterministic output, tree
  reconstruction, and pinned-tree verification.
- The existing repository test suite passes: 28 tests.
- The acceptance CLI guard was exercised: selecting the acceptance transition
  without the explicit unlock fails before creating output files.

## Inspection checkpoint

Stop here for human inspection of the complete FileDelta inventory. Do not
preregister or extract atomic-fact families until this checkpoint is accepted.
Pull-request metadata remains proposed provenance for later
`FeatureCandidate` discovery; it is not part of FileDelta classification and
must not become ground truth.
