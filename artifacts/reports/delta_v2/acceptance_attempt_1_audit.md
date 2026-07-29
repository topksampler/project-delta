# delta_v2 acceptance attempt 1

Status: failed acceptance coverage; preserved as negative evidence.

The frozen development recipe was validated and committed before this
transition was accessed. Acceptance attempt 1 then applied that exact recipe to
vLLM `v0.25.0` → `v0.25.1`.

## What the frozen recipe observed

- Snapshot verification matched both pinned commit and Git-tree hashes.
- The complete FileDelta inventory contained 5,864 rows: 5,860 stable and 4
  modified.
- The modified paths were:
  - `tests/compile/passes/distributed/test_fusion_all_reduce.py`
  - `vllm/compilation/passes/fusion/allreduce_rms_fusion.py`
  - `vllm/multimodal/video.py`
  - `vllm/utils/import_utils.py`
- The transition contains two bug-fix commits, whose subjects reference
  vLLM PRs `#48330` and `#47888`.
- Independent full-snapshot extraction produced 1,040 AtomicFactDeltas.
- All 1,040 facts were `stable`; there were 0 `added`, 0 `removed`, and 0
  `changed` facts in the four preregistered families.
- All 1,040 verified sources were assigned to `eval`.
- The frozen EvalItem selector emitted 0 items because it includes all
  nonstable facts plus a 1:1 stable-control match.

## Decision

This transition cannot be the final acceptance environment for the frozen
fact-level slice. A zero-item evaluation cannot measure source-change
knowledge. The tags will not be silently swapped and no post-hoc questions will
be invented from the four files.

The attempt also exposed a harness validation bug: the EvalItem build summary
reported `status: pass` for zero items and a zero-row audit packet. The next
recipe version must fail closed when an acceptance build is empty.

No target model, training, fine-tuning, Modal, Lambda, B2 write, or model
evaluation ran.

## Exact local evidence

| Artifact | SHA-256 |
|---|---|
| FileDelta inventory | `9c11fc7c46d97bd152e1e138c38eec931b99e9b903bec7b25ded7e18c276156f` |
| FileDelta summary | `3833dd8ccd65b32e9a17aef68320ed886ad71cddf5e69ea8bea68b8a8ee87ed7` |
| AtomicFactDelta corpus | `164fc60b9fe26a669496b0aa730888a9fbbf2a5d667ebd6984cb5c901a71c2e9` |
| Atomic-fact summary | `c455177db14be087e3f640ce743b9289e0d3e7b1d868530e1f6bc64b58954748` |
| Source assignments | `fc085045b8f7a9eb67bbf3c5e8171334092f1a14164272faec325d6b11677cd3` |
| Source-split summary | `b9d8dd99f90d751d25537eec29d9001e07c7851140304e46631178671a1a2088` |
| EvalItems | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| EvalItem summary | `86f7fccd13d53128694e1004c35f47a08a4a3a1a16254b898c25ee8dc8c6ffa7` |
| Audit packet | `e3b0c44298fc1c149afbf4c8996fb92427ae41e464b934ca495991b7852b855` |
