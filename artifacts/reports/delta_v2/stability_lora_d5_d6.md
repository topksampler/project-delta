# DELTA v2 stability LoRA iterations d5-d6

Date: 2026-07-30

Decision: reject both candidates before acquisition/full evaluation

## Comparable held-out stability results

| Probe | Base | d4 (25% replay) | d5 (50% replay) | d6 (50% replay, false 2:1) | Gate |
|---|---:|---:|---:|---:|---:|
| Choice parseable | 0/15 | 15/15 | 15/15 | 12/15 | at least 90% |
| Choice correct | 0/15 | 8/15 | 8/15 | 4/15 | diagnostic |
| Boolean true | 1/15 | 11/15 | 13/15 | 9/15 | at least 12/15 |
| Boolean false | 14/15 | 11/15 | 9/15 | 9/15 | at least 14/15 |
| Recall | 0/15 | 1/15 | 3/15 | 3/15 | advisory |

All results use the same 60 source-disjoint items, two byte-identical greedy
repeats per item, and no pooled score.

## d5: replay share

d5 changed only replay share from 25% to 50% at a fixed 240-unit, 60-step
budget. Training completed with 120 acquisition and 120 replay units. Its
adapter SHA-256 is
`05d5e66653e4c116c1e3d32bd5d7df463fecea3989ab89ffa28bf029482bfda3`.

The stable-true cell improved from 11/15 to 13/15, but stable-false fell from
11/15 to 9/15. More replay did not preserve false rejection.

Evidence:

- Training receipt:
  `448adf5dcd5b049d4cbf3edb2832c2b951ff11eaf059accb262305aa024f986c`
- Training metrics:
  `2971fb93488aeafcc88979327b00a951378b3b77bf4e0c56c95340d769191c7e`
- Stability receipt:
  `30fd04620b5b6194392e63279aaf73a8d2a0147aaab1abbff807f8f5b9a99dcc`
- Stability metrics:
  `eb4681387cf599dbc7babab55f36cc844270cb85c3acce6ff19f609e94a0be29`

## d6: false-target weighting

d6 held d5's base, LoRA topology, replay share, schedule size, optimizer
steps, learning rate, and seed fixed. It changed Boolean target exposure to
120 positive versus 240 negative targets.

Its adapter SHA-256 is
`6f3d198a0e2417bd717c7078d64989d345150137f6ef8487287aa7d131a5b2e5`.
The intervention did not recover false rejection and regressed true accuracy
and choice parseability.

Evidence:

- Training receipt:
  `db485d6e2e934c6f26ceb5c95b5dca8f96f0689d8c25e8dea1538fca31b10a79`
- Training metrics:
  `c53dd8a340a7b6d4e8e035b68749e7e7c296b49b0ece6b80b610410dea2b7495`
- Stability receipt:
  `4955cb3bedd334eb224130ee7d9749c9a51d04f4939d26b4bf9cfe5c0d5893a7`
- Stability metrics:
  `4e4ecf63127cecda46a19051caa331eb457025a58aaeece243f1e3a677a04bcf`

## Next controlled candidate

d7 returns to d5's balanced objective and 50% replay. It changes only peak
learning rate from `1e-4` to `5e-5`. This tests whether the stability failure
comes from moving the adapter too far from the fresh base rather than from
insufficient replay or false-target weighting.

Full evaluation, sealed `verify_v2`, QLoRA, full-weight tuning, reinforcement
learning, and promotion remain unauthorized.
