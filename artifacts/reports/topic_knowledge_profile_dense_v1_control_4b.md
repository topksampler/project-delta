# Dense profile control — 0.8B vs 4B

Same verified bank: 85 claims, 996 probes, closed-book, thinking disabled for 4B.

Invalid prior run: `e1-vllm-profile-dense-v1-qwen35-4b-modal` burned tokens inside `<think>` and never answered. Discarded.

Valid control: `e1-vllm-profile-dense-v1-qwen35-4b-nthink-modal` ($0.82).

## result

| behavior | 0.8B | 4B |
|----------|-----:|---:|
| free recall (advisory) | 76.3% | 80.7% |
| meaning recognition (3-choice) | 34.5% | **89.6%** |
| accepts true meanings | 95.6% | 22.9% |
| rejects false meanings | 23.7% | 93.6% |
| meaning known rate | 9.4% | 65.9% |
| honesty failure rate | 95.3% | 97.6% |

Random recognition baseline = 33.3%.

## what this means

1. The quiz is solvable. 4B recognition at 89.6% falsifies “the instrument is broken.”
2. 0.8B recognition at chance is a real knowledge hole, not a bad test.
3. Honesty failures differ by model:
   - 0.8B → acquiescent (says yes to almost anything)
   - 4B → rejection-biased (says no even to true documented meanings)
4. Forced-choice recognition and yes/no verification measure different skills. 4B can pick the right meaning but still refuses to affirm it.

## implication for the data wheel

Train the 0.8B on:
1. correct flag→meaning associations (recognition-shaped);
2. contrastive false meanings;
3. abstention when unsure — without flipping into 4B-style blanket rejection.
