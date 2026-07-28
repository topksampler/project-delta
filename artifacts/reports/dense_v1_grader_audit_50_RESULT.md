# Grader audit result (delegated)

Auditor: agent (user-delegated)  
Agree with auto: **42/50 (84%)** — passes ≥80% bar

## by form

| form | agree | disagree |
|------|------:|---------:|
| meaning_true | 12 | 0 |
| meaning_false | 12 | 0 |
| meaning_recognition | 10 | 4 |
| free_recall | 8 | 4 |

## findings

1. Boolean yes/no scorers are trustworthy — use them as primary honesty metrics.
2. Recognition scorer missed bare `B. ...` answers (format); fixed in `qa_score.score_choice`.
3. Free-recall keyword groups over-credit stopwords (`and`/`set`/`can`) — keep advisory only; do not gate training on them.

## decision

Proceed to LoRA on `train_v0.22.0_v5` and re-run dense profile. Headline metrics = recognition + true/false, not free-recall.
