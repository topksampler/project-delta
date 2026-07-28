# Grader audit pack — how to fill

## invariant

```text
You are checking the automatic scorer, not the model’s knowledge.
Agree = the auto label matches what a careful human would mark.
```

## what goes where

| file | role |
|------|------|
| `artifacts/reports/dense_v1_grader_audit_50.md` | readable worksheet |
| `artifacts/reports/dense_v1_grader_audit_50.jsonl` | machine form to edit |

## what can die

- your temporary notes while reading the markdown

## what must survive

- filled `human_agree_with_auto` / `human_correct_label` fields in the JSONL
- disagreement count by probe form

## command

Open the markdown, score each of the 50 items, then set fields in the JSONL:

```text
human_agree_with_auto: true | false
human_correct_label: correct | wrong | abstain_ok | unclear
human_notes: optional
```

Target: ≥40/50 agreements before we trust free-recall as a training gate.
Recognition / yes-no already use exact scorers; prioritize those if time is short.
