"""Generate held-out eval wording without allowing the teacher to define gold."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _tokens(text: str) -> set[str]:
    return set(WORD_RE.findall(text.lower()))


def _jaccard(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    return len(a & b) / len(a | b) if a and b else 0.0


def _required_literals(row: dict) -> list[str]:
    literals = [row["display_entity"]]
    literals.extend(re.findall(r"\b\d+\.\d+\.\d+\b", row["question"]))
    return literals


def _clean(text: str) -> str:
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.strip("`").strip()
    text = re.sub(r"^(?:question|rewritten question)\s*:\s*", "", text, flags=re.I)
    text = re.sub(
        r"\?\s*(?:please\s+)?(?:respond|answer)(?:\s+with)?\s+yes\s+or\s+no\.?\s*$",
        "?",
        text,
        flags=re.I,
    )
    return text.strip().strip('"').strip()


def _validate_surface(seed: dict, question: str, max_seed_jaccard: float) -> str | None:
    if not question.endswith("?"):
        return "not_a_question"
    if len(question) < 20 or len(question) > 300:
        return "length"
    for literal in _required_literals(seed):
        if literal.lower() not in question.lower():
            return f"missing_literal:{literal}"
    overlap = _jaccard(seed["question"], question)
    if overlap > max_seed_jaccard:
        return f"seed_jaccard:{overlap:.4f}"
    if seed["probe_form"] == "version_delta":
        neutral_markers = (
            "whether",
            "if any",
            "if so",
            "same",
            "differ",
            "difference",
        )
        if not any(marker in question.lower() for marker in neutral_markers):
            return "delta_question_presupposes_change"
    return None


def _prompt(row: dict) -> list[dict]:
    forbidden = (
        "Do not use the phrase 'is a recognized'."
        if row["probe_form"] == "versioned_existence"
        else (
            "Do not use the phrase 'what happened to'. Ask neutrally whether "
            "and how the flag differed; never assume that a change occurred."
        )
    )
    return [
        {
            "role": "system",
            "content": (
                "You rewrite technical evaluation questions. Preserve the exact "
                "technical intent, code identifier, and version numbers. Never "
                "answer the question or add facts. Return exactly one question."
            ),
        },
        {
            "role": "user",
            "content": (
                "Rewrite the question with substantially different syntax and "
                f"wording. {forbidden}\n\nOriginal: {row['question']}"
            ),
        },
    ]


def generate(config_path: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    repo_root = Path(__file__).resolve().parents[3]
    input_path = repo_root / cfg["eval"]["path"]
    output_dir = repo_root / cfg["eval"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    model_id = cfg["model"]["id"]
    generator_id = cfg["generation"]["id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=bool(cfg["model"].get("trust_remote_code", False)),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    seeds = _load_jsonl(input_path)
    batch_size = int(cfg["generation"].get("batch_size", 16))
    max_seed_jaccard = float(cfg["generation"].get("max_seed_jaccard", 0.7))
    accepted: list[dict] = []
    rejected: list[dict] = []
    prompt_payloads: list[tuple[dict, str]] = []
    for row in seeds:
        messages = _prompt(row)
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_payloads.append((row, rendered))

    for start in range(0, len(prompt_payloads), batch_size):
        batch = prompt_payloads[start : start + batch_size]
        inputs = tokenizer(
            [rendered for _, rendered in batch],
            return_tensors="pt",
            padding=True,
        ).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=int(cfg["generation"].get("max_new_tokens", 96)),
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        for (seed, rendered), token_ids in zip(batch, output_ids, strict=True):
            question = _clean(
                tokenizer.decode(token_ids[prompt_len:], skip_special_tokens=True)
            )
            reason = _validate_surface(seed, question, max_seed_jaccard)
            prompt_sha = hashlib.sha256(rendered.encode()).hexdigest()
            if reason:
                rejected.append(
                    {
                        "seed_id": seed["id"],
                        "candidate_question": question,
                        "reject_reason": reason,
                        "prompt_sha256": prompt_sha,
                    }
                )
                continue
            accepted.append(
                {
                    **seed,
                    "id": f"{seed['id']}:teacher",
                    "seed_id": seed["id"],
                    "question": question,
                    "seed_question": seed["question"],
                    "generator": {
                        "id": generator_id,
                        "model_id": model_id,
                        "decoding": "greedy",
                        "prompt_sha256": prompt_sha,
                    },
                }
            )

    _write_jsonl(output_dir / "probes_eval_generated.jsonl", accepted)
    _write_jsonl(output_dir / "rejected.jsonl", rejected)
    metrics = {
        "schema": "delta.eval_factory.surface_generation.v1",
        "run_id": cfg["run_id"],
        "model_id": model_id,
        "generator_id": generator_id,
        "n_seed": len(seeds),
        "n_accepted": len(accepted),
        "n_rejected": len(rejected),
        "accept_rate": len(accepted) / max(1, len(seeds)),
        "max_seed_jaccard": max_seed_jaccard,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.config), indent=2))


if __name__ == "__main__":
    main()
