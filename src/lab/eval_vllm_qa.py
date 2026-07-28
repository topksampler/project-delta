"""e1_vllm short-answer eval with optional LoRA adapter and failure-mode labels."""

from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from rich import print
from transformers import AutoModelForCausalLM, AutoTokenizer

from lab.eval_base import append_jsonl, choose_device, read_jsonl, write_json
from lab.qa_score import classify_failure_mode, score_gold, score_must_contain


def _resolve(path: str | Path, repo_root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root / p


class RetrievalBundle:
    """Load corpus (+ BM25 index); return context for a question.

    mode=bm25   → lexical top-k (c1/c2/c4)
    mode=shuffle → seeded random k chunks from same corpus (c6 control)
    """

    def __init__(
        self,
        corpus_path: Path,
        index_path: Path | None = None,
        *,
        top_k: int = 4,
        max_chars: int = 3500,
        mode: str = "bm25",
        seed: int = 1337,
    ):
        self.top_k = top_k
        self.max_chars = max_chars
        self.mode = mode
        self.seed = seed
        self.texts: dict[str, str] = {}
        for row in read_jsonl(str(corpus_path)):
            self.texts[row["chunk_id"]] = row["text"]
        self.all_ids = list(self.texts.keys())
        self.index = None
        if mode == "bm25":
            if index_path is None or not index_path.exists():
                raise FileNotFoundError("bm25 mode requires index_path")
            harness = Path(__file__).resolve().parents[2] / "experiments" / "e1_vllm"
            if str(harness) not in sys.path:
                sys.path.insert(0, str(harness))
            from bm25 import BM25Index  # noqa: WPS433 — experiment harness

            payload = json.loads(index_path.read_text(encoding="utf-8"))
            self.index = BM25Index.from_dict(payload["index"])

    def _format(self, chunk_ids: list[str]) -> tuple[str, list[str]]:
        parts: list[str] = []
        used = 0
        for cid in chunk_ids:
            text = self.texts.get(cid, "").strip()
            if not text:
                continue
            block = f"[{cid}]\n{text}"
            if used + len(block) + 2 > self.max_chars and parts:
                break
            parts.append(block)
            used += len(block) + 2
        if not parts:
            return "", chunk_ids
        return "Retrieved documentation:\n\n" + "\n\n".join(parts), chunk_ids

    def retrieve(self, question: str) -> tuple[str, list[str]]:
        if self.mode == "shuffle":
            rng = random.Random(f"{self.seed}:{question}")
            k = min(self.top_k, len(self.all_ids))
            chunk_ids = rng.sample(self.all_ids, k) if k else []
            return self._format(chunk_ids)
        assert self.index is not None
        hits = self.index.top_k(question, self.top_k)
        return self._format([cid for cid, _ in hits])


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def resolve_adapter_path(cfg: dict, repo_root: Path) -> Path | None:
    model_cfg = cfg.get("model", {})
    if model_cfg.get("adapter_path"):
        path = Path(model_cfg["adapter_path"])
        return path if path.is_absolute() else repo_root / path
    run_id = model_cfg.get("adapter_run_id")
    if run_id:
        return repo_root / "runs" / run_id / "adapter"
    return None


def load_model(cfg: dict, device: str, repo_root: Path):
    model_name = cfg["model"]["name"]
    trust = cfg["model"].get("trust_remote_code", False)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=trust,
    )
    adapter_path = resolve_adapter_path(cfg, repo_root)
    if adapter_path and adapter_path.exists():
        print(f"loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(base, str(adapter_path))
    else:
        if adapter_path:
            print(f"[yellow]adapter missing at {adapter_path}, using base[/yellow]")
        model = base
    model.to(device)
    model.eval()
    return model, tokenizer


def _chat_template_kwargs(eval_cfg: dict) -> dict:
    """Qwen3.5+ defaults to open <think>; disable unless explicitly requested."""
    return {"enable_thinking": bool(eval_cfg.get("enable_thinking", False))}


def generate_answer(
    model,
    tokenizer,
    *,
    question: str,
    device: str,
    eval_cfg: dict,
    context: str = "",
) -> str:
    user_content = question
    if context:
        user_content = f"{context}\n\nQuestion: {question}"
    messages = [
        {
            "role": "system",
            "content": (
                "You answer questions about vLLM documentation. "
                "Use the retrieved documentation when provided. "
                "Reply in 1–3 sentences. If unsure, say unknown."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **_chat_template_kwargs(eval_cfg),
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    temperature = float(eval_cfg.get("temperature", 0.0))
    gen_kwargs = {
        "max_new_tokens": int(eval_cfg.get("max_new_tokens", 256)),
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature <= 0.0:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = float(eval_cfg.get("top_p", 0.9))

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
    generated = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def generate_answers(
    model,
    tokenizer,
    *,
    questions: list[str],
    device: str,
    eval_cfg: dict,
    contexts: list[str] | None = None,
) -> list[str]:
    """Batched closed-book/context generation; materially faster for profile runs."""
    contexts = contexts or [""] * len(questions)
    texts: list[str] = []
    for question, context in zip(questions, contexts, strict=True):
        user_content = f"{context}\n\nQuestion: {question}" if context else question
        messages = [
            {
                "role": "system",
                "content": (
                    "You answer questions about vLLM documentation. "
                    "Use the retrieved documentation when provided. "
                    "Reply in 1–3 sentences. If unsure, say unknown."
                ),
            },
            {"role": "user", "content": user_content},
        ]
        texts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **_chat_template_kwargs(eval_cfg),
            )
        )

    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    temperature = float(eval_cfg.get("temperature", 0.0))
    gen_kwargs = {
        "max_new_tokens": int(eval_cfg.get("max_new_tokens", 256)),
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature <= 0.0:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs.update(
            do_sample=True,
            temperature=temperature,
            top_p=float(eval_cfg.get("top_p", 0.9)),
        )
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
    prompt_len = inputs["input_ids"].shape[1]
    outputs = [
        tokenizer.decode(row[prompt_len:], skip_special_tokens=True).strip()
        for row in output_ids
    ]
    tokenizer.padding_side = previous_padding_side
    return outputs


def aggregate(rows: list[dict]) -> dict:
    by_type: dict[str, list[float]] = defaultdict(list)
    by_class: dict[str, list[float]] = defaultdict(list)
    by_requires: dict[str, list[float]] = defaultdict(list)
    by_failure: dict[str, int] = defaultdict(int)
    for row in rows:
        by_type[row["type"]].append(row["content_score"])
        by_class[row.get("eval_class", "?")].append(row["content_score"])
        by_requires[row["requires_doc"]].append(row["content_score"])
        by_failure[row["failure_mode"]] += 1
    return {
        "by_type": {k: float(sum(v) / len(v)) for k, v in by_type.items()},
        "by_eval_class": {k: float(sum(v) / len(v)) for k, v in by_class.items()},
        "by_requires_doc": {k: float(sum(v) / len(v)) for k, v in by_requires.items()},
        "by_failure_mode": {k: int(v) for k, v in by_failure.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    repo_root = Path(__file__).resolve().parents[2]
    run_id = cfg["run_id"]
    eval_cfg = cfg["eval"]
    eval_path = Path(eval_cfg["path"])
    if not eval_path.is_absolute():
        eval_path = repo_root / eval_path
    out_dir = Path(eval_cfg["output_dir"])
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir

    samples_path = out_dir / "samples.jsonl"
    metrics_path = out_dir / "metrics.json"
    ledger_path = out_dir / "ledger.yaml"
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path.unlink(missing_ok=True)

    device = choose_device()
    print(f"[bold]e1_vllm eval[/bold] run_id={run_id} device={device}")
    model, tokenizer = load_model(cfg, device, repo_root)

    retrieval_cfg = cfg.get("retrieval") or {}
    retriever: RetrievalBundle | None = None
    if retrieval_cfg.get("enabled"):
        corpus_path = _resolve(retrieval_cfg["corpus_path"], repo_root)
        index_rel = retrieval_cfg.get("index_path")
        index_path = _resolve(index_rel, repo_root) if index_rel else None
        mode = str(retrieval_cfg.get("mode", "bm25"))
        retriever = RetrievalBundle(
            corpus_path,
            index_path,
            top_k=int(retrieval_cfg.get("top_k", 4)),
            max_chars=int(retrieval_cfg.get("max_chars", 3500)),
            mode=mode,
            seed=int(retrieval_cfg.get("seed", 1337)),
        )
        print(
            f"retrieval enabled mode={mode} corpus={corpus_path.name} "
            f"top_k={retriever.top_k}"
        )

    rows: list[dict] = []
    total_score = 0.0
    n = 0
    retrieval_nonempty = 0

    examples = read_jsonl(str(eval_path))
    prepared: list[tuple[dict, str, list[str]]] = []
    for ex in examples:
        context = ""
        retrieved_chunk_ids: list[str] = []
        if retriever is not None:
            context, retrieved_chunk_ids = retriever.retrieve(ex["question"])
            if retrieved_chunk_ids:
                retrieval_nonempty += 1
        prepared.append((ex, context, retrieved_chunk_ids))

    batch_size = max(1, int(eval_cfg.get("batch_size", 1)))
    for start in range(0, len(prepared), batch_size):
        batch = prepared[start : start + batch_size]
        outputs = generate_answers(
            model,
            tokenizer,
            questions=[item[0]["question"] for item in batch],
            contexts=[item[1] for item in batch],
            device=device,
            eval_cfg=eval_cfg,
        )
        for (ex, _context, retrieved_chunk_ids), output in zip(
            batch, outputs, strict=True
        ):
            n += 1
            question = ex["question"]
            gold = ex.get("gold") or {}
            needles = gold.get("must_contain", [])
            requires_doc = ex.get("requires_doc", "?")
            if (
                gold.get("choice")
                or gold.get("boolean")
                or gold.get("must_contain_any")
                or gold.get("must_not_contain")
                or gold.get("abstain_if_unknown")
            ):
                score, hits, misses = score_gold(output, gold)
            else:
                score, hits, misses = score_must_contain(output, needles)
            failure_mode = classify_failure_mode(
                score=score,
                output=output,
                requires_doc=requires_doc,
                misses=misses,
                needles=needles,
                gold=gold,
            )
            total_score += score

            row = {
                "id": ex["id"],
                "eval_class": ex.get("eval_class", "?"),
                "type": ex.get("type", "unknown"),
                "requires_doc": requires_doc,
                "question": question,
                "gold_must_contain": needles,
                "output": output,
                "content_score": score,
                "hits": hits,
                "misses": misses,
                "failure_mode": failure_mode,
            }
            for key in (
                "claim_id",
                "probe_form",
                "paraphrase_idx",
                "zone",
                "centrality",
            ):
                if key in ex:
                    row[key] = ex[key]
            if retriever is not None:
                row["retrieved_chunk_ids"] = retrieved_chunk_ids
            append_jsonl(samples_path, row)
            rows.append(row)
            print(f"{ex['id']}: score={score:.2f} mode={failure_mode}")

    avg = total_score / max(1, n)
    metrics = {
        "run_id": run_id,
        "experiment_id": cfg.get("experiment_id"),
        "condition_id": cfg.get("condition_id"),
        "model": cfg["model"]["name"],
        "adapter_run_id": cfg.get("model", {}).get("adapter_run_id"),
        "num_examples": n,
        "avg_content_score": round(avg, 4),
        "accuracy": round(sum(1 for r in rows if r["failure_mode"] == "correct") / max(1, n), 4),
        "breakdown": aggregate(rows),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if retriever is not None:
        metrics["retrieval"] = {
            "enabled": True,
            "mode": retriever.mode,
            "top_k": retriever.top_k,
            "seed": retriever.seed if retriever.mode == "shuffle" else None,
            "hit_rate_nonempty": round(retrieval_nonempty / max(1, n), 4),
        }

    write_json(metrics_path, metrics)
    ledger = {
        "run_id": run_id,
        "type": "e1_vllm_qa",
        "model": cfg["model"]["name"],
        "adapter_run_id": cfg.get("model", {}).get("adapter_run_id"),
        "device": device,
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "metrics": metrics,
    }
    with open(ledger_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(ledger, f, sort_keys=False)
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"\n[bold green]eval complete[/bold green] avg={avg:.3f} accuracy={metrics['accuracy']:.3f}")
    print(f"metrics: {metrics_path}")


if __name__ == "__main__":
    main()
