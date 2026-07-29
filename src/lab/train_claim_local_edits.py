"""Sequential claim-local LoRA micro-edits (Experiment B).

Loads a claim-local edits manifest, runs short DPO steps per claim with
stable replay, scores a fixed stable canary after each edit, aborts (rolls
back) an edit when canary exact drop exceeds budget, and writes the composed
adapter plus an interference curve.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from lab.eval_base import choose_device
from lab.eval_run_spec import read_jsonl
from lab.eval_vllm_qa import generate_answers
from lab.qa_score import score_gold
from lab.train_sft_lora import resolve_continue_adapter

DPO_COLUMNS = ("prompt", "chosen", "rejected")


def _resolve(path: str | Path, repo_root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root / p


def _snapshot_lora(model: PeftModel) -> dict[str, torch.Tensor]:
    return {
        k: v.detach().cpu().clone()
        for k, v in get_peft_model_state_dict(model).items()
    }


def _restore_lora(model: PeftModel, snap: dict[str, torch.Tensor]) -> None:
    device = next(model.parameters()).device
    payload = {k: v.to(device) for k, v in snap.items()}
    set_peft_model_state_dict(model, payload)


def _canary_exact(
    model,
    tokenizer,
    *,
    probes: list[dict],
    device: str,
    eval_cfg: dict,
    batch_size: int = 8,
) -> float:
    was_training = model.training
    model.eval()
    scores: list[float] = []
    for start in range(0, len(probes), batch_size):
        batch = probes[start : start + batch_size]
        answers = generate_answers(
            model,
            tokenizer,
            questions=[p["question"] for p in batch],
            device=device,
            eval_cfg=eval_cfg,
        )
        for probe, answer in zip(batch, answers, strict=True):
            score, _, _ = score_gold(answer, probe.get("gold") or {})
            scores.append(1.0 if score >= 1.0 else 0.0)
    if was_training:
        model.train()
    return sum(scores) / len(scores) if scores else 0.0


def _run_micro_dpo(
    *,
    model: PeftModel,
    tokenizer,
    train_rows: list[dict],
    out_dir: Path,
    training_config: dict,
) -> PeftModel:
    train_dataset = Dataset.from_list(train_rows).select_columns(list(DPO_COLUMNS))
    eval_rows = train_rows[: min(4, len(train_rows))]
    eval_dataset = Dataset.from_list(eval_rows).select_columns(list(DPO_COLUMNS))
    micro_steps = int(training_config.get("micro_steps", training_config["max_steps"]))
    dpo_args = DPOConfig(
        output_dir=str(out_dir / "_micro_trainer"),
        max_steps=micro_steps,
        learning_rate=float(training_config["learning_rate"]),
        per_device_train_batch_size=int(
            training_config["per_device_train_batch_size"]
        ),
        gradient_accumulation_steps=int(
            training_config["gradient_accumulation_steps"]
        ),
        logging_steps=int(training_config.get("logging_steps", 5)),
        save_strategy="no",
        bf16=bool(training_config.get("bf16", True)),
        fp16=bool(training_config.get("fp16", False)),
        beta=float(training_config.get("beta", 0.1)),
        max_length=int(training_config.get("max_length", 1024)),
        remove_unused_columns=False,
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        args=dpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    return trainer.model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_id = cfg["run_id"]
    model_name = cfg["model"]["name"]
    out_dir = Path(cfg["output"]["dir"])
    repo_root = Path(__file__).resolve().parents[2]
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = cfg.get("data") or {}
    manifest_rel = data_cfg.get("manifest_path")
    if not manifest_rel:
        raise ValueError("data.manifest_path is required for claim-local edits")
    manifest_path = _resolve(manifest_rel, repo_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    canary_path = _resolve(manifest["canary_path"], repo_root)
    canary_probes = list(read_jsonl(str(canary_path)))

    training_config = cfg["training"]
    canary_budget = float(
        training_config.get("canary_budget", manifest.get("canary_budget", 0.02))
    )
    eval_cfg = {
        "max_new_tokens": int(training_config.get("canary_max_new_tokens", 64)),
        "temperature": 0.0,
        "top_p": 1.0,
        "enable_thinking": False,
    }

    device = choose_device()
    print(f"Using device: {device}")
    print(
        f"claim-local edits: n={manifest['n_edits']} "
        f"canary_n={len(canary_probes)} budget={canary_budget}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=cfg["model"].get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=cfg["model"].get("trust_remote_code", False),
    )

    continue_adapter = resolve_continue_adapter(cfg, repo_root)
    if continue_adapter and continue_adapter.exists():
        print(f"continuing LoRA from {continue_adapter}")
        model = PeftModel.from_pretrained(
            base, str(continue_adapter), is_trainable=True
        )
    else:
        if continue_adapter:
            raise FileNotFoundError(
                f"continue adapter missing at {continue_adapter}"
            )
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=cfg["lora"]["r"],
            lora_alpha=cfg["lora"]["alpha"],
            lora_dropout=cfg["lora"]["dropout"],
            target_modules=cfg["lora"]["target_modules"],
        )
        # Attach zero LoRA before the loop so every edit can roll back.
        model = get_peft_model(base, peft_config)

    model.to(device)
    model.train()

    baseline = _canary_exact(
        model,
        tokenizer,
        probes=canary_probes,
        device=device,
        eval_cfg=eval_cfg,
    )

    curve: list[dict] = [
        {
            "edit_index": -1,
            "claim_id": None,
            "status": "baseline",
            "canary_exact": round(baseline, 4),
            "drop_vs_baseline": 0.0,
            "accepted": True,
        }
    ]
    print(f"canary baseline exact={baseline:.4f}")

    accepted = 0
    aborted = 0
    for edit in manifest["edits"]:
        train_path = _resolve(edit["train_path"], repo_root)
        train_rows = list(read_jsonl(str(train_path)))
        if not train_rows:
            raise RuntimeError(f"empty micro-edit train set: {train_path}")

        pre_snap = _snapshot_lora(model)
        edit_out = out_dir / "edits" / f"{edit['index']:02d}"
        edit_out.mkdir(parents=True, exist_ok=True)
        print(
            f"edit {edit['index']}: claim={edit['claim_id']} "
            f"drift={edit['primary_drift']} n_pairs={len(train_rows)}"
        )
        model = _run_micro_dpo(
            model=model,
            tokenizer=tokenizer,
            train_rows=train_rows,
            out_dir=edit_out,
            training_config=training_config,
        )
        model.train()

        canary = _canary_exact(
            model,
            tokenizer,
            probes=canary_probes,
            device=device,
            eval_cfg=eval_cfg,
        )
        drop = baseline - canary
        if drop > canary_budget + 1e-9:
            _restore_lora(model, pre_snap)
            canary_after = _canary_exact(
                model,
                tokenizer,
                probes=canary_probes,
                device=device,
                eval_cfg=eval_cfg,
            )
            aborted += 1
            point = {
                "edit_index": edit["index"],
                "claim_id": edit["claim_id"],
                "primary_drift": edit["primary_drift"],
                "status": "aborted_canary",
                "canary_exact_pre_rollback": round(canary, 4),
                "canary_exact": round(canary_after, 4),
                "drop_vs_baseline": round(baseline - canary_after, 4),
                "attempt_drop": round(drop, 4),
                "accepted": False,
            }
            print(
                f"  ABORT canary drop={drop:.4f} > budget={canary_budget}; "
                f"restored canary={canary_after:.4f}"
            )
        else:
            accepted += 1
            model.save_pretrained(edit_out / "adapter")
            point = {
                "edit_index": edit["index"],
                "claim_id": edit["claim_id"],
                "primary_drift": edit["primary_drift"],
                "status": "accepted",
                "canary_exact": round(canary, 4),
                "drop_vs_baseline": round(drop, 4),
                "accepted": True,
            }
            print(f"  OK canary={canary:.4f} drop={drop:.4f}")
        curve.append(point)

    adapter_dir = out_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    interference = {
        "schema": "delta.eval_factory.claim_local_interference.v1",
        "run_id": run_id,
        "manifest": str(manifest_path),
        "baseline_canary_exact": round(baseline, 4),
        "canary_budget": canary_budget,
        "n_edits_attempted": len(manifest["edits"]),
        "n_accepted": accepted,
        "n_aborted": aborted,
        "final_canary_exact": curve[-1]["canary_exact"],
        "final_drop_vs_baseline": curve[-1]["drop_vs_baseline"],
        "curve": curve,
        "caps": {
            "max_claims": manifest.get("max_claims"),
            "n_delta_claims_available": manifest.get("n_delta_claims_available"),
            "cap_reason": manifest.get("cap_reason"),
            "micro_steps": int(
                training_config.get("micro_steps", training_config["max_steps"])
            ),
            "stable_replay_n": manifest.get("stable_replay_n"),
            "canary_n": manifest.get("canary_n"),
        },
    }
    curve_path = out_dir / "interference_curve.json"
    curve_path.write_text(
        json.dumps(interference, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with open(out_dir / "ledger.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    (out_dir / "manifest_used.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        f"Training complete: run_id={run_id} accepted={accepted} aborted={aborted} "
        f"final_canary={curve[-1]['canary_exact']} output={out_dir}"
    )


if __name__ == "__main__":
    main()
