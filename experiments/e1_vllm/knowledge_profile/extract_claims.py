"""Structure index + corpus → typed claim bank (deterministic, no LLM)."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

from . import SCHEMA_CLAIM

# mill helpers for flag → prose answer (same snapshot corpus)
_MILL = Path(__file__).resolve().parent.parent / "mill"
if str(_MILL) not in sys.path:
    sys.path.insert(0, str(_MILL))
from generators.common import answers_for_flags  # noqa: E402

PRIORITY_FLAG_TOKENS = (
    "tensor-parallel",
    "pipeline-parallel",
    "data-parallel",
    "gpu-memory",
    "max-model-len",
    "dtype",
    "quant",
    "lora",
    "kv-cache",
    "max-num",
    "enable-",
    "trust-remote",
    "swap-space",
    "block-size",
    "seed",
    "port",
    "host",
    "api-key",
    "served-model",
    "tokenizer",
    "chat-template",
)

PRIORITY_PATH_TOKENS = (
    "cli/serve",
    "engine_args",
    "features/lora",
    "features/quantization",
    "serving/online_serving",
    "serving/parallelism",
    "configuration/",
    "getting_started",
    "models/",
    "README.md",
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_chunks(corpus_path: Path) -> dict[str, str]:
    chunks: dict[str, str] = {}
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            chunks[row["chunk_id"]] = row["text"]
    return chunks


def _symbol_index(symbols: list[dict]) -> dict[str, dict]:
    """First occurrence per (kind, name)."""
    out: dict[str, dict] = {}
    for sym in symbols:
        key = f"{sym['kind']}:{sym['name']}"
        if key not in out:
            out[key] = sym
    return out


def _centrality_flag(name: str) -> str:
    low = name.lower()
    if any(tok in low for tok in PRIORITY_FLAG_TOKENS):
        return "high"
    return "medium"


def _centrality_path(path: str) -> str:
    low = path.lower()
    if any(tok in low for tok in PRIORITY_PATH_TOKENS):
        return "high"
    if low.startswith("docs/"):
        return "medium"
    return "low"


def _claim(
    *,
    claim_id: str,
    claim_type: str,
    entity: str,
    predicate: str,
    obj,
    truth: dict,
    evidence: dict,
    tag: str,
    zone: str,
    centrality: str,
    commit_sha: str,
    content_sha: str,
) -> dict:
    return {
        "schema": SCHEMA_CLAIM,
        "claim_id": claim_id,
        "subject": "vllm",
        "source_revision": tag,
        "legacy_alias": "doc_0" if tag == "v0.22.0" else None,
        "claim_type": claim_type,
        "entity": entity,
        "predicate": predicate,
        "object": obj,
        "polarity": "affirmative",
        "truth": truth,
        "evidence": evidence,
        "valid_from_revision": tag,
        "valid_to_revision": None,
        "authority": "docs",
        "zone": zone,
        "centrality": centrality,
        "commit_sha": commit_sha,
        "content_sha": content_sha,
    }


def extract_claims(
    *,
    structure_path: Path,
    corpus_path: Path,
    include_meanings: bool = True,
) -> list[dict]:
    structure = _load_json(structure_path)
    tag = structure["tag"]
    commit_sha = structure["commit_sha"]
    content_sha = structure["content_sha"]
    symbols = structure["symbols"]
    paths = structure["paths"]
    by_sym = _symbol_index(symbols)
    chunks = _load_chunks(corpus_path)

    claims: list[dict] = []

    # --- flag exists ---
    flag_names: list[str] = []
    for key, sym in by_sym.items():
        if sym["kind"] != "flag":
            continue
        name = sym["name"]
        flag_names.append(name)
        cid = f"flag.exists.{name}"
        claims.append(
            _claim(
                claim_id=cid,
                claim_type="flag_exists",
                entity=name,
                predicate="documented_as_cli_flag",
                obj=True,
                truth={
                    "must_contain_any": [[name, "yes", "true"]],
                    "must_not_contain": ["unknown"],
                    "expected": "yes",
                },
                evidence={
                    "chunk_ids": [sym["chunk_id"]],
                    "source_paths": [sym.get("source_path", "")],
                    "span_hash": _sha(name),
                },
                tag=tag,
                zone="cli",
                centrality=_centrality_flag(name),
                commit_sha=commit_sha,
                content_sha=content_sha,
            )
        )

    # --- env exists ---
    for key, sym in by_sym.items():
        if sym["kind"] != "env":
            continue
        name = sym["name"]
        cid = f"env.exists.{name}"
        claims.append(
            _claim(
                claim_id=cid,
                claim_type="env_exists",
                entity=name,
                predicate="documented_as_env_var",
                obj=True,
                truth={
                    "must_contain_any": [[name, "yes", "true"]],
                    "must_not_contain": ["unknown"],
                    "expected": "yes",
                },
                evidence={
                    "chunk_ids": [sym["chunk_id"]],
                    "source_paths": [sym.get("source_path", "")],
                    "span_hash": _sha(name),
                },
                tag=tag,
                zone="env",
                centrality="high" if name.startswith("VLLM_") else "medium",
                commit_sha=commit_sha,
                content_sha=content_sha,
            )
        )

    # --- path exists ---
    for row in paths:
        src = row["source_path"]
        cid = f"path.exists.{src.replace('/', '.')}"
        claims.append(
            _claim(
                claim_id=cid,
                claim_type="path_exists",
                entity=src,
                predicate="exists_in_docs_tree",
                obj=True,
                truth={
                    "must_contain_any": [[src, Path(src).name, "yes", "true"]],
                    "must_not_contain": ["unknown"],
                    "expected": "yes",
                },
                evidence={
                    "chunk_ids": [],
                    "source_paths": [src],
                    "span_hash": _sha(src),
                },
                tag=tag,
                zone="docs_path",
                centrality=_centrality_path(src),
                commit_sha=commit_sha,
                content_sha=content_sha,
            )
        )

    # --- flag meaning (prose from corpus; only when answer extractable) ---
    if include_meanings and flag_names:
        answers = answers_for_flags(flag_names, chunks)
        for name, (answer, chunk_id) in answers.items():
            # require a few contentful tokens beyond the flag itself
            tokens = [
                t
                for t in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", answer)
                if t.lower() not in {name.lstrip("-").lower(), "vllm", "the", "and"}
            ]
            if len(tokens) < 2:
                continue
            needles = [name] + tokens[:3]
            sym = by_sym.get(f"flag:{name}", {})
            cid = f"flag.meaning.{name}"
            claims.append(
                _claim(
                    claim_id=cid,
                    claim_type="flag_meaning",
                    entity=name,
                    predicate="controls",
                    obj=answer[:400],
                    truth={"must_contain": needles[:4], "expected": "prose"},
                    evidence={
                        "chunk_ids": [chunk_id],
                        "source_paths": [sym.get("source_path", "")],
                        "span": answer[:400],
                        "span_hash": _sha(answer),
                    },
                    tag=tag,
                    zone="cli_meaning",
                    centrality=_centrality_flag(name),
                    commit_sha=commit_sha,
                    content_sha=content_sha,
                )
            )

    claims.sort(key=lambda c: c["claim_id"])
    return claims


def write_claims(claims: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in claims:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def claim_bank_hash(claims: list[dict]) -> str:
    payload = "\n".join(c["claim_id"] for c in claims)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
