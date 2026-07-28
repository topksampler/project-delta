"""CLI for TopicKnowledgeProfile build / aggregate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent
HARNESS = PKG.parent
ROOT = HARNESS.parents[1]
sys.path.insert(0, str(HARNESS))
sys.path.insert(0, str(ROOT / "src"))

from knowledge_profile.aggregate import (  # noqa: E402
    aggregate_profile,
    load_jsonl,
    write_profile,
)
from knowledge_profile.extract_claims import (  # noqa: E402
    claim_bank_hash,
    extract_claims,
    write_claims,
)
from knowledge_profile.probes import build_probes, probe_bank_hash, write_probes  # noqa: E402
from knowledge_profile.sample import sample_claims  # noqa: E402


def _default_snapshot() -> Path:
    return ROOT / "data/experiments/e1_vllm/snapshots/v0.22.0"


def cmd_build_claims(args: argparse.Namespace) -> None:
    snap = Path(args.snapshot) if args.snapshot else _default_snapshot()
    structure = snap / "structure_index.json"
    corpus = snap / "corpus.jsonl"
    if not structure.is_file():
        raise SystemExit(f"missing {structure} — run mill extract first")
    if not corpus.is_file():
        raise SystemExit(f"missing {corpus}")

    claims = extract_claims(structure_path=structure, corpus_path=corpus)
    out = Path(args.out)
    write_claims(claims, out)
    meta = {
        "n_claims": len(claims),
        "claim_bank_hash": claim_bank_hash(claims),
        "by_type": {},
    }
    for c in claims:
        meta["by_type"][c["claim_type"]] = meta["by_type"].get(c["claim_type"], 0) + 1
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(claims)} claims → {out}")
    print(f"hash={meta['claim_bank_hash']} types={meta['by_type']}")


def cmd_build_probes(args: argparse.Namespace) -> None:
    claims = load_jsonl(Path(args.claims))
    if args.sample:
        claims = sample_claims(claims, seed=args.seed)
        print(f"sampled {len(claims)} claims for pilot")
    probes = build_probes(claims, seed=args.seed, n_paraphrases=args.paraphrases)
    out = Path(args.out)
    write_probes(probes, out)
    if args.claims_out:
        write_claims(claims, Path(args.claims_out))
        print(f"wrote sampled claims → {args.claims_out}")
    print(f"wrote {len(probes)} probes → {out}")
    print(f"probe_hash={probe_bank_hash(probes)}")


def cmd_aggregate(args: argparse.Namespace) -> None:
    claims = load_jsonl(Path(args.claims))
    samples = load_jsonl(Path(args.samples))
    metrics = {}
    if args.metrics and Path(args.metrics).is_file():
        metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    profile = aggregate_profile(
        claims=claims,
        samples=samples,
        model_name=metrics.get("model") or args.model,
        run_id=metrics.get("run_id") or args.run_id,
        source_revision=args.revision,
    )
    out = Path(args.out)
    write_profile(profile, out)
    agg = profile["aggregate"]
    print(f"wrote profile → {out}")
    print(
        f"claims={agg['n_claims']} probes={agg['n_probes']} "
        f"known_rate={agg['known_rate']} statuses={agg['status_counts']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="knowledge_profile")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("build-claims", help="structure_index → claim bank")
    p1.add_argument("--snapshot", default=None)
    p1.add_argument(
        "--out",
        default=str(ROOT / "data/experiments/e1_vllm/profile/claims_v0.22.0.jsonl"),
    )
    p1.set_defaults(func=cmd_build_claims)

    p2 = sub.add_parser("build-probes", help="claims → probe JSONL (eval.path)")
    p2.add_argument("--claims", required=True)
    p2.add_argument(
        "--out",
        default=str(ROOT / "data/experiments/e1_vllm/profile/probes_v0.22.0_pilot.jsonl"),
    )
    p2.add_argument(
        "--claims-out",
        default=str(ROOT / "data/experiments/e1_vllm/profile/claims_v0.22.0_pilot.jsonl"),
    )
    p2.add_argument("--sample", action="store_true", default=True)
    p2.add_argument("--no-sample", action="store_false", dest="sample")
    p2.add_argument("--seed", type=int, default=20260718)
    p2.add_argument("--paraphrases", type=int, default=3)
    p2.set_defaults(func=cmd_build_probes)

    p3 = sub.add_parser("aggregate", help="samples.jsonl → TopicKnowledgeProfile")
    p3.add_argument("--claims", required=True)
    p3.add_argument("--samples", required=True)
    p3.add_argument("--metrics", default=None)
    p3.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    p3.add_argument("--run-id", default="local")
    p3.add_argument("--revision", default="v0.22.0")
    p3.add_argument(
        "--out",
        default=str(ROOT / "artifacts/reports/topic_knowledge_profile_v0.22.0.json"),
    )
    p3.set_defaults(func=cmd_aggregate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
