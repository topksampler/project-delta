"""Build TruthSource arm claim banks + shared bridge probes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
HARNESS = PKG.parent
ROOT = HARNESS.parents[1]
sys.path.insert(0, str(HARNESS))

from knowledge_profile.extract_claims import extract_claims, write_claims  # noqa: E402
from knowledge_profile.probes import build_probes, write_probes  # noqa: E402
from knowledge_profile.truthsource.extract_code import (  # noqa: E402
    extract_distill,
    extract_full,
    write_json,
)

SCHEMA = "delta.claim.v1"


def _claim(
    *,
    claim_id: str,
    claim_type: str,
    entity: str,
    arm: str,
    authority: str,
    evidence: dict,
    truth: dict,
    zone: str,
    tag: str = "v0.22.0",
    bridge: bool = False,
    obj=True,
) -> dict:
    return {
        "schema": SCHEMA,
        "claim_id": claim_id,
        "subject": "vllm",
        "source_revision": tag,
        "legacy_alias": "doc_0",
        "claim_type": claim_type,
        "entity": entity,
        "predicate": "exists_or_means",
        "object": obj,
        "polarity": "affirmative",
        "truth": truth,
        "evidence": evidence,
        "valid_from_revision": tag,
        "valid_to_revision": None,
        "authority": authority,
        "zone": zone,
        "centrality": "high" if bridge else "medium",
        "arm": arm,
        "bridge": bridge,
        "commit_sha": None,
        "content_sha": None,
    }


def _docs_flag_set(docs_claims: list[dict]) -> set[str]:
    return {
        c["entity"]
        for c in docs_claims
        if c["claim_type"] in ("flag_exists", "flag_meaning")
    }


def _meaning_by_flag(docs_claims: list[dict]) -> dict[str, dict]:
    return {c["entity"]: c for c in docs_claims if c["claim_type"] == "flag_meaning"}


def build_arms(
    *,
    snapshot: Path,
    out_dir: Path,
    bridge_limit: int = 40,
    private_docs: int = 30,
    private_full_symbols: int = 80,
) -> dict:
    structure = snapshot / "structure_index.json"
    corpus = snapshot / "corpus.jsonl"
    docs_claims = extract_claims(structure_path=structure, corpus_path=corpus)
    distill = extract_distill(snapshot)
    full = extract_full(snapshot, max_symbols=600)

    write_json(distill, out_dir / "extract_C_distill.json")
    write_json(full, out_dir / "extract_C_full.json")

    docs_flags = _docs_flag_set(docs_claims)
    distill_flags = {f["name"] for f in distill["flags"]}
    bridge_flags = sorted(docs_flags & distill_flags)[:bridge_limit]
    meanings = _meaning_by_flag(docs_claims)

    # --- bridge claims (identical IDs across arms; authority differs) ---
    bridge_docs: list[dict] = []
    bridge_distill: list[dict] = []
    bridge_full: list[dict] = []

    for name in bridge_flags:
        cid_exists = f"bridge.flag.exists.{name}"
        truth_exists = {
            "must_contain_any": [[name, "yes", "true"]],
            "must_not_contain": ["unknown"],
        }
        evid_docs = {"source_paths": ["docs"], "note": "structure_index"}
        evid_code = {"source_paths": ["vllm/engine/arg_utils.py"], "rule": "arg_utils"}

        bridge_docs.append(
            _claim(
                claim_id=cid_exists,
                claim_type="flag_exists",
                entity=name,
                arm="D",
                authority="docs",
                evidence=evid_docs,
                truth=truth_exists,
                zone="bridge",
                bridge=True,
            )
        )
        bridge_distill.append(
            _claim(
                claim_id=cid_exists,
                claim_type="flag_exists",
                entity=name,
                arm="C_distill",
                authority="code_distill",
                evidence=evid_code,
                truth=truth_exists,
                zone="bridge",
                bridge=True,
            )
        )
        bridge_full.append(
            _claim(
                claim_id=cid_exists,
                claim_type="flag_exists",
                entity=name,
                arm="C_full",
                authority="code_full",
                evidence=evid_code,
                truth=truth_exists,
                zone="bridge",
                bridge=True,
            )
        )

        if name in meanings:
            m = meanings[name]
            cid_m = f"bridge.flag.meaning.{name}"
            truth_m = m.get("truth") or {"must_contain": [name]}
            for arm, bucket, auth in (
                ("D", bridge_docs, "docs"),
                ("C_distill", bridge_distill, "code_distill"),
                ("C_full", bridge_full, "code_full"),
            ):
                # meaning gold always from docs span; code arms still ask same question
                bucket.append(
                    _claim(
                        claim_id=cid_m,
                        claim_type="flag_meaning",
                        entity=name,
                        arm=arm,
                        authority=auth if arm != "D" else "docs",
                        evidence=m.get("evidence") or evid_docs,
                        truth=truth_m,
                        zone="bridge",
                        bridge=True,
                        obj=m.get("object"),
                    )
                )

    # --- arm-private ---
    private_d: list[dict] = []
    # docs-only meanings not in bridge
    for c in docs_claims:
        if c["claim_type"] != "flag_meaning":
            continue
        if c["entity"] in bridge_flags:
            continue
        private_d.append(
            {
                **c,
                "claim_id": f"D.{c['claim_id']}",
                "arm": "D",
                "bridge": False,
                "centrality": c.get("centrality", "medium"),
            }
        )
        if len(private_d) >= private_docs:
            break

    private_distill: list[dict] = []
    for exp in distill["exports"][:40]:
        private_distill.append(
            _claim(
                claim_id=f"C_distill.export.exists.{exp['name']}",
                claim_type="export_exists",
                entity=exp["name"],
                arm="C_distill",
                authority="code_distill",
                evidence={"source_paths": [exp["source_path"]], "target": exp.get("target")},
                truth={
                    "must_contain_any": [[exp["name"], "yes", "true"]],
                    "must_not_contain": ["unknown"],
                },
                zone="export",
            )
        )
    for env in distill["envs"][:40]:
        private_distill.append(
            _claim(
                claim_id=f"C_distill.env.exists.{env['name']}",
                claim_type="env_exists",
                entity=env["name"],
                arm="C_distill",
                authority="code_distill",
                evidence={"source_paths": [env["source_path"]]},
                truth={
                    "must_contain_any": [[env["name"], "yes", "true"]],
                    "must_not_contain": ["unknown"],
                },
                zone="env",
            )
        )

    private_full: list[dict] = []
    for sym in full["symbols"][:private_full_symbols]:
        private_full.append(
            _claim(
                claim_id=f"C_full.symbol.exists.{sym['qualname'].replace('/', '.')}",
                claim_type="symbol_exists",
                entity=sym["qualname"],
                arm="C_full",
                authority="code_full",
                evidence={"source_paths": [sym["source_path"]]},
                truth={
                    "must_contain_any": [[sym["name"], "yes", "true"]],
                    "must_not_contain": ["unknown"],
                },
                zone="symbol",
            )
        )

    arms = {
        "D": bridge_docs + private_d,
        "C_distill": bridge_distill + private_distill,
        "C_full": bridge_full + private_full,
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    # Freeze bridge probes ONCE from bridge claims + bridge flag universe only.
    # All arms must share these exact questions (hard-negatives included).
    bridge_canonical = bridge_docs  # same entities/gold as other bridge lists
    write_claims(bridge_canonical, out_dir / "claims_bridge.jsonl")
    bridge_probes = build_probes(
        bridge_canonical,
        seed=20260718,
        n_paraphrases=3,
        flag_universe=bridge_flags,
    )
    write_probes(bridge_probes, out_dir / "probes_bridge.jsonl")

    meta = {
        "bridge_flags": bridge_flags,
        "n_bridge_flags": len(bridge_flags),
        "n_bridge_claims_per_arm": len(bridge_docs),
        "n_bridge_probes": len(bridge_probes),
        "bridge_probes_frozen": True,
        "arm_sizes": {k: len(v) for k, v in arms.items()},
        "distill_stats": distill["stats"],
        "full_stats": full["stats"],
    }
    (out_dir / "arms_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    probe_meta = {"bridge": {"n_claims": len(bridge_canonical), "n_probes": len(bridge_probes)}}
    for arm, claims in arms.items():
        cpath = out_dir / f"claims_{arm}.jsonl"
        ppath = out_dir / f"probes_{arm}.jsonl"
        write_claims(claims, cpath)
        private = [c for c in claims if not c.get("bridge")]
        private_probes = _build_arm_probes(
            private, flag_universe=bridge_flags if arm == "D" else None
        )
        # bridge first (identical bytes across arms), then private
        probes = list(bridge_probes) + private_probes
        write_probes(probes, ppath)
        # verify bridge prefix identity
        with open(ppath, encoding="utf-8") as f:
            head = [next(f) for _ in range(len(bridge_probes))]
        with open(out_dir / "probes_bridge.jsonl", encoding="utf-8") as f:
            expected = f.readlines()
        if head != expected:
            raise RuntimeError(f"bridge probe freeze broken for arm {arm}")
        probe_meta[arm] = {
            "n_claims": len(claims),
            "n_probes": len(probes),
            "n_bridge_probes": len(bridge_probes),
            "n_private_probes": len(private_probes),
        }
        print(f"{arm}: claims={len(claims)} probes={len(probes)} (bridge={len(bridge_probes)}) → {ppath}")

    (out_dir / "probe_meta.json").write_text(json.dumps(probe_meta, indent=2) + "\n", encoding="utf-8")
    return meta


def _build_arm_probes(
    claims: list[dict],
    *,
    flag_universe: list[str] | None = None,
) -> list[dict]:
    from knowledge_profile.probes import _probe  # noqa: WPS437

    standard = [
        c
        for c in claims
        if c["claim_type"] in ("flag_exists", "flag_meaning", "env_exists", "path_exists")
    ]
    probes = (
        build_probes(
            standard,
            seed=20260719,
            n_paraphrases=3,
            flag_universe=flag_universe,
        )
        if standard
        else []
    )

    for c in claims:
        if c["claim_type"] == "export_exists":
            name = c["entity"]
            qs = [
                f"Does the public vLLM Python API export `{name}` in v0.22.0?",
                f"Is `{name}` a documented public export of the vLLM package in 0.22.0?",
                f"Can users import `{name}` as a public vLLM symbol in version 0.22.0?",
            ]
            for i, q in enumerate(qs):
                probes.append(
                    _probe(
                        claim=c,
                        probe_form="recall",
                        paraphrase_idx=i,
                        question=q,
                        gold={
                            "must_contain_any": [[name, "yes", "true"]],
                            "must_not_contain": ["unknown"],
                        },
                    )
                )
            fake = f"{name}Shadow"
            nqs = [
                f"Does the public vLLM Python API export `{fake}` in v0.22.0?",
                f"Is `{fake}` a documented public export of the vLLM package in 0.22.0?",
                f"Can users import `{fake}` as a public vLLM symbol in version 0.22.0?",
            ]
            for i, q in enumerate(nqs):
                probes.append(
                    _probe(
                        claim=c,
                        probe_form="negation",
                        paraphrase_idx=i,
                        question=q,
                        gold={
                            "must_contain_any": [
                                ["no", "unknown", "not", "does not", "doesn't"]
                            ],
                            "abstain_if_unknown": True,
                        },
                    )
                )
        elif c["claim_type"] == "symbol_exists":
            name = c["entity"].split(":")[-1]
            qual = c["entity"]
            qs = [
                f"In vLLM 0.22.0 source, does `{qual}` exist as a top-level class or function?",
                f"Is `{name}` defined at `{qual}` in the vLLM 0.22.0 codebase?",
                f"Does the vLLM 0.22.0 tree include symbol `{name}` at `{qual}`?",
            ]
            for i, q in enumerate(qs):
                probes.append(
                    _probe(
                        claim=c,
                        probe_form="recall",
                        paraphrase_idx=i,
                        question=q,
                        gold={
                            "must_contain_any": [[name, "yes", "true"]],
                            "must_not_contain": ["unknown"],
                        },
                    )
                )
            fake_qual = qual + "_Missing"
            nqs = [
                f"In vLLM 0.22.0 source, does `{fake_qual}` exist as a top-level class or function?",
                f"Is `{name}Missing` defined at `{fake_qual}` in the vLLM 0.22.0 codebase?",
                f"Does the vLLM 0.22.0 tree include symbol `{name}Missing` at `{fake_qual}`?",
            ]
            for i, q in enumerate(nqs):
                probes.append(
                    _probe(
                        claim=c,
                        probe_form="negation",
                        paraphrase_idx=i,
                        question=q,
                        gold={
                            "must_contain_any": [
                                ["no", "unknown", "not", "does not", "doesn't"]
                            ],
                            "abstain_if_unknown": True,
                        },
                    )
                )

    probes.sort(key=lambda p: p["id"])
    return probes


if __name__ == "__main__":
    snap = ROOT / "data/experiments/e1_vllm/snapshots/v0.22.0"
    out = ROOT / "data/experiments/e1_vllm/profile/truthsource"
    build_arms(snapshot=snap, out_dir=out)
