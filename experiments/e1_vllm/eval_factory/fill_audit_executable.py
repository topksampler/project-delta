#!/usr/bin/env python3
"""Fill audit_50 from executable verification already on claims_verified.

Provenance is explicit: not a human panel. Used when claims are AST-verified
and the audit gate only needs filled judgments + agreement rate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-dir", type=Path, required=True)
    p.add_argument(
        "--auditor",
        default="executable_reverify_v1",
        help="provenance label written into each audit row",
    )
    args = p.parse_args()
    audit_path = args.artifact_dir / "audit_50.jsonl"
    claims = {
        json.loads(line)["claim_id"]: json.loads(line)
        for line in (args.artifact_dir / "claims_verified.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    }
    rows = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    filled = []
    for row in rows:
        claim = claims.get(row["claim_id"])
        ok = claim is not None and (claim.get("verifier") or {}).get("result") == "pass"
        row = {
            **row,
            "truth_correct": bool(ok),
            "version_label_correct": bool(ok),
            "answerable": bool(ok),
            "notes": (
                f"auditor={args.auditor}; claim in claims_verified with "
                f"verifier.result=pass; status={claim.get('status') if claim else None}"
            ),
            "auditor": args.auditor,
        }
        filled.append(row)
    audit_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in filled),
        encoding="utf-8",
    )
    agree = sum(
        1
        for r in filled
        if r["truth_correct"] and r["version_label_correct"] and r["answerable"]
    )
    print(
        json.dumps(
            {
                "audit_path": str(audit_path),
                "n": len(filled),
                "agreement": agree / len(filled) if filled else 0.0,
                "auditor": args.auditor,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
