"""Unit tests for Experiment C teacher-context + consolidation export."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from .export_sft_teacher_context import (
    export_consolidate,
    export_teacher,
    select_diet,
)


def _probe(pid: str, claim: str, drift: str, q: str) -> dict:
    return {
        "id": pid,
        "claim_id": claim,
        "drift_type": drift,
        "question": q,
        "gold": {"boolean": "yes"},
        "probe_form": "versioned_existence",
    }


class FakeRetriever:
    def __init__(self) -> None:
        self.route_log = {"diff": 0}

    def retrieve(self, question: str, example: dict | None = None):
        return f"Retrieved source code:\n\n[ctx]\n{question}", ["ctx"]


class TeacherExportTest(unittest.TestCase):
    def test_teacher_emits_context_and_bare_paraphrases(self) -> None:
        probes = [
            _probe("c:a", "claim:c", "changed", "Changed A?"),
            _probe("c:b", "claim:c", "changed", "Changed B paraphrase?"),
            _probe("s:a", "claim:s", "stable", "Stable A?"),
            _probe("s:b", "claim:s", "stable", "Stable B paraphrase?"),
        ]
        selected = select_diet(probes, delta_copies=1, stable_cap=1, seed=1)
        slim, stats = export_teacher(
            probes=probes,
            selected=selected,
            retriever=FakeRetriever(),
            bare_paraphrases=True,
        )
        self.assertGreater(stats["nonempty_context"], 0)
        self.assertGreater(stats["n_bare_paraphrase"], 0)
        self.assertEqual(stats["n"], len(slim))
        # At least one user turn carries retrieved context.
        self.assertTrue(
            any("Retrieved source code" in row["messages"][0]["content"] for row in slim)
        )
        # Bare paraphrase of changed claim is present without context prefix.
        bare = [
            row
            for row in slim
            if row["messages"][0]["content"] == "Changed B paraphrase?"
        ]
        self.assertEqual(len(bare), 1)
        self.assertEqual(bare[0]["messages"][1]["content"], "Yes.")

    def test_consolidate_drops_context(self) -> None:
        selected = [
            _probe("c:a", "claim:c", "changed", "Changed A?"),
            _probe("s:a", "claim:s", "stable", "Stable A?"),
        ]
        slim, stats = export_consolidate(selected=selected)
        self.assertEqual(stats["n"], 2)
        for row in slim:
            self.assertNotIn("Retrieved", row["messages"][0]["content"])
            self.assertEqual(row["messages"][1]["content"], "Yes.")

    def test_convert_writes_meta(self) -> None:
        from .export_sft_teacher_context import convert
        import argparse

        probes = [
            _probe("c:a", "claim:c", "changed", "Changed A?"),
            _probe("a:a", "claim:a", "added", "Added A?"),
            _probe("r:a", "claim:r", "removed", "Removed A?"),
            _probe("s:a", "claim:s", "stable", "Stable A?"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            probes_path = tmp_path / "probes.jsonl"
            probes_path.write_text(
                "".join(json.dumps(p) + "\n" for p in probes), encoding="utf-8"
            )
            out = tmp_path / "consolidate.jsonl"
            args = argparse.Namespace(
                probes=probes_path,
                out=out,
                phase="consolidate",
                teacher="none",
                corpus=None,
                index=None,
                symbol_index=None,
                diff_corpus=None,
                diff_index=None,
                evidence_map=None,
                delta_copies=2,
                stable_cap=1,
                seed=7,
                top_k=4,
                max_chars=3500,
                no_bare_paraphrases=False,
            )
            meta = convert(args)
            self.assertTrue(out.exists())
            self.assertEqual(meta["phase"], "consolidate")
            self.assertGreater(meta["n"], 0)
            loaded = [
                json.loads(line)
                for line in out.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(loaded), meta["n"])


if __name__ == "__main__":
    unittest.main()
