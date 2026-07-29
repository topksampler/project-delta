from __future__ import annotations

import unittest

from lab.delta.sense import build_factory_drift_event


class FactorySenseTest(unittest.TestCase):
    def test_builds_structural_and_behavioral_drift(self) -> None:
        event = build_factory_drift_event(
            manifest={
                "schema": "delta.eval_environment.v1",
                "source_before": {"revision": "v1"},
                "source_after": {"revision": "v2"},
                "counts": {
                    "claims": 2,
                    "by_status": {"changed": 1, "stable": 1},
                    "by_family_status": {},
                },
            },
            summary={
                "schema": "delta.eval_factory.run_summary.v1",
                "n_samples": 2,
                "by_drift_type": {"changed": 0.25, "stable": 1.0},
            },
            probes=[
                {"id": "p1", "claim_id": "c1", "drift_type": "changed"},
                {"id": "p2", "claim_id": "c2", "drift_type": "stable"},
            ],
            samples=[
                {"id": "p1", "content_score": 0.0},
                {"id": "p2", "content_score": 1.0},
            ],
        )
        self.assertEqual(event.drift_score, 0.75)
        self.assertEqual(event.affected_claim_ids, ("c1",))
        self.assertEqual(tuple(item.value for item in event.drift_types), ("changed",))
        self.assertEqual(event.behavioral_failures["scoreboard"], "eval_factory")

    def test_never_infers_failures_from_pooled_accuracy(self) -> None:
        event = build_factory_drift_event(
            manifest={
                "source_before": {"revision": "v1"},
                "source_after": {"revision": "v2"},
                "counts": {},
            },
            summary={"exact_accuracy": 0.0, "by_drift_type": {"stable": 1.0}},
            probes=[],
        )
        self.assertEqual(event.drift_score, 0.0)


if __name__ == "__main__":
    unittest.main()
