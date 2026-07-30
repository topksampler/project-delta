from __future__ import annotations

import unittest

from experiments.delta_v2.run_knowledge_train_surface_diagnostic import (
    DiagnosticError,
    build_requests,
    score,
)


def rows() -> list[dict]:
    result = []
    for surface, answer in (
        ("exact_recall", '{"x":1}'),
        ("verify_true", "yes"),
        ("verify_false", "no"),
    ):
        result.append(
            {
                "row_id": surface,
                "surface": surface,
                "messages": [
                    {"role": "user", "content": surface},
                    {"role": "assistant", "content": answer},
                ],
            }
        )
    return result


PROTOCOL = {
    "generation": {"repeats": 2, "max_new_tokens": 256, "seed": 1},
    "decision": {
        "boolean_surface_fit": {
            "verify_true_accuracy_minimum": 0.8,
            "verify_false_accuracy_minimum": 0.8,
        }
    },
}


class TrainSurfaceDiagnosticTests(unittest.TestCase):
    def test_requests_expose_only_user_message(self) -> None:
        requests = build_requests(rows(), PROTOCOL)
        self.assertEqual(len(requests), 6)
        self.assertEqual(
            requests[0]["model_input"]["messages"],
            [{"role": "user", "content": "exact_recall"}],
        )
        self.assertNotIn("gold", requests[0])

    def test_exact_fit_yields_paraphrase_diagnosis(self) -> None:
        data = rows()
        outputs = []
        for request in build_requests(data, PROTOCOL):
            answer = next(
                row["messages"][1]["content"]
                for row in data
                if row["row_id"] == request["row_id"]
            )
            outputs.append({**request, "raw_response": answer})
        metrics = score(
            rows=data,
            outputs=outputs,
            protocol=PROTOCOL,
        )
        self.assertTrue(metrics["boolean_surface_fit"])
        self.assertEqual(
            metrics["diagnosis"],
            "paraphrase-transfer-failure",
        )
        self.assertIsNone(metrics["pooled_overall_accuracy"])

    def test_repeat_disagreement_fails_closed(self) -> None:
        data = rows()
        outputs = [
            {**request, "raw_response": "same"}
            for request in build_requests(data, PROTOCOL)
        ]
        outputs[1]["raw_response"] = "different"
        with self.assertRaisesRegex(DiagnosticError, "nondeterministic"):
            score(rows=data, outputs=outputs, protocol=PROTOCOL)


if __name__ == "__main__":
    unittest.main()
