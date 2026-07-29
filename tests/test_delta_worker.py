from __future__ import annotations

import unittest

from lab.delta.worker import SURFACE_TASK, WorkerRequest, validate_proposal


class WorkerEnvelopeTest(unittest.TestCase):
    def test_request_id_is_deterministic(self) -> None:
        request = WorkerRequest(
            task=SURFACE_TASK,
            payload={"seed_id": "p1"},
            constraints={"required_literals": ["--foo"]},
            prompt="Rewrite the question.",
        )
        clone = WorkerRequest(
            task=SURFACE_TASK,
            payload={"seed_id": "p1"},
            constraints={"required_literals": ["--foo"]},
            prompt="Rewrite the question.",
        )
        self.assertEqual(request.request_id, clone.request_id)
        self.assertNotIn("credential", request.to_dict())

    def test_controller_validator_rejects_model_output(self) -> None:
        request = WorkerRequest(
            task=SURFACE_TASK,
            payload={"seed_id": "p1"},
            constraints={},
            prompt="Rewrite.",
        )
        result = validate_proposal(
            request=request,
            model_id="small/model",
            raw_output="not a question",
            validator=lambda text: None if text.endswith("?") else "not_a_question",
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.reject_reason, "not_a_question")
        self.assertEqual(len(result.prompt_sha256), 64)

    def test_unknown_task_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            WorkerRequest(task="choose_policy", payload={}, constraints={}, prompt="Pick.")


if __name__ == "__main__":
    unittest.main()
