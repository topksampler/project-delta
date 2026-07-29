from __future__ import annotations

import copy
import unittest
from pathlib import Path

from experiments.delta_v2.endpoint_plugins_probe import (
    EndpointPluginsProbeError,
    _exercise_lifecycle,
    _exercise_loader,
    _protocol_contract,
    evaluate_observations,
    load_probe,
    validate_probe,
)


PROBE_PATH = (
    Path(__file__).with_name("probes") / "endpoint_plugins_framework.yaml"
)


def probe_record() -> dict:
    import yaml

    return yaml.safe_load(PROBE_PATH.read_text(encoding="utf-8"))


LOADER_SOURCE = """
def load_endpoint_plugins(supported_tasks=None):
    from importlib.metadata import entry_points
    if envs.VLLM_PLUGINS is None:
        discovered = entry_points(group=ENDPOINT_PLUGINS_GROUP)
        if discovered:
            logger.warning("found", [p.name for p in discovered])
        return []
    factories = load_plugins_by_group(ENDPOINT_PLUGINS_GROUP)
    endpoint_plugins = []
    for name, factory in factories.items():
        try:
            plugin = factory()
        except Exception:
            logger.exception("failed", name)
            continue
        required_tasks = plugin.required_tasks
        if required_tasks is not None and (
            supported_tasks is None
            or not set(required_tasks) & set(supported_tasks)
        ):
            logger.info("skip", name)
            continue
        endpoint_plugins.append(plugin)
    return endpoint_plugins
"""


LIFECYCLE_SOURCE = """
def _attach_endpoint_plugins(app, supported_tasks):
    from vllm.plugins import load_endpoint_plugins
    endpoint_plugins = load_endpoint_plugins(supported_tasks)
    for plugin in endpoint_plugins:
        plugin.attach_router(app)
    app.state.endpoint_plugins = endpoint_plugins

async def _init_endpoint_plugins_state(engine_client, state, args):
    for plugin in getattr(state, "endpoint_plugins", []):
        await plugin.init_state(engine_client, state, args)
"""


PROTOCOL_SOURCE = """
@runtime_checkable
class EndpointPlugin(Protocol):
    def attach_router(self, app):
        ...

    async def init_state(self, engine_client, state, args):
        ...
"""


class ProbeContractTest(unittest.TestCase):
    def test_real_contract_is_complete_and_acceptance_only(self) -> None:
        probe = load_probe(PROBE_PATH)

        self.assertEqual(probe["transition"], "acceptance")
        self.assertEqual(probe["claim_scope"], "complete-candidate")

    def test_rejects_development_or_partial_claim(self) -> None:
        probe = probe_record()
        probe["transition"] = "development"
        with self.assertRaisesRegex(
            EndpointPluginsProbeError,
            "requires acceptance",
        ):
            validate_probe(probe)

        probe = probe_record()
        probe["claim_scope"] = "loader-only"
        with self.assertRaisesRegex(
            EndpointPluginsProbeError,
            "complete candidate",
        ):
            validate_probe(probe)


class ExactSourceExecutionTest(unittest.TestCase):
    def test_loader_enforces_opt_in_task_gate_and_failure_isolation(self) -> None:
        observed = _exercise_loader(LOADER_SOURCE)

        self.assertTrue(all(observed.values()))

    def test_two_phase_lifecycle_executes_both_hooks(self) -> None:
        observed = _exercise_lifecycle(LIFECYCLE_SOURCE)

        self.assertEqual(
            observed,
            {
                "route_phase_attaches": True,
                "state_phase_initializes": True,
            },
        )

    def test_protocol_requires_runtime_checkable_sync_and_async_hooks(self) -> None:
        observed = _protocol_contract(PROTOCOL_SOURCE)

        self.assertEqual(
            observed,
            {
                "runtime_checkable_protocol": True,
                "attach_router_hook": True,
                "async_init_state_hook": True,
            },
        )


class ObservationTest(unittest.TestCase):
    def test_exact_old_new_outcomes_pass(self) -> None:
        probe = probe_record()
        case = probe["cases"][0]

        result = evaluate_observations(
            probe,
            {
                "acceptance_before": case["expected_before"],
                "acceptance_after": case["expected_after"],
            },
        )

        self.assertEqual(result[0]["status"], "pass")

    def test_mismatch_fails_without_rewriting_expected_truth(self) -> None:
        probe = probe_record()
        original = copy.deepcopy(probe)
        case = probe["cases"][0]
        altered_after = {
            **case["expected_after"],
            "default_off": False,
        }

        result = evaluate_observations(
            probe,
            {
                "acceptance_before": case["expected_before"],
                "acceptance_after": altered_after,
            },
        )

        self.assertEqual(result[0]["status"], "fail")
        self.assertEqual(probe, original)


if __name__ == "__main__":
    unittest.main()
