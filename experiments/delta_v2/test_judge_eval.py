from __future__ import annotations

import copy
import unittest
from pathlib import Path

from experiments.delta_v2.build_eval import (
    build_fact_item,
    build_feature_item,
)
from experiments.delta_v2.judge_eval import (
    JudgeAuditError,
    _jsonl_bytes,
    _reviewed_ids_hash,
    _sha256_bytes,
    apply_attestation,
    build_evidence_bundle,
    validate_contract,
)


JUDGE_CONTRACT_PATH = Path(__file__).with_name("llm_judge_contract.yaml")
ACCEPTANCE_JUDGE_CONTRACT_PATH = Path(__file__).with_name(
    "acceptance_llm_judge_contract_v2.yaml"
)
ACCEPTANCE_FEATURE_JUDGE_CONTRACT_PATH = Path(__file__).with_name(
    "acceptance_llm_judge_contract_v3.yaml"
)
EVAL_CONTRACT_PATH = Path(__file__).with_name("eval_item_contract.yaml")
ACCEPTANCE_EVAL_CONTRACT_PATH = Path(__file__).with_name(
    "acceptance_eval_item_contract_v2.yaml"
)
ACCEPTANCE_FEATURE_EVAL_CONTRACT_PATH = Path(__file__).with_name(
    "acceptance_eval_item_contract_v3.yaml"
)


def load_contracts() -> tuple[dict, dict]:
    import yaml

    judge = yaml.safe_load(JUDGE_CONTRACT_PATH.read_text(encoding="utf-8"))
    eval_contract = yaml.safe_load(
        EVAL_CONTRACT_PATH.read_text(encoding="utf-8")
    )
    return judge, eval_contract


def fact_record(status: str = "changed") -> dict:
    before = {"path": "before.py", "snapshot_role": "development_before"}
    after = {"path": "after.py", "snapshot_role": "development_after"}
    values = {
        "stable": ({"value": 1}, {"value": 1}),
        "added": (None, {"value": 1}),
        "removed": ({"value": 1}, None),
        "changed": ({"value": 1}, {"value": 2}),
    }
    value_before, value_after = values[status]
    return {
        "schema": "delta.atomic_fact_delta.v1",
        "fact_id": "fact:one",
        "family": "python.environment_variable.v1",
        "status": status,
        "semantic_key": "python-environment-variable:VLLM_ONE",
        "value_before": value_before,
        "value_after": value_after,
        "evidence_before": before if value_before is not None else None,
        "evidence_after": after if value_after is not None else None,
        "verifier": {"extractor_version": "test-v1"},
    }


def feature_record() -> dict:
    return {
        "schema": "delta.feature_delta.v1",
        "feature_id": "feature:vllm-prefix-cache-retention",
        "source_candidate_id": "feature-candidate:one",
        "repository": "vllm-project/vllm",
        "transition": "development",
        "summary": "Prefix cache retention",
        "status": "added",
        "evidence_ids": [
            "fact:one",
            "file:one",
            "pull-request:vllm-project/vllm#1",
        ],
        "behavior_probe_ids": [
            "behavior-probe:vllm-prefix-cache-retention-mechanics-v1"
        ],
        "verification": {
            "old_revision": "unavailable",
            "new_revision": "pass",
            "recipe": "test-v1",
            "probe_results": {
                "behavior-probe:vllm-prefix-cache-retention-mechanics-v1": {
                    "sha256": "a" * 64,
                    "claim_scope": "complete-candidate",
                }
            },
        },
        "environment_status": "development-only-unfrozen",
    }


def probe_result() -> dict:
    cases = [
        (
            "dense_default",
            {"kind": "cache_state", "cached_indices": list(range(16))},
        ),
        (
            "interval_64",
            {"kind": "cache_state", "cached_indices": [3, 7, 11, 14, 15]},
        ),
        (
            "latest_only",
            {"kind": "cache_state", "cached_indices": [14]},
        ),
        ("negative_rejected", {"kind": "construction_error"}),
        ("misaligned_rejected", {"kind": "construction_error"}),
    ]
    return {
        "schema": "delta.behavior_probe_result.v1",
        "probe_id": "behavior-probe:vllm-prefix-cache-retention-mechanics-v1",
        "claim_scope": "complete-candidate",
        "status": "pass",
        "cases": [
            {
                "case_id": case_id,
                "status": "pass",
                "observed_after": observed,
            }
            for case_id, observed in cases
        ],
    }


def acceptance_feature_record() -> dict:
    return {
        "schema": "delta.feature_delta.v1",
        "feature_id": "feature:vllm-endpoint-plugins",
        "source_candidate_id": "feature-candidate:endpoint-plugins",
        "repository": "vllm-project/vllm",
        "transition": "acceptance",
        "summary": "Endpoint plugins",
        "status": "added",
        "fact_policy": "no-post-unseal-fact-family-expansion",
        "evidence_ids": [
            "file:one",
            "pull-request:vllm-project/vllm#1",
        ],
        "behavior_probe_ids": [
            "behavior-probe:vllm-endpoint-plugins-framework-v1"
        ],
        "verification": {
            "old_revision": "unavailable",
            "new_revision": "pass",
            "recipe": "test-v1",
            "probe_results": {
                "behavior-probe:vllm-endpoint-plugins-framework-v1": {
                    "sha256": "a" * 64,
                    "claim_scope": "complete-candidate",
                }
            },
        },
        "environment_status": "acceptance-feature-unfrozen",
    }


def endpoint_probe_result() -> dict:
    return {
        "schema": "delta.behavior_probe_result.v1",
        "probe_id": "behavior-probe:vllm-endpoint-plugins-framework-v1",
        "claim_scope": "complete-candidate",
        "status": "pass",
        "cases": [
            {
                "case_id": "complete_endpoint_plugin_framework",
                "status": "pass",
                "observed_before": {
                    "kind": "unavailable",
                    "loader": False,
                    "protocol": False,
                    "route_phase": False,
                    "state_phase": False,
                },
                "observed_after": {
                    "kind": "endpoint_plugin_framework",
                    "allowlisted_task_match_loads": True,
                    "async_init_state_hook": True,
                    "attach_router_hook": True,
                    "default_off": True,
                    "documentation_present": True,
                    "factory_failure_isolated": True,
                    "required_task_miss_skips": True,
                    "route_phase_attaches": True,
                    "runtime_checkable_protocol": True,
                    "state_phase_initializes": True,
                    "upstream_tests_present": True,
                },
            }
        ],
    }


def audit_row(item: dict, audit_index: int) -> dict:
    return {
        "schema": "delta.human_eval_audit_item.v1",
        "audit_index": audit_index,
        "eval_id": item["eval_id"],
        "source_id": item["source_id"],
        "split": item["split"],
        "task_kind": item["task_kind"],
        "prompt": item["prompt"],
        "proposed_gold": item["gold"],
        "provenance": item["provenance"],
        "review": {
            "truth": None,
            "version_status": None,
            "answerability": None,
            "notes": None,
        },
    }


def build_inputs() -> tuple[list[dict], list[dict], list[dict], dict[str, dict]]:
    _, eval_contract = load_contracts()
    facts: list[dict] = []
    rows: list[dict] = []
    for index in range(31):
        fact = fact_record(["stable", "added", "removed", "changed"][index % 4])
        fact["fact_id"] = f"fact:{index:02d}"
        fact["semantic_key"] = f"python-environment-variable:VLLM_{index:02d}"
        item = build_fact_item(
            fact,
            split="train" if index % 2 == 0 else "dev",
            eligibility_reason=(
                "stable_control" if fact["status"] == "stable" else "nonstable"
            ),
            contract=eval_contract,
        )
        facts.append(fact)
        rows.append(audit_row(item, index + 1))
    feature = feature_record()
    probe = probe_result()
    feature_item = build_feature_item(
        feature,
        split="train",
        probe_result=probe,
        contract=eval_contract,
    )
    rows.append(audit_row(feature_item, 32))
    return rows, facts, [feature], {probe["probe_id"]: probe}


def attestation(bundle: list[dict], verdict: str = "pass") -> dict:
    judge_contract, _ = load_contracts()
    return {
        "schema": "delta.llm_judge_attestation.v1",
        "contract_id": judge_contract["contract_id"],
        "evidence_bundle_sha256": _sha256_bytes(_jsonl_bytes(bundle)),
        "reviewed_eval_ids_sha256": _reviewed_ids_hash(bundle),
        "reviewed_items": len(bundle),
        "judge_run": {
            "judge_run_id": "judge:test",
            "kind": "llm",
            "runtime": "codex-desktop",
            "model_family": "GPT-5",
            "target_model": False,
        },
        "default_dimensions": {
            dimension: {
                "verdict": verdict,
                "reason": f"{dimension} was reviewed against the evidence.",
            }
            for dimension in ("truth", "version_status", "answerability")
        },
        "exceptions": [],
    }


class JudgeContractTest(unittest.TestCase):
    def test_contract_preserves_truth_authority(self) -> None:
        judge_contract, _ = load_contracts()

        validate_contract(judge_contract)
        self.assertFalse(
            judge_contract["truth_authority"]["llm_may_change_gold"]
        )

    def test_rejects_llm_as_gold_authority(self) -> None:
        judge_contract, _ = load_contracts()
        altered = copy.deepcopy(judge_contract)
        altered["truth_authority"]["llm_may_change_gold"] = True

        with self.assertRaisesRegex(JudgeAuditError, "ground truth"):
            validate_contract(altered)

    def test_acceptance_contract_discloses_post_unseal_implementation(
        self,
    ) -> None:
        import yaml

        contract = yaml.safe_load(
            ACCEPTANCE_JUDGE_CONTRACT_PATH.read_text(encoding="utf-8")
        )

        validate_contract(contract)
        amendment = contract["protocol_amendment"]
        self.assertEqual(
            amendment["implementation_timing"],
            "validator-parameterized-after-acceptance-unseal",
        )
        self.assertFalse(amendment["source_or_gold_changed"])
        self.assertTrue(amendment["selection_changed"])

    def test_acceptance_v3_discloses_added_feature_source_and_gold(self) -> None:
        import yaml

        contract = yaml.safe_load(
            ACCEPTANCE_FEATURE_JUDGE_CONTRACT_PATH.read_text(encoding="utf-8")
        )

        validate_contract(contract)
        amendment = contract["protocol_amendment"]
        self.assertTrue(amendment["source_or_gold_changed"])
        self.assertEqual(
            amendment["change_scope"],
            "one-promoted-fact-frozen-acceptance-feature",
        )


class EvidenceBundleTest(unittest.TestCase):
    def build(self) -> tuple[list[dict], dict]:
        judge_contract, eval_contract = load_contracts()
        rows, facts, features, probes = build_inputs()
        bundle = build_evidence_bundle(
            audit_rows=rows,
            fact_records=facts,
            feature_records=features,
            probe_results=probes,
            contract=judge_contract,
            eval_contract=eval_contract,
        )
        return bundle, judge_contract

    def test_roots_all_fact_and_feature_rows(self) -> None:
        bundle, _ = self.build()

        self.assertEqual(len(bundle), 32)
        self.assertTrue(all(row["root_check"] == "pass" for row in bundle))
        self.assertEqual(bundle[-1]["task_kind"], "feature_behavior_matrix")
        self.assertEqual(len(bundle[-1]["evidence"]), 6)

    def test_bundle_is_deterministic(self) -> None:
        first, _ = self.build()
        second, _ = self.build()

        self.assertEqual(first, second)
        self.assertEqual(_jsonl_bytes(first), _jsonl_bytes(second))

    def test_rejects_gold_that_disagrees_with_verified_fact(self) -> None:
        judge_contract, eval_contract = load_contracts()
        rows, facts, features, probes = build_inputs()
        rows[0]["proposed_gold"] = {"status": "removed"}

        with self.assertRaisesRegex(JudgeAuditError, "proposed gold"):
            build_evidence_bundle(
                audit_rows=rows,
                fact_records=facts,
                feature_records=features,
                probe_results=probes,
                contract=judge_contract,
                eval_contract=eval_contract,
            )

    def test_rejects_status_without_matching_revision_evidence(self) -> None:
        judge_contract, eval_contract = load_contracts()
        rows, facts, features, probes = build_inputs()
        facts[0]["evidence_after"] = None

        with self.assertRaisesRegex(JudgeAuditError, "after evidence"):
            build_evidence_bundle(
                audit_rows=rows,
                fact_records=facts,
                feature_records=features,
                probe_results=probes,
                contract=judge_contract,
                eval_contract=eval_contract,
            )

    def test_acceptance_fact_only_bundle_is_rooted(self) -> None:
        import yaml

        judge_contract = yaml.safe_load(
            ACCEPTANCE_JUDGE_CONTRACT_PATH.read_text(encoding="utf-8")
        )
        eval_contract = yaml.safe_load(
            ACCEPTANCE_EVAL_CONTRACT_PATH.read_text(encoding="utf-8")
        )
        facts: list[dict] = []
        rows: list[dict] = []
        for index in range(32):
            fact = fact_record(
                ["stable", "added", "removed", "changed"][index % 4]
            )
            fact["fact_id"] = f"fact:acceptance:{index:02d}"
            fact["semantic_key"] = (
                f"python-environment-variable:VLLM_ACCEPTANCE_{index:02d}"
            )
            item = build_fact_item(
                fact,
                split="eval",
                eligibility_reason=(
                    "stable_control"
                    if fact["status"] == "stable"
                    else "nonstable"
                ),
                contract=eval_contract,
            )
            facts.append(fact)
            rows.append(audit_row(item, index + 1))

        bundle = build_evidence_bundle(
            audit_rows=rows,
            fact_records=facts,
            feature_records=[],
            probe_results={},
            contract=judge_contract,
            eval_contract=eval_contract,
        )

        self.assertEqual(len(bundle), 32)
        self.assertTrue(all(row["root_check"] == "pass" for row in bundle))

    def test_acceptance_feature_bundle_roots_gold_in_complete_probe(self) -> None:
        import yaml

        judge_contract = yaml.safe_load(
            ACCEPTANCE_FEATURE_JUDGE_CONTRACT_PATH.read_text(encoding="utf-8")
        )
        eval_contract = yaml.safe_load(
            ACCEPTANCE_FEATURE_EVAL_CONTRACT_PATH.read_text(encoding="utf-8")
        )
        facts: list[dict] = []
        rows: list[dict] = []
        for index in range(31):
            status = "stable" if index % 2 == 0 else "added"
            fact = fact_record(status)
            fact["fact_id"] = f"fact:acceptance-v3:{index:02d}"
            fact["semantic_key"] = (
                f"python-environment-variable:VLLM_ACCEPTANCE_V3_{index:02d}"
            )
            item = build_fact_item(
                fact,
                split="eval",
                eligibility_reason=(
                    "stable_control" if status == "stable" else "nonstable"
                ),
                contract=eval_contract,
            )
            facts.append(fact)
            rows.append(audit_row(item, index + 1))
        feature = acceptance_feature_record()
        probe = endpoint_probe_result()
        feature_item = build_feature_item(
            feature,
            split="eval",
            probe_result=probe,
            contract=eval_contract,
        )
        rows.append(audit_row(feature_item, 32))

        bundle = build_evidence_bundle(
            audit_rows=rows,
            fact_records=facts,
            feature_records=[feature],
            probe_results={str(probe["probe_id"]): probe},
            contract=judge_contract,
            eval_contract=eval_contract,
        )
        feature_evidence = bundle[-1]

        self.assertEqual(len(bundle), 32)
        self.assertEqual(
            feature_evidence["proposed_gold"]["route_phase"],
            "attach_router",
        )
        self.assertEqual(
            [row["kind"] for row in feature_evidence["evidence"]],
            ["promoted-feature", "executable-after-observation"],
        )

    def test_acceptance_feature_bundle_rejects_probe_mismatch(self) -> None:
        import yaml

        judge_contract = yaml.safe_load(
            ACCEPTANCE_FEATURE_JUDGE_CONTRACT_PATH.read_text(encoding="utf-8")
        )
        eval_contract = yaml.safe_load(
            ACCEPTANCE_FEATURE_EVAL_CONTRACT_PATH.read_text(encoding="utf-8")
        )
        facts: list[dict] = []
        rows: list[dict] = []
        for index in range(31):
            status = "stable" if index % 2 == 0 else "added"
            fact = fact_record(status)
            fact["fact_id"] = f"fact:acceptance-v3:{index:02d}"
            fact["semantic_key"] = (
                f"python-environment-variable:VLLM_ACCEPTANCE_V3_{index:02d}"
            )
            item = build_fact_item(
                fact,
                split="eval",
                eligibility_reason=(
                    "stable_control" if status == "stable" else "nonstable"
                ),
                contract=eval_contract,
            )
            facts.append(fact)
            rows.append(audit_row(item, index + 1))
        feature = acceptance_feature_record()
        probe = endpoint_probe_result()
        feature_item = build_feature_item(
            feature,
            split="eval",
            probe_result=probe,
            contract=eval_contract,
        )
        rows.append(audit_row(feature_item, 32))
        probe["cases"][0]["observed_after"]["route_phase_attaches"] = False

        with self.assertRaisesRegex(
            JudgeAuditError,
            "observations do not support",
        ):
            build_evidence_bundle(
                audit_rows=rows,
                fact_records=facts,
                feature_records=[feature],
                probe_results={str(probe["probe_id"]): probe},
                contract=judge_contract,
                eval_contract=eval_contract,
            )


class AttestationTest(unittest.TestCase):
    def build(self) -> tuple[list[dict], dict]:
        return EvidenceBundleTest().build()

    def test_all_pass_attestation_passes_every_dimension(self) -> None:
        bundle, contract = self.build()

        verdicts, summary = apply_attestation(
            bundle=bundle,
            attestation=attestation(bundle),
            contract=contract,
        )

        self.assertEqual(summary["status"], "pass")
        self.assertEqual(len(verdicts), 32)
        self.assertEqual(
            summary["dimension_pass_rates"],
            {
                "truth": 1.0,
                "version_status": 1.0,
                "answerability": 1.0,
            },
        )

    def test_attestation_cannot_be_reused_for_other_evidence(self) -> None:
        bundle, contract = self.build()
        record = attestation(bundle)
        altered = copy.deepcopy(bundle)
        altered[0]["prompt"] = "changed"

        with self.assertRaisesRegex(JudgeAuditError, "evidence hash"):
            apply_attestation(
                bundle=altered,
                attestation=record,
                contract=contract,
            )

    def test_abstentions_count_against_threshold(self) -> None:
        bundle, contract = self.build()
        record = attestation(bundle)
        record["default_dimensions"]["truth"]["verdict"] = "abstain"

        _, summary = apply_attestation(
            bundle=bundle,
            attestation=record,
            contract=contract,
        )

        self.assertEqual(summary["status"], "fail")
        self.assertEqual(summary["dimension_pass_rates"]["truth"], 0.0)


if __name__ == "__main__":
    unittest.main()
