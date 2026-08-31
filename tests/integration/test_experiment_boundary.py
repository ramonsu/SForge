import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.v6_cognitive_profession_ablation import (
    _apply_run_metrics,
    build_causal_decomposition_conditions,
    build_conditions,
    build_dry_run_record,
    build_final_validation_conditions,
    build_policy_transmission_conditions,
    build_strength_sweep_conditions,
    evaluate_answer,
    load_causal_tasks,
    load_final_validation_tasks,
    load_tasks,
    main,
    validate_record_schema,
    write_results,
)


class V6AblationExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = load_tasks(["mutable_default_argument"])[0]

    def test_condition_construction_changes_only_requested_resources(self):
        conditions = build_conditions("INTJ", "software_engineering")
        self.assertEqual(
            [
                "base",
                "profession_only",
                "policy_a_only",
                "profession_and_policy_a",
                "policy_b_only",
            ],
            [item.id for item in conditions],
        )
        self.assertIsNone(conditions[0].cognitive_policy_id)
        self.assertEqual((), conditions[0].profession_ids)
        self.assertEqual(("software_engineering",), conditions[1].profession_ids)
        self.assertEqual("INTJ", conditions[2].cognitive_policy_id)
        self.assertEqual(("software_engineering",), conditions[3].profession_ids)

    def test_dry_run_uses_real_runtime_context_without_api(self):
        base, profession_only, policy_only, both, policy_b = build_conditions()
        base_record = build_dry_run_record(self.task, base)
        profession_record = build_dry_run_record(self.task, profession_only)
        policy_record = build_dry_run_record(self.task, policy_only)
        both_record = build_dry_run_record(self.task, both)
        policy_b_record = build_dry_run_record(self.task, policy_b)

        self.assertEqual("dry_run", base_record["status"])
        self.assertEqual([], base_record["model_calls"])
        self.assertIsNone(base_record["active_cognitive_policy"])
        self.assertEqual([], base_record["active_professions"])
        self.assertEqual("INTJ", policy_record["active_cognitive_policy"]["id"])
        self.assertEqual(
            "software_engineering",
            profession_record["active_professions"][0]["id"],
        )
        self.assertEqual("INTJ", both_record["active_cognitive_policy"]["id"])
        self.assertEqual("ENFP", policy_b_record["active_cognitive_policy"]["id"])
        profession_memory_ids = {
            item["id"]
            for item in profession_record["retrieved_resources"]["memories"]
        }
        base_memory_ids = {
            item["id"] for item in base_record["retrieved_resources"]["memories"]
        }
        self.assertIn("exp-profession-software-engineering", profession_memory_ids)
        self.assertNotIn("exp-profession-software-engineering", base_memory_ids)
        self.assertEqual(
            base_record["work_assignment"]["grants"],
            both_record["work_assignment"]["grants"],
        )

    def test_cli_dry_run_does_not_require_api_key(self):
        environment = {
            key: value
            for key, value in os.environ.items()
            if key != "DEEPSEEK_API_KEY"
        }
        with patch.dict(os.environ, environment, clear=True), patch(
            "builtins.print"
        ) as printer:
            result = main(
                ["--dry-run", "--task", "mutable_default_argument"]
            )
        self.assertEqual(0, result)
        self.assertEqual(5, printer.call_count)

    def test_mechanism_profession_fixture_crosses_real_retrieval_boundary(self):
        task = load_tasks(["mechanism_profession_sentinel"])[0]
        base, profession_only, *_ = build_conditions()
        base_ids = set(
            build_dry_run_record(task, base)["retrieved_memory_ids"]
        )
        profession_ids = set(
            build_dry_run_record(task, profession_only)[
                "retrieved_memory_ids"
            ]
        )
        target = "mechanism-profession-missing-sentinel"
        self.assertNotIn(target, base_ids)
        self.assertIn(target, profession_ids)

    def test_policy_a_and_b_reverse_real_memory_ranking(self):
        task = load_tasks(["mechanism_policy_cache"])[0]
        conditions = build_conditions()
        policy_a = build_dry_run_record(task, conditions[2])
        policy_b = build_dry_run_record(task, conditions[4])

        def ranks(record):
            return {
                item["id"]: item["rank"]
                for item in record["retrieved_resources"]["memories"]
            }

        a_ranks = ranks(policy_a)
        b_ranks = ranks(policy_b)
        risk = "mechanism-policy-risk-r17"
        novel = "mechanism-policy-novel-n42"
        self.assertLess(a_ranks[risk], a_ranks[novel])
        self.assertLess(b_ranks[novel], b_ranks[risk])

    def test_task_relevance_gate_isolates_policy_fixtures_before_ranking(self):
        task = load_tasks(["mechanism_policy_cache"])[0]
        policy_a, policy_b = build_policy_transmission_conditions()
        a_record = build_dry_run_record(task, policy_a)
        b_record = build_dry_run_record(task, policy_b)
        cache_ids = {
            "mechanism-policy-risk-r17",
            "mechanism-policy-novel-n42",
        }
        unrelated_ids = {
            "mechanism-policy-precedent-p08",
            "mechanism-policy-exploratory-x31",
        }

        self.assertTrue(unrelated_ids.issubset(a_record["legal_memory_ids"]))
        self.assertTrue(cache_ids.issubset(a_record["task_relevant_memory_ids"]))
        self.assertTrue(
            unrelated_ids.isdisjoint(a_record["task_relevant_memory_ids"])
        )
        self.assertEqual(
            cache_ids.intersection(a_record["task_relevant_memory_ids"]),
            cache_ids.intersection(b_record["task_relevant_memory_ids"]),
        )
        self.assertTrue(unrelated_ids.isdisjoint(a_record["ranked_memory_ids"]))
        self.assertTrue(unrelated_ids.isdisjoint(b_record["ranked_memory_ids"]))
        self.assertTrue(a_record["ranking_matches_policy"])
        self.assertTrue(b_record["ranking_matches_policy"])
        self.assertIsNone(a_record["primary_evidence_matches_policy"])
        self.assertIsNone(a_record["final_choice_matches_policy"])

    def test_profession_fixture_does_not_cross_task_relevance_gate(self):
        task = load_tasks(["mechanism_profession_sentinel"])[0]
        profession_only = build_conditions()[1]
        record = build_dry_run_record(task, profession_only)

        self.assertIn(
            "mechanism-profession-missing-sentinel",
            record["task_relevant_memory_ids"],
        )
        self.assertNotIn(
            "mechanism-profession-rollback-before-retry",
            record["task_relevant_memory_ids"],
        )

    def test_strength_sweep_keeps_profession_fixed_and_scales_only_bias(self):
        conditions = build_strength_sweep_conditions(
            "INTJ", "ENFP", "software_engineering", (0.0, 0.5, 1.0)
        )
        self.assertEqual(6, len(conditions))
        self.assertEqual(
            {("software_engineering",)},
            {item.profession_ids for item in conditions},
        )
        self.assertEqual(
            {0.0, 0.5, 1.0}, {item.policy_strength for item in conditions}
        )

    def test_causal_conditions_expose_only_the_declared_model_treatment(self):
        task = load_causal_tasks(["causal_cache_invalidation"])[0]
        conditions = build_causal_decomposition_conditions()
        self.assertEqual(
            [
                "causal_neutral",
                "causal_order_only",
                "causal_explicit_rank",
                "causal_reasoning_only",
                "causal_full",
            ],
            [item.id for item in conditions],
        )
        records = {
            item.projection_mode: build_dry_run_record(task, item)
            for item in conditions
        }
        neutral = records["neutral"]
        order_only = records["order_only"]
        explicit = records["explicit_rank"]
        reasoning = records["reasoning_only"]
        full = records["full"]

        self.assertEqual(
            ["causal-cache-b", "causal-cache-a"],
            neutral["model_visible_evidence_ids"],
        )
        self.assertEqual(
            ["causal-cache-a", "causal-cache-b"],
            order_only["model_visible_evidence_ids"],
        )
        self.assertIsNone(neutral["model_visible_reasoning_guidance"])
        self.assertIsNone(order_only["model_visible_reasoning_guidance"])
        self.assertFalse(order_only["model_visible_explicit_priority"])
        self.assertTrue(explicit["model_visible_explicit_priority"])
        self.assertEqual(
            [1, 2],
            [
                item["retrieval_rank"]
                for item in explicit["final_context"]["work"][
                    "relevant_archive_and_artifacts"
                ]
            ],
        )
        self.assertEqual(
            neutral["model_visible_evidence_ids"],
            reasoning["model_visible_evidence_ids"],
        )
        self.assertIsNotNone(reasoning["model_visible_reasoning_guidance"])
        self.assertEqual(
            order_only["model_visible_evidence_ids"],
            full["model_visible_evidence_ids"],
        )
        self.assertIsNotNone(full["model_visible_reasoning_guidance"])

        for record in records.values():
            visible = json.dumps(
                record["model_visible_context"], ensure_ascii=False
            ).casefold()
            for forbidden in (
                "intj",
                "enfp",
                "cognitivepolicy",
                "cognitive_policy",
                "policy_strength",
                "raw_parameters",
                "effective_parameters",
                "causal_direction",
                "software_engineering",
            ):
                self.assertNotIn(forbidden, visible)
            self.assertEqual(
                {"runtime_envelope", "life", "profession", "work"},
                set(record["model_visible_context"]),
            )
            self.assertEqual([], record["model_visible_context"]["profession"]["active_resources"])
            self.assertEqual([], record["active_profession"])
            self.assertEqual("generalist", record["work_assignment"]["role_id"])
            self.assertEqual(
                ["filesystem.read"], record["work_assignment"]["grants"]
            )
            self.assertEqual(
                2,
                len(
                    record["final_context"]["work"][
                        "relevant_archive_and_artifacts"
                    ]
                ),
            )
            self.assertEqual(
                "INTJ",
                record["runtime_control_context"]["cognitive_policy"]["id"],
            )

    def test_causal_evidence_pairs_are_counterbalanced_and_fixture_isolated(self):
        tasks = load_causal_tasks()
        self.assertEqual(8, len(tasks))
        risk_as_a = 0
        neutral_risk_first = 0
        neutral_a_first = 0
        all_ids = {
            item["id"]
            for task in tasks
            for item in task.mechanism["evidence"]
        }
        neutral = build_causal_decomposition_conditions()[0]
        for task in tasks:
            evidence = task.mechanism["evidence"]
            lengths = [len(item["content"].split()) for item in evidence]
            self.assertLessEqual(max(lengths) / min(lengths), 1.35)
            by_id = {item["id"]: item for item in evidence}
            risk_as_a += int(
                next(item for item in evidence if item["direction"] == "risk")[
                    "id"
                ].endswith("-a")
            )
            first = task.mechanism["neutral_order"][0]
            neutral_risk_first += int(by_id[first]["direction"] == "risk")
            neutral_a_first += int(first.endswith("-a"))
            record = build_dry_run_record(task, neutral)
            own_ids = {item["id"] for item in evidence}
            self.assertEqual(
                own_ids,
                all_ids.intersection(record["task_relevant_memory_ids"]),
            )
        self.assertEqual(4, risk_as_a)
        self.assertEqual(4, neutral_risk_first)
        self.assertEqual(4, neutral_a_first)

    def test_final_validation_conditions_are_bounded(self):
        self.assertEqual(
            ["final_neutral", "final_reasoning_only"],
            [item.id for item in build_final_validation_conditions()],
        )
        self.assertEqual(
            [
                "final_neutral",
                "final_reasoning_only",
                "final_explicit_rank",
            ],
            [
                item.id
                for item in build_final_validation_conditions(
                    include_explicit_rank=True
                )
            ],
        )

    def test_final_validation_pairs_are_strictly_counterbalanced(self):
        tasks = load_final_validation_tasks()
        self.assertEqual(16, len(tasks))
        target_risk = 0
        risk_as_a = 0
        risk_first = 0
        a_first = 0
        preferred_first = 0
        for task in tasks:
            mechanism = task.mechanism
            self.assertNotIn("reasoning_guidance", mechanism)
            evidence = mechanism["evidence"]
            self.assertEqual(2, len(evidence))
            self.assertEqual(
                {"risk", "exploration"},
                {item["direction"] for item in evidence},
            )
            lengths = [len(item["content"].split()) for item in evidence]
            self.assertLessEqual(max(lengths) / min(lengths), 1.35)
            by_id = {item["id"]: item for item in evidence}
            risk = next(
                item for item in evidence if item["direction"] == "risk"
            )
            target = mechanism["target_direction"]
            preferred = next(
                item for item in evidence if item["direction"] == target
            )
            first = mechanism["neutral_order"][0]
            target_risk += int(target == "risk")
            risk_as_a += int(risk["id"].endswith("-a"))
            risk_first += int(by_id[first]["direction"] == "risk")
            a_first += int(first.endswith("-a"))
            preferred_first += int(first == preferred["id"])
            self.assertEqual(
                "INTJ" if target == "risk" else "ENFP",
                mechanism["target_policy_id"],
            )
        self.assertEqual(8, target_risk)
        self.assertEqual(8, risk_as_a)
        self.assertEqual(8, risk_first)
        self.assertEqual(8, a_first)
        self.assertEqual(8, preferred_first)

    def test_final_guidance_is_compiled_and_hidden_from_model_metadata(self):
        neutral, reasoning = build_final_validation_conditions()
        for task_id, direction in (
            ("final_api_contract", "risk"),
            ("final_abstraction_shape", "exploration"),
        ):
            task = load_final_validation_tasks([task_id])[0]
            neutral_record = build_dry_run_record(task, neutral)
            reasoning_record = build_dry_run_record(task, reasoning)

            self.assertEqual(
                neutral_record["evidence_order"],
                reasoning_record["evidence_order"],
            )
            self.assertIsNone(
                neutral_record["model_visible_reasoning_guidance"]
            )
            self.assertEqual(direction, reasoning_record["policy_direction"])
            self.assertIsNotNone(
                reasoning_record["model_visible_reasoning_guidance"]
            )
            self.assertEqual([], reasoning_record["active_profession"])
            self.assertEqual(
                ["filesystem.read"],
                reasoning_record["work_assignment"]["grants"],
            )
            visible = json.dumps(
                reasoning_record["model_visible_context"],
                ensure_ascii=False,
            ).casefold()
            for forbidden in (
                "intj",
                "enfp",
                "cognitivepolicy",
                "cognitive_policy",
                "policy_strength",
                "effective_parameters",
                "causal_direction",
                "software_engineering",
            ):
                self.assertNotIn(forbidden, visible)
            self.assertEqual(
                {item["id"] for item in task.mechanism["evidence"]},
                set(reasoning_record["model_visible_evidence_ids"]),
            )

    def test_final_validation_smoke_dry_run_scope(self):
        with patch("builtins.print") as printer:
            result = main(["--dry-run", "--final-validation", "--smoke"])
        self.assertEqual(0, result)
        self.assertEqual(1, printer.call_count)
        single = json.loads(printer.call_args.args[0])
        self.assertEqual("final_api_contract", single["task_id"])
        self.assertEqual("final_neutral", single["condition"]["id"])

        with patch("builtins.print") as printer:
            result = main(
                ["--dry-run", "--final-validation", "--smoke-all-conditions"]
            )
        self.assertEqual(0, result)
        self.assertEqual(2, printer.call_count)

        with patch("builtins.print") as printer:
            result = main(
                [
                    "--dry-run",
                    "--final-validation",
                    "--include-explicit-rank",
                    "--smoke-all-conditions",
                ]
            )
        self.assertEqual(0, result)
        self.assertEqual(3, printer.call_count)

    def test_causal_cli_dry_run_prints_one_selected_model_projection(self):
        with patch("builtins.print") as printer:
            result = main(
                [
                    "--dry-run",
                    "--causal-decomposition",
                    "--task",
                    "causal_cache_invalidation",
                    "--condition",
                    "causal_explicit_rank",
                ]
            )
        self.assertEqual(0, result)
        self.assertEqual(1, printer.call_count)
        output = json.loads(printer.call_args.args[0])
        self.assertEqual("explicit_rank", output["model_projection_mode"])
        self.assertEqual(
            2,
            len(
                output["model_visible_context"]["work"][
                    "relevant_archive_and_artifacts"
                ]
            ),
        )

    def test_protocol_failure_metrics_remain_distinct_from_completion(self):
        record = build_dry_run_record(self.task, build_conditions()[0])
        _apply_run_metrics(
            record,
            self.task,
            "",
            [
                {
                    "phase": "decision",
                    "raw_response": '{"type":"final","content": }',
                }
            ],
            [],
            capability_violation=False,
            exception={"type": "DecisionProtocolError", "message": "bad JSON"},
        )
        self.assertFalse(record["structured_decision_valid"])
        self.assertTrue(record["decision_received"])
        self.assertTrue(record["protocol_evaluated"])
        self.assertEqual("protocol", record["failure_stage"])
        self.assertEqual("parse", record["protocol_failure_kind"])
        self.assertFalse(record["protocol_success"])
        self.assertFalse(record["runtime_completion_success"])
        self.assertEqual("protocol_error", record["actual_runtime_action"])
        self.assertIsNone(record["presentation_preservation_score"])

    def test_provider_failure_is_not_misclassified_as_invalid_json(self):
        task = load_causal_tasks(["causal_cache_invalidation"])[0]
        record = build_dry_run_record(
            task, build_causal_decomposition_conditions()[0]
        )
        provider_error = {
            "type": "AgentWorkerError",
            "message": "provider rejected request",
            "failure_stage": "api",
            "api_status_code": 400,
            "api_error_type": "invalid_request_error",
            "api_error_message": "provider rejected request",
        }
        _apply_run_metrics(
            record,
            task,
            "",
            [
                {
                    "phase": "decision",
                    "raw_response": None,
                    "error": provider_error,
                }
            ],
            [],
            capability_violation=False,
            exception=provider_error,
        )
        self.assertFalse(record["decision_received"])
        self.assertFalse(record["protocol_evaluated"])
        self.assertIsNone(record["structured_decision_valid"])
        self.assertEqual("not_received", record["decision_parse_mode"])
        self.assertEqual("api", record["failure_stage"])
        self.assertEqual(400, record["api_status_code"])
        self.assertFalse(record["protocol_success"])
        self.assertFalse(record["runtime_completion_success"])
        self.assertFalse(record["presentation_completion_success"])
        self.assertIsNone(record["causal_metrics"]["primary_is_risk"])
        self.assertIsNone(record["presentation_preservation_score"])

    def test_valid_json_with_invalid_schema_is_a_schema_protocol_failure(self):
        record = build_dry_run_record(self.task, build_conditions()[0])
        _apply_run_metrics(
            record,
            self.task,
            "",
            [
                {
                    "phase": "decision",
                    "raw_response": '{"type":"final","content":42}',
                    "error": None,
                }
            ],
            [],
            capability_violation=False,
            exception={
                "type": "DecisionProtocolError",
                "message": "content must be a string",
                "failure_stage": "protocol",
            },
        )
        self.assertTrue(record["decision_received"])
        self.assertTrue(record["structured_decision_valid"])
        self.assertEqual("strict_json", record["decision_parse_mode"])
        self.assertEqual("protocol", record["failure_stage"])
        self.assertEqual("schema", record["protocol_failure_kind"])
        self.assertFalse(record["protocol_success"])

    def test_round4_smoke_dry_runs_have_fixed_scope(self):
        with patch("builtins.print") as printer:
            result = main(["--dry-run", "--smoke"])
        self.assertEqual(0, result)
        self.assertEqual(1, printer.call_count)
        single = json.loads(printer.call_args.args[0])
        self.assertEqual("causal_cache_invalidation", single["task_id"])
        self.assertEqual("causal_neutral", single["condition"]["id"])

        with patch("builtins.print") as printer:
            result = main(["--dry-run", "--smoke-all-conditions"])
        self.assertEqual(0, result)
        self.assertEqual(5, printer.call_count)

    def test_output_schema_and_jsonl_csv_writers(self):
        record = build_dry_run_record(self.task, build_conditions()[0])
        validate_record_schema(record)
        self.assertIn("protocol_metrics", record)
        self.assertIn("presentation_metrics", record)
        self.assertIn("policy_parameters", record)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "runs.jsonl"
            jsonl, csv_path = write_results([record], output)
            self.assertTrue(jsonl.is_file())
            self.assertTrue(csv_path.is_file())
            self.assertEqual(1, len(jsonl.read_text(encoding="utf-8").splitlines()))
            with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                fields = csv.DictReader(handle).fieldnames or []
            for field in (
                "api_request_success_rate",
                "decision_received_rate",
                "protocol_success_rate_among_received",
                "runtime_completion_rate",
                "presentation_completion_rate",
            ):
                self.assertIn(field, fields)

    def test_summary_excludes_not_received_from_protocol_denominator(self):
        task = load_causal_tasks(["causal_cache_invalidation"])[0]
        condition = build_causal_decomposition_conditions()[0]
        api_failure = build_dry_run_record(task, condition)
        malformed = build_dry_run_record(task, condition)
        error = {
            "failure_stage": "api",
            "api_status_code": 400,
            "api_error_type": "invalid_request_error",
            "api_error_message": "rejected",
        }
        _apply_run_metrics(
            api_failure,
            task,
            "",
            [{"phase": "decision", "raw_response": None, "error": error}],
            [],
            capability_violation=False,
            exception=error,
        )
        _apply_run_metrics(
            malformed,
            task,
            "",
            [
                {
                    "phase": "decision",
                    "raw_response": '{"type":"final","content": }',
                    "error": None,
                }
            ],
            [],
            capability_violation=False,
            exception={
                "type": "DecisionProtocolError",
                "failure_stage": "protocol",
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            _, csv_path = write_results(
                [api_failure, malformed], Path(directory) / "rates.jsonl"
            )
            with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        aggregate = next(row for row in rows if row["row_type"] == "aggregate")
        self.assertEqual("0.5", aggregate["api_request_success_rate"])
        self.assertEqual("0.5", aggregate["decision_received_rate"])
        self.assertEqual(
            "0.0", aggregate["protocol_success_rate_among_received"]
        )
        self.assertEqual("0.0", aggregate["presentation_completion_rate"])
        self.assertEqual("0", aggregate["selection_observation_count"])

    def test_causal_summary_reports_condition_delta_against_neutral(self):
        task = load_causal_tasks(["causal_cache_invalidation"])[0]
        neutral, order_only = build_causal_decomposition_conditions()[:2]
        neutral_record = build_dry_run_record(task, neutral)
        order_record = build_dry_run_record(task, order_only)
        for record, is_risk in (
            (neutral_record, False),
            (order_record, True),
        ):
            record["protocol_success"] = True
            record["runtime_completion_success"] = True
            record["causal_metrics"].update(
                {
                    "primary_is_risk": is_risk,
                    "primary_matches_target_direction": is_risk,
                    "primary_is_first_visible": True,
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            _, csv_path = write_results(
                [neutral_record, order_record],
                Path(directory) / "causal.jsonl",
            )
            with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        order_summary = next(
            row
            for row in rows
            if row["row_type"] == "aggregate"
            and row["condition"] == "causal_order_only"
        )
        self.assertEqual("1.0", order_summary["risk_selection_rate"])
        self.assertEqual("1.0", order_summary["delta_risk_vs_neutral"])
        self.assertEqual("1", order_summary["selection_observation_count"])

    def test_final_summary_reports_paired_switches_and_token_overhead(self):
        task = load_final_validation_tasks(["final_api_contract"])[0]
        neutral, reasoning = build_final_validation_conditions()
        records = []
        for replicate, baseline_value, guidance_value in (
            (1, False, True),
            (2, True, False),
        ):
            for condition, selected, tokens in (
                (neutral, baseline_value, (100, 50, 150)),
                (reasoning, guidance_value, (110, 60, 170)),
            ):
                record = build_dry_run_record(
                    task, condition, replicate=replicate
                )
                record["causal_metrics"].update(
                    {
                        "primary_is_risk": selected,
                        "primary_matches_target_direction": selected,
                        "primary_is_first_visible": selected,
                    }
                )
                record["usage"] = {
                    "input_tokens": tokens[0],
                    "output_tokens": tokens[1],
                    "total_tokens": tokens[2],
                }
                records.append(record)
        with tempfile.TemporaryDirectory() as directory:
            _, csv_path = write_results(
                records, Path(directory) / "final.jsonl"
            )
            with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        summary = next(
            row
            for row in rows
            if row["row_type"] == "final_validation_overall"
            and row["condition"] == "final_reasoning_only"
        )
        self.assertEqual("0.5", summary["neutral_baseline_preferred_rate"])
        self.assertEqual("0.5", summary["preferred_selection_rate"])
        self.assertEqual("0.0", summary["delta_preferred_vs_neutral"])
        self.assertEqual("1", summary["paired_switches_toward_preferred"])
        self.assertEqual("1", summary["paired_switches_away_from_preferred"])
        self.assertEqual("2", summary["paired_observation_count"])
        self.assertEqual("20.0", summary["token_overhead_vs_neutral"])

    def test_deterministic_evaluator(self):
        answer = (
            "This is a mutable default shared across calls: both calls reuse the "
            "same list. Use tags=None and initialize it with `if tags is None`."
        )
        result = evaluate_answer(self.task, answer)
        self.assertTrue(result["task_success"])
        self.assertEqual(1.0, result["score"])
        denied = evaluate_answer(
            self.task, answer, capability_violation=True
        )
        self.assertFalse(denied["task_success"])
        self.assertTrue(denied["capability_violation"])


if __name__ == "__main__":
    unittest.main()
