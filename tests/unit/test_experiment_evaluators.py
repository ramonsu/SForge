import unittest

from experiments.evaluators import (
    evaluate_causal_selection,
    evaluate_decision,
    evaluate_mechanism,
    evaluate_policy_transmission,
    evaluate_presentation,
    normalize_text,
)


class ExperimentEvaluatorTests(unittest.TestCase):
    def test_normalization_handles_unicode_markdown_and_spacing(self):
        self.assertEqual("choice: c", normalize_text(" **Ｃｈｏｉｃｅ：  C**\n"))

    def test_choice_variants_are_one_semantic_selection(self):
        task = {
            "required_findings": [
                {"id": "choose_c", "any_of": ["option c"], "weight": 1}
            ],
            "forbidden_claims": [],
        }
        for answer in ("Choice: C", "option C", "I choose C", "选择：C"):
            result = evaluate_decision(task, answer, protocol_success=True)
            self.assertEqual(1.0, result["semantic_score"], answer)
            self.assertEqual("C", result["selected_option"], answer)
            self.assertTrue(result["evidence"][0]["matched"])

    def test_python_ast_detects_none_default_fix(self):
        task = {
            "required_findings": [
                {"id": "none_fix", "any_of": [], "weight": 1}
            ],
            "forbidden_claims": [],
        }
        answer = """```python
def add_tag(tag, tags=None):
    if tags is None:
        tags = []
    return tags
```"""
        result = evaluate_decision(task, answer, protocol_success=True)
        self.assertEqual("python_ast", result["evidence"][0]["method"])
        self.assertTrue(result["semantic_success"])

    def test_profession_semantic_variants_avoid_phrase_false_negatives(self):
        task = {
            "required_findings": [
                {"id": "none_is_valid", "any_of": [], "weight": 0.5},
                {
                    "id": "rollback_before_retry",
                    "any_of": [],
                    "weight": 0.5,
                },
            ],
            "forbidden_claims": [],
        }
        result = evaluate_decision(
            task,
            "None is a valid cached value; roll back the failed transaction "
            "before retrying.",
            protocol_success=True,
        )
        self.assertEqual(1.0, result["semantic_score"])
        self.assertEqual(
            {"semantic_none_value", "semantic_transaction_order"},
            {item["method"] for item in result["evidence"]},
        )

    def test_presentation_omission_does_not_change_decision_score(self):
        task = {
            "required_findings": [
                {"id": "root", "any_of": ["root cause"], "weight": 0.5},
                {"id": "fix", "any_of": ["safe fix"], "weight": 0.5},
            ],
            "forbidden_claims": [],
        }
        decision = evaluate_decision(
            task, "root cause and safe fix", protocol_success=True
        )
        presentation = evaluate_presentation(
            task, "root cause and safe fix", "root cause"
        )
        self.assertEqual(1.0, decision["semantic_score"])
        self.assertEqual(0.5, presentation["preservation_score"])
        self.assertEqual(["fix"], presentation["omitted_findings"])

    def test_presentation_context_misread_is_formatting_issue_only(self):
        task = {
            "required_findings": [
                {"id": "fact", "any_of": ["fact"], "weight": 1}
            ],
            "forbidden_claims": [],
        }
        result = evaluate_presentation(
            task, "fact", "Here is the fact for your 演示文稿."
        )
        self.assertEqual(1.0, result["preservation_score"])
        self.assertFalse(result["formatting_integrity"])

    def test_presentation_reports_semantic_claim_added_after_decision(self):
        task = {
            "required_findings": [
                {"id": "fact", "any_of": ["critical fact"], "weight": 1}
            ],
            "forbidden_claims": [],
        }
        result = evaluate_presentation(task, "uncertain", "critical fact")
        self.assertIn(
            "required_finding_added:fact", result["introduced_claims"]
        )
        self.assertFalse(result["preservation_success"])

    def test_presentation_internal_term_leak_and_absence_are_observable(self):
        task = {"required_findings": [], "forbidden_claims": []}
        leaked = evaluate_presentation(
            task,
            "safe answer",
            "这是格式化后的最终回复：safe answer",
        )
        self.assertIn(
            "internal_response_rendering_term_leaked",
            leaked["formatting_issues"],
        )
        absent = evaluate_presentation(task, "safe answer", None)
        self.assertFalse(absent["applicable"])
        self.assertIsNone(absent["preservation_score"])
        self.assertIsNone(absent["formatting_integrity"])

    def test_secondary_evidence_cannot_create_policy_focus_false_positive(self):
        task = {
            "mechanism": {
                "kind": "policy_transmission",
                "candidate_memory_ids": ["risk", "novel"],
                "policy_preferences": {
                    "ENFP": {
                        "primary_evidence_id": "novel",
                        "final_choice": "explore",
                    }
                },
            }
        }
        result = evaluate_policy_transmission(
            task,
            policy_id="ENFP",
            policy_strength=1.0,
            retrieval_trace={
                "task_relevant_memory_ids": ["risk", "novel"],
                "retrieval_scores": [
                    {"id": "novel", "rank": 1},
                    {"id": "risk", "rank": 2},
                ],
            },
            decision_payload={
                "primary_evidence_id": "risk",
                "secondary_evidence_ids": ["novel"],
                "final_choice": "guarded",
            },
        )
        self.assertTrue(result["ranking_matches_policy"])
        self.assertFalse(result["primary_evidence_matches_policy"])
        self.assertFalse(result["final_choice_matches_policy"])

    def test_causal_evaluator_reads_primary_from_structured_field_only(self):
        task = {
            "suite": "causal",
            "mechanism": {
                "target_direction": "exploration",
                "evidence": [
                    {
                        "id": "a",
                        "direction": "risk",
                        "choice_id": "safe",
                    },
                    {
                        "id": "b",
                        "direction": "exploration",
                        "choice_id": "explore",
                    },
                ],
            },
        }
        result = evaluate_causal_selection(
            task,
            condition={"id": "reasoning", "projection_mode": "reasoning_only"},
            retrieval_trace={"model_visible_evidence_ids": ["a", "b"]},
            decision_payload={
                "content": "The prose praises a repeatedly.",
                "primary_evidence_id": "b",
                "secondary_evidence_ids": ["a"],
                "final_choice": "explore",
            },
        )
        self.assertEqual("exploration", result["primary_direction"])
        self.assertTrue(result["primary_matches_target_direction"])
        self.assertFalse(result["primary_is_first_visible"])
        self.assertTrue(result["primary_choice_consistent"])

    def test_presentation_detects_policy_and_retrieval_metadata_leaks(self):
        task = {"required_findings": [], "forbidden_claims": []}
        result = evaluate_presentation(
            task,
            "choose b",
            "ENFP CognitivePolicy novelty_weight=0.85; retrieval_rank=1.",
        )
        self.assertIn(
            "internal_policy_metadata_leaked",
            result["formatting_issues"],
        )

    def test_profession_mechanism_checks_retrieval_and_evidence(self):
        task = {
            "suite": "mechanism",
            "mechanism": {
                "kind": "profession_sensitive",
                "memory_ids": ["se-memory"],
                "evidence_patterns": ["se-17"],
            },
        }
        result = evaluate_mechanism(
            task,
            condition={"profession_ids": ["software_engineering"]},
            retrieved_memories=[{"id": "se-memory", "rank": 1}],
            decision_text="Apply SE-17.",
        )
        self.assertEqual(1.0, result["mechanism_score"])
        self.assertTrue(result["retrieval_alignment"])

    def test_policy_mechanism_compares_rank_and_answer_focus(self):
        task = {
            "suite": "mechanism",
            "mechanism": {
                "kind": "policy_contrast",
                "expected_focus": {"INTJ": "risk"},
                "branches": {
                    "risk": {
                        "memory_ids": ["risk"],
                        "evidence_patterns": ["r-17"],
                    },
                    "novel": {
                        "memory_ids": ["novel"],
                        "evidence_patterns": ["n-42"],
                    },
                },
            },
        }
        result = evaluate_mechanism(
            task,
            condition={"cognitive_policy_id": "INTJ", "profession_ids": []},
            retrieved_memories=[
                {"id": "risk", "rank": 1},
                {"id": "novel", "rank": 2},
            ],
            decision_text="R-17 is the controlling evidence.",
        )
        self.assertEqual(1.0, result["mechanism_score"])
        self.assertEqual("risk", result["selected_focus"])


if __name__ == "__main__":
    unittest.main()
