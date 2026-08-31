import json
import unittest

from agent.agent import Agent
from agent.decision_protocol import parse_decision_payload
from harness.models import FinalAnswer


class DecisionProtocolTests(unittest.TestCase):
    def test_valid_escaped_multiline_and_code_are_strict_json(self):
        raw = json.dumps(
            {
                "type": "final",
                "content": "line one\n```python\nprint(\"quoted\")\n```",
            }
        )
        result = parse_decision_payload(raw)
        self.assertTrue(result.structured_decision_valid)
        self.assertFalse(result.fallback_used)
        self.assertIn('print("quoted")', result.payload["content"])

    def test_json_code_fence_is_valid_transport_decoration(self):
        result = parse_decision_payload(
            '```json\n{"type":"final","content":"ok"}\n```'
        )
        self.assertTrue(result.structured_decision_valid)
        self.assertEqual("fenced_json", result.decision_parse_mode)

    def test_literal_newline_is_recovered_but_remains_protocol_failure(self):
        result = parse_decision_payload(
            '{"type":"final","content":"first line\nsecond line"}'
        )
        self.assertFalse(result.structured_decision_valid)
        self.assertTrue(result.fallback_used)
        self.assertEqual("control_character_repair", result.decision_parse_mode)
        self.assertIn("Invalid control character", result.decision_parse_error)
        self.assertEqual("first line\nsecond line", result.payload["content"])

    def test_malformed_json_is_not_converted_to_final_answer(self):
        result = parse_decision_payload('{"type":"final","content": }')
        self.assertIsNone(result.payload)
        self.assertFalse(result.structured_decision_valid)
        self.assertFalse(result.fallback_used)
        self.assertEqual("invalid_json", result.decision_parse_mode)

    def test_one_missing_closing_brace_has_visible_bounded_repair(self):
        result = parse_decision_payload(
            '{"type":"final","content":"ok"'
        )
        self.assertIsNotNone(result.payload)
        self.assertFalse(result.structured_decision_valid)
        self.assertTrue(result.fallback_used)
        self.assertEqual(
            "missing_closing_brace_repair", result.decision_parse_mode
        )

    def test_final_decision_accepts_optional_structured_evidence_fields(self):
        decision = Agent._parse_decision(
            json.dumps(
                {
                    "type": "final",
                    "content": "Use the guarded path.",
                    "primary_evidence_id": "risk-r17",
                    "secondary_evidence_ids": ["support-s2"],
                    "final_choice": "guarded_invalidation",
                }
            )
        )
        self.assertIsInstance(decision, FinalAnswer)
        self.assertEqual("risk-r17", decision.primary_evidence_id)
        self.assertEqual(("support-s2",), decision.secondary_evidence_ids)
        self.assertEqual("guarded_invalidation", decision.final_choice)


if __name__ == "__main__":
    unittest.main()
