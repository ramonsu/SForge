import tempfile
import unittest
from datetime import datetime, timezone

from harness.models import MemoryRecord, ResourceBindingRequest
from tests.support.runtime_factory import build_harness


class CognitiveContextIntegrationTests(unittest.TestCase):
    def test_profession_changes_memory_candidates_and_skill_view_then_unbinds(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, memory, _ = build_harness(workspace)
            memory.write(
                MemoryRecord(
                    id="software-memory",
                    scope="identity:ada",
                    kind="identity.work_experience",
                    content="trace interface callers",
                    metadata={
                        "professional_tags": ["software_engineering"]
                    },
                )
            )
            try:
                process = harness.create_agent("review interfaces")
                base = harness.build_context(process.id)
                self.assertNotIn(
                    "software-memory", {item.id for item in base.memory}
                )
                self.assertFalse(
                    any(
                        "profession:software_engineering" in item["sources"]
                        for item in base.skills
                    )
                )

                mounted = harness.request_binding(
                    process.id,
                    ResourceBindingRequest(
                        "profession", "activate", "software_engineering"
                    ),
                )
                self.assertEqual("success", mounted.status)
                professional = harness.build_context(process.id)
                self.assertIn(
                    "software-memory", {item.id for item in professional.memory}
                )
                self.assertTrue(
                    any(
                        "profession:software_engineering" in item["sources"]
                        for item in professional.skills
                    )
                )

                removed = harness.request_binding(
                    process.id,
                    ResourceBindingRequest(
                        "profession", "deactivate", "software_engineering"
                    ),
                )
                self.assertEqual("success", removed.status)
                refreshed = harness.build_context(process.id)
                self.assertNotIn(
                    "software-memory", {item.id for item in refreshed.memory}
                )
                self.assertFalse(
                    any(
                        "profession:software_engineering" in item["sources"]
                        for item in refreshed.skills
                    )
                )
            finally:
                harness.close()

    def test_policy_switch_rebuilds_ranking_without_stale_context(self):
        timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, memory, _ = build_harness(workspace)
            risk = MemoryRecord(
                id="z-risk",
                scope="core",
                kind="runtime.risk",
                content="contradiction risk",
                created_at=timestamp,
            )
            novel = MemoryRecord(
                id="a-novel",
                scope="core",
                kind="runtime.idea",
                content="novel new idea",
                created_at=timestamp,
            )
            memory.write(risk)
            memory.write(novel)
            try:
                process = harness.create_agent("choose")
                harness.request_binding(
                    process.id,
                    ResourceBindingRequest(
                        "cognitive_policy", "activate", "ESTJ"
                    ),
                )
                first = [
                    item.id for item in harness.build_context(process.id).memory
                ]
                harness.request_binding(
                    process.id,
                    ResourceBindingRequest(
                        "cognitive_policy", "activate", "ENFP"
                    ),
                )
                second_context = harness.build_context(process.id)
                second = [item.id for item in second_context.memory]

                self.assertEqual("ENFP", second_context.cognitive_policy["id"])
                self.assertEqual(set(first), set(second))
                self.assertLess(first.index(risk.id), first.index(novel.id))
                self.assertLess(second.index(novel.id), second.index(risk.id))
            finally:
                harness.close()

    def test_policy_never_expands_the_legal_memory_candidate_set(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, memory, _ = build_harness(workspace)
            memory.write(
                MemoryRecord(
                    id="private-risk",
                    scope="private:outside",
                    kind="risk.evidence",
                    content="highly relevant contradiction risk test",
                )
            )
            try:
                process = harness.create_agent("find contradiction risk")
                harness.request_binding(
                    process.id,
                    ResourceBindingRequest(
                        "cognitive_policy", "activate", "INTJ"
                    ),
                )
                context = harness.build_context(process.id)

                self.assertEqual("INTJ", context.cognitive_policy["id"])
                self.assertNotIn(
                    "private-risk", {item.id for item in context.memory}
                )
            finally:
                harness.close()


if __name__ == "__main__":
    unittest.main()
