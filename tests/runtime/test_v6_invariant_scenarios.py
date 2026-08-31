import tempfile
import unittest
from pathlib import Path

from harness.models import (
    ActionRequest,
    FinalAnswer,
    ResourceBindingRequest,
    WorkAssignmentRequest,
)
from tests.support.runtime_factory import build_harness


class V6RuntimeInvariantScenarios(unittest.TestCase):
    def test_ada_review_story_preserves_resources_but_revokes_assignment(self):
        with tempfile.TemporaryDirectory() as workspace:
            Path(workspace, "README.md").write_text("SForge", encoding="utf-8")
            harness, _, _, _ = build_harness(workspace, workspace_id="SForge")
            try:
                process = harness.create_agent("review SForge")
                base = harness.runtime_state(process.id)
                self.assertEqual("ada", base.identity_id)
                self.assertEqual({"echo"}, set(base.allowed_capabilities))

                profession = harness.request_binding(
                    process.id,
                    ResourceBindingRequest(
                        "profession", "activate", "software_engineering"
                    ),
                )
                policy = harness.request_binding(
                    process.id,
                    ResourceBindingRequest(
                        "cognitive_policy", "activate", "INTJ"
                    ),
                )
                assignment = harness.request_work_assignment(
                    process.id,
                    WorkAssignmentRequest(
                        "reviewer",
                        workspace_id="SForge",
                        workflow_id="general_task",
                        requested_capabilities=("filesystem.read",),
                    ),
                )
                context = harness.build_context(process.id)
                read = harness.execute_action(
                    process.id,
                    ActionRequest("filesystem.read", {"path": "README.md"}),
                )
                write = harness.execute_action(
                    process.id,
                    ActionRequest(
                        "filesystem.write",
                        {"path": "forbidden.txt", "content": "no"},
                    ),
                )

                self.assertEqual("success", profession.status)
                self.assertEqual("success", policy.status)
                self.assertEqual("success", assignment.status)
                self.assertEqual("software_engineering", context.professions[0]["id"])
                self.assertEqual("INTJ", context.cognitive_policy["id"])
                self.assertEqual("reviewer", context.work_role["id"])
                self.assertEqual("success", read.status)
                self.assertEqual("rejected", write.status)
                self.assertEqual("admission", write.metadata["stage"])
                self.assertFalse(Path(workspace, "forbidden.txt").exists())

                harness.end_work_assignment(process.id)
                ended_state = harness.runtime_state(process.id)
                ended_context = harness.build_context(process.id)
                read_after_close = harness.execute_action(
                    process.id,
                    ActionRequest("filesystem.read", {"path": "README.md"}),
                )

                self.assertIsNone(ended_state.assignment_id)
                self.assertEqual({"echo"}, set(ended_state.allowed_capabilities))
                self.assertEqual("INTJ", ended_state.cognitive_policy_id)
                self.assertEqual(
                    ("software_engineering",), ended_state.profession_ids
                )
                self.assertIsNone(ended_context.workspace)
                self.assertIsNone(ended_context.work_assignment)
                self.assertEqual("rejected", read_after_close.status)
                self.assertEqual("admission", read_after_close.metadata["stage"])
            finally:
                harness.close()

    def test_assignment_replacement_does_not_union_old_and_new_grants(self):
        with tempfile.TemporaryDirectory() as workspace:
            Path(workspace, "input.txt").write_text("input", encoding="utf-8")
            harness, _, _, _ = build_harness(
                workspace, workspace_id="SForge"
            )
            try:
                process = harness.create_agent("change assignment")
                first = harness.request_work_assignment(
                    process.id,
                    WorkAssignmentRequest(
                        "reviewer",
                        workspace_id="SForge",
                        workflow_id="general_task",
                        requested_capabilities=("filesystem.read",),
                    ),
                )
                second = harness.request_work_assignment(
                    process.id,
                    WorkAssignmentRequest(
                        "developer",
                        workspace_id="SForge",
                        workflow_id="general_task",
                        requested_capabilities=("filesystem.write",),
                    ),
                )
                state = harness.runtime_state(process.id)
                stale_read = harness.execute_action(
                    process.id,
                    ActionRequest("filesystem.read", {"path": "input.txt"}),
                )
                current_write = harness.execute_action(
                    process.id,
                    ActionRequest(
                        "filesystem.write",
                        {"path": "output.txt", "content": "verified"},
                    ),
                )

                self.assertEqual("success", first.status)
                self.assertEqual("success", second.status)
                self.assertNotEqual(first.assignment_id, second.assignment_id)
                self.assertEqual(
                    {"echo", "filesystem.write"},
                    set(state.allowed_capabilities),
                )
                self.assertEqual("rejected", stale_read.status)
                self.assertEqual("admission", stale_read.metadata["stage"])
                self.assertEqual("success", current_write.status)
            finally:
                harness.close()

    def test_final_answer_closes_assignment_and_records_revocation(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, memory, _ = build_harness(
                workspace,
                responses=['{"type":"final","content":"reviewed"}'],
                workspace_id="SForge",
            )
            try:
                process = harness.create_agent("finish review")
                harness.request_binding(
                    process.id,
                    ResourceBindingRequest(
                        "profession", "activate", "software_engineering"
                    ),
                )
                harness.request_binding(
                    process.id,
                    ResourceBindingRequest(
                        "cognitive_policy", "activate", "INTJ"
                    ),
                )
                harness.request_work_assignment(
                    process.id,
                    WorkAssignmentRequest(
                        "reviewer",
                        workspace_id="SForge",
                        workflow_id="general_task",
                        requested_capabilities=("filesystem.read",),
                    ),
                )

                result = harness.step(process.id)
                state = harness.runtime_state(process.id)
                archive = memory.retrieve(scope="workspace:SForge", limit=50)

                self.assertIsInstance(result, FinalAnswer)
                self.assertEqual("reviewed", result.content)
                self.assertIsNone(state.assignment_id)
                self.assertEqual({"echo"}, set(state.allowed_capabilities))
                self.assertEqual("INTJ", state.cognitive_policy_id)
                self.assertEqual(("software_engineering",), state.profession_ids)
                self.assertIn(
                    "workspace.assignment_ended", {item.kind for item in archive}
                )
            finally:
                harness.close()

    def test_workspace_archive_survives_disposable_agent_processes(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, _, _ = build_harness(
                workspace, workspace_id="SForge"
            )
            try:
                first = harness.create_agent("first review")
                harness.request_work_assignment(
                    first.id,
                    WorkAssignmentRequest(
                        "reviewer", workspace_id="SForge"
                    ),
                )
                harness.end_work_assignment(first.id, reason="verified review")
                harness.terminate_agent(first.id)

                second = harness.create_agent("second review")
                harness.request_work_assignment(
                    second.id,
                    WorkAssignmentRequest(
                        "reviewer", workspace_id="SForge"
                    ),
                )
                context = harness.build_context(second.id)
                archive_kinds = {item.kind for item in context.memory}

                self.assertIn("workspace.assignment_started", archive_kinds)
                self.assertIn("workspace.assignment_ended", archive_kinds)
                self.assertEqual("SForge", context.workspace["id"])
            finally:
                harness.close()


if __name__ == "__main__":
    unittest.main()
