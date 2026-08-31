import tempfile
import unittest
from pathlib import Path

from harness.models import ActionRequest, TaskSpec, WorkflowRequest
from tests.support.runtime_factory import build_harness


class WorkflowAdmissionTests(unittest.TestCase):
    def test_workflow_choice_is_only_a_hint_until_agent_admission(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, _, _ = build_harness(workspace)
            try:
                process = harness.create_agent(
                    TaskSpec("write a story", id="hint-task"),
                    "novel_writing",
                )
                state = harness.runtime_state(process.id)
                context = harness.build_context(process.id)
                self.assertEqual("direct", state.mode)
                self.assertIsNone(state.workflow_id)
                self.assertEqual(("core",), state.memory_scopes)
                self.assertEqual({"echo"}, set(state.allowed_capabilities))
                self.assertEqual(
                    "novel_writing",
                    context.task["context"]["requested_workflow_id"],
                )
                self.assertIsNone(context.workflow)
            finally:
                harness.close()

    def test_initial_admission_mounts_one_state_atomically(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, _, _ = build_harness(workspace)
            try:
                process = harness.create_agent(
                    TaskSpec("write", id="novel-task")
                )
                before = harness.execute_action(
                    process.id,
                    ActionRequest("filesystem.list", {"path": "."}),
                )
                self.assertEqual("rejected", before.status)

                admission = harness.request_workflow(
                    process.id, WorkflowRequest("novel_writing")
                )
                self.assertEqual("success", admission.status)
                self.assertEqual("creation", admission.workflow_state_id)
                self.assertEqual(
                    "workflow:novel_writing:creative_memory",
                    admission.memory_scope,
                )
                self.assertEqual(
                    (
                        "core",
                        "workflow:novel_writing",
                        "workflow:novel_writing:creative_memory",
                    f"workspace:{Path(workspace).name}",
                    ),
                    admission.memory_scopes,
                )
                self.assertIn("filesystem.write", admission.allowed_capabilities)

                state = harness.runtime_state(process.id)
                context = harness.build_context(process.id)
                self.assertEqual("workflow", state.mode)
                self.assertEqual(2, state.version)
                self.assertEqual("creation", state.workflow_state_id)
                self.assertEqual(
                    "creation", context.workflow["current_state"]["id"]
                )
                self.assertEqual(
                    [{"condition": "draft_completed", "target": "revision"}],
                    context.workflow["outgoing_transitions"],
                )
            finally:
                harness.close()

    def test_invalid_admission_and_edge_leave_runtime_unchanged(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, _, _ = build_harness(workspace)
            try:
                process = harness.create_agent("novel")
                bootstrap = harness.runtime_state(process.id)
                rejected = harness.request_workflow(
                    process.id,
                    WorkflowRequest(
                        "novel_writing", target_state_id="review"
                    ),
                )
                self.assertEqual("rejected", rejected.status)
                self.assertEqual(bootstrap, harness.runtime_state(process.id))

                harness.request_workflow(
                    process.id, WorkflowRequest("novel_writing")
                )
                creation = harness.runtime_state(process.id)
                bad_edge = harness.request_workflow(
                    process.id,
                    WorkflowRequest(
                        "novel_writing",
                        target_state_id="review",
                        transition_condition="draft_completed",
                    ),
                )
                self.assertEqual("rejected", bad_edge.status)
                self.assertEqual(creation, harness.runtime_state(process.id))

                cross_workflow = harness.request_workflow(
                    process.id, WorkflowRequest("general_task")
                )
                self.assertEqual("rejected", cross_workflow.status)
                self.assertEqual(creation, harness.runtime_state(process.id))
            finally:
                harness.close()

    def test_declared_cycle_can_be_traversed_by_agent_requests(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, _, _ = build_harness(workspace)
            try:
                process = harness.create_agent("revise a novel")
                harness.request_workflow(
                    process.id, WorkflowRequest("novel_writing")
                )
                revision = harness.request_workflow(
                    process.id,
                    WorkflowRequest(
                        "novel_writing",
                        target_state_id="revision",
                        transition_condition="draft_completed",
                    ),
                )
                review = harness.request_workflow(
                    process.id,
                    WorkflowRequest(
                        "novel_writing",
                        target_state_id="review",
                        transition_condition="revision_completed",
                    ),
                )
                revision_again = harness.request_workflow(
                    process.id,
                    WorkflowRequest(
                        "novel_writing",
                        target_state_id="revision",
                        transition_condition="feedback_generated",
                    ),
                )
                self.assertEqual(
                    ["revision", "review", "revision"],
                    [
                        revision.workflow_state_id,
                        review.workflow_state_id,
                        revision_again.workflow_state_id,
                    ],
                )
                self.assertNotIn(
                    "filesystem.write", review.allowed_capabilities
                )
                self.assertIn(
                    "filesystem.write", revision_again.allowed_capabilities
                )
                self.assertEqual(5, harness.runtime_state(process.id).version)
            finally:
                harness.close()


if __name__ == "__main__":
    unittest.main()
