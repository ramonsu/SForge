import json
import tempfile
import unittest
from pathlib import Path

from harness.memory_manager import SQLiteMemoryProvider
from harness.models import (
    ActionRequest,
    AgentStatus,
    ResourceBindingRequest,
    TaskSpec,
    WorkAssignmentRequest,
    WorkflowRequest,
)
from tests.support.runtime_factory import build_harness


class RuntimeContinuityTests(unittest.TestCase):
    def _prepare_long_work(self, workspace, *, memory_provider=None):
        Path(workspace, "input.txt").write_text(
            "continuity input", encoding="utf-8"
        )
        harness, supervisor, memory, _ = build_harness(
            workspace,
            workspace_id="SForge",
            memory_provider=memory_provider,
        )
        process = harness.create_agent(
            TaskSpec("review the continuity boundary", id="continuity-task")
        )
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
        admission = harness.request_work_assignment(
            process.id,
            WorkAssignmentRequest(
                "reviewer",
                workspace_id="SForge",
                workflow_id="general_task",
                requested_capabilities=(
                    "filesystem.read",
                    "filesystem.write",
                ),
            ),
        )
        action = harness.execute_action(
            process.id,
            ActionRequest(
                "filesystem.write",
                {
                    "path": "continuity.txt",
                    "content": "created by process one",
                },
            ),
        )
        experience = harness.record_work_experience(
            process.id,
            "Reconstruct context from durable facts after process replacement.",
            objective_outcome="continuity.txt was written and verified",
            professional_tags=("software_engineering", "testing"),
            artifact_refs=("continuity.txt",),
        )
        transition = harness.request_workflow(
            process.id,
            WorkflowRequest(
                "general_task",
                target_state_id="audit",
                transition_condition="audit_requested",
            ),
        )
        self.assertEqual("success", admission.status)
        self.assertEqual("success", action.status)
        self.assertEqual("success", transition.status)
        return harness, supervisor, memory, process, admission, experience

    def test_process_destruction_does_not_destroy_identity(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, _, first, _, _ = self._prepare_long_work(workspace)
            try:
                harness.terminate_agent(first.id)
                second = harness.create_agent("continue the same work")
                self.assertNotEqual(first.id, second.id)
                self.assertEqual(first.identity_id, second.identity_id)
                self.assertEqual(
                    "ada", harness.runtime_state(second.id).identity_id
                )
            finally:
                harness.close()

    def test_process_destruction_does_not_destroy_durable_memory(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, memory, first, _, experience = (
                self._prepare_long_work(workspace)
            )
            try:
                harness.terminate_agent(first.id)
                self.assertEqual(experience, memory.get(experience.id))
                second = harness.create_agent("continue the same work")
                model = harness.build_context(second.id).for_model()
                ids = {
                    item["id"]
                    for item in model["profession"]["professional_memory"]
                }
                self.assertIn(experience.id, ids)
            finally:
                harness.close()

    def test_process_destruction_does_not_destroy_workspace_archive(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, memory, first, _, _ = self._prepare_long_work(
                workspace
            )
            try:
                harness.terminate_agent(first.id)
                self.assertTrue(Path(workspace, "continuity.txt").exists())
                archive = memory.retrieve(
                    scope="workspace:SForge", limit=100
                )
                self.assertIn(
                    "workspace.action_result",
                    {item.kind for item in archive},
                )
                second = harness.create_agent("continue the same work")
                work = harness.build_context(second.id).for_model()["work"]
                self.assertIn(
                    "workspace.action_result",
                    {
                        item["kind"]
                        for item in work[
                            "relevant_archive_and_artifacts"
                        ]
                    },
                )
            finally:
                harness.close()

    def test_process_destruction_does_not_end_active_assignment(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, memory, first, admission, _ = (
                self._prepare_long_work(workspace)
            )
            try:
                terminated = harness.terminate_agent(first.id)
                assignment = harness.work_assignment(first.id)
                archive = memory.retrieve(
                    scope="workspace:SForge", limit=100
                )
                self.assertEqual(AgentStatus.TERMINATED, terminated.status)
                self.assertIsNotNone(assignment)
                self.assertEqual(admission.assignment_id, assignment.id)
                self.assertEqual("active", assignment.status)
                self.assertNotIn(
                    "workspace.assignment_ended",
                    {item.kind for item in archive},
                )
            finally:
                harness.close()

    def test_new_process_reconstructs_life_profession_work_context(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, _, first, admission, experience = (
                self._prepare_long_work(workspace)
            )
            try:
                context_one = harness.build_context(first.id)
                harness.terminate_agent(first.id)
                second = harness.create_agent(
                    TaskSpec("continue from durable evidence", id="new-turn")
                )
                context_two = harness.build_context(second.id)
                model = context_two.for_model()

                self.assertEqual(
                    "INTJ",
                    model["life"]["cognitive_configuration"]["id"],
                )
                self.assertEqual(
                    "software_engineering",
                    model["profession"]["active_resources"][0]["id"],
                )
                self.assertEqual(
                    admission.assignment_id,
                    model["work"]["assignment"]["id"],
                )
                self.assertEqual(
                    second.id,
                    model["work"]["assignment"]["agent_process_id"],
                )
                self.assertEqual(
                    "audit",
                    model["work"]["workflow"]["current_state"]["id"],
                )
                self.assertIn(
                    experience.id,
                    {
                        item["id"]
                        for item in model["profession"][
                            "professional_memory"
                        ]
                    },
                )
                continued = harness.execute_action(
                    second.id,
                    ActionRequest(
                        "filesystem.read", {"path": "continuity.txt"}
                    ),
                )
                self.assertEqual("success", continued.status)
                self.assertEqual(
                    "created by process one", continued.output
                )
                self.assertIsNot(context_one, context_two)
            finally:
                harness.close()

    def test_new_process_does_not_reuse_old_context_object(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, _, first, _, _ = self._prepare_long_work(workspace)
            try:
                first_context = harness.build_context(first.id)
                first_projection = first_context.model_projection
                harness.terminate_agent(first.id)
                second = harness.create_agent("continue the same work")
                second_context = harness.build_context(second.id)
                self.assertIsNot(first_context, second_context)
                self.assertIsNot(
                    first_projection, second_context.model_projection
                )
                self.assertEqual(
                    set(first_projection),
                    set(second_context.model_projection),
                )
            finally:
                harness.close()

    def test_ephemeral_process_state_is_not_carried_forward(self):
        responses = [
            json.dumps(
                {
                    "type": "action",
                    "capability_id": "echo",
                    "arguments": {"text": "turn one"},
                }
            ),
            json.dumps({"type": "final", "content": "turn two"}),
        ]
        with tempfile.TemporaryDirectory() as workspace:
            harness, supervisor, _, _ = build_harness(
                workspace, responses=responses
            )
            try:
                first = harness.create_agent("first turn")
                harness.step(first.id)
                harness.terminate_agent(first.id)
                second = harness.create_agent("second turn")
                harness.step(second.id)
                second_request = json.loads(
                    supervisor.action_calls[1][1][1]["content"]
                )
                self.assertIsNone(second_request["observation"])
                self.assertNotEqual(
                    supervisor.action_calls[0][0],
                    supervisor.action_calls[1][0],
                )
            finally:
                harness.close()

    def test_runtime_reconstruction_recovers_only_durable_sources(self):
        with tempfile.TemporaryDirectory() as workspace:
            database = Path(workspace, "continuity.sqlite3")
            first_memory = SQLiteMemoryProvider(database)
            first, _, _, process, _, experience = self._prepare_long_work(
                workspace, memory_provider=first_memory
            )
            first.terminate_agent(process.id)
            first.close()

            second_memory = SQLiteMemoryProvider(database)
            second, _, _, _ = build_harness(
                workspace,
                workspace_id="SForge",
                memory_provider=second_memory,
            )
            try:
                process_two = second.create_agent("inspect durable state")
                base = second.runtime_state(process_two.id)
                base_context = second.build_context(process_two.id)
                self.assertEqual("ada", base_context.identity["id"])
                self.assertIsNone(base.cognitive_policy_id)
                self.assertEqual((), base.profession_ids)
                self.assertIsNone(base.assignment_id)

                second.request_binding(
                    process_two.id,
                    ResourceBindingRequest(
                        "profession",
                        "activate",
                        "software_engineering",
                    ),
                )
                second.request_work_assignment(
                    process_two.id,
                    WorkAssignmentRequest(
                        "reviewer",
                        workspace_id="SForge",
                        workflow_id="general_task",
                        requested_capabilities=("filesystem.read",),
                    ),
                )
                reconstructed = second.build_context(
                    process_two.id
                ).for_model()
                self.assertIn(
                    experience.id,
                    {
                        item["id"]
                        for item in reconstructed["profession"][
                            "professional_memory"
                        ]
                    },
                )
                self.assertIn(
                    "workspace.action_result",
                    {
                        item["kind"]
                        for item in reconstructed["work"][
                            "relevant_archive_and_artifacts"
                        ]
                    },
                )
                self.assertEqual(
                    "created by process one",
                    Path(workspace, "continuity.txt").read_text(
                        encoding="utf-8"
                    ),
                )
            finally:
                second.close()


if __name__ == "__main__":
    unittest.main()
