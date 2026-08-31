import inspect
import tempfile
import unittest
from dataclasses import replace
from types import MappingProxyType

from harness.core import Harness
from harness.identity import Identity, load_identity
from harness.memory_manager import InMemoryMemoryProvider, MemoryRecord
from harness.models import (
    ActionRequest,
    WorkAssignmentRequest,
    WorkflowRequest,
)
from harness.runtime_engine import RuntimeEngine
from harness.work_role import load_work_roles
from tests.support.runtime_factory import build_harness, make_persona

Personality = object

@unittest.skip("V1.5 Personality/Role 架构已由 V1.6 资源边界取代")
class IdentityAndRoleTests(unittest.TestCase):
    def test_personality_survives_role_rebinding(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, _, _ = build_harness(
                workspace, workspace_id="novel_A"
            )
            try:
                process = harness.create_agent("write then review")
                baseline = harness.runtime_state(process.id)
                author = harness.request_work_assignment(
                    process.id, WorkAssignmentRequest("author")
                )
                editor = harness.request_work_assignment(
                    process.id, WorkAssignmentRequest("editor")
                )
                current = harness.runtime_state(process.id)

                self.assertEqual("success", author.status)
                self.assertEqual("success", editor.status)
                self.assertNotEqual(
                    author.assignment_id, editor.assignment_id
                )
                self.assertEqual(baseline.identity_id, current.identity_id)
                self.assertEqual(
                    baseline.personality_id, current.personality_id
                )
                self.assertEqual("editor", current.work_role_id)
            finally:
                harness.close()

    def test_same_personality_supports_different_professions(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, _, _ = build_harness(workspace)
            try:
                first = harness.create_agent("implement")
                second = harness.create_agent("edit")
                developer = harness.request_work_assignment(
                    first.id, WorkAssignmentRequest("developer")
                )
                editor = harness.request_work_assignment(
                    second.id, WorkAssignmentRequest("editor")
                )
                first_state = harness.runtime_state(first.id)
                second_state = harness.runtime_state(second.id)

                self.assertEqual("success", developer.status)
                self.assertEqual("success", editor.status)
                self.assertEqual(
                    first_state.personality_id,
                    second_state.personality_id,
                )
                self.assertNotEqual(
                    first_state.work_role_id, second_state.work_role_id
                )
            finally:
                harness.close()

    def test_same_role_supports_different_personalities(self):
        persona = make_persona()
        identity_03 = Identity(
            "ada_03",
            Personality(
                "personality_03",
                "1.0",
                MappingProxyType({"creative": 0.9}),
            ),
            persona.reference,
        )
        identity_07 = replace(
            load_identity(), persona_reference=persona.reference
        )
        with tempfile.TemporaryDirectory() as first_root, tempfile.TemporaryDirectory() as second_root:
            first, _, _, _ = build_harness(
                first_root, persona=persona, identity=identity_03
            )
            second, _, _, _ = build_harness(
                second_root, persona=persona, identity=identity_07
            )
            try:
                first_process = first.create_agent("edit one")
                second_process = second.create_agent("edit two")
                first.request_work_assignment(
                    first_process.id, WorkAssignmentRequest("editor")
                )
                second.request_work_assignment(
                    second_process.id, WorkAssignmentRequest("editor")
                )
                self.assertEqual(
                    "editor",
                    first.runtime_state(first_process.id).work_role_id,
                )
                self.assertEqual(
                    "editor",
                    second.runtime_state(second_process.id).work_role_id,
                )
                self.assertNotEqual(
                    first.runtime_state(first_process.id).personality_id,
                    second.runtime_state(second_process.id).personality_id,
                )
            finally:
                first.close()
                second.close()

    def test_role_owns_neither_workspace_nor_workflow(self):
        roles = load_work_roles()
        editor = roles.get("editor")
        self.assertFalse(hasattr(editor, "workspace_id"))
        self.assertFalse(hasattr(editor, "workflow_id"))
        self.assertFalse(hasattr(editor, "grants"))

        with tempfile.TemporaryDirectory() as first_root, tempfile.TemporaryDirectory() as second_root:
            first, _, _, _ = build_harness(
                first_root, workspace_id="novel_A"
            )
            second, _, _, _ = build_harness(
                second_root, workspace_id="documentation_project"
            )
            try:
                first_process = first.create_agent("edit novel")
                second_process = second.create_agent("edit docs")
                novel = first.request_work_assignment(
                    first_process.id,
                    WorkAssignmentRequest(
                        "editor", workflow_id="novel_writing"
                    ),
                )
                docs = second.request_work_assignment(
                    second_process.id, WorkAssignmentRequest("editor")
                )
                self.assertEqual("novel_A", novel.workspace_id)
                self.assertEqual(
                    "documentation_project", docs.workspace_id
                )
                self.assertEqual("novel_writing", novel.workflow_id)
                self.assertIsNone(docs.workflow_id)
            finally:
                first.close()
                second.close()


@unittest.skip("V1.5 grants 表示已由 V1.6 effective grants 派生规则取代")
class WorkAssignmentAuthorityTests(unittest.TestCase):
    def test_same_role_can_receive_different_valid_grants(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, _, _ = build_harness(workspace)
            try:
                process = harness.create_agent("implement in two phases")
                read_only = harness.request_work_assignment(
                    process.id,
                    WorkAssignmentRequest(
                        "developer",
                        workflow_id="general_task",
                        requested_capabilities=("filesystem.read",),
                    ),
                )
                harness.end_work_assignment(process.id)
                read_write = harness.request_work_assignment(
                    process.id,
                    WorkAssignmentRequest(
                        "developer",
                        workflow_id="general_task",
                        requested_capabilities=(
                            "filesystem.read",
                            "filesystem.write",
                        ),
                    ),
                )

                self.assertEqual("developer", read_only.role_id)
                self.assertEqual("developer", read_write.role_id)
                self.assertNotIn("filesystem.write", read_only.grants)
                self.assertIn("filesystem.write", read_write.grants)
            finally:
                harness.close()

    def test_workspace_entry_and_exit_mount_and_revoke_grants(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, _, _ = build_harness(workspace)
            try:
                process = harness.create_agent("read project")
                before = harness.execute_action(
                    process.id,
                    ActionRequest("filesystem.list", {"path": "."}),
                )
                admitted = harness.request_work_assignment(
                    process.id,
                    WorkAssignmentRequest(
                        "developer", workflow_id="general_task"
                    ),
                )
                during = harness.execute_action(
                    process.id,
                    ActionRequest("filesystem.list", {"path": "."}),
                )
                harness.end_work_assignment(process.id)
                after = harness.execute_action(
                    process.id,
                    ActionRequest("filesystem.list", {"path": "."}),
                )

                self.assertEqual("rejected", before.status)
                self.assertEqual("success", admitted.status)
                self.assertEqual("success", during.status)
                self.assertEqual("rejected", after.status)
                state = harness.runtime_state(process.id)
                self.assertIsNone(state.assignment_id)
                self.assertEqual({"echo"}, set(state.allowed_capabilities))
            finally:
                harness.close()

    def test_workflow_transition_cannot_exceed_assignment_grants(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, _, _ = build_harness(workspace)
            try:
                process = harness.create_agent("review without write")
                admitted = harness.request_work_assignment(
                    process.id,
                    WorkAssignmentRequest(
                        "editor",
                        workflow_id="novel_writing",
                        requested_capabilities=("filesystem.read",),
                    ),
                )
                transition = harness.request_workflow(
                    process.id,
                    WorkflowRequest(
                        "novel_writing",
                        target_state_id="revision",
                        transition_condition="draft_completed",
                    ),
                )
                self.assertEqual("success", admitted.status)
                self.assertEqual("success", transition.status)
                self.assertNotIn(
                    "filesystem.write", transition.allowed_capabilities
                )
                self.assertEqual(
                    {"echo", "filesystem.read"},
                    set(transition.allowed_capabilities),
                )
            finally:
                harness.close()

    def test_unknown_workspace_or_escalated_grant_is_atomic(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, _, _ = build_harness(
                workspace, workspace_id="workspace_A"
            )
            try:
                process = harness.create_agent("stay bounded")
                baseline = harness.runtime_state(process.id)
                wrong_workspace = harness.request_work_assignment(
                    process.id,
                    WorkAssignmentRequest(
                        "developer", workspace_id="workspace_B"
                    ),
                )
                escalation = harness.request_work_assignment(
                    process.id,
                    WorkAssignmentRequest(
                        "developer",
                        requested_capabilities=("filesystem.write",),
                    ),
                )
                self.assertEqual("rejected", wrong_workspace.status)
                self.assertEqual("rejected", escalation.status)
                self.assertEqual(baseline, harness.runtime_state(process.id))
            finally:
                harness.close()


@unittest.skip("V1.5 记忆挂载测试已迁移到 V1.6 组合测试")
class MemoryOwnershipTests(unittest.TestCase):
    def test_workspace_archive_survives_process_and_stays_project_local(self):
        memory = InMemoryMemoryProvider()
        with tempfile.TemporaryDirectory() as first_root, tempfile.TemporaryDirectory() as second_root:
            first, _, _, _ = build_harness(
                first_root,
                workspace_id="workspace_A",
                memory_provider=memory,
            )
            process = first.create_agent("project A")
            first.request_work_assignment(
                process.id, WorkAssignmentRequest("developer")
            )
            first.terminate_agent(process.id, "task stopped")
            archive = memory.retrieve(scope="workspace:workspace_A")
            self.assertIn(
                "workspace.assignment_ended",
                {record.kind for record in archive},
            )
            first.close()

            second, _, _, _ = build_harness(
                second_root,
                workspace_id="workspace_B",
                memory_provider=memory,
            )
            try:
                other = second.create_agent("project B")
                second.request_work_assignment(
                    other.id, WorkAssignmentRequest("developer")
                )
                contents = {
                    record.content
                    for record in second.build_context(other.id).memory
                }
                self.assertNotIn("task stopped", contents)
            finally:
                second.close()

    def test_grounded_professional_experience_crosses_workspaces(self):
        memory = InMemoryMemoryProvider()
        with tempfile.TemporaryDirectory() as first_root, tempfile.TemporaryDirectory() as second_root:
            first, _, _, _ = build_harness(
                first_root,
                workspace_id="workspace_A",
                memory_provider=memory,
            )
            process = first.create_agent("learn")
            first.request_work_assignment(
                process.id, WorkAssignmentRequest("developer")
            )
            first.record_work_experience(
                process.id,
                "Check callers before changing a public interface.",
                objective_outcome="42 tests passed",
                professional_tags=("software_engineering",),
                artifact_refs=("test-result:42",),
            )
            first.end_work_assignment(process.id)
            first.terminate_agent(process.id)
            first.close()

            second, _, _, _ = build_harness(
                second_root,
                workspace_id="workspace_B",
                memory_provider=memory,
            )
            try:
                other = second.create_agent("reuse experience")
                second.request_work_assignment(
                    other.id, WorkAssignmentRequest("developer")
                )
                context = second.build_context(other.id)
                self.assertIn(
                    "Check callers before changing a public interface.",
                    {record.content for record in context.memory},
                )
            finally:
                second.close()


@unittest.skip("V1.5 组合边界测试已迁移到 V1.6 组合测试")
class CompositionBoundaryTests(unittest.TestCase):
    def test_operational_context_composes_independent_layers(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, _, _ = build_harness(
                workspace, workspace_id="SForge"
            )
            try:
                process = harness.create_agent("implement refactor")
                harness.request_work_assignment(
                    process.id,
                    WorkAssignmentRequest(
                        "developer", workflow_id="general_task"
                    ),
                )
                context = harness.build_context(process.id)
                self.assertEqual("ada", context.identity["id"])
                self.assertEqual(
                    "personality_07", context.personality["id"]
                )
                self.assertEqual("SForge", context.workspace["id"])
                self.assertEqual("developer", context.work_role["id"])
                self.assertEqual(
                    "general_task", context.workflow["id"]
                )
                self.assertEqual(
                    context.work_assignment["grants"],
                    sorted(
                        item.id for item in context.capabilities
                    ),
                )
                self.assertNotIn("persona", context.as_dict())
            finally:
                harness.close()

    def test_thin_kernel_contains_no_profession_or_domain_branches(self):
        source = inspect.getsource(Harness) + inspect.getsource(RuntimeEngine)
        for semantic in (
            "author",
            "editor",
            "developer",
            "researcher",
            "novel",
            "coding",
        ):
            self.assertNotIn(f'"{semantic}"', source.casefold())


if __name__ == "__main__":
    unittest.main()
