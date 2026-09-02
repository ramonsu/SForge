import inspect
import tempfile
import unittest
from dataclasses import fields
from datetime import datetime, timezone

from harness.core import Harness
from harness.cognitive_policy import load_cognitive_policies
from harness.memory_manager import MemoryRecord
from harness.models import ResourceBindingRequest, WorkAssignmentRequest
from harness.models import RuntimeState
from harness.profession import load_professions
from harness.skill import load_skills
from tests.support.runtime_factory import build_harness


class V16ResourceModelTests(unittest.TestCase):
    def test_policy_profession_and_skill_catalogs_are_independent(self):
        policies = load_cognitive_policies()
        professions = load_professions()
        skills = load_skills()

        self.assertEqual(16, len(policies.available()))
        self.assertEqual("INTJ", policies.get("intj").id)
        self.assertIn(
            "evidence_review",
            professions.get("software_engineering").preferred_skills,
        )
        self.assertIn(
            "evidence_review",
            professions.get("scientific_research").preferred_skills,
        )
        self.assertEqual("evidence_review", skills.get("evidence_review").id)

    def test_thin_harness_contains_no_semantic_resource_branches(self):
        source = inspect.getsource(Harness)
        for semantic_id in (
            "INTJ",
            "ENFP",
            "software_engineering",
            "scientific_research",
            "developer",
            "reviewer",
        ):
            self.assertNotIn(semantic_id, source)

    def test_work_role_is_not_an_independent_runtime_state_field(self):
        runtime_fields = {item.name for item in fields(RuntimeState)}
        self.assertNotIn("work_role_id", runtime_fields)
        self.assertNotIn("active_workspace_id", runtime_fields)
        self.assertIn("assignment_id", runtime_fields)


class V16BindingAndAuthorityTests(unittest.TestCase):
    def test_startup_has_core_only_and_workspace_metadata_only(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, _, _ = build_harness(
                workspace, workspace_id="project_alpha"
            )
            try:
                process = harness.create_agent("inspect")
                state = harness.runtime_state(process.id)
                context = harness.build_context(process.id)

                self.assertEqual(("core",), state.memory_scopes)
                self.assertEqual({"echo"}, set(state.allowed_capabilities))
                self.assertIsNone(state.assignment_id)
                self.assertIsNone(context.workspace)
                self.assertIsNone(context.work_role)
                self.assertEqual(
                    "project_alpha", context.system["workspace_catalog"][0]["id"]
                )
                self.assertEqual(
                    "project_alpha", harness.available_workspaces()[0]["id"]
                )
                self.assertFalse(
                    any(
                        "workspace:project_alpha" in item["sources"]
                        for item in context.skills
                    )
                )
            finally:
                harness.close()

    def test_policy_and_profession_bind_in_either_order_without_authority(self):
        with tempfile.TemporaryDirectory() as workspace:
            first, _, _, _ = build_harness(workspace)
            second, _, _, _ = build_harness(workspace)
            try:
                first_process = first.create_agent("first")
                second_process = second.create_agent("second")
                first_identity = first.runtime_state(first_process.id).identity_id
                second_identity = second.runtime_state(second_process.id).identity_id

                first.request_binding(
                    first_process.id,
                    ResourceBindingRequest(
                        "cognitive_policy", "activate", "INTJ"
                    ),
                )
                first.request_binding(
                    first_process.id,
                    ResourceBindingRequest(
                        "profession", "activate", "software_engineering"
                    ),
                )
                second.request_binding(
                    second_process.id,
                    ResourceBindingRequest(
                        "profession", "activate", "software_engineering"
                    ),
                )
                second.request_binding(
                    second_process.id,
                    ResourceBindingRequest(
                        "cognitive_policy", "activate", "ENFP"
                    ),
                )

                first_state = first.runtime_state(first_process.id)
                second_state = second.runtime_state(second_process.id)
                self.assertEqual("INTJ", first_state.cognitive_policy_id)
                self.assertEqual("ENFP", second_state.cognitive_policy_id)
                self.assertEqual(
                    ("software_engineering",), first_state.profession_ids
                )
                self.assertEqual(
                    ("software_engineering",), second_state.profession_ids
                )
                self.assertEqual({"echo"}, set(first_state.allowed_capabilities))
                self.assertEqual({"echo"}, set(second_state.allowed_capabilities))
                self.assertEqual(first_identity, first_state.identity_id)
                self.assertEqual(second_identity, second_state.identity_id)
                self.assertIn("identity:ada", first_state.memory_scopes)
                self.assertFalse(
                    any(
                        scope.startswith("workspace:")
                        for scope in first_state.memory_scopes
                    )
                )
            finally:
                first.close()
                second.close()

    def test_assignment_is_the_only_workspace_role_and_temporary_grant_source(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, memory, _ = build_harness(
                workspace, workspace_id="project_alpha"
            )
            memory.write(
                MemoryRecord(
                    scope="workspace:project_alpha",
                    kind="workspace.test_result",
                    content="42 tests passed",
                )
            )
            try:
                process = harness.create_agent("review project")
                before = harness.build_context(process.id)
                self.assertIsNone(before.workspace)
                self.assertNotIn(
                    "42 tests passed", [item.content for item in before.memory]
                )

                admitted = harness.request_work_assignment(
                    process.id,
                    WorkAssignmentRequest(
                        "reviewer",
                        workspace_id="project_alpha",
                        workflow_id="general_task",
                        requested_capabilities=("filesystem.read",),
                    ),
                )
                self.assertEqual("success", admitted.status)
                self.assertEqual(
                    {"filesystem.read"}, set(admitted.grants)
                )
                state = harness.runtime_state(process.id)
                context = harness.build_context(process.id)
                self.assertEqual(
                    {"echo", "filesystem.read"},
                    set(state.allowed_capabilities),
                )
                self.assertEqual("project_alpha", context.workspace["id"])
                self.assertEqual("reviewer", context.work_role["id"])
                self.assertEqual(
                    "reviewer", context.work_assignment["role_id"]
                )
                self.assertIn(
                    "42 tests passed", [item.content for item in context.memory]
                )
                self.assertTrue(
                    any(
                        "workspace:project_alpha" in item["sources"]
                        for item in context.skills
                    )
                )

                harness.end_work_assignment(process.id)
                ended_state = harness.runtime_state(process.id)
                ended_context = harness.build_context(process.id)
                self.assertEqual({"echo"}, set(ended_state.allowed_capabilities))
                self.assertIsNone(ended_state.assignment_id)
                self.assertNotIn(
                    "workspace:project_alpha", ended_state.memory_scopes
                )
                self.assertIsNone(ended_context.workspace)
                self.assertIsNone(ended_context.work_role)
                self.assertFalse(
                    any(
                        "workspace:project_alpha" in item["sources"]
                        for item in ended_context.skills
                    )
                )
            finally:
                harness.close()

    def test_assignment_end_preserves_policy_and_profession_lifetimes(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, _, _ = build_harness(workspace)
            try:
                process = harness.create_agent("implement")
                harness.request_binding(
                    process.id,
                    ResourceBindingRequest(
                        "cognitive_policy", "activate", "INTJ"
                    ),
                )
                harness.request_binding(
                    process.id,
                    ResourceBindingRequest(
                        "profession", "activate", "software_engineering"
                    ),
                )
                harness.request_work_assignment(
                    process.id,
                    WorkAssignmentRequest(
                        "developer", workflow_id="general_task"
                    ),
                )
                harness.end_work_assignment(process.id)
                state = harness.runtime_state(process.id)

                self.assertEqual("INTJ", state.cognitive_policy_id)
                self.assertEqual(
                    ("software_engineering",), state.profession_ids
                )
                self.assertEqual(
                    ("core", "identity:ada"), state.memory_scopes
                )
                self.assertEqual({"echo"}, set(state.allowed_capabilities))
            finally:
                harness.close()

    def test_resource_binding_noop_and_request_replay_do_not_mutate_state(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, _, _ = build_harness(workspace)
            try:
                process = harness.create_agent("bind once")
                request = ResourceBindingRequest(
                    "profession",
                    "activate",
                    "software_engineering",
                    request_id="binding-1",
                )
                first = harness.request_binding(process.id, request)
                first_state = harness.runtime_state(process.id)
                replay = harness.request_binding(process.id, request)
                replay_state = harness.runtime_state(process.id)
                noop = harness.request_binding(
                    process.id,
                    ResourceBindingRequest(
                        "profession",
                        "activate",
                        "software_engineering",
                        request_id="binding-2",
                    ),
                )
                noop_state = harness.runtime_state(process.id)
                collision = harness.request_binding(
                    process.id,
                    ResourceBindingRequest(
                        "cognitive_policy",
                        "activate",
                        "INTJ",
                        request_id="binding-1",
                    ),
                )

                self.assertTrue(first.changed)
                self.assertFalse(first.replayed)
                self.assertFalse(replay.changed)
                self.assertTrue(replay.replayed)
                self.assertFalse(noop.changed)
                self.assertFalse(noop.replayed)
                self.assertEqual("rejected", collision.status)
                self.assertTrue(collision.replayed)
                self.assertEqual(first_state, replay_state)
                self.assertEqual(first_state, noop_state)
                self.assertEqual(
                    ("software_engineering",),
                    noop_state.profession_ids,
                )
                self.assertIsNone(noop_state.cognitive_policy_id)
                binding_events = [
                    event
                    for event in harness.recent_events(
                        20, agent_id=process.id
                    )
                    if event.type.value == "resource_binding_completed"
                ]
                self.assertFalse(binding_events[-2].data["changed"])
                self.assertFalse(binding_events[-2].data["replayed"])
                self.assertFalse(binding_events[-1].data["changed"])
                self.assertTrue(binding_events[-1].data["replayed"])
            finally:
                harness.close()

    def test_workspace_has_no_independent_binding_path(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, _, _ = build_harness(workspace)
            try:
                process = harness.create_agent("discover")
                before = harness.runtime_state(process.id)
                result = harness.request_binding(
                    process.id,
                    ResourceBindingRequest(
                        "workspace", "activate", "forbidden_workspace"
                    ),
                )
                self.assertEqual("rejected", result.status)
                self.assertEqual(before, harness.runtime_state(process.id))
                self.assertIsNone(harness.build_context(process.id).workspace)
            finally:
                harness.close()


class V16MemoryCompositionTests(unittest.TestCase):
    def test_profession_filters_identity_memory_and_scope_beats_policy(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, memory, _ = build_harness(workspace)
            memory.write(
                MemoryRecord(
                    scope="identity:ada",
                    kind="identity.work_experience",
                    content="trace downstream callers",
                    metadata={"professional_tags": ["software_engineering"]},
                )
            )
            memory.write(
                MemoryRecord(
                    scope="identity:ada",
                    kind="identity.work_experience",
                    content="preserve character motivation",
                    metadata={"professional_tags": ["writing"]},
                )
            )
            memory.write(
                MemoryRecord(
                    scope="private:outside",
                    kind="private.secret",
                    content="unauthorized secret",
                )
            )
            try:
                process = harness.create_agent("debug")
                harness.request_binding(
                    process.id,
                    ResourceBindingRequest(
                        "cognitive_policy", "activate", "ESTJ"
                    ),
                )
                before = harness.build_context(process.id)
                self.assertNotIn(
                    "trace downstream callers",
                    [item.content for item in before.memory],
                )
                harness.request_binding(
                    process.id,
                    ResourceBindingRequest(
                        "profession", "activate", "software_engineering"
                    ),
                )
                after = harness.build_context(process.id)
                contents = [item.content for item in after.memory]
                self.assertIn("trace downstream callers", contents)
                self.assertNotIn("preserve character motivation", contents)
                self.assertNotIn("unauthorized secret", contents)
            finally:
                harness.close()

    def test_policies_rank_the_same_legal_candidates_differently(self):
        timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as workspace:
            first, _, memory, _ = build_harness(workspace)
            second, _, _, _ = build_harness(
                workspace, memory_provider=memory
            )
            risk = MemoryRecord(
                scope="core",
                kind="runtime.risk",
                content="contradiction risk",
                created_at=timestamp,
            )
            novel = MemoryRecord(
                scope="core",
                kind="runtime.idea",
                content="novel new idea",
                created_at=timestamp,
            )
            memory.write(risk)
            memory.write(novel)
            try:
                first_process = first.create_agent("choose")
                second_process = second.create_agent("choose")
                first.request_binding(
                    first_process.id,
                    ResourceBindingRequest(
                        "cognitive_policy", "activate", "ESTJ"
                    ),
                )
                second.request_binding(
                    second_process.id,
                    ResourceBindingRequest(
                        "cognitive_policy", "activate", "ENFP"
                    ),
                )
                first_ids = [item.id for item in first.build_context(first_process.id).memory]
                second_ids = [item.id for item in second.build_context(second_process.id).memory]

                self.assertEqual(set(first_ids), set(second_ids))
                self.assertLess(first_ids.index(risk.id), first_ids.index(novel.id))
                self.assertLess(second_ids.index(novel.id), second_ids.index(risk.id))
            finally:
                first.close()
                second.close()


if __name__ == "__main__":
    unittest.main()
