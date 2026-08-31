import inspect
import json
import tempfile
import unittest

from agent.agent import Agent
from harness.context_manager import ContextManager
from harness.models import MemoryRecord, ResourceBindingRequest, WorkAssignmentRequest
from harness.runtime_engine import RuntimeEngine
from tests.support.runtime_factory import build_harness


class ThinCoreContextProjectionTests(unittest.TestCase):
    def test_agent_holds_process_references_but_no_runtime_truth(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, _, _ = build_harness(workspace)
            try:
                process = harness.create_agent("inspect")
                agent = harness._runtime._agents.agent(process.id)
                self.assertIsInstance(agent, Agent)
                self.assertEqual(
                    {
                        "_process",
                        "_identity",
                        "_persona",
                        "_reasoning_process",
                    },
                    set(vars(agent)),
                )
                for forbidden in (
                    "runtime_state",
                    "memory",
                    "profession",
                    "cognitive_policy",
                    "workflow",
                    "workspace",
                    "grants",
                ):
                    self.assertNotIn(forbidden, vars(agent))
            finally:
                harness.close()

    def test_model_context_is_life_profession_work_plus_envelope(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, memory, _ = build_harness(
                workspace, workspace_id="projection_project"
            )
            memory.write(
                MemoryRecord(
                    id="life-core",
                    scope="core",
                    kind="core.fact",
                    content="stable life fact",
                )
            )
            memory.write(
                MemoryRecord(
                    id="profession-memory",
                    scope="identity:ada",
                    kind="identity.work_experience",
                    content="trace interface contracts",
                    metadata={
                        "professional_tags": ["software_engineering"]
                    },
                )
            )
            memory.write(
                MemoryRecord(
                    id="workspace-archive",
                    scope="workspace:projection_project",
                    kind="workspace.test_result",
                    content="integration evidence",
                )
            )
            try:
                process = harness.create_agent("review the project")
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
                        "reviewer",
                        workspace_id="projection_project",
                        workflow_id="general_task",
                        requested_capabilities=("filesystem.read",),
                    ),
                )

                operational = harness.build_context(process.id)
                projected = operational.for_model()
                self.assertEqual(
                    {"runtime_envelope", "life", "profession", "work"},
                    set(projected),
                )
                self.assertEqual(
                    "INTJ",
                    projected["life"]["cognitive_configuration"]["id"],
                )
                self.assertTrue(
                    projected["life"]["cognitive_configuration"]
                    ["orientation"]["reasoning_guidance"]
                )
                self.assertIn(
                    "life-core",
                    {
                        item["id"]
                        for item in projected["life"]["core_memory"]
                    },
                )
                self.assertEqual(
                    ["software_engineering"],
                    [
                        item["id"]
                        for item in projected["profession"][
                            "active_resources"
                        ]
                    ],
                )
                self.assertIn(
                    "profession-memory",
                    {
                        item["id"]
                        for item in projected["profession"][
                            "professional_memory"
                        ]
                    },
                )
                self.assertEqual(
                    "projection_project",
                    projected["work"]["workspace"]["id"],
                )
                self.assertEqual(
                    "reviewer", projected["work"]["role"]["id"]
                )
                self.assertEqual(
                    "reviewer",
                    projected["work"]["assignment"]["role_id"],
                )
                self.assertEqual(
                    "general_task", projected["work"]["workflow"]["id"]
                )
                self.assertIn(
                    "workspace-archive",
                    {
                        item["id"]
                        for item in projected["work"][
                            "relevant_archive_and_artifacts"
                        ]
                    },
                )
                self.assertTrue(projected["work"]["local_skills"])
                self.assertEqual(
                    {"echo", "filesystem.read"},
                    {
                        item["id"]
                        for item in projected["work"][
                            "capability_boundary"
                        ]["available"]
                    },
                )
                visible = json.dumps(projected, ensure_ascii=False).casefold()
                self.assertNotIn("communication_style", visible)
                self.assertNotIn("test persona", visible)

                harness.end_work_assignment(process.id)
                ended = harness.build_context(process.id).for_model()
                self.assertIsNone(ended["work"]["assignment"])
                self.assertIsNone(ended["work"]["workspace"])
                self.assertIsNone(ended["work"]["role"])
                self.assertEqual([], ended["work"]["local_skills"])
                self.assertEqual(
                    "INTJ", ended["life"]["cognitive_configuration"]["id"]
                )
                self.assertEqual(
                    ["software_engineering"],
                    [
                        item["id"]
                        for item in ended["profession"]["active_resources"]
                    ],
                )
            finally:
                harness.close()

    def test_core_has_no_experiment_treatments_or_resource_instance_ids(self):
        context_source = inspect.getsource(ContextManager)
        for treatment in (
            '"neutral"',
            '"order_only"',
            '"explicit_rank"',
            '"reasoning_only"',
            '"full"',
            "_experiment_projection",
        ):
            self.assertNotIn(treatment, context_source)

        core_source = inspect.getsource(RuntimeEngine) + context_source
        for resource_id in (
            "INTJ",
            "ENFP",
            "software_engineering",
            "scientific_research",
            "general_task",
            "novel_writing",
            "reviewer",
            "developer",
        ):
            self.assertNotIn(resource_id, core_source)


if __name__ == "__main__":
    unittest.main()
