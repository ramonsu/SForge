import json
import tempfile
import unittest
from pathlib import Path

from harness.errors import InvalidWorkflowStateError, WorkflowNotFoundError
from harness.models import TaskSpec
from harness.workflow_loader import WorkflowLoader
from harness.workflow_manager import WorkflowRegistry
from tests.test_harness import build_harness


def workflow_definition():
    return {
        "id": "demo",
        "initial_state": "start",
        "states": {
            "start": {
                "allowed_capabilities": ["echo"],
                "memory_scope": "task",
                "context_sources": [],
            },
            "future": {
                "allowed_capabilities": ["read_text"],
                "memory_scope": "workflow",
                "context_sources": ["repo"],
            },
        },
    }


class WorkflowRegistryTests(unittest.TestCase):
    def _loader(self, definition):
        temporary = tempfile.TemporaryDirectory()
        directory = Path(temporary.name) / "demo"
        directory.mkdir()
        (directory / "workflow.json").write_text(
            json.dumps(definition), encoding="utf-8"
        )
        (directory / "workflow.md").write_text(
            "# Demo instruction", encoding="utf-8"
        )
        self.addCleanup(temporary.cleanup)
        return WorkflowLoader(temporary.name)

    def test_loads_thin_definition_without_dag_execution(self):
        definition = self._loader(workflow_definition()).load("demo")
        self.assertEqual("demo", definition.id)
        self.assertEqual("start", definition.initial_state)
        self.assertEqual(
            frozenset({"echo"}),
            definition.states["start"].allowed_capabilities,
        )
        self.assertFalse(hasattr(definition.states["start"], "next_states"))

    def test_registry_resolves_and_caches_definition(self):
        registry = WorkflowRegistry(self._loader(workflow_definition()))
        self.assertIs(registry.get("demo"), registry.get("demo"))

    def test_unknown_and_invalid_workflows_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(WorkflowNotFoundError):
                WorkflowLoader(temporary).load("../outside")
        raw = workflow_definition()
        raw["initial_state"] = "missing"
        with self.assertRaises(InvalidWorkflowStateError):
            self._loader(raw).load("demo")

    def test_workflow_initial_state_changes_runtime_visibility(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, _, _ = build_harness(
                workspace, direct_capabilities=frozenset({"echo"})
            )
            try:
                direct = harness.create_agent("direct")
                workflow = harness.create_agent(
                    TaskSpec("workflow", id="workflow-task"),
                    workflow_id="general_task",
                )
                direct_ids = {
                    item.id for item in harness.build_context(direct.id).capabilities
                }
                workflow_context = harness.build_context(workflow.id)
                workflow_ids = {item.id for item in workflow_context.capabilities}
                self.assertEqual({"echo"}, direct_ids)
                self.assertEqual(
                    {"echo", "read_text", "write_text"}, workflow_ids
                )
                self.assertEqual("general_task", workflow_context.workflow["id"])
                self.assertEqual(
                    "task:workflow-task",
                    harness.runtime_state(workflow.id).memory_scope,
                )
                self.assertFalse(hasattr(harness, "transition_workflow"))
            finally:
                harness.close()


if __name__ == "__main__":
    unittest.main()
