import json
import tempfile
import unittest
from pathlib import Path

from capabilities import builtins
from harness.agent_manager import AgentManager
from harness.capability import (
    CapabilityRegistry,
    DefaultAdmissionPolicy,
    FunctionCapability,
)
from harness.context_manager import ContextManager
from harness.core import Harness
from harness.errors import AgentNotFoundError, InvalidAgentStateError
from harness.memory_manager import InMemoryMemoryProvider
from harness.models import (
    ActionRequest,
    AgentStatus,
    CapabilityDescriptor,
    TaskSpec,
)
from harness.persona import Persona
from harness.workflow_loader import WorkflowLoader
from harness.workflow_manager import WorkflowRegistry
from tests.fakes import FakeSupervisor


WORKFLOWS = Path(__file__).resolve().parent.parent / "workflows"


def make_persona(name="Ada", persona_id="persona_ada"):
    return Persona(
        persona_id=persona_id,
        version="1.0",
        name=name,
        description="test persona",
        traits=("careful",),
        communication_style="structured",
    )


def build_harness(
    workspace,
    responses=None,
    *,
    persona=None,
    direct_capabilities=frozenset({"echo", "read_text", "write_text"}),
    extra_capabilities=(),
    presentation_responses=None,
    max_steps=8,
):
    memory = InMemoryMemoryProvider()
    registry = CapabilityRegistry()
    for capability in (*builtins(workspace), *extra_capabilities):
        registry.register(capability)
    workflows = WorkflowRegistry(WorkflowLoader(WORKFLOWS))
    supervisor = FakeSupervisor(
        responses, presentation_responses=presentation_responses
    )
    harness = Harness(
        AgentManager(supervisor),
        ContextManager(memory, workflows, registry),
        memory,
        registry,
        workflows,
        DefaultAdmissionPolicy(),
        persona or make_persona(),
        direct_capabilities=direct_capabilities,
        max_steps=max_steps,
    )
    return harness, supervisor, memory, registry


class HarnessLifecycleTests(unittest.TestCase):
    def test_create_exposes_process_and_runtime_state_then_terminates(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, supervisor, _, _ = build_harness(workspace)
            try:
                process = harness.create_agent(TaskSpec("hello", id="task-1"))
                state = harness.runtime_state(process.id)
                self.assertEqual(AgentStatus.RUNNING, process.status)
                self.assertEqual(process.id, state.agent_id)
                self.assertEqual("direct", state.mode)
                self.assertEqual("task:task-1", state.memory_scope)
                self.assertEqual(
                    {"echo", "read_text", "write_text"},
                    set(state.allowed_capabilities),
                )
                terminated = harness.terminate_agent(process.id)
                self.assertEqual(AgentStatus.TERMINATED, terminated.status)
                self.assertEqual(1, len(supervisor.terminated))
                with self.assertRaises(InvalidAgentStateError):
                    harness.execute_action(
                        process.id, ActionRequest("echo", {"text": "x"})
                    )
            finally:
                harness.close()

    def test_unknown_agent_and_illegal_transition_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, _, _ = build_harness(workspace)
            try:
                with self.assertRaises(AgentNotFoundError):
                    harness.process("missing")
                process = harness.create_agent("hello")
                with self.assertRaises(InvalidAgentStateError):
                    harness._agents.mark_running(process.id)
            finally:
                harness.close()

    def test_agent_process_owns_no_privileged_runtime_resources(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, _, _ = build_harness(workspace)
            try:
                process = harness.create_agent("hello")
                agent = harness._agents.agent(process.id)
                for forbidden in (
                    "tool_manager",
                    "capability_registry",
                    "memory_manager",
                    "workflow_registry",
                    "child_agents",
                ):
                    self.assertFalse(hasattr(agent, forbidden))
            finally:
                harness.close()


class HarnessActionBoundaryTests(unittest.TestCase):
    def test_allowed_unknown_disallowed_and_invalid_actions_are_structured(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, memory, _ = build_harness(
                workspace, direct_capabilities=frozenset({"echo"})
            )
            try:
                process = harness.create_agent("test actions")
                success = harness.execute_action(
                    process.id, ActionRequest("echo", {"text": "hello"})
                )
                unknown = harness.execute_action(
                    process.id, ActionRequest("missing", {})
                )
                disallowed = harness.execute_action(
                    process.id, ActionRequest("read_text", {"path": "x.txt"})
                )
                invalid = harness.execute_action(
                    process.id, ActionRequest("echo", {})
                )
                self.assertEqual(("success", "hello"), (success.status, success.output))
                self.assertEqual("rejected", unknown.status)
                self.assertEqual("resolve", unknown.metadata["stage"])
                self.assertEqual("rejected", disallowed.status)
                self.assertEqual("admission", disallowed.metadata["stage"])
                self.assertEqual("rejected", invalid.status)
                self.assertEqual("validation", invalid.metadata["stage"])
                records = memory.retrieve(
                    scope=harness.runtime_state(process.id).memory_scope
                )
                self.assertEqual(4, len(records))
            finally:
                harness.close()

    def test_capability_exception_becomes_failed_action_result(self):
        failing = FunctionCapability(
            CapabilityDescriptor(
                id="fail",
                description="always fail",
                input_schema={"type": "object"},
            ),
            lambda _: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, _, _ = build_harness(
                workspace,
                direct_capabilities=frozenset({"fail"}),
                extra_capabilities=(failing,),
            )
            try:
                process = harness.create_agent("fail safely")
                result = harness.execute_action(
                    process.id, ActionRequest("fail", {})
                )
                self.assertEqual("failed", result.status)
                self.assertIn("boom", result.error)
                self.assertEqual(AgentStatus.RUNNING, harness.process(process.id).status)
            finally:
                harness.close()

    def test_structured_operational_context_lists_only_visible_capabilities(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, _, _ = build_harness(
                workspace, direct_capabilities=frozenset({"echo"})
            )
            try:
                process = harness.create_agent(
                    TaskSpec("inspect", context={"language": "zh"})
                )
                context = harness.build_context(process.id)
                self.assertEqual("SForge V1", context.system["runtime"])
                self.assertEqual("zh", context.task["context"]["language"])
                self.assertEqual(
                    ["echo"], [item.id for item in context.capabilities]
                )
                self.assertNotIn("persona", context.as_dict())
            finally:
                harness.close()


if __name__ == "__main__":
    unittest.main()
