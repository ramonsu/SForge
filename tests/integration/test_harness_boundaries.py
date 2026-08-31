import json
import tempfile
import unittest

from harness.capability import FunctionCapability
from harness.errors import AgentNotFoundError, InvalidAgentStateError
from harness.models import (
    ActionRequest,
    AgentStatus,
    CapabilityDescriptor,
    TaskSpec,
)
from tests.support.runtime_factory import build_harness


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
                self.assertEqual("core", state.memory_scope)
                self.assertEqual(("core",), state.memory_scopes)
                self.assertEqual({"echo"}, set(state.allowed_capabilities))
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
                harness.terminate_agent(process.id)
                with self.assertRaises(InvalidAgentStateError):
                    harness.step(process.id)
            finally:
                harness.close()

    def test_agent_process_owns_no_privileged_runtime_resources(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, _, _ = build_harness(workspace)
            try:
                process = harness.create_agent("hello")
                for forbidden in (
                    "tool_manager",
                    "capability_registry",
                    "memory_manager",
                    "workflow_registry",
                    "child_agents",
                ):
                    self.assertFalse(hasattr(process, forbidden))
                for hidden_mechanism in (
                    "_agents",
                    "_contexts",
                    "_memory",
                    "_capabilities",
                    "_workflows",
                    "_admission",
                    "_events",
                ):
                    self.assertFalse(hasattr(harness, hidden_mechanism))
            finally:
                harness.close()


class HarnessActionBoundaryTests(unittest.TestCase):
    def test_empty_capability_state_exposes_and_executes_nothing(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, _, _ = build_harness(
                workspace, basic_capabilities=frozenset()
            )
            try:
                process = harness.create_agent("no capabilities")
                self.assertEqual((), harness.build_context(process.id).capabilities)
                result = harness.execute_action(
                    process.id, ActionRequest("echo", {"text": "blocked"})
                )
                self.assertEqual("rejected", result.status)
                self.assertEqual("admission", result.metadata["stage"])
            finally:
                harness.close()

    def test_allowed_unknown_disallowed_and_invalid_actions_are_structured(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, memory, _ = build_harness(
                workspace, basic_capabilities=frozenset({"echo"})
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
                    process.id,
                    ActionRequest("filesystem.read", {"path": "x.txt"}),
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
                self.assertEqual(
                    0,
                    len(records),
                    "Bootstrap runtime observations must not pollute Core Memory",
                )
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
                basic_capabilities=frozenset({"fail"}),
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
                workspace, basic_capabilities=frozenset({"echo"})
            )
            try:
                process = harness.create_agent(
                    TaskSpec("inspect", context={"language": "zh"})
                )
                context = harness.build_context(process.id)
                self.assertEqual("SForge V1.6", context.system["runtime"])
                self.assertEqual("zh", context.task["context"]["language"])
                self.assertEqual(
                    ["echo"], [item.id for item in context.capabilities]
                )
                self.assertNotIn("persona", context.as_dict())
                self.assertIn(
                    "novel_writing",
                    {item["id"] for item in context.system["workflow_catalog"]},
                )
            finally:
                harness.close()


if __name__ == "__main__":
    unittest.main()
