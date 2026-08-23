import unittest

from harness.models import (
    ActionRequest,
    AgentProcess,
    CapabilityDescriptor,
    OperationalContext,
    RuntimeState,
)


class RuntimeModelTests(unittest.TestCase):
    def test_process_identity_and_runtime_state_are_separate_objects(self):
        process = AgentProcess(runtime_state_id="state-1", id="agent-1")
        state = RuntimeState(
            id="state-1",
            agent_id="agent-1",
            task_id="task-1",
            mode="direct",
            allowed_capabilities=frozenset({"echo"}),
            memory_scope="task:task-1",
        )
        self.assertEqual(state.id, process.runtime_state_id)
        self.assertFalse(hasattr(process, "persona"))
        self.assertFalse(hasattr(state, "persona"))

    def test_action_request_has_stable_request_id(self):
        request = ActionRequest("echo", {"text": "hello"})
        self.assertTrue(request.request_id)
        self.assertEqual(request.request_id, request.request_id)

    def test_operational_context_is_structured_and_serializable(self):
        context = OperationalContext(
            system={"runtime": "SForge V1"},
            task={"request": "hello"},
            runtime={"mode": "direct"},
            workflow=None,
            memory=(),
            capabilities=(
                CapabilityDescriptor(
                    id="echo",
                    description="echo text",
                    input_schema={"type": "object"},
                ),
            ),
        )
        value = context.as_dict()
        self.assertEqual("hello", value["task"]["request"])
        self.assertEqual("echo", value["capabilities"][0]["id"])


if __name__ == "__main__":
    unittest.main()
