import json
import tempfile
import unittest

from harness.capability import FunctionCapability
from harness.events import EventType
from harness.models import (
    ActionRequest,
    AgentStatus,
    CapabilityDescriptor,
    MemoryRecord,
    TaskSpec,
    WorkAssignmentRequest,
    WorkflowRequest,
)
from tests.support.runtime_factory import build_harness


def event_types(harness, agent_id):
    return [event.type for event in harness.recent_events(100, agent_id=agent_id)]


class RuntimeTraceTests(unittest.TestCase):
    def test_invalid_json_protocol_failure_is_visible_before_runtime_error(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, _, _ = build_harness(
                workspace, ['{"type":"final","content": }']
            )
            try:
                process = harness.create_agent("invalid json")
                with self.assertRaises(Exception):
                    harness.run(process.id)
                events = harness.recent_events(100, agent_id=process.id)
                completed = next(
                    event
                    for event in events
                    if event.type is EventType.REASONING_COMPLETED
                )
                protocol = completed.data["decision_protocol"]
                self.assertFalse(completed.data["protocol_success"])
                self.assertFalse(protocol["structured_decision_valid"])
                self.assertEqual("invalid_json", protocol["decision_parse_mode"])
                self.assertFalse(protocol["fallback_used"])
                self.assertTrue(protocol["decision_parse_error"])
                self.assertFalse(
                    any(
                        event.data.get("phase") == "presentation"
                        for event in events
                        if event.type is EventType.REASONING_STARTED
                    )
                )
            finally:
                harness.close()

    def test_control_character_recovery_completes_but_protocol_stays_failed(self):
        raw = '{"type":"final","content":"line one\nline two"}'
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, _, _ = build_harness(workspace, [raw])
            try:
                process = harness.create_agent("literal newline")
                self.assertEqual("line one\nline two", harness.run(process.id))
                completed = next(
                    event
                    for event in harness.recent_events(100, agent_id=process.id)
                    if event.type is EventType.REASONING_COMPLETED
                    and event.data.get("phase") == "decision"
                )
                protocol = completed.data["decision_protocol"]
                self.assertFalse(completed.data["protocol_success"])
                self.assertTrue(protocol["fallback_used"])
                self.assertEqual(
                    "control_character_repair",
                    protocol["decision_parse_mode"],
                )
                self.assertEqual(
                    AgentStatus.COMPLETED, harness.process(process.id).status
                )
            finally:
                harness.close()

    def test_successful_action_has_correlated_ordered_trace(self):
        responses = [
            json.dumps(
                {
                    "type": "action",
                    "capability_id": "echo",
                    "arguments": {"text": "private argument"},
                    "request_id": "request-1",
                }
            ),
            json.dumps({"type": "final", "content": "private output"}),
        ]
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, _, _ = build_harness(workspace, responses)
            try:
                process = harness.create_agent(
                    TaskSpec("private task request", id="trace-task")
                )
                self.assertEqual("private output", harness.run(process.id))
                events = harness.recent_events(100, agent_id=process.id)
                types = [event.type for event in events]
                requested = types.index(EventType.CAPABILITY_REQUESTED)
                invoked = types.index(EventType.CAPABILITY_COMPLETED)
                action = types.index(EventType.ACTION_COMPLETED)
                self.assertLess(requested, invoked)
                self.assertLess(invoked, action)
                correlated = events[requested : action + 1]
                self.assertTrue(
                    all(
                        event.request_id == "request-1"
                        for event in correlated
                        if event.type
                        in {
                            EventType.CAPABILITY_REQUESTED,
                            EventType.CAPABILITY_COMPLETED,
                            EventType.ACTION_COMPLETED,
                        }
                    )
                )
                self.assertIn(EventType.CONTEXT_BUILT, types)
                self.assertIn(EventType.ACTION_COMPLETED, types)
                self.assertEqual(
                    AgentStatus.COMPLETED, harness.process(process.id).status
                )

                serialized = json.dumps(
                    [event.as_dict() for event in events],
                    ensure_ascii=False,
                )
                for private_value in (
                    "private task request",
                    "private argument",
                    "private output",
                    "Ada",
                ):
                    self.assertNotIn(private_value, serialized)
            finally:
                harness.close()

    def test_rejected_action_does_not_claim_capability_invocation(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, _, _ = build_harness(
                workspace, basic_capabilities=frozenset({"echo"})
            )
            try:
                process = harness.create_agent("reject")
                result = harness.execute_action(
                    process.id,
                    ActionRequest(
                        "filesystem.read", {"path": "secret.txt"}
                    ),
                )
                self.assertEqual("rejected", result.status)
                types = event_types(harness, process.id)
                self.assertIn(EventType.CAPABILITY_REQUESTED, types)
                self.assertIn(EventType.ACTION_COMPLETED, types)
                self.assertNotIn(EventType.CAPABILITY_COMPLETED, types)
            finally:
                harness.close()

    def test_workflow_and_runtime_error_are_observable(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, _, _ = build_harness(
                workspace,
                [
                    json.dumps(
                        {
                            "type": "workflow",
                            "workflow_id": "novel_writing",
                        }
                    ),
                    json.dumps({"type": "unknown"}),
                ],
            )
            try:
                process = harness.create_agent("malformed", "novel_writing")
                with self.assertRaises(Exception):
                    harness.run(process.id)
                events = harness.recent_events(100, agent_id=process.id)
                self.assertIn(
                    EventType.WORKFLOW_ADMISSION_COMPLETED,
                    [event.type for event in events],
                )
                error = next(
                    event for event in events if event.type is EventType.ERROR
                )
                self.assertEqual("reasoning", error.data["stage"])
                self.assertEqual("DecisionProtocolError", error.data["error_type"])
                protocol_event = next(
                    event
                    for event in reversed(events)
                    if event.type is EventType.REASONING_COMPLETED
                    and event.data.get("phase") == "decision"
                )
                self.assertFalse(protocol_event.data["protocol_success"])
                self.assertTrue(
                    protocol_event.data["decision_protocol"][
                        "structured_decision_valid"
                    ]
                )
                self.assertEqual(
                    AgentStatus.FAILED, harness.process(process.id).status
                )
            finally:
                harness.close()

    def test_capability_failure_is_structured_without_runtime_error_event(self):
        failing = FunctionCapability(
            CapabilityDescriptor(
                id="fail",
                description="controlled failure",
                input_schema={"type": "object"},
            ),
            lambda _: (_ for _ in ()).throw(RuntimeError("private failure")),
        )
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, _, _ = build_harness(
                workspace,
                basic_capabilities=frozenset({"fail"}),
                extra_capabilities=(failing,),
            )
            try:
                process = harness.create_agent("failure")
                result = harness.execute_action(
                    process.id, ActionRequest("fail", {})
                )
                self.assertEqual("failed", result.status)
                events = harness.recent_events(100, agent_id=process.id)
                completed = next(
                    event
                    for event in events
                    if event.type is EventType.CAPABILITY_COMPLETED
                )
                self.assertEqual("failed", completed.data["status"])
                self.assertNotIn(
                    EventType.ERROR, [event.type for event in events]
                )
                self.assertNotIn(
                    "private failure",
                    json.dumps(
                        [event.as_dict() for event in events],
                        ensure_ascii=False,
                    ),
                )
            finally:
                harness.close()

class RuntimeInspectorTests(unittest.TestCase):
    def test_inspector_is_read_only_and_uses_runtime_visibility(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, memory, _ = build_harness(
                workspace,
                [json.dumps({"type": "final", "content": "done"})],
                basic_capabilities=frozenset({"echo"}),
            )
            memory.write(
                MemoryRecord(
                    scope="task:inspect-task",
                    kind="runtime.note",
                    content="local inspector memory",
                )
            )
            try:
                process = harness.create_agent(
                    TaskSpec("do not expose this request", id="inspect-task")
                )
                before_events = harness.recent_events(100, agent_id=process.id)
                running = harness.inspect(process.id).as_dict()
                after_events = harness.recent_events(100, agent_id=process.id)
                self.assertEqual(before_events, after_events)
                self.assertEqual("running", running["agent"]["status"])
                self.assertEqual(
                    ["echo"],
                    [
                        item["id"]
                        for item in running["available_capabilities"]
                    ],
                )
                self.assertNotIn(
                    "local inspector memory",
                    [item["content"] for item in running["loaded_memory"]],
                )
                self.assertNotIn(
                    "do not expose this request",
                    json.dumps(running, ensure_ascii=False),
                )
                running["agent"]["status"] = "tampered"
                self.assertEqual(
                    AgentStatus.RUNNING, harness.process(process.id).status
                )

                admission = harness.request_work_assignment(
                    process.id,
                    WorkAssignmentRequest(
                        "generalist", workflow_id="general_task"
                    ),
                )
                self.assertEqual("success", admission.status)
                mounted = harness.inspect(process.id).as_dict()
                self.assertIn(
                    "local inspector memory",
                    [item["content"] for item in mounted["loaded_memory"]],
                )

                harness.run(process.id)
                completed = harness.inspect().as_dict()
                self.assertEqual("completed", completed["agent"]["status"])
                self.assertEqual(process.id, completed["agent"]["id"])
            finally:
                harness.close()

    def test_workflow_snapshot_exposes_ids_not_domain_control(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, _, _ = build_harness(workspace)
            try:
                process = harness.create_agent("novel", "novel_writing")
                harness.request_work_assignment(
                    process.id,
                    WorkAssignmentRequest(
                        "author", workflow_id="novel_writing"
                    ),
                )
                snapshot = harness.inspect(process.id).as_dict()
                self.assertEqual(
                    {"id": "novel_writing", "state_id": "creation"},
                    snapshot["workflow"],
                )
                self.assertEqual(
                    "workflow", snapshot["runtime_state"]["mode"]
                )
            finally:
                harness.close()


if __name__ == "__main__":
    unittest.main()
