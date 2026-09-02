import json
import tempfile
import unittest
from pathlib import Path

from harness.capability import FunctionCapability
from harness.errors import InvalidAgentStateError, InvalidDecisionError
from harness.models import AgentStatus, CapabilityDescriptor, TaskSpec
from harness.process_supervisor import AgentProcessSupervisor
from tests.support.runtime_factory import build_harness


class RuntimeLoopIntegrationTests(unittest.TestCase):
    def test_direct_mode_write_read_final_answer_e2e(self):
        responses = [
            json.dumps(
                {
                    "type": "assignment",
                    "role_id": "developer",
                    "workflow_id": "general_task",
                    "requested_capabilities": [
                        "filesystem.write",
                        "filesystem.list",
                        "filesystem.read",
                    ],
                }
            ),
            json.dumps(
                {
                    "type": "action",
                    "capability_id": "filesystem.write",
                    "arguments": {
                        "path": "artifacts/hello.txt",
                        "content": "hello",
                    },
                }
            ),
            json.dumps(
                {
                    "type": "action",
                    "capability_id": "filesystem.list",
                    "arguments": {"path": "artifacts"},
                }
            ),
            json.dumps(
                {
                    "type": "action",
                    "capability_id": "filesystem.read",
                    "arguments": {"path": "artifacts/hello.txt"},
                }
            ),
            json.dumps({"type": "final", "content": "hello was written and read"}),
        ]
        with tempfile.TemporaryDirectory() as workspace:
            harness, supervisor, memory, _ = build_harness(workspace, responses)
            try:
                process = harness.create_agent(
                    TaskSpec(
                        "write hello to a test file and read it back",
                        id="e2e-task",
                    )
                )
                answer = harness.run(process.id)
                self.assertEqual("hello was written and read", answer)
                self.assertEqual(
                    "hello",
                    (Path(workspace) / "artifacts" / "hello.txt").read_text(
                        encoding="utf-8"
                    ),
                )
                self.assertEqual(
                    AgentStatus.COMPLETED, harness.process(process.id).status
                )
                self.assertEqual(5, len(supervisor.action_calls))
                second_prompt = supervisor.action_calls[2][1][1]["content"]
                self.assertIn('"status": "success"', second_prompt)
                third_prompt = supervisor.action_calls[3][1][1]["content"]
                self.assertIn("artifacts/hello.txt", third_prompt)
                records = memory.retrieve(scope="task:e2e-task")
                self.assertEqual(
                    3,
                    sum(item.kind == "runtime.action_result" for item in records),
                )
                self.assertEqual("runtime.final_answer", records[-1].kind)
            finally:
                harness.close()

    def test_capability_failure_is_observed_and_agent_can_finish(self):
        failing = FunctionCapability(
            CapabilityDescriptor(
                id="fail",
                description="controlled failure",
                input_schema={"type": "object"},
            ),
            lambda _: (_ for _ in ()).throw(RuntimeError("expected failure")),
        )
        responses = [
            json.dumps(
                {
                    "type": "action",
                    "capability_id": "fail",
                    "arguments": {},
                }
            ),
            json.dumps({"type": "final", "content": "recovered"}),
        ]
        with tempfile.TemporaryDirectory() as workspace:
            harness, supervisor, _, _ = build_harness(
                workspace,
                responses,
                basic_capabilities=frozenset({"fail"}),
                extra_capabilities=(failing,),
            )
            try:
                process = harness.create_agent("recover from failure")
                self.assertEqual("recovered", harness.run(process.id))
                self.assertIn(
                    '"status": "failed"',
                    supervisor.action_calls[1][1][1]["content"],
                )
                self.assertEqual(
                    AgentStatus.COMPLETED, harness.process(process.id).status
                )
            finally:
                harness.close()

    def test_bounded_loop_marks_agent_failed(self):
        response = json.dumps(
            {
                "type": "action",
                "capability_id": "echo",
                "arguments": {"text": "again"},
            }
        )
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, _, _ = build_harness(
                workspace, [response, response], max_steps=2
            )
            try:
                process = harness.create_agent("loop")
                with self.assertRaisesRegex(InvalidAgentStateError, "最大步骤"):
                    harness.run(process.id)
                self.assertEqual(
                    AgentStatus.FAILED, harness.process(process.id).status
                )
            finally:
                harness.close()

    def test_reasoning_failure_marks_agent_failed_and_stops_process(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, supervisor, _, _ = build_harness(
                workspace, [RuntimeError("Connection error")]
            )
            try:
                process = harness.create_agent("network failure")
                with self.assertRaisesRegex(RuntimeError, "Connection error"):
                    harness.run(process.id)
                failed = harness.process(process.id)
                self.assertEqual(AgentStatus.FAILED, failed.status)
                self.assertIsNone(failed.host_process_id)
                self.assertEqual(1, len(supervisor.terminated))
            finally:
                harness.close()

    def test_malformed_structured_decision_fails_without_side_effect(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, supervisor, memory, _ = build_harness(
                workspace, [json.dumps({"type": "unknown"})]
            )
            try:
                process = harness.create_agent("malformed")
                memory_scope = harness.runtime_state(process.id).memory_scope
                with self.assertRaisesRegex(InvalidDecisionError, "Decision type"):
                    harness.run(process.id)
                self.assertEqual(
                    AgentStatus.FAILED, harness.process(process.id).status
                )
                self.assertEqual([], memory.retrieve(scope=memory_scope))
                self.assertEqual(1, len(supervisor.action_calls))
            finally:
                harness.close()

    def test_minimal_novel_workflow_uses_same_agent_abstraction(self):
        responses = [
            json.dumps(
                {
                    "type": "assignment",
                    "role_id": "author",
                    "workflow_id": "novel_writing",
                }
            ),
            json.dumps(
                {
                    "type": "action",
                    "capability_id": "echo",
                    "arguments": {"text": "draft"},
                }
            ),
            json.dumps({"type": "final", "content": "novel draft ready"}),
        ]
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, _, _ = build_harness(workspace, responses)
            try:
                process = harness.create_agent(
                    "write a novel", workflow_id="novel_writing"
                )
                self.assertEqual("novel draft ready", harness.run(process.id))
                state = harness.runtime_state(process.id)
                self.assertEqual("workflow", state.mode)
                self.assertEqual("novel_writing", state.workflow_id)
                self.assertEqual("creation", state.workflow_state_id)
                self.assertIsNotNone(state.assignment_id)
            finally:
                harness.close()

    def test_workflow_memory_connects_two_disposable_agents(self):
        responses = [
            json.dumps(
                {
                    "type": "assignment",
                    "role_id": "author",
                    "workflow_id": "novel_writing",
                }
            ),
            json.dumps(
                {
                    "type": "action",
                    "capability_id": "filesystem.write",
                    "arguments": {
                        "path": "novel/seed.txt",
                        "content": "a city that forgets its name",
                    },
                }
            ),
            json.dumps({"type": "final", "content": "seed saved"}),
            json.dumps({"type": "final", "content": "memory continued"}),
        ]
        with tempfile.TemporaryDirectory() as workspace:
            harness, supervisor, memory, _ = build_harness(
                workspace, responses
            )
            try:
                first = harness.create_agent(
                    "save a story seed", workflow_id="novel_writing"
                )
                self.assertEqual("seed saved", harness.run(first.id))
                finished = harness.process(first.id)
                self.assertEqual(AgentStatus.COMPLETED, finished.status)
                self.assertIsNone(finished.host_process_id)

                second = harness.create_agent(
                    "continue the story", workflow_id="novel_writing"
                )
                self.assertNotEqual(first.id, second.id)
                self.assertEqual("memory continued", harness.run(second.id))

                contract = json.loads(
                    supervisor.action_calls[3][1][0]["content"]
                )
                work = contract["context"]["work"]
                mounted = [
                    *work["relevant_archive_and_artifacts"],
                    *work["recent_observations"],
                ]
                mounted_kinds = [record["kind"] for record in mounted]
                self.assertTrue(
                    {
                        "runtime.final_answer",
                        "workspace.assignment_started",
                    }.issubset(set(mounted_kinds))
                )
                self.assertEqual(
                    1,
                    sum(
                        kind.endswith(".action_result")
                        for kind in mounted_kinds
                    ),
                )
                self.assertNotIn(
                    "workspace.assignment_ended",
                    set(mounted_kinds),
                )
                self.assertNotIn("persona", contract["context"])
                self.assertEqual(
                    3,
                    len(
                        memory.retrieve(
                            scope="workflow:novel_writing:creative_memory"
                        )
                    ),
                )
            finally:
                harness.close()


class AgentProcessSupervisorTests(unittest.TestCase):
    def test_process_is_independent_and_disposable(self):
        supervisor = AgentProcessSupervisor()
        process_id = supervisor.spawn("test-runtime-state")
        try:
            self.assertTrue(supervisor.is_alive(process_id))
            arguments = supervisor._processes[process_id].args
            self.assertIn("-X", arguments)
            self.assertIn("utf8", arguments)
        finally:
            supervisor.terminate(process_id)
        self.assertFalse(supervisor.is_alive(process_id))


if __name__ == "__main__":
    unittest.main()
