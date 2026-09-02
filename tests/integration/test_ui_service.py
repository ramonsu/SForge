import json
import tempfile
import unittest

from harness.events import EventLogger
from harness.models import ReasoningResponse, TokenUsage
from tests.support.runtime_factory import build_harness
from ui import ProgressKind, RunService, RunStatus


class RunServiceTests(unittest.TestCase):
    def test_work_assignment_progress_keeps_active_work_after_process(self):
        responses = [
            ReasoningResponse(
                json.dumps(
                    {
                        "type": "assignment",
                        "role_id": "developer",
                        "workflow_id": "general_task",
                        "requested_capabilities": ["filesystem.read"],
                    }
                ),
                TokenUsage(5, 2),
            ),
            ReasoningResponse(
                json.dumps({"type": "final", "content": "done"}),
                TokenUsage(2, 1),
            ),
        ]
        with tempfile.TemporaryDirectory() as workspace:
            events = EventLogger()
            harness, _, _, _ = build_harness(
                workspace, responses, events=events
            )
            service = RunService(harness, events)
            progress = []
            service.subscribe(progress.append)
            try:
                snapshot = service.wait(service.start("implement"), timeout=2)
                self.assertEqual("developer", snapshot.work_role_id)
                self.assertIsNotNone(snapshot.assignment_id)
                kinds = {event.kind for event in progress}
                self.assertIn(ProgressKind.WORK_ASSIGNMENT_REQUESTED, kinds)
                self.assertIn(
                    ProgressKind.WORK_ASSIGNMENT_ADMISSION_COMPLETED, kinds
                )
                self.assertNotIn(ProgressKind.WORK_ASSIGNMENT_ENDED, kinds)
            finally:
                service.close()

    def test_resource_binding_progress_updates_frontend_snapshot(self):
        responses = [
            ReasoningResponse(
                json.dumps(
                    {
                        "type": "binding",
                        "resource_type": "cognitive_policy",
                        "operation": "activate",
                        "resource_id": "INTJ",
                    }
                ),
                TokenUsage(3, 1),
            ),
            ReasoningResponse(
                json.dumps({"type": "final", "content": "reviewed"}),
                TokenUsage(4, 1),
            ),
        ]
        with tempfile.TemporaryDirectory() as workspace:
            events = EventLogger()
            harness, _, _, _ = build_harness(
                workspace, responses, events=events
            )
            service = RunService(harness, events)
            progress = []
            service.subscribe(progress.append)
            try:
                snapshot = service.wait(
                    service.start("review this"), timeout=2
                )
                self.assertEqual("INTJ", snapshot.cognitive_policy_id)
                kinds = [event.kind for event in progress]
                self.assertIn(
                    ProgressKind.RESOURCE_BINDING_REQUESTED, kinds
                )
                self.assertIn(
                    ProgressKind.RESOURCE_BINDING_COMPLETED,
                    kinds,
                )
                self.assertEqual(
                    16,
                    len(
                        {
                            item["id"]
                            for item in service.available_cognitive_policies()
                        }
                    ),
                )
            finally:
                service.close()

    def test_ui_workflow_selection_is_a_hint_until_runtime_admission(self):
        responses = [
            ReasoningResponse(
                json.dumps(
                    {
                        "type": "assignment",
                        "role_id": "author",
                        "workflow_id": "novel_writing",
                    }
                ),
                TokenUsage(4, 1),
            ),
            ReasoningResponse(
                json.dumps({"type": "final", "content": "draft"}),
                TokenUsage(6, 2),
            ),
        ]
        with tempfile.TemporaryDirectory() as workspace:
            events = EventLogger()
            harness, _, _, _ = build_harness(
                workspace, responses, events=events
            )
            service = RunService(harness, events)
            progress = []
            service.subscribe(progress.append)
            try:
                snapshot = service.wait(
                    service.start("write", "novel_writing"), timeout=2
                )
                self.assertEqual("novel_writing", snapshot.workflow_id)
                self.assertEqual("creation", snapshot.workflow_state_id)
                kinds = [event.kind for event in progress]
                self.assertIn(ProgressKind.WORK_ASSIGNMENT_REQUESTED, kinds)
                self.assertIn(
                    ProgressKind.WORK_ASSIGNMENT_ADMISSION_COMPLETED, kinds
                )
            finally:
                service.close()

    def test_progress_usage_and_result_are_frontend_neutral(self):
        responses = [
            ReasoningResponse(
                json.dumps(
                    {
                        "type": "action",
                        "capability_id": "echo",
                        "arguments": {"text": "hello"},
                    }
                ),
                TokenUsage(10, 2),
            ),
            ReasoningResponse(
                json.dumps({"type": "final", "content": "draft"}),
                TokenUsage(20, 3),
            ),
        ]
        presentation = [ReasoningResponse("polished", TokenUsage(5, 4))]
        with tempfile.TemporaryDirectory() as workspace:
            events = EventLogger()
            harness, _, _, _ = build_harness(
                workspace,
                responses,
                presentation_responses=presentation,
                events=events,
            )
            service = RunService(harness, events)
            progress = []
            unsubscribe = service.subscribe(progress.append)
            service.subscribe(
                lambda _: (_ for _ in ()).throw(RuntimeError("UI failed"))
            )
            try:
                run_id = service.start("say hello")
                snapshot = service.wait(run_id, timeout=2)
                self.assertEqual(RunStatus.COMPLETED, snapshot.status)
                self.assertEqual("polished", snapshot.answer)
                self.assertEqual(
                    {
                        "input_tokens": 35,
                        "output_tokens": 9,
                        "total_tokens": 44,
                    },
                    snapshot.token_usage.as_dict(),
                )
                kinds = [event.kind for event in progress]
                self.assertIn(ProgressKind.REASONING_STARTED, kinds)
                self.assertIn(ProgressKind.CAPABILITY_REQUESTED, kinds)
                self.assertIn(ProgressKind.CAPABILITY_COMPLETED, kinds)
                self.assertEqual(ProgressKind.RUN_COMPLETED, kinds[-1])
                usage_event = next(
                    event
                    for event in progress
                    if event.kind is ProgressKind.REASONING_COMPLETED
                )
                with self.assertRaises(TypeError):
                    usage_event.data["usage"]["input_tokens"] = 0
                json.dumps(snapshot.as_dict())
                json.dumps([event.as_dict() for event in progress])
                self.assertIn(
                    "novel_writing",
                    {item["id"] for item in service.available_workflows()},
                )
            finally:
                unsubscribe()
                service.close()

    def test_failure_is_a_user_facing_terminal_snapshot(self):
        with tempfile.TemporaryDirectory() as workspace:
            events = EventLogger()
            harness, _, _, _ = build_harness(
                workspace,
                [RuntimeError("provider unavailable")],
                events=events,
            )
            service = RunService(harness, events)
            progress = []
            service.subscribe(progress.append)
            try:
                snapshot = service.wait(
                    service.start("fail predictably"), timeout=2
                )
                self.assertEqual(RunStatus.FAILED, snapshot.status)
                self.assertIn("provider unavailable", snapshot.error)
                self.assertEqual(
                    ProgressKind.RUN_FAILED, progress[-1].kind
                )
            finally:
                service.close()


if __name__ == "__main__":
    unittest.main()
