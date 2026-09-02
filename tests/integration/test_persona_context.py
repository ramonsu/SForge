import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from harness.memory_manager import MemoryRecord
from harness.models import (
    ActionRequest,
    TaskSpec,
    WorkAssignmentRequest,
    WorkflowRequest,
)
from harness.persona import load_persona
from tests.support.runtime_factory import build_harness, make_persona


class PersonaLoaderTests(unittest.TestCase):
    def test_loads_versioned_immutable_persona(self):
        persona = load_persona()
        self.assertEqual(
            f"{persona.persona_id}@{persona.version}", persona.reference
        )
        self.assertTrue(persona.name)
        self.assertTrue(persona.traits)
        self.assertTrue(persona.communication_style)
        with self.assertRaises(FrozenInstanceError):
            persona.name = "changed"
        with self.assertRaises(TypeError):
            persona.presentation_metadata["new"] = "value"

    def test_rejects_execution_metadata_but_allows_presentation_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "persona.json"
            raw = {
                "id": "persona_test",
                "version": "1.0",
                "name": "Test",
                "description": "test",
                "traits": ["curious"],
                "communication_style": "clear",
                "favorite_topics": ["science"],
            }
            path.write_text(json.dumps(raw), encoding="utf-8")
            self.assertEqual(
                ["science"],
                load_persona(path).as_context()["presentation"]["favorite_topics"],
            )

            raw["permissions"] = ["admin"]
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "运行控制字段"):
                load_persona(path)


class PersonaBoundaryTests(unittest.TestCase):
    def test_operational_context_never_contains_persona_or_communication_memory(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, memory, _ = build_harness(workspace)
            memory.write(
                MemoryRecord(
                    scope="core",
                    kind="communication.preference.tone",
                    content="warm",
                )
            )
            memory.write(
                MemoryRecord(
                    scope="core",
                    kind="runtime.note",
                    content="operational fact",
                )
            )
            try:
                process = harness.create_agent(
                    TaskSpec("inspect", id="persona-task")
                )
                context = harness.build_context(process.id)
                serialized = context.as_dict()
                self.assertNotIn("persona", serialized)
                self.assertNotIn("Ada", json.dumps(serialized, ensure_ascii=False))
                self.assertEqual(
                    ["runtime.note"], [record.kind for record in context.memory]
                )
            finally:
                harness.close()

    def test_personas_do_not_change_admission_or_workflow_selection(self):
        with tempfile.TemporaryDirectory() as first_workspace, tempfile.TemporaryDirectory() as second_workspace:
            first, _, _, _ = build_harness(
                first_workspace,
                persona=make_persona("Ada", "persona_a"),
                basic_capabilities=frozenset({"echo"}),
            )
            second, _, _, _ = build_harness(
                second_workspace,
                persona=make_persona("Lin", "persona_b"),
                basic_capabilities=frozenset({"echo"}),
            )
            try:
                first_process = first.create_agent("first", "general_task")
                second_process = second.create_agent("second", "general_task")
                first_admission = first.request_workflow(
                    first_process.id, WorkflowRequest("general_task")
                )
                second_admission = second.request_workflow(
                    second_process.id, WorkflowRequest("general_task")
                )
                self.assertEqual(
                    first_admission.allowed_capabilities,
                    second_admission.allowed_capabilities,
                )
                self.assertEqual(
                    first.runtime_state(first_process.id).allowed_capabilities,
                    second.runtime_state(second_process.id).allowed_capabilities,
                )
                for harness, process in (
                    (first, first_process),
                    (second, second_process),
                ):
                    result = harness.execute_action(
                        process.id, ActionRequest("missing", {})
                    )
                    self.assertEqual("rejected", result.status)
                    self.assertEqual("resolve", result.metadata["stage"])
            finally:
                first.close()
                second.close()

    def test_same_persona_can_run_direct_and_workflow_modes(self):
        persona = make_persona()
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, _, _ = build_harness(workspace, persona=persona)
            try:
                direct = harness.create_agent("direct")
                workflow = harness.create_agent("workflow", "novel_writing")
                self.assertEqual("direct", harness.runtime_state(direct.id).mode)
                harness.request_work_assignment(
                    workflow.id,
                    WorkAssignmentRequest(
                        "author", workflow_id="novel_writing"
                    ),
                )
                workflow_state = harness.runtime_state(workflow.id)
                self.assertEqual("workflow", workflow_state.mode)
                self.assertEqual("novel_writing", workflow_state.workflow_id)
            finally:
                harness.close()

    def test_persona_and_communication_memory_only_format_final_text(self):
        response = json.dumps({"type": "final", "content": "draft"})
        tool_like_text = json.dumps(
            {"type": "action", "capability_id": "echo", "arguments": {"text": "x"}}
        )
        with tempfile.TemporaryDirectory() as workspace:
            harness, supervisor, memory, _ = build_harness(
                workspace,
                [response],
                presentation_responses=[tool_like_text],
            )
            memory.write(
                MemoryRecord(
                    scope="core",
                    kind="communication.preference.tone",
                    content="warm",
                )
            )
            try:
                process = harness.create_agent(
                    TaskSpec("question", id="persona-final")
                )
                self.assertEqual(tool_like_text, harness.run(process.id))
                self.assertEqual(1, len(supervisor.action_calls))
                self.assertEqual(1, len(supervisor.presentation_calls))
                action_prompt = supervisor.action_calls[0][1][0]["content"]
                presentation_messages = supervisor.presentation_calls[0][1]
                presentation_prompt = presentation_messages[0]["content"]
                presentation_input = presentation_messages[1]["content"]
                self.assertNotIn("Ada", action_prompt)
                self.assertNotIn("warm", action_prompt)
                self.assertIn("Return exactly one valid JSON object", action_prompt)
                self.assertIn("Do not wrap the JSON in Markdown fences", action_prompt)
                self.assertIn("Ada", presentation_input)
                self.assertIn("warm", presentation_input)
                self.assertEqual(
                    "question", json.loads(presentation_input)["request"]
                )
                self.assertTrue(
                    presentation_prompt.startswith("Formatting instructions:")
                )
                self.assertNotIn("user_response_rendering", presentation_prompt)
                self.assertNotIn("presentation context", presentation_prompt.casefold())
                self.assertNotIn("runtime", presentation_prompt.casefold())
                self.assertEqual(
                    1,
                    len(supervisor.action_calls),
                    "Presentation 输出不得重新进入动作循环",
                )
            finally:
                harness.close()

    def test_presentation_failure_returns_semantic_draft(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, _, _ = build_harness(
                workspace,
                [json.dumps({"type": "final", "content": "safe draft"})],
                presentation_responses=[RuntimeError("formatter unavailable")],
            )
            try:
                process = harness.create_agent("question")
                self.assertEqual("safe draft", harness.run(process.id))
            finally:
                harness.close()

    def test_presentation_preserves_user_visible_resource_identifiers(self):
        draft = (
            "CognitivePolicy INTJ and Profession software_engineering "
            "are active."
        )
        with tempfile.TemporaryDirectory() as workspace:
            harness, supervisor, _, _ = build_harness(
                workspace,
                [json.dumps({"type": "final", "content": draft})],
            )
            try:
                process = harness.create_agent("question")
                answer = harness.run(process.id)
                rendering = json.loads(
                    supervisor.presentation_calls[0][1][1]["content"]
                )
                for value in (answer, rendering["draft"]):
                    self.assertIn("CognitivePolicy", value)
                    self.assertIn("INTJ", value)
                    self.assertIn("software_engineering", value)
            finally:
                harness.close()


if __name__ == "__main__":
    unittest.main()
