import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from harness.memory_manager import MemoryRecord
from harness.models import ActionRequest, TaskSpec
from harness.persona import load_persona
from tests.test_harness import build_harness, make_persona


class PersonaLoaderTests(unittest.TestCase):
    def test_loads_versioned_immutable_persona(self):
        persona = load_persona()
        self.assertEqual("persona_ada@1.0", persona.reference)
        self.assertEqual("Ada", persona.name)
        self.assertEqual(("analytical", "patient"), persona.traits)
        self.assertEqual("concise and structured", persona.communication_style)
        self.assertEqual(
            ["science", "engineering"],
            persona.as_context()["presentation"]["favorite_topics"],
        )
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
                    scope="task:persona-task",
                    kind="communication.preference.tone",
                    content="warm",
                )
            )
            memory.write(
                MemoryRecord(
                    scope="task:persona-task",
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
                direct_capabilities=frozenset({"echo"}),
            )
            second, _, _, _ = build_harness(
                second_workspace,
                persona=make_persona("Lin", "persona_b"),
                direct_capabilities=frozenset({"echo"}),
            )
            try:
                first_process = first.create_agent("first", "general_task")
                second_process = second.create_agent("second", "general_task")
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
                workflow_state = harness.runtime_state(workflow.id)
                self.assertEqual("workflow", workflow_state.mode)
                self.assertEqual("novel_writing", workflow_state.workflow_id)
                self.assertIs(persona, harness._agents.agent(direct.id).persona)
                self.assertIs(persona, harness._agents.agent(workflow.id).persona)
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
                    scope="task:persona-final",
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
                presentation_prompt = supervisor.presentation_calls[0][1][0]["content"]
                self.assertNotIn("Ada", action_prompt)
                self.assertNotIn("warm", action_prompt)
                self.assertIn("Ada", presentation_prompt)
                self.assertIn("warm", presentation_prompt)
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


if __name__ == "__main__":
    unittest.main()
