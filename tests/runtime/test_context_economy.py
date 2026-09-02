import json
import tempfile
import unittest
from pathlib import Path

from harness.events import EventType
from harness.models import (
    ActionRequest,
    MemoryRecord,
    ResourceBindingRequest,
    TaskSpec,
    WorkAssignmentRequest,
)
from tests.support.runtime_factory import build_harness


class ContextEconomyRuntimeTests(unittest.TestCase):
    def test_large_read_is_durable_but_not_replayed_in_full(self):
        large_payload = "LARGE-EVIDENCE\n" + ("x" * 50_000)
        responses = [
            json.dumps(
                {
                    "type": "assignment",
                    "role_id": "reviewer",
                    "workspace_id": "SForge",
                    "workflow_id": "general_task",
                    "requested_capabilities": ["filesystem.read"],
                }
            ),
            json.dumps(
                {
                    "type": "action",
                    "capability_id": "filesystem.read",
                    "arguments": {"path": "large.txt"},
                    "request_id": "large-read",
                }
            ),
            json.dumps(
                {"type": "final", "content": "Large evidence reviewed."}
            ),
        ]
        with tempfile.TemporaryDirectory() as workspace:
            Path(workspace, "large.txt").write_text(
                large_payload, encoding="utf-8"
            )
            harness, _, memory, _ = build_harness(
                workspace,
                responses,
                workspace_id="SForge",
            )
            try:
                first = harness.create_agent(
                    TaskSpec("review large evidence", id="context-economy-task")
                )
                harness.request_binding(
                    first.id,
                    ResourceBindingRequest(
                        "profession", "activate", "software_engineering"
                    ),
                )
                harness.request_binding(
                    first.id,
                    ResourceBindingRequest(
                        "cognitive_policy", "activate", "INTJ"
                    ),
                )
                self.assertEqual("Large evidence reviewed.", harness.run(first.id))
                first_assignment = harness.work_assignment(first.id)
                self.assertIsNotNone(first_assignment)

                durable_results = [
                    *memory.retrieve(
                        scope="task:context-economy-task", limit=100
                    ),
                    *memory.retrieve(scope="workspace:SForge", limit=100),
                ]
                durable_actions = [
                    item
                    for item in durable_results
                    if item.kind.endswith(".action_result")
                ]
                self.assertEqual(2, len(durable_actions))
                self.assertTrue(
                    all(
                        json.loads(item.content)["output"] == large_payload
                        for item in durable_actions
                    )
                )

                second = harness.create_agent("continue the same review")
                context = harness.build_context(second.id)
                projected = context.for_model()
                serialized = json.dumps(projected, ensure_ascii=False)
                trace = harness.retrieval_trace(second.id)
                resumed = harness.work_assignment(second.id)

                self.assertEqual(
                    {"runtime_envelope", "life", "profession", "work"},
                    set(projected),
                )
                self.assertEqual(first.identity_id, second.identity_id)
                self.assertEqual(first_assignment.id, resumed.id)
                self.assertEqual(first_assignment.created_at, resumed.created_at)
                self.assertEqual(second.id, resumed.agent_process_id)
                self.assertEqual(
                    "INTJ", projected["life"]["cognitive_configuration"]["id"]
                )
                self.assertEqual(
                    "software_engineering",
                    projected["profession"]["active_resources"][0]["id"],
                )
                self.assertEqual(
                    "general_task", projected["work"]["workflow"]["id"]
                )
                self.assertEqual(
                    {"echo", "filesystem.read"},
                    {
                        item["id"]
                        for item in projected["work"]["capability_boundary"][
                            "available"
                        ]
                    },
                )
                self.assertNotIn(large_payload, serialized)
                self.assertIn("bounded_excerpt", serialized)
                projected_actions = [
                    item
                    for item in (
                        *projected["work"]["relevant_archive_and_artifacts"],
                        *projected["work"]["recent_observations"],
                    )
                    if item["kind"].endswith(".action_result")
                ]
                self.assertEqual(1, len(projected_actions))
                self.assertGreaterEqual(trace["deduplicated_memory_count"], 1)
                self.assertLessEqual(
                    trace["estimated_context_tokens"],
                    trace["context_budget_tokens"],
                )
                for region, size in trace["region_size_estimates"].items():
                    self.assertLessEqual(
                        size, trace["context_region_budgets"][region]
                    )
                self.assertGreater(
                    trace["unbounded_context_tokens_estimate"],
                    trace["estimated_context_tokens"] * 2,
                )

                event = harness.recent_events(
                    1,
                    agent_id=second.id,
                    event_types=(EventType.CONTEXT_BUILT,),
                )[0]
                self.assertEqual(
                    trace["estimated_context_tokens"],
                    event.as_dict()["data"]["estimated_context_tokens"],
                )
                self.assertNotIn("LARGE-EVIDENCE", json.dumps(event.as_dict()))
                denied = harness.execute_action(
                    second.id,
                    ActionRequest(
                        "filesystem.write",
                        {"path": "forbidden.txt", "content": "no"},
                    ),
                )
                self.assertEqual("rejected", denied.status)
                self.assertFalse(Path(workspace, "forbidden.txt").exists())
            finally:
                harness.close()

    def test_repeated_read_deduplicates_by_resource_and_output(self):
        with tempfile.TemporaryDirectory() as workspace:
            Path(workspace, "same.txt").write_text("same evidence", encoding="utf-8")
            harness, _, memory, _ = build_harness(
                workspace, workspace_id="SForge"
            )
            try:
                process = harness.create_agent(
                    TaskSpec("inspect duplicate reads", id="duplicate-read-task")
                )
                harness.request_work_assignment(
                    process.id,
                    WorkAssignmentRequest(
                        "reviewer",
                        workspace_id="SForge",
                        workflow_id="general_task",
                        requested_capabilities=("filesystem.read",),
                    ),
                )
                for request_id in ("read-one", "read-two"):
                    harness.execute_action(
                        process.id,
                        ActionRequest(
                            "filesystem.read",
                            {"path": "same.txt"},
                            request_id=request_id,
                        ),
                    )
                trace = harness.retrieval_trace(process.id)
                projected = harness.build_context(process.id).for_model()
                projected_actions = [
                    item
                    for item in (
                        *projected["work"]["relevant_archive_and_artifacts"],
                        *projected["work"]["recent_observations"],
                    )
                    if item["kind"].endswith(".action_result")
                ]
                stored_actions = [
                    *[
                        item
                        for item in memory.retrieve(
                            scope="task:duplicate-read-task", limit=100
                        )
                        if item.kind.endswith(".action_result")
                    ],
                    *[
                        item
                        for item in memory.retrieve(
                            scope="workspace:SForge", limit=100
                        )
                        if item.kind.endswith(".action_result")
                    ],
                ]
                self.assertEqual(4, len(stored_actions))
                self.assertEqual(1, len(projected_actions))
                self.assertGreaterEqual(trace["deduplicated_memory_count"], 3)
            finally:
                harness.close()

    def test_duplicate_facts_and_final_answers_project_once(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, memory, _ = build_harness(
                workspace, workspace_id="SForge"
            )
            try:
                process = harness.create_agent(
                    TaskSpec("inspect repeated facts", id="repeated-facts-task")
                )
                harness.request_work_assignment(
                    process.id,
                    WorkAssignmentRequest(
                        "reviewer",
                        workspace_id="SForge",
                        workflow_id="general_task",
                    ),
                )
                for record_id in ("fact-one", "fact-two"):
                    memory.write(
                        MemoryRecord(
                            id=record_id,
                            scope="workspace:SForge",
                            kind="workspace.fact",
                            content="the same durable project fact",
                        )
                    )
                for record_id in ("answer-one", "answer-two"):
                    memory.write(
                        MemoryRecord(
                            id=record_id,
                            scope="task:repeated-facts-task",
                            kind="runtime.final_answer",
                            content="the same final answer",
                        )
                    )
                projected = harness.build_context(process.id).for_model()
                mounted = [
                    *projected["work"]["relevant_archive_and_artifacts"],
                    *projected["work"]["recent_observations"],
                ]
                self.assertEqual(
                    1,
                    sum(item["kind"] == "workspace.fact" for item in mounted),
                )
                self.assertEqual(
                    1,
                    sum(
                        item["kind"] == "runtime.final_answer"
                        for item in mounted
                    ),
                )
                self.assertGreaterEqual(
                    harness.retrieval_trace(process.id)[
                        "deduplicated_memory_count"
                    ],
                    2,
                )
            finally:
                harness.close()

    def test_oversized_non_action_memory_is_dropped_by_work_budget(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, memory, _ = build_harness(
                workspace, workspace_id="SForge"
            )
            memory.write(
                MemoryRecord(
                    id="small-work-fact",
                    scope="workspace:SForge",
                    kind="workspace.fact",
                    content="small relevant fact",
                )
            )
            memory.write(
                MemoryRecord(
                    id="oversized-work-fact",
                    scope="workspace:SForge",
                    kind="workspace.fact",
                    content="z" * 50_000,
                )
            )
            try:
                process = harness.create_agent("inspect the workspace facts")
                harness.request_work_assignment(
                    process.id,
                    WorkAssignmentRequest(
                        "reviewer",
                        workspace_id="SForge",
                        workflow_id="general_task",
                    ),
                )
                trace = harness.retrieval_trace(process.id)
                self.assertIn(
                    "oversized-work-fact", trace["ranked_memory_ids"]
                )
                self.assertIn(
                    "oversized-work-fact", trace["budget_dropped_memory_ids"]
                )
                self.assertNotIn(
                    "oversized-work-fact", trace["context_memory_ids"]
                )
                self.assertIn("small-work-fact", trace["context_memory_ids"])
            finally:
                harness.close()


if __name__ == "__main__":
    unittest.main()
