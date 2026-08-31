import tempfile
import unittest
from pathlib import Path

from harness.models import (
    ActionRequest,
    ResourceBindingRequest,
    WorkAssignmentRequest,
)
from tests.support.runtime_factory import build_harness


class WorkRuntimeIntegrationTests(unittest.TestCase):
    def test_assignment_atomically_mounts_workspace_context_skills_and_grants(self):
        with tempfile.TemporaryDirectory() as workspace:
            Path(workspace, "evidence.txt").write_text("verified", encoding="utf-8")
            harness, _, _, _ = build_harness(
                workspace, workspace_id="project_alpha"
            )
            try:
                process = harness.create_agent("review evidence")
                before = harness.build_context(process.id)
                self.assertIsNone(before.workspace)
                self.assertEqual(
                    {"echo"},
                    set(harness.runtime_state(process.id).allowed_capabilities),
                )
                self.assertFalse(
                    any(
                        "workspace:project_alpha" in item["sources"]
                        for item in before.skills
                    )
                )

                admission = harness.request_work_assignment(
                    process.id,
                    WorkAssignmentRequest(
                        "reviewer",
                        workspace_id="project_alpha",
                        workflow_id="general_task",
                        requested_capabilities=("filesystem.read",),
                    ),
                )
                context = harness.build_context(process.id)

                self.assertEqual("success", admission.status)
                self.assertEqual("project_alpha", context.workspace["id"])
                self.assertIn(
                    "workspace:project_alpha",
                    harness.runtime_state(process.id).memory_scopes,
                )
                self.assertTrue(
                    any(
                        "workspace:project_alpha" in item["sources"]
                        for item in context.skills
                    )
                )
                self.assertEqual(
                    {"echo", "filesystem.read"},
                    set(harness.runtime_state(process.id).allowed_capabilities),
                )
            finally:
                harness.close()

    def test_profession_and_skills_never_grant_capabilities(self):
        with tempfile.TemporaryDirectory() as workspace:
            harness, _, _, _ = build_harness(workspace)
            try:
                process = harness.create_agent("review")
                harness.request_binding(
                    process.id,
                    ResourceBindingRequest(
                        "profession", "activate", "software_engineering"
                    ),
                )
                context = harness.build_context(process.id)
                denied = harness.execute_action(
                    process.id, ActionRequest("filesystem.read", {"path": "x"})
                )

                self.assertTrue(context.professions)
                self.assertTrue(context.skills)
                self.assertEqual(
                    {"echo"},
                    set(harness.runtime_state(process.id).allowed_capabilities),
                )
                self.assertEqual("rejected", denied.status)
                self.assertEqual("admission", denied.metadata["stage"])
            finally:
                harness.close()

    def test_assignment_close_invalidates_context_skills_grants_and_admission(self):
        with tempfile.TemporaryDirectory() as workspace:
            Path(workspace, "evidence.txt").write_text("verified", encoding="utf-8")
            harness, _, _, _ = build_harness(
                workspace, workspace_id="project_alpha"
            )
            try:
                process = harness.create_agent("review")
                harness.request_work_assignment(
                    process.id,
                    WorkAssignmentRequest(
                        "reviewer",
                        workspace_id="project_alpha",
                        workflow_id="general_task",
                        requested_capabilities=("filesystem.read",),
                    ),
                )
                allowed = harness.execute_action(
                    process.id,
                    ActionRequest("filesystem.read", {"path": "evidence.txt"}),
                )
                harness.end_work_assignment(process.id)
                context = harness.build_context(process.id)
                denied = harness.execute_action(
                    process.id,
                    ActionRequest("filesystem.read", {"path": "evidence.txt"}),
                )

                self.assertEqual("success", allowed.status)
                self.assertIsNone(context.workspace)
                self.assertIsNone(context.work_assignment)
                self.assertFalse(
                    any(
                        "workspace:project_alpha" in item["sources"]
                        for item in context.skills
                    )
                )
                self.assertEqual(
                    {"echo"},
                    set(harness.runtime_state(process.id).allowed_capabilities),
                )
                self.assertEqual("rejected", denied.status)
                self.assertEqual("admission", denied.metadata["stage"])
            finally:
                harness.close()


if __name__ == "__main__":
    unittest.main()
