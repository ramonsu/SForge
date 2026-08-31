import json
import tempfile
import unittest
from pathlib import Path

from harness.capability import DefaultAdmissionPolicy
from harness.cognitive_policy import load_cognitive_policies
from harness.models import (
    ActionRequest,
    CapabilityDescriptor,
    RuntimeState,
    WorkAssignment,
)
from harness.profession import load_professions
from harness.skill import load_skills
from harness.workspace import load_workspace


CONFIG = Path(__file__).resolve().parents[2] / "config"


class CognitivePolicyContractTests(unittest.TestCase):
    def test_presets_parse_to_immutable_bounded_parameters(self):
        policies = load_cognitive_policies()
        policy = policies.get("intj")

        self.assertEqual(16, len(policies.available()))
        self.assertEqual("INTJ", policy.id)
        self.assertGreater(policy.value("cognition", "planning_weight"), 0.5)
        self.assertGreater(policy.value("memory", "contradiction_weight"), 0.5)
        for section in policy.parameters.values():
            self.assertTrue(all(0.0 <= value <= 1.0 for value in section.values()))
        with self.assertRaises(TypeError):
            policy.parameters["memory"]["novelty_weight"] = 0.0

    def test_policy_config_rejects_execution_authority(self):
        raw = json.loads(
            (CONFIG / "cognitive_policies.json").read_text(encoding="utf-8")
        )
        raw["capabilities"] = ["filesystem.write"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "运行控制字段"):
                load_cognitive_policies(path)

    def test_policy_compiles_to_authority_free_operational_guidance(self):
        policies = load_cognitive_policies()
        risk = policies.get("INTJ").compile_model_projection()
        exploration = policies.get("ENFP").compile_model_projection()

        self.assertEqual("risk", risk["direction"])
        self.assertEqual("exploration", exploration["direction"])
        self.assertIn("verification evidence", risk["reasoning_guidance"])
        self.assertIn("novel alternatives", exploration["reasoning_guidance"])
        for projection in (risk, exploration):
            self.assertEqual(
                {"direction", "reasoning_guidance"}, set(projection)
            )
            visible = json.dumps(projection).casefold()
            for forbidden in (
                "intj",
                "enfp",
                "strength",
                "permission",
                "capability",
                "evidence_id",
            ):
                self.assertNotIn(forbidden, visible)


class ProfessionAndSkillContractTests(unittest.TestCase):
    def test_profession_exposes_memory_knowledge_and_skill_references_only(self):
        profession = load_professions().get("software_engineering")
        context = profession.as_context()

        self.assertIn("testing", context["memory"]["tags"])
        self.assertIn("interface contracts", context["knowledge_references"])
        self.assertIn("code_review", context["skills"]["preferred"])
        self.assertNotIn("capabilities", context)
        self.assertNotIn("grants", context)

    def test_skill_is_declarative_and_has_no_execution_surface(self):
        skill = load_skills().get("debugging")
        context = skill.as_context(sources=("profession:software_engineering",))

        self.assertTrue(context["instructions"])
        self.assertEqual(
            ["profession:software_engineering"], context["sources"]
        )
        self.assertFalse(hasattr(skill, "execute"))
        self.assertFalse(hasattr(skill, "invoke"))
        self.assertNotIn("capability_id", context)


class AssignmentGrantAndWorkspaceContractTests(unittest.TestCase):
    def test_work_assignment_schema_keeps_role_grants_and_lifecycle_explicit(self):
        assignment = WorkAssignment(
            id="assignment-1",
            agent_process_id="agent-1",
            identity_id="ada",
            workspace_id="SForge",
            role_id="reviewer",
            task_id="task-1",
            workflow_id="general_task",
            grants=frozenset({"filesystem.read"}),
        )
        context = assignment.as_context()

        self.assertEqual("active", context["status"])
        self.assertEqual("reviewer", context["role_id"])
        self.assertEqual(["filesystem.read"], context["grants"])
        self.assertEqual(assignment, assignment.snapshot())
        self.assertNotIn("profession_id", context)
        self.assertNotIn("cognitive_policy_id", context)

    def test_admission_reads_current_effective_grants_for_every_request(self):
        descriptor = CapabilityDescriptor(
            "filesystem.read", "read", {"type": "object"}
        )
        request = ActionRequest("filesystem.read", {})
        policy = DefaultAdmissionPolicy()
        allowed = RuntimeState(
            agent_id="agent",
            task_id="task",
            mode="direct",
            allowed_capabilities=frozenset({"echo", "filesystem.read"}),
            memory_scope="core",
        )
        revoked = allowed.snapshot()
        revoked.allowed_capabilities = frozenset({"echo"})

        self.assertTrue(policy.authorize(allowed, request, descriptor).allowed)
        self.assertFalse(policy.authorize(revoked, request, descriptor).allowed)

    def test_workspace_loads_retrieval_metadata_without_authority(self):
        workspace = load_workspace("SForge")
        context = workspace.as_context()

        self.assertTrue(context["retrieval"]["sources"])
        self.assertTrue(context["retrieval"]["prefer"])
        self.assertTrue(context["local_knowledge"]["references"])
        self.assertIn("evidence_review", context["local_skills"])
        self.assertNotIn("capabilities", context)
        self.assertNotIn("grants", context)


if __name__ == "__main__":
    unittest.main()
