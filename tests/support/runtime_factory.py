"""Compose the real SForge Runtime with only its reasoning process replaced."""

from dataclasses import replace
from pathlib import Path

from capabilities import builtins
from harness.agent_manager import AgentManager
from harness.capability import CapabilityRegistry, DefaultAdmissionPolicy
from harness.context_manager import ContextManager
from harness.cognitive_policy import load_cognitive_policies
from harness.core import Harness
from harness.identity import load_identity
from harness.memory_manager import InMemoryMemoryProvider
from harness.persona import Persona
from harness.profession import load_professions
from harness.runtime_engine import RuntimeEngine
from harness.skill import load_skills
from harness.workflow_loader import WorkflowLoader
from harness.workflow_manager import WorkflowRegistry
from harness.work_role import load_work_roles
from harness.workspace import load_workspace
from tests.support.fakes import FakeSupervisor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = PROJECT_ROOT / "workflows"


def make_persona(name="Ada", persona_id="persona_ada"):
    return Persona(
        persona_id=persona_id,
        version="1.0",
        name=name,
        description="test persona",
        traits=("careful",),
        communication_style="structured",
    )


def build_harness(
    workspace,
    responses=None,
    *,
    persona=None,
    identity=None,
    memory_provider=None,
    basic_capabilities=frozenset({"echo"}),
    extra_capabilities=(),
    presentation_responses=None,
    max_steps=8,
    events=None,
    workspace_id=None,
):
    memory = memory_provider or InMemoryMemoryProvider()
    registry = CapabilityRegistry()
    for capability in (*builtins(workspace), *extra_capabilities):
        registry.register(capability)
    workflows = WorkflowRegistry(WorkflowLoader(WORKFLOWS))
    roles = load_work_roles()
    policies = load_cognitive_policies()
    professions = load_professions()
    skills = load_skills()
    resolved_workspace_id = workspace_id or Path(workspace).name
    workspace_resource = load_workspace(resolved_workspace_id)
    supervisor = FakeSupervisor(
        responses, presentation_responses=presentation_responses
    )
    active_persona = persona or make_persona()
    active_identity = identity or replace(
        load_identity(), persona_reference=active_persona.reference
    )
    harness = Harness(
        RuntimeEngine(
            AgentManager(supervisor),
            ContextManager(
                memory,
                workflows,
                roles,
                registry,
                policies,
                professions,
                skills,
                workspace_resource,
            ),
            memory,
            registry,
            workflows,
            roles,
            policies,
            professions,
            skills,
            workspace_resource,
            DefaultAdmissionPolicy(),
            active_persona,
            identity=active_identity,
            default_work_role_id="generalist",
            events=events,
            basic_capabilities=basic_capabilities,
            max_steps=max_steps,
        )
    )
    return harness, supervisor, memory, registry
