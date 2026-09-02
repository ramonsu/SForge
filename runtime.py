"""Application composition root for SForge V1.6."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from capabilities import builtins
from harness.agent_manager import AgentManager, ProcessSupervisor
from harness.capability import CapabilityRegistry, DefaultAdmissionPolicy
from harness.context_manager import ContextManager, ModelProjectionOverride
from harness.cognitive_policy import load_cognitive_policies
from harness.core import Harness
from harness.events import EventLogger
from harness.inspector import RuntimeSnapshot
from harness.identity import load_identity
from harness.memory_manager import MemoryProvider, SQLiteMemoryProvider
from harness.models import MemoryRecord, TaskSpec
from harness.persona import load_persona
from harness.profession import load_professions
from harness.process_supervisor import AgentProcessSupervisor
from harness.runtime_engine import RuntimeEngine
from harness.skill import load_skills
from harness.workflow_loader import WorkflowLoader
from harness.workflow_manager import WorkflowRegistry
from harness.work_role import load_work_roles
from harness.workspace import load_workspace


PROJECT_ROOT = Path(__file__).resolve().parent


def create_runtime(
    *,
    processes: ProcessSupervisor | None = None,
    memory: MemoryProvider | None = None,
    events: EventLogger | None = None,
    workspace_root: str | Path | None = None,
    workspace_id: str | None = None,
    policy_strength: float = 1.0,
    model_projection_override: ModelProjectionOverride | None = None,
    total_context_budget: int = 12_000,
    region_context_budgets: Mapping[str, int] | None = None,
    max_memory_records: int = 20,
    action_result_excerpt_characters: int = 1_200,
) -> Harness:
    memory_provider = memory or SQLiteMemoryProvider()
    _bootstrap_core_memory(memory_provider)
    registry = CapabilityRegistry()
    workspace_path = Path(workspace_root or PROJECT_ROOT).resolve()
    resolved_workspace_id = workspace_id or workspace_path.name
    for capability in builtins(workspace_path):
        registry.register(capability)
    workflows = WorkflowRegistry(WorkflowLoader())
    roles = load_work_roles()
    policies = load_cognitive_policies()
    professions = load_professions()
    skills = load_skills()
    workspace = load_workspace(resolved_workspace_id)
    persona = load_persona()
    identity = load_identity()
    contexts = ContextManager(
        memory_provider,
        workflows,
        roles,
        registry,
        policies,
        professions,
        skills,
        workspace,
        policy_strength=policy_strength,
        model_projection_override=model_projection_override,
        total_context_budget=total_context_budget,
        region_context_budgets=region_context_budgets,
        max_memory_records=max_memory_records,
        action_result_excerpt_characters=(
            action_result_excerpt_characters
        ),
    )
    return Harness(
        RuntimeEngine(
            AgentManager(processes or AgentProcessSupervisor()),
            contexts,
            memory_provider,
            registry,
            workflows,
            roles,
            policies,
            professions,
            skills,
            workspace,
            DefaultAdmissionPolicy(),
            persona,
            identity=identity,
            default_work_role_id="generalist",
            events=events,
            basic_capabilities=frozenset({"echo"}),
        )
    )


class AgentApplication:
    def __init__(self):
        self.harness = create_runtime()
        self._closed = False
        self._last_agent_id: str | None = None

    def handle(self, user_input: str, workflow_id: str | None = None) -> str:
        request = user_input.strip()
        if not request:
            return "我在听，请告诉我你想做什么。"
        process = self.harness.create_agent(
            TaskSpec(request=request), workflow_id=workflow_id
        )
        self._last_agent_id = process.id
        try:
            return self.harness.run(process.id).strip()
        finally:
            self.harness.terminate_agent(process.id)

    def inspect(self) -> RuntimeSnapshot:
        return self.harness.inspect(self._last_agent_id)

    def close(self) -> None:
        if not self._closed:
            self.harness.close()
            self._closed = True


def _bootstrap_core_memory(memory: MemoryProvider) -> None:
    path = PROJECT_ROOT / "memory" / "core_memory.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    existing = {item.kind: item.content for item in memory.retrieve(scope="core", limit=100)}
    for kind, content in records.items():
        serialized = json.dumps(content, ensure_ascii=False)
        if existing.get(kind) != serialized:
            memory.write(
                MemoryRecord(
                    scope="core",
                    kind=kind,
                    content=serialized,
                    metadata={"source": "bootstrap"},
                )
            )
