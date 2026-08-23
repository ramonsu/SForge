"""Application composition root for SForge V1."""

from __future__ import annotations

import json
from pathlib import Path

from capabilities import builtins
from harness.agent_manager import AgentManager, ProcessSupervisor
from harness.capability import CapabilityRegistry, DefaultAdmissionPolicy
from harness.context_manager import ContextManager
from harness.core import Harness
from harness.memory_manager import MemoryProvider, SQLiteMemoryProvider
from harness.models import MemoryRecord, TaskSpec
from harness.persona import load_persona
from harness.process_supervisor import AgentProcessSupervisor
from harness.workflow_loader import WorkflowLoader
from harness.workflow_manager import WorkflowRegistry


PROJECT_ROOT = Path(__file__).resolve().parent


def create_runtime(
    *,
    processes: ProcessSupervisor | None = None,
    memory: MemoryProvider | None = None,
    workspace_root: str | Path | None = None,
) -> Harness:
    memory_provider = memory or SQLiteMemoryProvider()
    _bootstrap_core_memory(memory_provider)
    registry = CapabilityRegistry()
    for capability in builtins(workspace_root or PROJECT_ROOT):
        registry.register(capability)
    workflows = WorkflowRegistry(WorkflowLoader())
    contexts = ContextManager(memory_provider, workflows, registry)
    return Harness(
        AgentManager(processes or AgentProcessSupervisor()),
        contexts,
        memory_provider,
        registry,
        workflows,
        DefaultAdmissionPolicy(),
        load_persona(),
        direct_capabilities=frozenset(
            {"echo", "read_text", "write_text"}
        ),
    )


class AgentApplication:
    def __init__(self):
        self.harness = create_runtime()
        self._closed = False

    def handle(self, user_input: str, workflow_id: str | None = None) -> str:
        request = user_input.strip()
        if not request:
            return "我在听，请告诉我你想做什么。"
        process = self.harness.create_agent(
            TaskSpec(request=request), workflow_id=workflow_id
        )
        try:
            return self.harness.run(process.id).strip()
        finally:
            self.harness.terminate_agent(process.id)

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
