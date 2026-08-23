"""Assemble the complete Agent Context from runtime-owned resources."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol

from harness.capability import CapabilityRegistry
from harness.memory_manager import MemoryProvider
from harness.models import (
    AgentProcess,
    MemoryRecord,
    OperationalContext,
    RuntimeState,
    TaskSpec,
)
from harness.persona import Persona
from harness.workflow_manager import WorkflowRegistry


class ContextProvider(Protocol):
    def provide(self, runtime_state: RuntimeState) -> dict[str, Any]: ...


class ContextManager:
    def __init__(
        self,
        memory: MemoryProvider,
        workflows: WorkflowRegistry,
        capabilities: CapabilityRegistry,
        *,
        providers: tuple[ContextProvider, ...] = (),
    ):
        self.memory = memory
        self.workflows = workflows
        self.capabilities = capabilities
        self.providers = providers

    def build(
        self,
        process: AgentProcess,
        runtime: RuntimeState,
        task: TaskSpec,
        persona: Persona,
    ) -> "ContextBundle":
        workflow_context = None
        if runtime.workflow_id:
            definition = self.workflows.get(runtime.workflow_id)
            state = definition.states[runtime.workflow_state_id or ""]
            workflow_context = {
                "id": definition.id,
                "instruction": definition.instruction,
                "state": state.as_context(),
            }

        scopes = ("core",) if runtime.memory_scope == "core" else (
            "core",
            runtime.memory_scope,
        )
        memory_records = []
        for scope in scopes:
            memory_records.extend(self.memory.retrieve(scope=scope))
        operational_memory, preferences, history, communication = (
            self._partition_memory(memory_records)
        )
        extensions = {}
        for provider in self.providers:
            extensions.update(provider.provide(runtime))
        operational = OperationalContext(
            system={
                "runtime": "SForge V1",
                "agent_id": process.id,
                "status": process.status.value,
                "rules": [
                    "Agent 只返回 FinalAnswer 或 ActionRequest",
                    "外部操作必须通过 Harness Capability boundary",
                    "只有 Operational Context 中列出的 Capability 可请求",
                ],
            },
            task={
                "id": task.id,
                "request": task.request,
                "context": deepcopy(task.context),
            },
            runtime={**runtime.as_context(), "extensions": extensions},
            workflow=workflow_context,
            capabilities=self.capabilities.descriptors(
                runtime.allowed_capabilities
            ),
            memory=tuple(operational_memory),
        )
        presentation = {
            "persona": persona.as_context(),
            "user_preferences": [self._memory_context(item) for item in preferences],
            "interaction_history": [self._memory_context(item) for item in history],
            "communication_memory": [
                self._memory_context(item) for item in communication
            ],
        }
        return ContextBundle(operational, presentation)

    @staticmethod
    def _memory_context(record: MemoryRecord) -> dict[str, Any]:
        return {
            "id": record.id,
            "scope": record.scope,
            "kind": record.kind,
            "content": record.content,
            "metadata": deepcopy(record.metadata),
        }

    @staticmethod
    def _partition_memory(records: list[MemoryRecord]) -> tuple[
        list[MemoryRecord],
        list[MemoryRecord],
        list[MemoryRecord],
        list[MemoryRecord],
    ]:
        operational: list[MemoryRecord] = []
        preferences: list[MemoryRecord] = []
        history: list[MemoryRecord] = []
        communication: list[MemoryRecord] = []
        for record in records:
            kind = record.kind.casefold()
            if kind == "core.style" or kind.startswith(
                ("communication.preference", "user.communication.preference")
            ):
                preferences.append(record)
            elif kind.startswith(("communication.history", "interaction.")):
                history.append(record)
            elif kind.startswith("communication."):
                communication.append(record)
            else:
                operational.append(record)
        return operational, preferences, history, communication


@dataclass(frozen=True)
class ContextBundle:
    """Two independently assembled views with no shared control authority."""

    operational: OperationalContext
    presentation: dict[str, Any]

    def presentation_for(self, draft: str) -> dict[str, Any]:
        context = deepcopy(self.presentation)
        context["draft_answer"] = draft
        return context
