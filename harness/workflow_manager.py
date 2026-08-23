"""Thin Workflow registry; V1 does not execute a DAG."""

from __future__ import annotations

from harness.models import WorkflowDefinition
from harness.workflow_loader import WorkflowLoader


class WorkflowRegistry:
    def __init__(self, loader: WorkflowLoader):
        self.loader = loader
        self._definitions: dict[str, WorkflowDefinition] = {}

    def available(self) -> list[dict]:
        return self.loader.list_available()

    def get(self, workflow_id: str) -> WorkflowDefinition:
        if workflow_id not in self._definitions:
            self._definitions[workflow_id] = self.loader.load(workflow_id)
        return self._definitions[workflow_id]
