"""Load a V1 Workflow package: workflow.json + workflow.md."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from harness.errors import InvalidWorkflowStateError, WorkflowNotFoundError
from harness.models import WorkflowDefinition, WorkflowStateDefinition


class WorkflowLoader:
    def __init__(self, workflows_dir: str | Path | None = None):
        root = Path(__file__).resolve().parent.parent
        self.workflows_dir = Path(workflows_dir) if workflows_dir else root / "workflows"
        self.errors: dict[str, str] = {}

    def list_available(self) -> list[dict[str, Any]]:
        self.errors.clear()
        summaries = []
        if not self.workflows_dir.is_dir():
            return summaries
        for directory in sorted(self.workflows_dir.iterdir()):
            if not directory.is_dir() or directory.name.startswith((".", "_")):
                continue
            try:
                summaries.append(self.load(directory.name).summary())
            except (WorkflowNotFoundError, InvalidWorkflowStateError) as exc:
                self.errors[directory.name] = str(exc)
        return summaries

    def load(self, name: str) -> WorkflowDefinition:
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", name):
            raise WorkflowNotFoundError(f"非法 Workflow 名称: {name}")
        directory = self.workflows_dir / name
        json_path = directory / "workflow.json"
        markdown_path = directory / "workflow.md"
        if not json_path.is_file() or not markdown_path.is_file():
            raise WorkflowNotFoundError(
                f"Workflow '{name}' 必须同时包含 workflow.json 和 workflow.md"
            )
        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
            instruction = markdown_path.read_text(encoding="utf-8").strip()
        except (OSError, json.JSONDecodeError) as exc:
            raise InvalidWorkflowStateError(
                f"无法读取 Workflow '{name}': {exc}"
            ) from exc
        return self._validate(name, raw, instruction)

    def _validate(
        self, directory_name: str, raw: Any, instruction: str
    ) -> WorkflowDefinition:
        if not isinstance(raw, dict):
            raise InvalidWorkflowStateError("workflow.json 顶层必须是对象")
        workflow_id = self._text(raw, "id")
        if workflow_id != directory_name:
            raise InvalidWorkflowStateError("Workflow id 必须与目录名一致")
        initial_state = self._text(raw, "initial_state")
        if not instruction:
            raise InvalidWorkflowStateError("workflow.md 不能为空")
        raw_states = raw.get("states")
        if not isinstance(raw_states, dict) or not raw_states:
            raise InvalidWorkflowStateError("states 必须是非空对象")

        states = {}
        for state_name, value in raw_states.items():
            if not isinstance(state_name, str) or not isinstance(value, dict):
                raise InvalidWorkflowStateError("State 名称和定义无效")
            states[state_name] = WorkflowStateDefinition(
                id=state_name,
                allowed_capabilities=frozenset(
                    self._optional_string_list(
                        value, "allowed_capabilities", []
                    )
                ),
                memory_scope=self._text(value, "memory_scope"),
                context_sources=tuple(
                    self._optional_string_list(value, "context_sources", [])
                ),
            )

        if initial_state not in states:
            raise InvalidWorkflowStateError("initial_state 不存在")
        return WorkflowDefinition(workflow_id, initial_state, states, instruction)

    @staticmethod
    def _text(mapping: dict[str, Any], key: str) -> str:
        value = mapping.get(key)
        if not isinstance(value, str) or not value.strip():
            raise InvalidWorkflowStateError(f"'{key}' 必须是非空字符串")
        return value.strip()

    @staticmethod
    def _string_list(mapping: dict[str, Any], key: str) -> list[str]:
        value = mapping.get(key)
        if not isinstance(value, list) or not all(
            isinstance(item, str) and item.strip() for item in value
        ):
            raise InvalidWorkflowStateError(f"'{key}' 必须是字符串数组")
        if len(value) != len(set(value)):
            raise InvalidWorkflowStateError(f"'{key}' 不允许重复项")
        return value

    @staticmethod
    def _optional_string_list(
        mapping: dict[str, Any], key: str, default: list[str]
    ) -> list[str]:
        if key not in mapping:
            return list(default)
        return WorkflowLoader._string_list(mapping, key)
