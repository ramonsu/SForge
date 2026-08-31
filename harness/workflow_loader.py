"""Load and validate a declarative, cyclic Workflow state space."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from harness.errors import InvalidWorkflowStateError, WorkflowNotFoundError
from harness.models import (
    WorkflowDefinition,
    WorkflowStateDefinition,
    WorkflowTransitionDefinition,
)


_IDENTIFIER = re.compile(r"[a-z][a-z0-9_-]{0,63}")


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
        if not _IDENTIFIER.fullmatch(name):
            raise WorkflowNotFoundError(f"非法 Workflow 名称: {name}")
        directory = self.workflows_dir / name
        json_path = directory / "workflow.json"
        markdown_path = directory / "workflow.md"
        if not markdown_path.is_file():
            markdown_path = directory / "WORKFLOW.md"
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
        initial_state = self._identifier(raw, "initial_state")
        if not instruction:
            raise InvalidWorkflowStateError("workflow.md 不能为空")
        description = self._optional_text(raw, "description") or ""
        raw_states = raw.get("states")
        if not isinstance(raw_states, dict) or not raw_states:
            raise InvalidWorkflowStateError("states 必须是非空对象")

        states: dict[str, WorkflowStateDefinition] = {}
        for state_name, value in raw_states.items():
            if not isinstance(state_name, str) or not _IDENTIFIER.fullmatch(
                state_name
            ):
                raise InvalidWorkflowStateError("State 名称无效")
            if not isinstance(value, dict):
                raise InvalidWorkflowStateError("State 定义必须是对象")
            forbidden = {"role", "role_context", "persona"}.intersection(value)
            if forbidden:
                raise InvalidWorkflowStateError(
                    "Workflow State 不能固定 Agent 身份或角色: "
                    + ", ".join(sorted(forbidden))
                )
            evaluation = value.get("evaluation", {})
            if not isinstance(evaluation, dict):
                raise InvalidWorkflowStateError("evaluation 必须是对象")
            scopes = self._memory_scopes(value)
            write_scope = (
                self._optional_text(value, "memory_write_scope")
                or scopes[-1]
            )
            if write_scope not in scopes:
                raise InvalidWorkflowStateError(
                    "memory_write_scope 必须属于 memory_scope"
                )
            states[state_name] = WorkflowStateDefinition(
                id=state_name,
                allowed_capabilities=frozenset(
                    self._optional_string_list(
                        value, "allowed_capabilities", []
                    )
                ),
                memory_scopes=tuple(scopes),
                memory_write_scope=write_scope,
                context=self._optional_text(value, "context") or "",
                goal=self._optional_text(value, "goal") or "",
                context_sources=tuple(
                    self._optional_string_list(value, "context_sources", [])
                ),
                memory_hints=tuple(
                    self._optional_string_list(value, "memory_hints", [])
                ),
                evaluation_criteria=tuple(
                    self._optional_string_list(evaluation, "criteria", [])
                ),
            )

        if initial_state not in states:
            raise InvalidWorkflowStateError("initial_state 不存在")

        raw_transitions = raw.get("transitions", {})
        if not isinstance(raw_transitions, dict):
            raise InvalidWorkflowStateError("transitions 必须是对象")
        unknown_sources = set(raw_transitions).difference(states)
        if unknown_sources:
            raise InvalidWorkflowStateError(
                "Transition source 不存在: " + ", ".join(sorted(unknown_sources))
            )
        transitions: dict[
            str, tuple[WorkflowTransitionDefinition, ...]
        ] = {}
        for source in states:
            raw_edges = raw_transitions.get(source, [])
            if not isinstance(raw_edges, list):
                raise InvalidWorkflowStateError(
                    f"State '{source}' 的 transitions 必须是数组"
                )
            edges = []
            seen: set[tuple[str, str]] = set()
            for raw_edge in raw_edges:
                if not isinstance(raw_edge, dict):
                    raise InvalidWorkflowStateError("Transition 必须是对象")
                condition = self._text(raw_edge, "condition")
                target = self._identifier(raw_edge, "target")
                if target not in states:
                    raise InvalidWorkflowStateError(
                        f"Transition target 不存在: {target}"
                    )
                key = (condition, target)
                if key in seen:
                    raise InvalidWorkflowStateError(
                        f"State '{source}' 存在重复 Transition"
                    )
                seen.add(key)
                edges.append(WorkflowTransitionDefinition(condition, target))
            transitions[source] = tuple(edges)

        return WorkflowDefinition(
            id=workflow_id,
            initial_state=initial_state,
            states=states,
            instruction=instruction,
            description=description,
            transitions=transitions,
        )

    @staticmethod
    def _text(mapping: dict[str, Any], key: str) -> str:
        value = mapping.get(key)
        if not isinstance(value, str) or not value.strip():
            raise InvalidWorkflowStateError(f"'{key}' 必须是非空字符串")
        return value.strip()

    @staticmethod
    def _identifier(mapping: dict[str, Any], key: str) -> str:
        value = WorkflowLoader._text(mapping, key)
        if not _IDENTIFIER.fullmatch(value):
            raise InvalidWorkflowStateError(f"'{key}' 不是合法标识符")
        return value

    @staticmethod
    def _string_list(mapping: dict[str, Any], key: str) -> list[str]:
        value = mapping.get(key)
        if not isinstance(value, list) or not all(
            isinstance(item, str) and item.strip() for item in value
        ):
            raise InvalidWorkflowStateError(f"'{key}' 必须是字符串数组")
        value = [item.strip() for item in value]
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

    @staticmethod
    def _optional_text(mapping: dict[str, Any], key: str) -> str | None:
        if key not in mapping:
            return None
        return WorkflowLoader._text(mapping, key)

    @staticmethod
    def _memory_scopes(mapping: dict[str, Any]) -> list[str]:
        value = mapping.get("memory_scope")
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        if isinstance(value, list):
            scopes = WorkflowLoader._string_list(mapping, "memory_scope")
            if scopes:
                return scopes
        raise InvalidWorkflowStateError(
            "'memory_scope' 必须是非空字符串或字符串数组"
        )
