"""Disposable reasoning process and its structured V1 action contract."""

from __future__ import annotations

import json
from typing import Any, Protocol

from harness.errors import InvalidAgentStateError, InvalidDecisionError
from harness.models import (
    ActionRequest,
    ActionResult,
    AgentDecision,
    AgentProcess,
    AgentStatus,
    FinalAnswer,
    OperationalContext,
)
from harness.persona import Persona


class ReasoningProcess(Protocol):
    def reason(self, process_id: str, messages: list[dict]) -> str: ...


class Agent:
    """Think and return a proposal; never execute effects directly."""

    def __init__(
        self,
        identity: AgentProcess,
        persona: Persona,
        process: ReasoningProcess,
    ):
        self._identity = identity
        self._persona = persona
        self._process = process

    @property
    def identity(self) -> AgentProcess:
        return self._identity.snapshot()

    @property
    def persona(self) -> Persona:
        return self._persona

    def run(
        self,
        context: OperationalContext,
        user_input: str,
        *,
        observation: ActionResult | None = None,
    ) -> AgentDecision:
        if (
            self._identity.status is not AgentStatus.RUNNING
            or not self._identity.host_process_id
        ):
            raise InvalidAgentStateError("Agent 没有运行中的推理进程")
        contract = {
            "rule": (
                "只负责推理并提出 FinalAnswer 或 ActionRequest；任何外部操作都必须由 "
                "Harness 执行"
            ),
            "decision_schema": {
                "final": {"type": "final", "content": "semantic answer"},
                "action": {
                    "type": "action",
                    "capability_id": "visible capability id",
                    "arguments": {},
                    "request_id": "optional id",
                },
            },
            "context": context.as_dict(),
        }
        messages = [
            {
                "role": "system",
                "content": json.dumps(contract, ensure_ascii=False, indent=2),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "request": user_input,
                        "observation": observation.as_dict() if observation else None,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        raw = self._process.reason(
            self._identity.host_process_id, messages
        ).strip()
        return self._parse_decision(raw)

    def format_response(self, presentation_context: dict[str, Any]) -> str:
        """Apply human-facing style once, outside the structured action loop."""

        if (
            self._identity.status is not AgentStatus.RUNNING
            or not self._identity.host_process_id
        ):
            raise InvalidAgentStateError("Agent 没有运行中的推理进程")
        draft = str(presentation_context.get("draft_answer", ""))
        messages = [
            {
                "role": "system",
                "content": json.dumps(
                    {
                        "purpose": "persona_response_formatting",
                        "rules": [
                            "只调整面向用户的表达方式，保留草稿的事实、结论和限制",
                            "不得提出或执行 Workflow、Skill、Tool、Memory 或生命周期动作",
                            "只返回最终自然语言文本，不返回结构化动作",
                        ],
                        "presentation_context": presentation_context,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            },
            {
                "role": "user",
                "content": "请只按 Presentation Context 格式化 draft_answer。",
            },
        ]
        try:
            formatted = self._process.reason(
                self._identity.host_process_id, messages
            ).strip()
        except Exception:
            return draft
        return formatted or draft

    @staticmethod
    def _parse_decision(raw: str) -> AgentDecision:
        candidate = raw
        if candidate.startswith("```"):
            lines = candidate.splitlines()
            candidate = "\n".join(lines[1:-1]).strip()
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            return FinalAnswer(raw)
        if not isinstance(payload, dict):
            raise InvalidDecisionError("Agent Decision 必须是对象")
        if payload.get("type") == "final":
            content = payload.get("content")
            if not isinstance(content, str):
                raise InvalidDecisionError("FinalAnswer content 必须是字符串")
            return FinalAnswer(content)
        if payload.get("type") != "action":
            raise InvalidDecisionError("Agent 返回了未知 Decision type")
        capability_id = payload.get("capability_id")
        arguments = payload.get("arguments")
        metadata = payload.get("metadata", {})
        if not isinstance(capability_id, str) or not capability_id.strip():
            raise InvalidDecisionError("ActionRequest capability_id 无效")
        if not isinstance(arguments, dict) or not isinstance(metadata, dict):
            raise InvalidDecisionError("ActionRequest arguments/metadata 必须是对象")
        request_id = payload.get("request_id")
        if request_id is not None and not isinstance(request_id, str):
            raise InvalidDecisionError("ActionRequest request_id 必须是字符串")
        options = {"metadata": metadata}
        if request_id:
            options["request_id"] = request_id
        return ActionRequest(capability_id, arguments, **options)
