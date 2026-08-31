"""OpenAI-compatible LLM adapter used only inside Agent workers."""

from __future__ import annotations

import json
import os

from harness.errors import (
    JSONModePromptConfigurationError,
    LLMProviderError,
)
from harness.models import ReasoningResponse, TokenUsage


class LLMClient:
    def __init__(self, client=None, *, model: str | None = None):
        dotenv_available = True
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            dotenv_available = False
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.temperature = self._float_setting(
            "DEEPSEEK_TEMPERATURE", default=0.1
        )
        self.max_tokens = self._optional_int_setting("DEEPSEEK_MAX_TOKENS")
        self.seed = self._optional_int_setting("DEEPSEEK_SEED")
        self.json_mode = self._bool_setting("DEEPSEEK_JSON_MODE", default=False)
        if client is not None:
            self.client = client
            return
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("缺少 openai 依赖，请先安装 requirements.txt") from exc
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            if not dotenv_available:
                raise RuntimeError(
                    "未配置 DEEPSEEK_API_KEY，且当前 Python 环境缺少 "
                    "python-dotenv，无法加载项目 .env"
                )
            raise RuntimeError("未配置 DEEPSEEK_API_KEY")
        self.client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )

    def chat(self, messages: list[dict]):
        request = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            request["max_tokens"] = self.max_tokens
        if self.seed is not None:
            request["seed"] = self.seed
        if self.json_mode and self._is_decision_request(messages):
            if not self._mentions_json(messages):
                raise JSONModePromptConfigurationError(
                    "JSON mode requires the Decision prompt to contain the "
                    "literal word 'JSON'"
                )
            request["response_format"] = {"type": "json_object"}
        try:
            response = self.client.chat.completions.create(**request)
        except LLMProviderError:
            raise
        except Exception as exc:
            raise self._provider_error(exc) from exc
        usage = getattr(response, "usage", None)
        return ReasoningResponse(
            content=response.choices[0].message.content or "",
            usage=TokenUsage(
                input_tokens=self._usage_value(usage, "prompt_tokens"),
                output_tokens=self._usage_value(usage, "completion_tokens"),
            ),
        )

    @staticmethod
    def _usage_value(usage, name: str) -> int:
        if usage is None:
            return 0
        value = getattr(usage, name, 0)
        return int(value or 0)

    @staticmethod
    def _float_setting(name: str, *, default: float) -> float:
        raw = os.getenv(name)
        if raw is None or not raw.strip():
            return default
        try:
            return float(raw)
        except ValueError as exc:
            raise RuntimeError(f"{name} 必须是数字") from exc

    @staticmethod
    def _optional_int_setting(name: str) -> int | None:
        raw = os.getenv(name)
        if raw is None or not raw.strip():
            return None
        try:
            value = int(raw)
        except ValueError as exc:
            raise RuntimeError(f"{name} 必须是整数") from exc
        if value < 1:
            raise RuntimeError(f"{name} 必须大于零")
        return value

    @staticmethod
    def _bool_setting(name: str, *, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None or not raw.strip():
            return default
        normalized = raw.strip().casefold()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise RuntimeError(f"{name} 必须是布尔值")

    @staticmethod
    def _is_decision_request(messages: list[dict]) -> bool:
        if not messages:
            return False
        try:
            payload = json.loads(str(messages[0].get("content", "")))
        except (json.JSONDecodeError, TypeError):
            return False
        return isinstance(payload, dict) and "decision_schema" in payload

    @staticmethod
    def _mentions_json(messages: list[dict]) -> bool:
        return any(
            "json" in str(message.get("content", "")).casefold()
            for message in messages
        )

    @staticmethod
    def _provider_error(exc: Exception) -> LLMProviderError:
        status_code = getattr(exc, "status_code", None)
        body = getattr(exc, "body", None)
        details = body.get("error", body) if isinstance(body, dict) else {}
        error_type = (
            details.get("type") if isinstance(details, dict) else None
        ) or type(exc).__name__
        message = (
            details.get("message") if isinstance(details, dict) else None
        ) or str(exc)
        return LLMProviderError(
            str(message),
            status_code=(
                int(status_code) if isinstance(status_code, int) else None
            ),
            error_type=str(error_type),
        )
