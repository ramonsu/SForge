"""JSON-lines host for one isolated Agent reasoning process."""

from __future__ import annotations

import json
import sys

from agent.llm_client import LLMClient
from harness.errors import (
    JSONModePromptConfigurationError,
    LLMProviderError,
)


def main() -> int:
    if len(sys.argv) < 2:
        return 2
    llm = None
    for line in sys.stdin:
        text = line.strip()
        if text == "shutdown":
            return 0
        try:
            request = json.loads(text)
        except Exception as exc:
            result = {
                "success": False,
                "stage": "request_decode",
                "error": str(exc),
            }
            print(json.dumps(result, ensure_ascii=False), flush=True)
            continue
        try:
            if request.get("command") != "reason":
                raise ValueError("未知 Agent Worker 命令")
            llm = llm or LLMClient()
            response = llm.chat(request["messages"])
            result = {
                "success": True,
                "content": response.content,
                "usage": response.usage.as_dict(),
            }
        except Exception as exc:
            stage = "reasoning"
            if isinstance(exc, LLMProviderError):
                stage = "api"
            elif isinstance(exc, JSONModePromptConfigurationError):
                stage = "configuration"
            result = {
                "success": False,
                "stage": stage,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "status_code": getattr(exc, "status_code", None),
                    "provider_error_type": getattr(exc, "error_type", None),
                },
            }
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
