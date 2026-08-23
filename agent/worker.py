"""JSON-lines host for one isolated Agent reasoning process."""

from __future__ import annotations

import json
import sys

from agent.llm_client import LLMClient


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
            result = {"success": True, "content": response.content or ""}
        except Exception as exc:
            result = {
                "success": False,
                "stage": "reasoning",
                "error": str(exc),
            }
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
