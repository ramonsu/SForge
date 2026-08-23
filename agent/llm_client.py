"""OpenAI-compatible LLM adapter used only inside Agent workers."""

from __future__ import annotations

import os


class LLMClient:
    def __init__(self, client=None, *, model: str | None = None):
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        if client is not None:
            self.client = client
            return
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("缺少 openai 依赖，请先安装 requirements.txt") from exc
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("未配置 DEEPSEEK_API_KEY")
        self.client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )

    def chat(self, messages: list[dict]):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
        )
        return response.choices[0].message
