import os
import json
import sys
import types
import unittest
from unittest.mock import patch

from agent.llm_client import LLMClient
from harness.errors import (
    JSONModePromptConfigurationError,
    LLMProviderError,
)


class LLMClientConfigurationTests(unittest.TestCase):
    def test_provider_usage_is_preserved(self):
        response = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content="answer")
                )
            ],
            usage=types.SimpleNamespace(
                prompt_tokens=120, completion_tokens=30
            ),
        )
        completions = types.SimpleNamespace(create=lambda **_: response)
        client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=completions)
        )
        result = LLMClient(client=client).chat(
            [{"role": "user", "content": "x"}]
        )
        self.assertEqual("answer", result.content)
        self.assertEqual(150, result.usage.total_tokens)

    def test_optional_generation_settings_are_forwarded(self):
        captured = {}
        response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(
                message=types.SimpleNamespace(content="answer")
            )],
            usage=None,
        )

        def create(**kwargs):
            captured.update(kwargs)
            return response

        client = types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=create)
            )
        )
        with patch.dict(
            os.environ,
            {
                "DEEPSEEK_TEMPERATURE": "0.25",
                "DEEPSEEK_MAX_TOKENS": "900",
                "DEEPSEEK_SEED": "7",
            },
        ):
            LLMClient(client=client, model="fixed-model").chat(
                [{"role": "user", "content": "x"}]
            )

        self.assertEqual("fixed-model", captured["model"])
        self.assertEqual(0.25, captured["temperature"])
        self.assertEqual(900, captured["max_tokens"])
        self.assertEqual(7, captured["seed"])

    def test_missing_dotenv_reports_the_actual_environment_problem(self):
        fake_openai = types.ModuleType("openai")
        fake_openai.OpenAI = object
        environment = {
            key: value
            for key, value in os.environ.items()
            if key != "DEEPSEEK_API_KEY"
        }
        with patch.dict(
            sys.modules,
            {"dotenv": None, "openai": fake_openai},
        ), patch.dict(os.environ, environment, clear=True):
            with self.assertRaisesRegex(RuntimeError, "python-dotenv"):
                LLMClient()

    def test_json_mode_is_sent_only_for_decision_contract(self):
        requests = []
        response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(
                message=types.SimpleNamespace(content='{"type":"final","content":"ok"}')
            )],
            usage=None,
        )
        client = types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(
                    create=lambda **kwargs: requests.append(kwargs) or response
                )
            )
        )
        with patch.dict(os.environ, {"DEEPSEEK_JSON_MODE": "true"}):
            adapter = LLMClient(client=client)
            adapter.chat([
                {
                    "role": "system",
                    "content": json.dumps(
                        {
                            "output_protocol": "Return one JSON object",
                            "decision_schema": {},
                        }
                    ),
                }
            ])
            adapter.chat([
                {
                    "role": "system",
                    "content": json.dumps({"purpose": "user_response_rendering"}),
                }
            ])
        self.assertEqual(
            {"type": "json_object"}, requests[0]["response_format"]
        )
        self.assertNotIn("response_format", requests[1])

    def test_json_mode_without_literal_json_fails_before_provider_call(self):
        requests = []
        client = types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(
                    create=lambda **kwargs: requests.append(kwargs)
                )
            )
        )
        with patch.dict(os.environ, {"DEEPSEEK_JSON_MODE": "true"}):
            adapter = LLMClient(client=client)
            with self.assertRaises(JSONModePromptConfigurationError):
                adapter.chat(
                    [
                        {
                            "role": "system",
                            "content": json.dumps({"decision_schema": {}}),
                        }
                    ]
                )
        self.assertEqual([], requests)

    def test_provider_http_error_keeps_status_and_provider_type(self):
        class FakeBadRequest(Exception):
            status_code = 400
            body = {
                "error": {
                    "type": "invalid_request_error",
                    "message": "provider rejected request",
                }
            }

        client = types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(
                    create=lambda **_: (_ for _ in ()).throw(FakeBadRequest())
                )
            )
        )
        with self.assertRaises(LLMProviderError) as raised:
            LLMClient(client=client).chat(
                [{"role": "user", "content": "hello"}]
            )
        self.assertEqual(400, raised.exception.status_code)
        self.assertEqual(
            "invalid_request_error", raised.exception.error_type
        )
        self.assertEqual("provider rejected request", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
