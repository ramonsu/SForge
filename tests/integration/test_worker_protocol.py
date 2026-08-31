import io
import json
import types
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from agent import worker
from harness.errors import AgentWorkerError, LLMProviderError
from harness.models import ReasoningResponse, TokenUsage
from harness.process_supervisor import AgentProcessSupervisor


class WorkerProtocolTests(unittest.TestCase):
    def test_worker_serializes_provider_usage(self):
        fake_client = types.SimpleNamespace(
            chat=lambda _: ReasoningResponse("answer", TokenUsage(8, 3))
        )
        request = json.dumps({"command": "reason", "messages": []})
        output = io.StringIO()
        with patch.object(worker, "LLMClient", return_value=fake_client), patch.object(
            worker.sys, "stdin", io.StringIO(f"{request}\nshutdown\n")
        ), patch.object(worker.sys, "argv", ["worker", "run-id"]), redirect_stdout(
            output
        ):
            self.assertEqual(0, worker.main())
        payload = json.loads(output.getvalue().splitlines()[0])
        self.assertEqual("answer", payload["content"])
        self.assertEqual(11, payload["usage"]["total_tokens"])

    def test_supervisor_deserializes_worker_usage(self):
        payload = {
            "success": True,
            "content": "answer",
            "usage": {"input_tokens": 13, "output_tokens": 5},
        }
        process = types.SimpleNamespace(
            stdin=io.StringIO(),
            stdout=io.StringIO(json.dumps(payload) + "\n"),
            poll=lambda: None,
        )
        supervisor = AgentProcessSupervisor()
        supervisor._processes["process"] = process
        result = supervisor.reason("process", [])
        supervisor._processes.clear()
        self.assertEqual("answer", result.content)
        self.assertEqual(18, result.usage.total_tokens)

    def test_provider_error_metadata_crosses_worker_boundary(self):
        def reject(_):
            raise LLMProviderError(
                "provider rejected request",
                status_code=400,
                error_type="invalid_request_error",
            )

        fake_client = types.SimpleNamespace(chat=reject)
        request = json.dumps({"command": "reason", "messages": []})
        output = io.StringIO()
        with patch.object(worker, "LLMClient", return_value=fake_client), patch.object(
            worker.sys, "stdin", io.StringIO(f"{request}\nshutdown\n")
        ), patch.object(worker.sys, "argv", ["worker", "run-id"]), redirect_stdout(
            output
        ):
            self.assertEqual(0, worker.main())
        payload = json.loads(output.getvalue().splitlines()[0])
        self.assertEqual("api", payload["stage"])
        self.assertEqual(400, payload["error"]["status_code"])
        self.assertEqual(
            "invalid_request_error",
            payload["error"]["provider_error_type"],
        )

        process = types.SimpleNamespace(
            stdin=io.StringIO(),
            stdout=io.StringIO(json.dumps(payload) + "\n"),
            poll=lambda: None,
        )
        supervisor = AgentProcessSupervisor()
        supervisor._processes["process"] = process
        with self.assertRaises(AgentWorkerError) as raised:
            supervisor.reason("process", [])
        supervisor._processes.clear()
        self.assertEqual("api", raised.exception.stage)
        self.assertEqual(400, raised.exception.status_code)
        self.assertEqual(
            "invalid_request_error", raised.exception.error_type
        )


if __name__ == "__main__":
    unittest.main()
