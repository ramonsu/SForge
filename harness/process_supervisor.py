"""Disposable operating-system process supervision for Agent runs."""

from __future__ import annotations

import subprocess
import sys
import json
from pathlib import Path
from threading import RLock
from uuid import uuid4

from harness.errors import AgentWorkerError
from harness.models import ReasoningResponse, TokenUsage


class AgentProcessSupervisor:
    def __init__(self):
        self._processes: dict[str, subprocess.Popen] = {}
        self._lock = RLock()

    def spawn(self, run_id: str) -> str:
        process_id = uuid4().hex
        project_root = Path(__file__).resolve().parent.parent
        process = subprocess.Popen(
            [sys.executable, "-X", "utf8", "-m", "agent.worker", run_id],
            cwd=str(project_root),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        with self._lock:
            self._processes[process_id] = process
        return process_id

    def reason(
        self, process_id: str, messages: list[dict]
    ) -> ReasoningResponse:
        with self._lock:
            process = self._processes.get(process_id)
        if process is None or process.poll() is not None:
            raise RuntimeError("Agent Instance 进程不存在或已退出")
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("Agent Instance 通信通道不可用")
        try:
            process.stdin.write(
                json.dumps({"command": "reason", "messages": messages}, ensure_ascii=False)
                + "\n"
            )
            process.stdin.flush()
            line = process.stdout.readline()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise RuntimeError("Agent Instance 通信中断") from exc
        if not line:
            raise RuntimeError("Agent Instance 未返回推理结果")
        response = json.loads(line)
        if not response.get("success"):
            stage = response.get("stage", "unknown")
            error = response.get("error", "Agent Instance 推理失败")
            if isinstance(error, dict):
                message = str(error.get("message") or "Agent Instance 推理失败")
                error_type = str(error.get("type") or "AgentWorkerError")
                provider_error_type = error.get("provider_error_type")
                status_code = error.get("status_code")
            else:
                message = str(error)
                error_type = "AgentWorkerError"
                provider_error_type = None
                status_code = None
            raise AgentWorkerError(
                f"Agent Worker {stage} 失败: {message}",
                stage=str(stage),
                error_type=str(provider_error_type or error_type),
                status_code=(
                    int(status_code) if isinstance(status_code, int) else None
                ),
                error_message=message,
            )
        return ReasoningResponse(
            str(response.get("content", "")),
            TokenUsage.from_dict(response.get("usage")),
        )

    def terminate(self, process_id: str | None) -> None:
        if process_id is None:
            return
        with self._lock:
            process = self._processes.pop(process_id, None)
        if process is None:
            return
        if process.poll() is not None:
            if process.stdin:
                process.stdin.close()
            if process.stdout:
                process.stdout.close()
            return
        try:
            if process.stdin:
                process.stdin.write("shutdown\n")
                process.stdin.flush()
            process.wait(timeout=2)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
        finally:
            if process.stdin:
                process.stdin.close()
            if process.stdout:
                process.stdout.close()

    def is_alive(self, process_id: str) -> bool:
        with self._lock:
            process = self._processes.get(process_id)
        return process is not None and process.poll() is None

    def close(self) -> None:
        with self._lock:
            process_ids = list(self._processes)
        for process_id in process_ids:
            self.terminate(process_id)
