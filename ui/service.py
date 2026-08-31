"""Frontend-neutral asynchronous run service for SForge user interfaces."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Event, RLock
from typing import Callable
from uuid import uuid4

from harness.core import Harness
from harness.events import EventLogger, EventType, RuntimeEvent
from harness.models import TaskSpec, TokenUsage
from runtime import create_runtime
from ui.contracts import ProgressEvent, ProgressKind, RunSnapshot, RunStatus


@dataclass
class _RunState:
    id: str
    request: str
    workflow_hint_id: str | None
    status: RunStatus = RunStatus.QUEUED
    workflow_id: str | None = None
    workflow_state_id: str | None = None
    assignment_id: str | None = None
    work_role_id: str | None = None
    workspace_id: str | None = None
    cognitive_policy_id: str | None = None
    profession_ids: tuple[str, ...] = ()
    agent_id: str | None = None
    stage: str = "queued"
    current_capability: str | None = None
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    answer: str | None = None
    error: str | None = None
    sequence: int = 0
    done: Event = field(default_factory=Event)
    future: Future | None = None


class RunService:
    """Own user-facing run sessions without exposing runtime resources."""

    def __init__(
        self,
        harness: Harness | None = None,
        events: EventLogger | None = None,
    ):
        if harness is not None and events is None:
            raise ValueError("注入 Harness 时必须同时注入其 EventLogger")
        self._events = events or EventLogger()
        self._harness = harness or create_runtime(events=self._events)
        self._runtime_unsubscribe = self._events.subscribe(
            self._on_runtime_event
        )
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="sforge-run"
        )
        self._runs: dict[str, _RunState] = {}
        self._listeners: dict[int, Callable[[ProgressEvent], None]] = {}
        self._listener_sequence = 0
        self._lock = RLock()
        self._closed = False

    def available_workflows(self) -> tuple[dict, ...]:
        return self._harness.available_workflows()

    def available_workspaces(self) -> tuple[dict, ...]:
        return self._harness.available_workspaces()

    def available_cognitive_policies(self) -> tuple[dict, ...]:
        return self._harness.available_cognitive_policies()

    def available_professions(self) -> tuple[dict, ...]:
        return self._harness.available_professions()

    def available_skills(self) -> tuple[dict, ...]:
        return self._harness.available_skills()

    def available_work_roles(self) -> tuple[dict, ...]:
        return self._harness.available_work_roles()

    def subscribe(
        self, listener: Callable[[ProgressEvent], None]
    ) -> Callable[[], None]:
        if not callable(listener):
            raise TypeError("Progress listener 必须可调用")
        with self._lock:
            self._require_open()
            self._listener_sequence += 1
            listener_id = self._listener_sequence
            self._listeners[listener_id] = listener

        def unsubscribe() -> None:
            with self._lock:
                self._listeners.pop(listener_id, None)

        return unsubscribe

    def start(
        self, request: str, workflow_id: str | None = None
    ) -> str:
        request = request.strip()
        if not request:
            raise ValueError("请求不能为空")
        run_id = uuid4().hex
        state = _RunState(run_id, request, workflow_id)
        with self._lock:
            self._require_open()
            self._runs[run_id] = state
        self._publish(
            run_id,
            ProgressKind.RUN_QUEUED,
            "任务已加入运行队列",
        )
        state.future = self._executor.submit(self._execute, run_id)
        return run_id

    def snapshot(self, run_id: str) -> RunSnapshot:
        with self._lock:
            return self._snapshot(self._state(run_id))

    def snapshots(self) -> tuple[RunSnapshot, ...]:
        with self._lock:
            return tuple(self._snapshot(state) for state in self._runs.values())

    def wait(
        self, run_id: str, timeout: float | None = None
    ) -> RunSnapshot:
        with self._lock:
            done = self._state(run_id).done
        if not done.wait(timeout):
            raise TimeoutError(f"等待 Run 超时: {run_id}")
        return self.snapshot(run_id)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            futures = tuple(
                state.future for state in self._runs.values() if state.future
            )
        for future in futures:
            future.cancel()
        self._harness.close()
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._runtime_unsubscribe()
        with self._lock:
            self._listeners.clear()

    def _execute(self, run_id: str) -> None:
        process_id = None
        self._set(run_id, status=RunStatus.RUNNING, stage="preparing")
        self._publish(run_id, ProgressKind.RUN_STARTED, "正在准备运行环境")
        try:
            with self._lock:
                state = self._state(run_id)
                task = TaskSpec(state.request, id=state.id)
                workflow_id = state.workflow_hint_id
            process = self._harness.create_agent(task, workflow_id)
            process_id = process.id
            self._set(run_id, agent_id=process_id)
            answer = self._harness.run(process_id).strip()
            self._set(
                run_id,
                status=RunStatus.COMPLETED,
                stage="completed",
                current_capability=None,
                answer=answer,
            )
            self._publish(run_id, ProgressKind.RUN_COMPLETED, "任务已完成")
        except Exception as exc:
            self._set(
                run_id,
                status=RunStatus.FAILED,
                stage="failed",
                current_capability=None,
                error=str(exc),
            )
            self._publish(
                run_id,
                ProgressKind.RUN_FAILED,
                f"运行失败：{exc}",
            )
        finally:
            if process_id is not None:
                self._harness.terminate_agent(process_id)
            with self._lock:
                self._state(run_id).done.set()

    def _on_runtime_event(self, event: RuntimeEvent) -> None:
        run_id = event.trace_id
        with self._lock:
            if run_id not in self._runs:
                return
        data = dict(event.data)
        if event.type is EventType.AGENT_CREATED:
            self._set(run_id, agent_id=event.agent_id, stage="preparing")
            self._publish(
                run_id, ProgressKind.AGENT_CREATED, "Agent 运行实例已创建"
            )
        elif event.type is EventType.WORKFLOW_REQUESTED:
            workflow_id = str(data.get("workflow_id", "")) or None
            target_state_id = str(data.get("target_state_id", "")) or None
            self._set(run_id, stage="workflow_admission")
            self._publish(
                run_id,
                ProgressKind.WORKFLOW_REQUESTED,
                f"Agent 正在申请 {workflow_id}",
                {
                    "workflow_id": workflow_id,
                    "target_state_id": target_state_id,
                },
            )
        elif event.type is EventType.WORKFLOW_ADMISSION_COMPLETED:
            workflow_id = str(data.get("workflow_id", "")) or None
            state_id = str(data.get("state_id", "")) or None
            status = str(data.get("status", ""))
            if status == "success":
                self._set(
                    run_id,
                    workflow_id=workflow_id,
                    workflow_state_id=state_id,
                )
            self._publish(
                run_id,
                ProgressKind.WORKFLOW_ADMISSION_COMPLETED,
                (
                    f"已进入 {workflow_id} / {state_id}"
                    if status == "success"
                    else f"未能进入 {workflow_id}"
                ),
                {
                    "workflow_id": workflow_id,
                    "state_id": state_id,
                    "status": status,
                },
            )
        elif event.type is EventType.WORK_ASSIGNMENT_REQUESTED:
            role_id = str(data.get("role_id", "")) or None
            self._set(run_id, stage="work_assignment_admission")
            self._publish(
                run_id,
                ProgressKind.WORK_ASSIGNMENT_REQUESTED,
                f"Agent 正在申请工作角色 {role_id}",
                {
                    "role_id": role_id,
                    "workspace_id": data.get("workspace_id"),
                    "workflow_id": data.get("workflow_id"),
                },
            )
        elif event.type is EventType.WORK_ASSIGNMENT_ADMISSION_COMPLETED:
            role_id = str(data.get("role_id", "")) or None
            assignment_id = str(data.get("assignment_id", "")) or None
            status = str(data.get("status", ""))
            if status == "success":
                self._set(
                    run_id,
                    assignment_id=assignment_id,
                    work_role_id=role_id,
                    workspace_id=(
                        str(data.get("workspace_id", "")) or None
                    ),
                    workflow_id=(
                        str(data.get("workflow_id", "")) or None
                    ),
                    workflow_state_id=(
                        str(data.get("workflow_state_id", "")) or None
                    ),
                )
            self._publish(
                run_id,
                ProgressKind.WORK_ASSIGNMENT_ADMISSION_COMPLETED,
                (
                    f"已建立工作关系：{role_id}"
                    if status == "success"
                    else f"未能建立工作关系：{role_id}"
                ),
                {
                    "assignment_id": assignment_id,
                    "role_id": role_id,
                    "status": status,
                },
            )
        elif event.type is EventType.WORK_ASSIGNMENT_ENDED:
            self._set(
                run_id,
                assignment_id=None,
                work_role_id=None,
                workspace_id=None,
                workflow_id=None,
                workflow_state_id=None,
            )
            self._publish(
                run_id,
                ProgressKind.WORK_ASSIGNMENT_ENDED,
                "工作关系已结束，临时权限已撤销",
                {
                    "role_id": data.get("role_id"),
                    "workspace_id": data.get("workspace_id"),
                },
            )
        elif event.type is EventType.RESOURCE_BINDING_REQUESTED:
            resource_type = str(data.get("resource_type", ""))
            resource_id = str(data.get("resource_id", "")) or None
            operation = str(data.get("operation", ""))
            self._set(run_id, stage="resource_binding_admission")
            self._publish(
                run_id,
                ProgressKind.RESOURCE_BINDING_REQUESTED,
                f"Agent 正在申请 {operation} {resource_type} {resource_id or ''}".strip(),
                {
                    "resource_type": resource_type,
                    "resource_id": resource_id,
                    "operation": operation,
                },
            )
        elif event.type is EventType.RESOURCE_BINDING_COMPLETED:
            resource_type = str(data.get("resource_type", ""))
            resource_id = str(data.get("resource_id", "")) or None
            operation = str(data.get("operation", ""))
            status = str(data.get("status", ""))
            if status == "success":
                self._set(
                    run_id,
                    cognitive_policy_id=(
                        str(data.get("cognitive_policy_id", "")) or None
                    ),
                    profession_ids=tuple(data.get("profession_ids", ())),
                )
            self._publish(
                run_id,
                ProgressKind.RESOURCE_BINDING_COMPLETED,
                (
                    f"资源绑定已更新：{resource_type} {resource_id or ''}".strip()
                    if status == "success"
                    else f"资源绑定被拒绝：{resource_type} {resource_id or ''}".strip()
                ),
                {
                    "resource_type": resource_type,
                    "resource_id": resource_id,
                    "operation": operation,
                    "status": status,
                },
            )
        elif event.type is EventType.CONTEXT_BUILT:
            self._set(run_id, stage="context")
            self._publish(
                run_id,
                ProgressKind.CONTEXT_READY,
                "运行上下文已装载",
                {
                    "memory_count": data.get("memory_count", 0),
                    "capability_count": data.get("capability_count", 0),
                },
            )
        elif event.type is EventType.REASONING_STARTED:
            phase = str(data.get("phase", "decision"))
            stage = "presentation" if phase == "presentation" else "reasoning"
            message = (
                "正在整理最终回答"
                if phase == "presentation"
                else "正在分析并决定下一步"
            )
            self._set(run_id, stage=stage)
            self._publish(
                run_id,
                ProgressKind.REASONING_STARTED,
                message,
                {"phase": phase},
            )
        elif event.type is EventType.REASONING_COMPLETED:
            usage = TokenUsage.from_dict(dict(data.get("usage", {})))
            with self._lock:
                state = self._state(run_id)
                state.token_usage = state.token_usage + usage
            self._publish(
                run_id,
                ProgressKind.REASONING_COMPLETED,
                f"模型返回 · 本次 {usage.total_tokens} tokens",
                {"phase": data.get("phase"), "usage": usage.as_dict()},
            )
        elif event.type is EventType.CAPABILITY_REQUESTED:
            capability_id = str(data.get("capability_id", ""))
            self._set(
                run_id,
                stage="capability",
                current_capability=capability_id,
            )
            self._publish(
                run_id,
                ProgressKind.CAPABILITY_REQUESTED,
                f"正在请求 {capability_id}",
                {"capability_id": capability_id},
            )
        elif event.type is EventType.CAPABILITY_COMPLETED:
            capability_id = str(data.get("capability_id", ""))
            status = str(data.get("status", ""))
            self._publish(
                run_id,
                ProgressKind.CAPABILITY_COMPLETED,
                f"{capability_id} · {status}",
                {"capability_id": capability_id, "status": status},
            )
        elif event.type is EventType.ACTION_COMPLETED:
            status = str(data.get("status", ""))
            capability_id = str(data.get("capability_id", ""))
            self._set(
                run_id, stage="reasoning", current_capability=None
            )
            self._publish(
                run_id,
                ProgressKind.ACTION_COMPLETED,
                f"动作结果 · {status}",
                {"capability_id": capability_id, "status": status},
            )

    def _publish(
        self,
        run_id: str,
        kind: ProgressKind,
        message: str,
        data: dict | None = None,
    ) -> None:
        with self._lock:
            state = self._state(run_id)
            state.sequence += 1
            progress = ProgressEvent(
                run_id,
                state.sequence,
                kind,
                message,
                data=data or {},
            )
            listeners = tuple(self._listeners.values())
        for listener in listeners:
            try:
                listener(progress)
            except Exception:
                continue

    def _set(self, run_id: str, **values) -> None:
        with self._lock:
            state = self._state(run_id)
            for name, value in values.items():
                setattr(state, name, value)

    def _state(self, run_id: str) -> _RunState:
        try:
            return self._runs[run_id]
        except KeyError as exc:
            raise KeyError(f"Run 不存在: {run_id}") from exc

    @staticmethod
    def _snapshot(state: _RunState) -> RunSnapshot:
        return RunSnapshot(
            id=state.id,
            status=state.status,
            request=state.request,
            workflow_id=state.workflow_id,
            workflow_state_id=state.workflow_state_id,
            assignment_id=state.assignment_id,
            work_role_id=state.work_role_id,
            workspace_id=state.workspace_id,
            cognitive_policy_id=state.cognitive_policy_id,
            profession_ids=state.profession_ids,
            agent_id=state.agent_id,
            stage=state.stage,
            current_capability=state.current_capability,
            token_usage=state.token_usage,
            answer=state.answer,
            error=state.error,
        )

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("RunService 已关闭")
