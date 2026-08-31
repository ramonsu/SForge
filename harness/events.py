"""Immutable, privacy-safe observations emitted by the SForge runtime."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from threading import RLock
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping


class EventType(str, Enum):
    AGENT_CREATED = "agent_created"
    WORKFLOW_REQUESTED = "workflow_requested"
    WORKFLOW_ADMISSION_COMPLETED = "workflow_admission_completed"
    WORK_ASSIGNMENT_REQUESTED = "work_assignment_requested"
    WORK_ASSIGNMENT_ADMISSION_COMPLETED = (
        "work_assignment_admission_completed"
    )
    WORK_ASSIGNMENT_ENDED = "work_assignment_ended"
    RESOURCE_BINDING_REQUESTED = "resource_binding_requested"
    RESOURCE_BINDING_COMPLETED = "resource_binding_completed"
    CONTEXT_BUILT = "context_built"
    REASONING_STARTED = "reasoning_started"
    REASONING_COMPLETED = "reasoning_completed"
    CAPABILITY_REQUESTED = "capability_requested"
    CAPABILITY_COMPLETED = "capability_completed"
    ACTION_COMPLETED = "action_completed"
    ERROR = "error"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _freeze_json(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        frozen = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("RuntimeEvent data 的键必须是字符串")
            frozen[key] = _freeze_json(item)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    raise TypeError(f"RuntimeEvent data 不支持类型: {type(value).__name__}")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True)
class RuntimeEvent:
    """One diagnostic fact. It is never authoritative runtime state."""

    type: EventType
    trace_id: str
    sequence: int = 0
    timestamp: datetime = field(default_factory=_utc_now)
    agent_id: str | None = None
    request_id: str | None = None
    data: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        if not isinstance(self.type, EventType):
            raise TypeError("RuntimeEvent type 必须是 EventType")
        if not isinstance(self.trace_id, str) or not self.trace_id.strip():
            raise ValueError("RuntimeEvent trace_id 不能为空")
        if self.sequence < 0:
            raise ValueError("RuntimeEvent sequence 不能为负数")
        if self.timestamp.tzinfo is None:
            raise ValueError("RuntimeEvent timestamp 必须包含时区")
        object.__setattr__(
            self, "timestamp", self.timestamp.astimezone(timezone.utc)
        )
        object.__setattr__(self, "data", _freeze_json(dict(self.data)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "type": self.type.value,
            "timestamp": self.timestamp.isoformat(),
            "trace_id": self.trace_id,
            "agent_id": self.agent_id,
            "request_id": self.request_id,
            "data": _thaw_json(self.data),
        }


class EventLogger:
    """Thread-safe bounded event history for one Harness lifetime."""

    def __init__(self, max_events: int = 1000):
        if max_events < 1:
            raise ValueError("max_events 必须大于零")
        self._events: deque[RuntimeEvent] = deque(maxlen=max_events)
        self._sequence = 0
        self._lock = RLock()
        self._listeners: dict[int, Callable[[RuntimeEvent], None]] = {}
        self._listener_sequence = 0

    def emit(self, event: RuntimeEvent) -> RuntimeEvent:
        if event.sequence != 0:
            raise ValueError("RuntimeEvent sequence 只能由 EventLogger 分配")
        with self._lock:
            self._sequence += 1
            recorded = replace(event, sequence=self._sequence)
            self._events.append(recorded)
            listeners = tuple(self._listeners.values())
        for listener in listeners:
            try:
                listener(recorded)
            except Exception:
                continue
        return recorded

    def subscribe(
        self, listener: Callable[[RuntimeEvent], None]
    ) -> Callable[[], None]:
        if not callable(listener):
            raise TypeError("Event listener 必须可调用")
        with self._lock:
            self._listener_sequence += 1
            listener_id = self._listener_sequence
            self._listeners[listener_id] = listener

        def unsubscribe() -> None:
            with self._lock:
                self._listeners.pop(listener_id, None)

        return unsubscribe

    def recent(
        self,
        limit: int = 20,
        *,
        agent_id: str | None = None,
        trace_id: str | None = None,
        event_types: Iterable[EventType] | None = None,
    ) -> tuple[RuntimeEvent, ...]:
        if limit < 1:
            raise ValueError("Event limit 必须大于零")
        allowed_types = set(event_types) if event_types is not None else None
        with self._lock:
            events = tuple(self._events)
        filtered = (
            event
            for event in events
            if (agent_id is None or event.agent_id == agent_id)
            and (trace_id is None or event.trace_id == trace_id)
            and (allowed_types is None or event.type in allowed_types)
        )
        return tuple(filtered)[-limit:]
