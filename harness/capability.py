"""Generic Capability registry, validation and admission contracts."""

from __future__ import annotations

from typing import Any, Callable, Protocol

from harness.errors import CapabilityNotFoundError, InvalidActionArgumentsError
from harness.models import ActionRequest, AdmissionDecision, CapabilityDescriptor, RuntimeState


class Capability(Protocol):
    @property
    def descriptor(self) -> CapabilityDescriptor: ...

    def invoke(self, arguments: dict[str, Any]) -> Any: ...


class AdmissionPolicy(Protocol):
    def authorize(
        self,
        runtime_state: RuntimeState,
        request: ActionRequest,
        capability: CapabilityDescriptor,
    ) -> AdmissionDecision: ...


class FunctionCapability:
    def __init__(self, descriptor: CapabilityDescriptor, handler: Callable[[dict[str, Any]], Any]):
        self._descriptor = descriptor
        self._handler = handler

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, arguments: dict[str, Any]) -> Any:
        return self._handler(arguments)


class CapabilityRegistry:
    def __init__(self):
        self._capabilities: dict[str, Capability] = {}

    def register(self, capability: Capability) -> None:
        capability_id = capability.descriptor.id
        if not capability_id.strip():
            raise ValueError("Capability id 不能为空")
        if capability_id in self._capabilities:
            raise ValueError(f"Capability 已存在: {capability_id}")
        self._capabilities[capability_id] = capability

    def get(self, capability_id: str) -> Capability:
        try:
            return self._capabilities[capability_id]
        except KeyError as exc:
            raise CapabilityNotFoundError(f"Capability 不存在: {capability_id}") from exc

    def descriptors(self, capability_ids: frozenset[str] | None = None) -> tuple[CapabilityDescriptor, ...]:
        ids = sorted(capability_ids or self._capabilities)
        return tuple(self.get(capability_id).descriptor for capability_id in ids)

    def validate_input(self, request: ActionRequest) -> None:
        descriptor = self.get(request.capability_id).descriptor
        self._validate_schema(request.arguments, descriptor.input_schema, "input")

    def validate_output(self, capability_id: str, output: Any) -> None:
        schema = self.get(capability_id).descriptor.output_schema
        if schema:
            self._validate_schema(output, schema, "output")

    @classmethod
    def _validate_schema(cls, value: Any, schema: dict[str, Any], label: str) -> None:
        expected = schema.get("type")
        if expected and not cls._matches_type(value, expected):
            raise InvalidActionArgumentsError(f"Capability {label} 类型应为 {expected}")
        if expected != "object":
            return
        missing = set(schema.get("required", [])) - set(value)
        if missing:
            raise InvalidActionArgumentsError(
                f"Capability {label} 缺少字段: {', '.join(sorted(missing))}"
            )
        for key, child_schema in schema.get("properties", {}).items():
            if key in value:
                cls._validate_schema(value[key], child_schema, f"{label}.{key}")

    @staticmethod
    def _matches_type(value: Any, expected: str) -> bool:
        mapping = {
            "object": dict,
            "array": list,
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "null": type(None),
        }
        python_type = mapping.get(expected)
        if python_type is None:
            raise InvalidActionArgumentsError(f"不支持的 schema type: {expected}")
        return isinstance(value, python_type) and not (
            expected in {"integer", "number"} and isinstance(value, bool)
        )


class DefaultAdmissionPolicy:
    def authorize(
        self,
        runtime_state: RuntimeState,
        request: ActionRequest,
        capability: CapabilityDescriptor,
    ) -> AdmissionDecision:
        if capability.id not in runtime_state.allowed_capabilities:
            return AdmissionDecision(False, f"Capability 当前不可用: {capability.id}")
        return AdmissionDecision(True)
