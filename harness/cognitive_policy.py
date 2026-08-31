"""Structured cognitive bias presets with no runtime authority."""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


_AXES = (("I", "E"), ("S", "N"), ("T", "F"), ("J", "P"))
_AUTHORITY_FIELDS = {
    "capabilities",
    "grants",
    "memory_scopes",
    "permissions",
    "skills",
    "tools",
    "workflow",
    "workspace",
}


@dataclass(frozen=True)
class CognitivePolicy:
    id: str
    axes: tuple[str, str, str, str]
    parameters: Mapping[str, Mapping[str, float]]

    def value(self, section: str, name: str, default: float = 0.0) -> float:
        return float(self.parameters.get(section, {}).get(name, default))

    def as_context(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "axes": list(self.axes),
            "parameters": {
                section: dict(values)
                for section, values in self.parameters.items()
            },
            "boundary": (
                "CognitivePolicy 只改变合法候选信息的处理和排序偏置；"
                "不得扩大 Memory scope、Skill、Capability 或权限。"
            ),
        }

    def compile_model_projection(self) -> dict[str, str]:
        """Compile parameter bias into authority-free operational guidance."""

        precedent_bias = sum(
            (
                self.value("memory", "precedent_weight", 0.5),
                self.value("cognition", "verification_weight", 0.5),
                self.value("cognition", "planning_weight", 0.5),
            )
        ) / 3
        exploration_bias = sum(
            (
                self.value("memory", "novelty_weight", 0.5),
                self.value("memory", "cross_domain_weight", 0.5),
                self.value("cognition", "exploration_weight", 0.5),
            )
        ) / 3
        if precedent_bias >= exploration_bias:
            return {
                "direction": "risk",
                "reasoning_guidance": (
                    "When multiple viable approaches exist, initially give more "
                    "weight to established failure modes and verification evidence "
                    "before converging."
                ),
            }
        return {
            "direction": "exploration",
            "reasoning_guidance": (
                "When multiple viable approaches exist, initially give more "
                "weight to novel alternatives and flexible designs before "
                "converging."
            ),
        }

    def summary(self) -> dict[str, Any]:
        return {"id": self.id, "axes": list(self.axes)}


class CognitivePolicyRegistry:
    def __init__(self, policies: tuple[CognitivePolicy, ...]):
        self._policies = {policy.id: policy for policy in policies}
        if len(self._policies) != len(policies):
            raise ValueError("CognitivePolicy id 不允许重复")

    def get(self, policy_id: str) -> CognitivePolicy:
        try:
            return self._policies[policy_id.upper()]
        except KeyError as exc:
            raise KeyError(f"CognitivePolicy 不存在: {policy_id}") from exc

    def available(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            self._policies[key].summary() for key in sorted(self._policies)
        )


def load_cognitive_policies(
    path: str | Path | None = None,
) -> CognitivePolicyRegistry:
    source = Path(path) if path else (
        Path(__file__).resolve().parent.parent
        / "config"
        / "cognitive_policies.json"
    )
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 CognitivePolicy 配置: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("CognitivePolicy 配置必须是对象")
    forbidden = sorted(_AUTHORITY_FIELDS.intersection(raw))
    if forbidden:
        raise ValueError(
            "CognitivePolicy 不能声明运行控制字段: "
            + ", ".join(forbidden)
        )
    base = _parameter_map(raw.get("base"), "base")
    deltas_raw = raw.get("axis_deltas")
    if not isinstance(deltas_raw, dict):
        raise ValueError("axis_deltas 必须是对象")
    expected = {choice for axis in _AXES for choice in axis}
    if set(deltas_raw) != expected:
        raise ValueError("axis_deltas 必须完整声明 I/E/S/N/T/F/J/P")
    deltas = {
        axis: _parameter_map(value, f"axis_deltas.{axis}", allow_negative=True)
        for axis, value in deltas_raw.items()
    }
    policies = tuple(
        CognitivePolicy(
            "".join(axes),
            axes,
            _freeze_parameters(_compose(base, (deltas[axis] for axis in axes))),
        )
        for axes in product(*_AXES)
    )
    return CognitivePolicyRegistry(policies)


def _parameter_map(
    value: Any, label: str, *, allow_negative: bool = False
) -> dict[str, dict[str, float]]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label} 必须是非空对象")
    result: dict[str, dict[str, float]] = {}
    for section, entries in value.items():
        if not isinstance(section, str) or not isinstance(entries, dict):
            raise ValueError(f"{label} section 无效")
        values: dict[str, float] = {}
        for name, raw_number in entries.items():
            if (
                not isinstance(name, str)
                or not isinstance(raw_number, (int, float))
                or isinstance(raw_number, bool)
            ):
                raise ValueError(f"{label}.{section} 参数无效")
            number = float(raw_number)
            if not allow_negative and not 0 <= number <= 1:
                raise ValueError(f"{label}.{section}.{name} 必须在 0 到 1 之间")
            if allow_negative and not -1 <= number <= 1:
                raise ValueError(f"{label}.{section}.{name} delta 必须在 -1 到 1 之间")
            values[name] = number
        result[section] = values
    return result


def _compose(
    base: dict[str, dict[str, float]],
    deltas: Any,
) -> dict[str, dict[str, float]]:
    result = {section: dict(values) for section, values in base.items()}
    for delta in deltas:
        for section, values in delta.items():
            target = result.setdefault(section, {})
            for name, value in values.items():
                target[name] = max(0.0, min(1.0, target.get(name, 0.0) + value))
    return result


def _freeze_parameters(
    value: dict[str, dict[str, float]],
) -> Mapping[str, Mapping[str, float]]:
    return MappingProxyType(
        {
            section: MappingProxyType(dict(entries))
            for section, entries in value.items()
        }
    )
