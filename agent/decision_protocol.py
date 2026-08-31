"""Observable, minimal decoding for the existing Agent Decision protocol.

The compatibility pass accepts literal control characters inside JSON strings.
It is deliberately reported as a fallback rather than being treated as valid
structured output, so experiments can distinguish runtime completion from
protocol conformance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DecisionParseResult:
    payload: dict[str, Any] | None
    structured_decision_valid: bool
    decision_parse_mode: str
    decision_parse_error: str | None
    fallback_used: bool

    def instrumentation(self) -> dict[str, Any]:
        return {
            "structured_decision_valid": self.structured_decision_valid,
            "decision_parse_mode": self.decision_parse_mode,
            "decision_parse_error": self.decision_parse_error,
            "fallback_used": self.fallback_used,
        }


def parse_decision_payload(raw: str) -> DecisionParseResult:
    """Decode one Decision response without changing the protocol schema.

    Markdown JSON fences are transport decoration and therefore remain valid.
    After strict parsing, at most one bounded repair is attempted: literal
    control characters inside strings, or one missing final top-level brace.
    A repaired payload remains an observable protocol failure.
    """

    candidate, fenced = _strip_json_fence(raw.strip())
    strict_mode = "fenced_json" if fenced else "strict_json"
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as strict_error:
        strict_error_text = _format_error(strict_error)
        repaired = None
        repair_mode = None
        repair_error_text = None
        if strict_error.msg == "Invalid control character at":
            try:
                repaired = json.loads(candidate, strict=False)
                repair_mode = (
                    "fenced_control_character_repair"
                    if fenced
                    else "control_character_repair"
                )
            except json.JSONDecodeError as repair_error:
                repair_error_text = _format_error(repair_error)
        else:
            closed = _repair_one_missing_closing_brace(candidate)
            if closed is not None:
                try:
                    repaired = json.loads(closed)
                    repair_mode = (
                        "fenced_missing_closing_brace_repair"
                        if fenced
                        else "missing_closing_brace_repair"
                    )
                except json.JSONDecodeError as repair_error:
                    repair_error_text = _format_error(repair_error)
        if repaired is None:
            suffix = (
                f"; bounded repair: {repair_error_text}"
                if repair_error_text
                else "; bounded repair: not applicable"
            )
            return DecisionParseResult(
                payload=None,
                structured_decision_valid=False,
                decision_parse_mode="invalid_json",
                decision_parse_error=f"strict: {strict_error_text}{suffix}",
                fallback_used=False,
            )
        if not isinstance(repaired, dict):
            return DecisionParseResult(
                payload=None,
                structured_decision_valid=False,
                decision_parse_mode="invalid_shape",
                decision_parse_error="Decision JSON 顶层必须是对象",
                fallback_used=True,
            )
        return DecisionParseResult(
            payload=repaired,
            structured_decision_valid=False,
            decision_parse_mode=str(repair_mode),
            decision_parse_error=strict_error_text,
            fallback_used=True,
        )
    if not isinstance(payload, dict):
        return DecisionParseResult(
            payload=None,
            structured_decision_valid=False,
            decision_parse_mode="invalid_shape",
            decision_parse_error="Decision JSON 顶层必须是对象",
            fallback_used=False,
        )
    return DecisionParseResult(
        payload=payload,
        structured_decision_valid=True,
        decision_parse_mode=strict_mode,
        decision_parse_error=None,
        fallback_used=False,
    )


def _strip_json_fence(raw: str) -> tuple[str, bool]:
    if not raw.startswith("```"):
        return raw, False
    lines = raw.splitlines()
    if len(lines) < 3 or not lines[-1].strip().startswith("```"):
        return raw, False
    opening = lines[0].strip().casefold()
    if opening not in {"```", "```json"}:
        return raw, False
    return "\n".join(lines[1:-1]).strip(), True


def _format_error(error: json.JSONDecodeError) -> str:
    return f"{error.msg} (line {error.lineno}, column {error.colno})"


def _repair_one_missing_closing_brace(candidate: str) -> str | None:
    """Repair only one truncated top-level object delimiter."""

    stripped = candidate.rstrip()
    if (
        stripped.startswith("{")
        and not stripped.endswith("}")
        and stripped.count("{") == stripped.count("}") + 1
        and stripped.endswith(('"', "]"))
    ):
        return stripped + "}"
    return None
