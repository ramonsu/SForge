"""Minimal built-in capabilities registered by the application layer."""

from __future__ import annotations

from pathlib import Path

from harness.capability import FunctionCapability
from harness.models import CapabilityDescriptor


def builtins(workspace_root: str | Path) -> tuple[FunctionCapability, ...]:
    root = Path(workspace_root).resolve()

    def safe_path(value: str) -> Path:
        candidate = Path(value)
        resolved = (
            candidate.resolve()
            if candidate.is_absolute()
            else (root / candidate).resolve()
        )
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError("路径必须位于 SForge workspace 内") from exc
        return resolved

    def read_text(arguments: dict) -> str:
        return safe_path(arguments["path"]).read_text(encoding="utf-8")

    def write_text(arguments: dict) -> dict[str, object]:
        path = safe_path(arguments["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        content = arguments["content"]
        path.write_text(content, encoding="utf-8")
        return {
            "path": path.relative_to(root).as_posix(),
            "characters": len(content),
        }

    return (
        FunctionCapability(
            CapabilityDescriptor(
                id="echo",
                description="Return the provided text unchanged",
                input_schema={
                    "type": "object",
                    "required": ["text"],
                    "properties": {"text": {"type": "string"}},
                },
                output_schema={"type": "string"},
            ),
            lambda arguments: arguments["text"],
        ),
        FunctionCapability(
            CapabilityDescriptor(
                id="read_text",
                description="Read one UTF-8 text file inside the workspace",
                input_schema={
                    "type": "object",
                    "required": ["path"],
                    "properties": {"path": {"type": "string"}},
                },
                output_schema={"type": "string"},
            ),
            read_text,
        ),
        FunctionCapability(
            CapabilityDescriptor(
                id="write_text",
                description="Write one UTF-8 text file inside the workspace",
                input_schema={
                    "type": "object",
                    "required": ["path", "content"],
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                },
                output_schema={
                    "type": "object",
                    "required": ["path", "characters"],
                    "properties": {
                        "path": {"type": "string"},
                        "characters": {"type": "integer"},
                    },
                },
                side_effects=True,
            ),
            write_text,
        ),
    )
