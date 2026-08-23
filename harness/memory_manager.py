"""Replaceable MemoryProvider contract and minimal V1 implementations."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Protocol

from harness.errors import MemoryProviderError
from harness.models import MemoryRecord


class MemoryProvider(Protocol):
    def write(self, record: MemoryRecord) -> MemoryRecord: ...

    def retrieve(
        self, *, scope: str, query: str | None = None, limit: int = 20
    ) -> list[MemoryRecord]: ...

    def get(self, memory_id: str) -> MemoryRecord | None: ...

    def close(self) -> None: ...


class InMemoryMemoryProvider:
    def __init__(self):
        self._records: dict[str, MemoryRecord] = {}

    def write(self, record: MemoryRecord) -> MemoryRecord:
        _validate_record(record)
        if record.id in self._records:
            raise MemoryProviderError(f"Memory id 已存在: {record.id}")
        self._records[record.id] = record
        return record

    def retrieve(
        self, *, scope: str, query: str | None = None, limit: int = 20
    ) -> list[MemoryRecord]:
        if limit < 1:
            return []
        records = [item for item in self._records.values() if item.scope == scope]
        if query:
            needle = query.casefold()
            records = [
                item
                for item in records
                if needle in item.kind.casefold() or needle in item.content.casefold()
            ]
        return sorted(records, key=lambda item: item.created_at)[-limit:]

    def get(self, memory_id: str) -> MemoryRecord | None:
        return self._records.get(memory_id)

    def close(self) -> None:
        return None


class SQLiteMemoryProvider:
    def __init__(self, database_path: str | Path | None = None):
        path = Path(database_path) if database_path else (
            Path(__file__).resolve().parent.parent / "memory" / "runtime.sqlite3"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = RLock()
        with self._connection:
            self._connection.execute(
                """CREATE TABLE IF NOT EXISTS sforge_memories (
                    id TEXT PRIMARY KEY,
                    scope TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    importance REAL,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )"""
            )

    def write(self, record: MemoryRecord) -> MemoryRecord:
        _validate_record(record)
        try:
            with self._lock, self._connection:
                self._connection.execute(
                    """INSERT INTO sforge_memories
                       (id, scope, kind, content, importance, metadata, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        record.id,
                        record.scope,
                        record.kind,
                        record.content,
                        record.importance,
                        json.dumps(record.metadata, ensure_ascii=False),
                        record.created_at.isoformat(),
                    ),
                )
        except sqlite3.Error as exc:
            raise MemoryProviderError(f"Memory 写入失败: {exc}") from exc
        return record

    def retrieve(
        self, *, scope: str, query: str | None = None, limit: int = 20
    ) -> list[MemoryRecord]:
        if limit < 1:
            return []
        sql = "SELECT * FROM sforge_memories WHERE scope = ?"
        parameters: list[object] = [scope]
        if query:
            sql += " AND (LOWER(kind) LIKE ? OR LOWER(content) LIKE ?)"
            pattern = f"%{query.casefold()}%"
            parameters.extend((pattern, pattern))
        sql += " ORDER BY created_at DESC LIMIT ?"
        parameters.append(limit)
        try:
            with self._lock:
                rows = self._connection.execute(sql, parameters).fetchall()
        except sqlite3.Error as exc:
            raise MemoryProviderError(f"Memory 读取失败: {exc}") from exc
        return [self._row(row) for row in reversed(rows)]

    def get(self, memory_id: str) -> MemoryRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM sforge_memories WHERE id = ?", (memory_id,)
            ).fetchone()
        return self._row(row) if row else None

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    @staticmethod
    def _row(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"],
            scope=row["scope"],
            kind=row["kind"],
            content=row["content"],
            importance=row["importance"],
            metadata=json.loads(row["metadata"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )


def _validate_record(record: MemoryRecord) -> None:
    if not record.scope.strip() or not record.kind.strip():
        raise MemoryProviderError("Memory scope 和 kind 不能为空")
    if record.importance is not None and not 0 <= record.importance <= 1:
        raise MemoryProviderError("Memory importance 必须在 0 到 1 之间")
