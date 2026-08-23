import tempfile
import unittest
from pathlib import Path

from capabilities import builtins
from harness.capability import CapabilityRegistry
from harness.errors import InvalidActionArgumentsError
from harness.memory_manager import (
    InMemoryMemoryProvider,
    SQLiteMemoryProvider,
)
from harness.models import ActionRequest, MemoryRecord


class MemoryProviderContractTests(unittest.TestCase):
    def test_in_memory_and_sqlite_share_write_retrieve_get_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            providers = (
                InMemoryMemoryProvider(),
                SQLiteMemoryProvider(Path(temporary) / "memory.sqlite3"),
            )
            try:
                for provider in providers:
                    record = MemoryRecord(
                        scope="task:one",
                        kind="note",
                        content="hello",
                        importance=0.5,
                        metadata={"source": "test"},
                    )
                    self.assertIs(record, provider.write(record))
                    self.assertEqual(record, provider.get(record.id))
                    self.assertEqual(
                        [record],
                        provider.retrieve(scope="task:one", query="hello"),
                    )
                    self.assertEqual(
                        [], provider.retrieve(scope="task:other")
                    )
            finally:
                for provider in providers:
                    provider.close()

    def test_memory_importance_is_validated(self):
        provider = InMemoryMemoryProvider()
        with self.assertRaisesRegex(Exception, "importance"):
            provider.write(
                MemoryRecord(
                    scope="core",
                    kind="invalid",
                    content="x",
                    importance=1.5,
                )
            )


class CapabilityRegistryTests(unittest.TestCase):
    def test_registry_rejects_duplicates_and_validates_schema(self):
        with tempfile.TemporaryDirectory() as workspace:
            echo = builtins(workspace)[0]
            registry = CapabilityRegistry()
            registry.register(echo)
            with self.assertRaisesRegex(ValueError, "已存在"):
                registry.register(echo)
            request = ActionRequest("echo", {})
            with self.assertRaises(InvalidActionArgumentsError):
                registry.validate_input(request)

    def test_workspace_capabilities_read_write_and_block_escape(self):
        with tempfile.TemporaryDirectory() as workspace:
            registry = CapabilityRegistry()
            for capability in builtins(workspace):
                registry.register(capability)
            write = registry.get("write_text")
            result = write.invoke({"path": "data/hello.txt", "content": "hello"})
            self.assertEqual(5, result["characters"])
            self.assertEqual(
                "hello",
                registry.get("read_text").invoke({"path": "data/hello.txt"}),
            )
            outside = str(Path(workspace).resolve().parent / "outside.txt")
            with self.assertRaisesRegex(ValueError, "workspace"):
                write.invoke({"path": outside, "content": "blocked"})


if __name__ == "__main__":
    unittest.main()
