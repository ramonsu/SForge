import unittest

from harness.events import EventLogger, EventType, RuntimeEvent


class RuntimeEventTests(unittest.TestCase):
    def test_event_data_is_copied_frozen_and_serializable(self):
        source = {"status": "success", "nested": {"count": 1}}
        event = RuntimeEvent(
            type=EventType.ACTION_COMPLETED,
            trace_id="trace-1",
            data=source,
        )
        source["status"] = "changed"
        source["nested"]["count"] = 9
        self.assertEqual("success", event.data["status"])
        self.assertEqual(1, event.data["nested"]["count"])
        with self.assertRaises(TypeError):
            event.data["status"] = "changed"
        serialized = event.as_dict()
        serialized["data"]["nested"]["count"] = 10
        self.assertEqual(1, event.data["nested"]["count"])

    def test_event_rejects_non_json_payload(self):
        with self.assertRaisesRegex(TypeError, "不支持类型"):
            RuntimeEvent(
                type=EventType.ERROR,
                trace_id="trace-1",
                data={"exception": RuntimeError("boom")},
            )


class EventLoggerTests(unittest.TestCase):
    def test_subscribers_are_realtime_isolated_and_removable(self):
        logger = EventLogger()
        received = []
        unsubscribe = logger.subscribe(received.append)
        logger.subscribe(
            lambda _: (_ for _ in ()).throw(RuntimeError("listener failed"))
        )
        first = logger.emit(
            RuntimeEvent(type=EventType.AGENT_CREATED, trace_id="trace")
        )
        self.assertEqual([first], received)
        unsubscribe()
        logger.emit(
            RuntimeEvent(type=EventType.CONTEXT_BUILT, trace_id="trace")
        )
        self.assertEqual([first], received)

    def test_sequence_capacity_and_filters_are_deterministic(self):
        logger = EventLogger(max_events=3)
        emitted = []
        for index, event_type in enumerate(
            (
                EventType.AGENT_CREATED,
                EventType.CONTEXT_BUILT,
                EventType.ACTION_COMPLETED,
                EventType.ERROR,
            )
        ):
            emitted.append(
                logger.emit(
                    RuntimeEvent(
                        type=event_type,
                        trace_id="trace-a" if index < 3 else "trace-b",
                        agent_id="agent-a" if index != 3 else "agent-b",
                    )
                )
            )

        self.assertEqual([1, 2, 3, 4], [event.sequence for event in emitted])
        self.assertEqual(
            [2, 3, 4], [event.sequence for event in logger.recent(10)]
        )
        self.assertEqual(
            [2, 3],
            [
                event.sequence
                for event in logger.recent(10, agent_id="agent-a")
            ],
        )
        self.assertEqual(
            [3],
            [
                event.sequence
                for event in logger.recent(
                    10, event_types=(EventType.ACTION_COMPLETED,)
                )
            ],
        )

    def test_invalid_logger_configuration_fails_explicitly(self):
        with self.assertRaises(ValueError):
            EventLogger(0)
        logger = EventLogger()
        with self.assertRaises(ValueError):
            logger.recent(0)
        with self.assertRaisesRegex(ValueError, "sequence"):
            logger.emit(
                RuntimeEvent(
                    type=EventType.ERROR,
                    trace_id="trace",
                    sequence=2,
                )
            )


if __name__ == "__main__":
    unittest.main()
