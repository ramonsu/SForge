import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import main


class _Snapshot:
    def as_dict(self):
        return {
            "agent": {"id": "agent-1", "status": "completed"},
            "recent_events": [{"type": "context_built"}],
        }


class _Application:
    def __init__(self):
        self.closed = False

    def handle(self, user_input):
        return f"answer:{user_input}"

    def inspect(self):
        return _Snapshot()

    def close(self):
        self.closed = True


class InspectorCliTests(unittest.TestCase):
    def test_default_entrypoint_launches_desktop_adapter(self):
        with patch("ui.desktop.launch_desktop", return_value=0) as launch:
            self.assertEqual(0, main.main([]))
        launch.assert_called_once_with()

    def test_inspect_flag_prints_snapshot_without_changing_handle(self):
        application = _Application()
        output = io.StringIO()
        with patch.object(main, "AgentApplication", return_value=application), patch.object(
            main.signal, "signal"
        ), patch("builtins.input", side_effect=["hello", "exit"]), redirect_stdout(
            output
        ):
            with self.assertRaises(SystemExit):
                main.main(["--inspect"])

        rendered = output.getvalue()
        self.assertIn("Agent：answer:hello", rendered)
        self.assertIn("Runtime Inspector", rendered)
        self.assertIn('"status": "completed"', rendered)
        self.assertTrue(application.closed)


if __name__ == "__main__":
    unittest.main()
