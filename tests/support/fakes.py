import json

from harness.models import ReasoningResponse


class FakeSupervisor:
    """Minimal ReasoningProcess double; the real Runtime remains in the test."""

    def __init__(self, responses=None, *, presentation_responses=None):
        self.spawned = []
        self.terminated = []
        self.action_calls = []
        self.presentation_calls = []
        self.responses = list(responses or [])
        self.presentation_responses = list(presentation_responses or [])

    def spawn(self, runtime_state_id):
        process_id = f"process-{len(self.spawned) + 1}"
        self.spawned.append((process_id, runtime_state_id))
        return process_id

    def terminate(self, process_id):
        if process_id:
            self.terminated.append(process_id)

    def reason(self, process_id, messages):
        system_message = str(messages[0]["content"])
        if system_message.startswith("Formatting instructions:"):
            self.presentation_calls.append((process_id, messages))
            if self.presentation_responses:
                response = self.presentation_responses.pop(0)
                if isinstance(response, Exception):
                    raise response
                if isinstance(response, ReasoningResponse):
                    return response
                return ReasoningResponse(str(response))
            request = json.loads(messages[1]["content"])
            return ReasoningResponse(request["draft"])
        contract = json.loads(system_message)
        self.action_calls.append((process_id, messages))
        if not self.responses:
            return ReasoningResponse(
                json.dumps({"type": "final", "content": "done"})
            )
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, ReasoningResponse):
            return response
        return ReasoningResponse(str(response))

    def close(self):
        return None
