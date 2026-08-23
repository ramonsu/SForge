import json


class FakeSupervisor:
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
        contract = json.loads(messages[0]["content"])
        if contract.get("purpose") == "persona_response_formatting":
            self.presentation_calls.append((process_id, messages))
            if self.presentation_responses:
                response = self.presentation_responses.pop(0)
                if isinstance(response, Exception):
                    raise response
                return response
            return contract["presentation_context"]["draft_answer"]
        self.action_calls.append((process_id, messages))
        if not self.responses:
            return json.dumps({"type": "final", "content": "done"})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def close(self):
        return None
