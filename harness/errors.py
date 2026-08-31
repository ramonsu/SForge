"""Explicit SForge runtime boundary errors."""


class SForgeError(RuntimeError):
    pass


class AgentNotFoundError(SForgeError):
    pass


class InvalidAgentStateError(SForgeError):
    pass


class CapabilityNotFoundError(SForgeError):
    pass


class InvalidActionArgumentsError(SForgeError):
    pass


class InvalidDecisionError(SForgeError):
    pass


class DecisionProtocolError(InvalidDecisionError):
    """A Decision could not cross the existing structured protocol boundary."""

    def __init__(self, message: str, protocol: dict | None = None):
        super().__init__(message)
        self.protocol = dict(protocol or {})


class JSONModePromptConfigurationError(SForgeError):
    """JSON-object mode was requested without a compatible prompt contract."""


class LLMProviderError(SForgeError):
    """A provider rejected or failed an LLM request before a usable response."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_type: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type


class AgentWorkerError(SForgeError):
    """Structured error propagated across the disposable worker boundary."""

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        error_type: str,
        status_code: int | None = None,
        error_message: str | None = None,
    ):
        super().__init__(message)
        self.stage = stage
        self.error_type = error_type
        self.status_code = status_code
        self.error_message = error_message or message


class WorkflowNotFoundError(SForgeError):
    pass


class InvalidWorkflowStateError(SForgeError):
    pass


class CognitiveProfileNotFoundError(SForgeError):
    pass


class InvalidCognitiveProfileError(SForgeError):
    pass


class WorkRoleNotFoundError(SForgeError):
    pass


class InvalidWorkRoleError(SForgeError):
    pass


class InvalidWorkAssignmentError(SForgeError):
    pass


class InvalidResourceBindingError(SForgeError):
    pass


class InvalidIdentityError(SForgeError):
    pass


class MemoryProviderError(SForgeError):
    pass


class ContextResolutionError(SForgeError):
    pass
