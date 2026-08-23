"""Explicit SForge V1 boundary errors."""


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


class WorkflowNotFoundError(SForgeError):
    pass


class InvalidWorkflowStateError(SForgeError):
    pass


class MemoryProviderError(SForgeError):
    pass


class ContextResolutionError(SForgeError):
    pass
