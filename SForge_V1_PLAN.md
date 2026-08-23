# SForge V1 Development Plan

> Audience: Codex / implementation agent  
> Project: **SForge**  
> Stage: **V1 — Core Runtime Foundation**  
> Status: Architecture freeze for initial implementation

---

## 0. Executive Summary

SForge V1 is **not** intended to become a feature-rich agent framework.

The purpose of V1 is to implement the smallest stable runtime foundation that can support future long-running agent systems without forcing future concepts into the core prematurely.

The central architectural principle is:

> **SForge Harness provides stable runtime mechanisms and execution boundaries. It does not perform intelligence, business reasoning, or domain-specific orchestration.**

The long-lived entity is the **Harness Runtime**.

An **Agent Process** is a short-lived, homogeneous cognitive process created and controlled by the Harness. Agent processes do not own lifecycle, permissions, tools, memory stores, workflows, or other agents.

Skills, tools, browser integrations, coding utilities, robotics controllers, etc. are all future **Capability implementations**. They are important for what an Agent can do, but they are not the defining architecture of SForge V1.

V1 must prove this minimal loop:

```text
User Request
    ↓
Harness
    ↓
Create Agent Process
    ↓
Build Operational Context
    ↓
Agent reasons
    ↓
Structured ActionRequest or Final Answer
    ↓
Harness validates ActionRequest
    ↓
Capability execution
    ↓
ActionResult
    ↓
State / Memory update
    ↓
Agent continues or terminates
```

Everything in V1 should serve this loop.

---

# 1. Core Design Invariants

These invariants are architectural rules. Codex should treat violations as design defects.

## 1.1 Harness owns runtime, not intelligence

Harness responsibilities:

- create and terminate Agent Processes
- maintain runtime state
- assemble operational context
- enforce capability admission
- validate structured actions
- mediate memory access
- mediate workflow activation/state
- record execution results
- expose stable extension interfaces

Harness must NOT:

- decide how to solve the user's domain task
- implement planner/reviewer/researcher/coder semantics
- contain task-specific prompts
- interpret business meanings of memory fields
- directly embed browser, Git, robot, email, research, or coding logic

---

## 1.2 Agent is a process, not a role

There must be one general Agent Process abstraction.

Do NOT create classes such as:

```text
PlannerAgent
CoderAgent
ReviewerAgent
AuditorAgent
ResearchAgent
ManagerAgent
ChildAgent
```

If future workflows need these concepts, they must be represented through runtime configuration:

```text
Workflow State
+ Context View
+ Memory View
+ Capability View
+ Permission View
```

In V1 these views may be minimal, but the architecture must not block them.

---

## 1.3 No ontological parent/child agents

If one Agent later requests another Agent Process, the Harness remains the lifecycle owner.

Conceptually:

```text
Agent A
  ↓ request

Harness
  ↓ admission

Agent B
```

Agent B is not a different species and does not belong to Agent A.

V1 does not need multi-agent spawning yet, but its interfaces must not assume a permanent "main agent".

---

## 1.4 Workflow is optional

Simple requests must be able to run without a workflow.

Supported conceptual modes:

```text
Direct Mode
Workflow Mode
```

V1 only needs:

- Direct Mode
- one minimal optional Workflow mechanism

Workflow V1 should remain declarative and thin.

It must NOT become a hard-coded DAG engine or business orchestration script.

---

## 1.5 Capability is the core action abstraction

Harness Core should not care whether a capability is implemented by:

- Python function
- local tool
- remote API
- MCP tool
- shell command
- skill package
- another process
- robot adapter
- future embodied controller

Core sees only a stable capability contract.

Prefer:

```text
Capability
```

as the core abstraction.

"Skill" and "Tool" can later exist as higher-level module/package concepts, but SForge Core should not depend on their distinction.

---

## 1.6 Memory is persistent runtime state, but V1 stays simple

V1 memory must be intentionally small.

Do NOT implement yet:

- graph database
- memory centrality
- memory tier promotion
- automatic reflection
- Memory Lens weighting
- semantic consolidation pipeline
- personality evolution

But V1 memory interfaces must allow those to be added later without changing Agent Process or Harness call sites.

---

## 1.7 Harness knows schemas/contracts, not meanings

Example:

Harness may know:

```text
importance: float
confidence: float
type: enum
```

Harness may validate these fields.

Harness must NOT contain logic such as:

```python
if memory.importance > 0.8 and task == "research":
    promote_to_core()
```

Semantic policies belong to future policy/plugin layers.

---

## 1.8 External effects must cross the Harness boundary

An Agent Process must not directly invoke side-effecting capabilities.

Bad:

```python
agent.file_system.delete(...)
agent.robot.move(...)
```

Required:

```python
request = ActionRequest(...)
result = harness.execute_action(agent_id, request)
```

This is one of the defining SForge guarantees.

---

# 2. V1 Goals

V1 should provide a working, testable runtime skeleton with the following capabilities.

## 2.1 Required

1. Harness Runtime lifecycle
2. Agent Process creation
3. Agent Process termination
4. Runtime state tracking
5. Operational Context assembly
6. Basic Memory Provider contract
7. Basic in-memory or SQLite memory implementation
8. Capability registry
9. Capability admission / availability checks
10. Structured ActionRequest
11. Structured ActionResult
12. Capability invocation through Harness
13. Minimal Direct Mode
14. Minimal optional Workflow definition/state
15. Unit tests around all core boundaries
16. One end-to-end runtime test

---

## 2.2 Explicit non-goals

Do NOT implement these in V1 unless necessary only as a tiny stub/interface:

- full Skill marketplace
- browser automation
- Git automation
- coding agent ecosystem
- multi-agent orchestration
- autonomous subagents
- scheduler
- cron
- distributed runtime
- event sourcing platform
- robotics integration
- world model
- multimodal perception stack
- memory graph
- Memory Lens
- automatic core-memory promotion
- reflection engine
- Persona Registry
- long-term persona evolution
- complex Workflow DSL
- prompt role simulation framework
- workflow-generated-workflow
- self-modifying plugins
- production sandbox system

These are future modules, not V1 prerequisites.

---

# 3. Proposed V1 Architecture

```text
                    ┌─────────────────────────────┐
                    │         Application         │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │        SForge Harness       │
                    │                             │
                    │  Lifecycle                  │
                    │  Runtime State              │
                    │  Context Assembly           │
                    │  Memory Boundary            │
                    │  Capability Admission       │
                    │  Action Validation          │
                    │  Execution Boundary         │
                    └───────┬─────────┬───────────┘
                            │         │
                 ┌──────────┘         └───────────┐
                 ▼                                ▼
        ┌─────────────────┐              ┌─────────────────┐
        │  Agent Process  │              │ Memory Provider │
        │                 │              └─────────────────┘
        │ reason(context) │
        └────────┬────────┘
                 │ ActionRequest
                 ▼
        ┌─────────────────┐
        │ Capability API  │
        └────────┬────────┘
                 ▼
        ┌─────────────────┐
        │ Implementations │
        │ echo/read/write │
        └─────────────────┘
```

Workflow is optional:

```text
Harness
   │
   ├── Direct Runtime State
   │
   └── Workflow Runtime State
          ├── workflow_id
          ├── state_id
          ├── allowed_capabilities
          └── memory_scope
```

---

# 4. Core Domain Objects

Keep these objects small and explicit.

## 4.1 AgentProcess

Represents one short-lived reasoning process.

Suggested fields:

```python
@dataclass
class AgentProcess:
    id: str
    status: AgentStatus
    created_at: datetime
    runtime_state_id: str
    model_ref: str | None = None
```

AgentProcess must NOT own:

- CapabilityRegistry
- MemoryProvider
- WorkflowRegistry
- permissions engine
- child-agent list
- tool manager

It may reference runtime state IDs but must not become the god object.

---

## 4.2 AgentStatus

Suggested V1:

```text
CREATED
RUNNING
WAITING
COMPLETED
FAILED
TERMINATED
```

Keep transitions explicit and validated.

---

## 4.3 RuntimeState

Represents state owned by Harness.

Suggested:

```python
@dataclass
class RuntimeState:
    id: str
    agent_id: str
    task_id: str
    mode: Literal["direct", "workflow"]
    workflow_id: str | None
    workflow_state_id: str | None
    allowed_capabilities: frozenset[str]
    memory_scope: str
    metadata: dict[str, Any]
```

Do not overload RuntimeState with domain semantics.

---

## 4.4 OperationalContext

Use a structured object rather than immediately flattening everything into a prompt string.

Suggested:

```python
@dataclass(frozen=True)
class OperationalContext:
    system: dict[str, Any]
    task: dict[str, Any]
    runtime: dict[str, Any]
    memory: list["MemoryRecord"]
    capabilities: list["CapabilityDescriptor"]
    workflow: dict[str, Any] | None = None
```

Future versions may add:

- Memory View
- Embodiment Context
- Sensor Context
- artifacts
- multimodal content

Therefore avoid making `build_prompt()` the fundamental abstraction.

Preferred:

```text
ContextSpec
→ ContextResolver
→ OperationalContext
→ Model Adapter
```

Prompt rendering can remain an adapter concern.

---

## 4.5 ActionRequest

This is one of the most important V1 contracts.

Suggested:

```python
@dataclass(frozen=True)
class ActionRequest:
    capability_id: str
    arguments: dict[str, Any]
    request_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
```

Future-safe optional concepts may be reserved but not implemented:

- expected state version
- idempotency key
- risk metadata
- requested resources
- timeout policy

Do not add them unless needed now.

---

## 4.6 ActionResult

Suggested:

```python
@dataclass(frozen=True)
class ActionResult:
    request_id: str
    capability_id: str
    status: Literal["success", "rejected", "failed"]
    output: Any | None
    error: str | None
    metadata: dict[str, Any]
```

All external execution should return through this contract.

---

## 4.7 CapabilityDescriptor

Suggested:

```python
@dataclass(frozen=True)
class CapabilityDescriptor:
    id: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None
    side_effects: bool
    metadata: dict[str, Any]
```

Harness should treat description as opaque user/model-facing metadata.

---

## 4.8 Capability

Suggested protocol:

```python
class Capability(Protocol):
    @property
    def descriptor(self) -> CapabilityDescriptor:
        ...

    def invoke(self, arguments: dict[str, Any]) -> Any:
        ...
```

Future async support should be easy to add.

If async is already natural in the repository, prefer:

```python
async def invoke(...)
```

Do not build both sync and async frameworks unnecessarily.

---

## 4.9 MemoryRecord

V1 minimal form:

```python
@dataclass(frozen=True)
class MemoryRecord:
    id: str
    scope: str
    kind: str
    content: str
    importance: float | None
    created_at: datetime
    metadata: dict[str, Any]
```

Do not add graph edges or tiers yet unless already trivial in the repository.

However, `metadata` and stable IDs should make future migration possible.

---

# 5. Core Interfaces

Interfaces matter more than implementations in V1.

## 5.1 AgentRunner / ReasoningEngine

Separate Agent Process identity from model execution.

Suggested:

```python
class ReasoningEngine(Protocol):
    def reason(
        self,
        context: OperationalContext,
    ) -> "AgentDecision":
        ...
```

AgentDecision may contain either:

```text
FinalAnswer
ActionRequest
```

Example:

```python
AgentDecision = FinalAnswer | ActionRequest
```

Avoid letting raw LLM text directly trigger side effects.

---

## 5.2 ContextProvider

Suggested:

```python
class ContextProvider(Protocol):
    def provide(
        self,
        runtime_state: RuntimeState,
    ) -> dict[str, Any]:
        ...
```

A ContextResolver may combine multiple providers.

Future providers:

- task provider
- workflow provider
- memory provider
- repo provider
- robot/body provider
- sensor provider

Harness should combine structured context without understanding domain meaning.

---

## 5.3 MemoryProvider

Suggested:

```python
class MemoryProvider(Protocol):
    def write(self, record: MemoryRecord) -> None:
        ...

    def retrieve(
        self,
        *,
        scope: str,
        query: str | None = None,
        limit: int = 20,
    ) -> list[MemoryRecord]:
        ...
```

Optional:

```python
def get(memory_id: str) -> MemoryRecord | None
```

V1 implementation can be in-memory or SQLite.

Important future compatibility:

Later Memory Graph, vector retrieval, Memory Lens, tiering, promotion, consolidation, and reflection must be implementable behind this boundary or through additive interfaces.

Do not expose backend-specific queries to Agent Process.

---

## 5.4 CapabilityRegistry

Suggested:

```python
class CapabilityRegistry:
    def register(self, capability: Capability) -> None:
        ...

    def get(self, capability_id: str) -> Capability:
        ...

    def descriptors(self) -> list[CapabilityDescriptor]:
        ...
```

Registration should reject duplicate IDs unless explicitly versioned in the future.

---

## 5.5 AdmissionPolicy

V1 can be deliberately simple.

Suggested:

```python
class AdmissionPolicy(Protocol):
    def authorize(
        self,
        runtime_state: RuntimeState,
        request: ActionRequest,
        capability: CapabilityDescriptor,
    ) -> "AdmissionDecision":
        ...
```

V1 default rule:

```text
capability exists
AND capability_id is in runtime_state.allowed_capabilities
AND request schema is valid
```

Future policy plugins may add:

- risk limits
- resource limits
- human approval
- robot safety
- network policy
- filesystem constraints

Harness invokes policy; Harness Core must not hardcode domain policy.

---

## 5.6 WorkflowProvider / WorkflowDefinition

Keep V1 thin.

Suggested:

```python
@dataclass(frozen=True)
class WorkflowStateDefinition:
    id: str
    allowed_capabilities: frozenset[str]
    memory_scope: str
    context_sources: tuple[str, ...] = ()
```

```python
@dataclass(frozen=True)
class WorkflowDefinition:
    id: str
    initial_state: str
    states: dict[str, WorkflowStateDefinition]
```

Future additions:

- Memory View / Lens
- Permission View
- transition guards
- required schema extensions
- resource budgets
- agent-process requests
- role labels

Do not implement these now unless needed to preserve obvious extension fields.

---

# 6. Suggested Repository Structure

Adapt to the current repository where reasonable. Do not perform gratuitous file churn.

Suggested target:

```text
sforge/
├── __init__.py
│
├── core/
│   ├── harness.py
│   ├── models.py
│   ├── errors.py
│   └── lifecycle.py
│
├── agent/
│   ├── process.py
│   ├── decision.py
│   └── reasoning.py
│
├── context/
│   ├── models.py
│   ├── provider.py
│   └── resolver.py
│
├── memory/
│   ├── models.py
│   ├── provider.py
│   └── in_memory.py
│
├── capability/
│   ├── models.py
│   ├── base.py
│   ├── registry.py
│   ├── admission.py
│   └── builtin/
│       ├── echo.py
│       ├── read_text.py
│       └── write_text.py
│
├── workflow/
│   ├── models.py
│   ├── provider.py
│   └── registry.py
│
├── runtime/
│   ├── state.py
│   └── state_store.py
│
└── adapters/
    └── model/
        ├── base.py
        └── fake.py
```

Tests:

```text
tests/
├── unit/
│   ├── test_lifecycle.py
│   ├── test_runtime_state.py
│   ├── test_context.py
│   ├── test_memory.py
│   ├── test_capability_registry.py
│   ├── test_admission.py
│   └── test_workflow.py
│
└── integration/
    └── test_v1_runtime_loop.py
```

If the current repository already has equivalent modules, preserve them and refactor minimally instead of blindly recreating this tree.

---

# 7. Minimal Built-in Capabilities

V1 should include only enough capabilities to exercise the runtime.

Recommended:

## 7.1 echo

```text
echo(text) -> text
```

Purpose:

- deterministic
- no side effects
- validates ActionRequest routing

## 7.2 read_text

```text
read_text(path) -> text
```

Purpose:

- read-only external interaction
- capability admission
- error handling

Restrict to a test/workspace root if practical.

## 7.3 write_text

```text
write_text(path, content) -> result
```

Purpose:

- side-effect test
- permission/admission validation

Again restrict to a test/workspace root.

Do not add shell, browser, email, Git, or MCP in V1.

---

# 8. Harness Main API

The API should be small.

Conceptual interface:

```python
class Harness:
    def create_agent(
        self,
        task: TaskSpec,
        workflow_id: str | None = None,
    ) -> AgentProcess:
        ...

    def terminate_agent(
        self,
        agent_id: str,
        reason: str | None = None,
    ) -> None:
        ...

    def build_context(
        self,
        agent_id: str,
    ) -> OperationalContext:
        ...

    def execute_action(
        self,
        agent_id: str,
        request: ActionRequest,
    ) -> ActionResult:
        ...

    def step(
        self,
        agent_id: str,
    ) -> AgentDecision | ActionResult:
        ...
```

Whether `step()` remains in the final API depends on the existing design. The important point is:

- Harness owns the loop boundary.
- ReasoningEngine proposes.
- Harness validates.
- Capability executes only after admission.

Avoid building dozens of public methods.

---

# 9. Execution Semantics

The side-effect path must be explicit.

Required sequence:

```text
1. Confirm Agent exists and is active
2. Resolve RuntimeState
3. Resolve CapabilityDescriptor
4. Check capability is visible/allowed
5. Validate ActionRequest input schema
6. Ask AdmissionPolicy
7. Reject or execute
8. Convert execution result to ActionResult
9. Record minimal runtime event/state
10. Return ActionResult to Agent loop
```

No code path should allow an Agent Process to bypass steps 1–8.

---

# 10. Error Model

Define explicit core errors.

Suggested:

```text
AgentNotFound
InvalidAgentState
CapabilityNotFound
CapabilityNotAllowed
InvalidActionArguments
ActionRejected
CapabilityExecutionError
WorkflowNotFound
InvalidWorkflowState
MemoryError
ContextResolutionError
```

Do not swallow errors silently.

Where possible, Harness should convert execution failures into structured `ActionResult(status="failed")` while programmer/configuration defects may raise exceptions.

---

# 11. Implementation Milestones

Codex should implement V1 incrementally. Each milestone must leave the repository passing tests.

---

## Milestone 0 — Repository Audit

Before editing:

1. inspect current repository
2. identify already existing equivalents
3. map existing modules to this plan
4. avoid duplicating working abstractions
5. record architecture conflicts in a short implementation note

Acceptance:

- existing tests run
- current architecture is understood
- no unnecessary rewrite performed

---

## Milestone 1 — Freeze Core Models

Implement or normalize:

- AgentProcess
- AgentStatus
- RuntimeState
- OperationalContext
- ActionRequest
- ActionResult
- CapabilityDescriptor
- MemoryRecord
- WorkflowDefinition / WorkflowStateDefinition

Acceptance:

- all models have unit tests
- no role-specific Agent subclasses exist
- no capability implementation leaks into core models

---

## Milestone 2 — Lifecycle and Runtime State

Implement:

- create Agent Process
- runtime state creation
- legal status transitions
- terminate Agent Process
- state lookup

Acceptance tests:

```text
create → RUNNING
terminate → TERMINATED
terminated agent cannot execute action
unknown agent fails explicitly
illegal state transition rejected
```

---

## Milestone 3 — Capability Boundary

Implement:

- Capability protocol/base
- CapabilityRegistry
- input schema validation
- AdmissionPolicy
- Harness.execute_action()
- echo capability

Acceptance:

```text
allowed capability succeeds
unknown capability rejected
disallowed capability rejected
invalid arguments rejected
capability exception returns controlled failure
```

Critical acceptance:

> Agent Process has no direct reference that permits bypassing Harness admission.

---

## Milestone 4 — Context Assembly

Implement:

- ContextProvider
- ContextResolver
- OperationalContext
- capability descriptor injection
- runtime/task context

Acceptance:

- context is structured
- only currently allowed capabilities appear
- context assembly has no role-specific conditional logic
- prompt rendering, if any, happens outside core context representation

---

## Milestone 5 — Memory Boundary

Implement:

- MemoryProvider
- simple implementation
- write
- retrieve
- memory injection into context

Acceptance:

- Agent-specific/runtime-specific scopes can be represented
- backend can be replaced without changing Harness/Agent public contracts
- no graph/tier/promotion logic exists in V1 core

---

## Milestone 6 — Minimal Workflow

Implement:

- optional Workflow Registry
- WorkflowDefinition
- initial state
- allowed capability set
- memory scope
- Direct Mode fallback

Acceptance:

```text
agent can run without workflow
agent can run with minimal workflow
workflow changes capability visibility
unknown workflow rejected
```

No DAG executor.

No hard-coded `if workflow == ...`.

---

## Milestone 7 — End-to-End V1 Loop

Use a deterministic fake ReasoningEngine.

Scenario:

```text
Task:
"write hello to a test file and read it back"

Agent step 1:
ActionRequest(write_text)

Harness:
validate → execute → ActionResult

Agent step 2:
ActionRequest(read_text)

Harness:
validate → execute → ActionResult

Agent step 3:
FinalAnswer
```

Acceptance:

- complete runtime path passes
- external action crossed Harness boundary
- state and memory remain valid
- Agent can terminate cleanly
- test uses no real LLM

---

## Milestone 8 — Real Model Adapter Smoke Test

Only after deterministic integration tests pass.

Implement/reuse one model adapter.

Requirements:

- model output parsed into `FinalAnswer | ActionRequest`
- malformed output handled safely
- model cannot directly execute capabilities
- API/network failures do not corrupt runtime state

Do not make real-model tests required for normal unit test suite.

---

# 12. Test Strategy

V1 is a runtime architecture project. Boundary tests matter more than feature count.

## 12.1 Unit tests

Test:

- model validation
- lifecycle transitions
- registry behavior
- policy decisions
- schema validation
- context assembly
- memory provider behavior
- workflow resolution

---

## 12.2 Integration tests

At minimum:

### Direct Mode E2E

```text
Harness
→ Agent
→ ActionRequest
→ Capability
→ ActionResult
→ FinalAnswer
```

### Workflow Mode E2E

Verify a capability permitted in one state is unavailable in another configuration.

### Failure E2E

Capability throws exception:

```text
Harness survives
Agent state remains coherent
failure returned structurally
```

---

## 12.3 Architecture tests

Where practical, add tests preventing accidental coupling.

Examples:

- AgentProcess has no `tool_manager`
- AgentProcess has no `memory_manager`
- Harness does not import built-in capability implementations directly except registration/bootstrap layer
- workflow IDs do not appear as hard-coded branches in Harness Core

These can be lightweight structural tests or code-review requirements.

---

# 13. Future Extension Interfaces

These are not V1 features. Only preserve architectural room for them.

---

## 13.1 Memory Lens

Future:

```text
Workflow State
→ MemoryViewSpec
→ retrieval policy
→ MemoryProvider
→ projected memories
```

Possible future interface:

```python
class MemoryRetrievalPolicy(Protocol):
    def rank(
        self,
        query: str,
        candidates: list[MemoryRecord],
        runtime_state: RuntimeState,
    ) -> list[MemoryRecord]:
        ...
```

Do not implement now.

---

## 13.2 Memory Graph

Future MemoryRecord may gain relations:

```text
supports
contradicts
derived_from
caused_by
related_to
depends_on
```

Stable IDs in V1 are therefore required.

Avoid designing V1 storage in a way that makes relations impossible to add.

---

## 13.3 Memory Tiers and Promotion

Future conceptual lifecycle:

```text
episodic
→ established
→ consolidated
→ core_candidate
→ core
```

Promotion/demotion must be policy-driven.

Harness should provide mechanisms, not domain meanings.

Do not implement promotion rules in V1.

---

## 13.4 Checkpoint / Restore

Future Agent Process termination should be able to preserve:

- RuntimeState
- task state
- workflow state
- memory references
- resumable execution metadata

Therefore keep process identity separate from task/runtime identity where practical.

Do not assume:

```text
task lifetime == agent process lifetime
```

---

## 13.5 Multi-Agent Processes

Future:

```text
Agent A proposes need for additional cognition
→ Harness admission
→ Harness creates Agent B
```

No parent/child subclasses.

Potential lineage metadata can exist for auditability, but not ownership.

---

## 13.6 Rich Workflow Views

Future Workflow State may add:

```text
Context View
Memory View
Capability View
Permission View
Transition Constraints
Resource Budget
Role Label
```

V1 should not hardcode semantics that block these fields.

---

## 13.7 Persona

Future Persona remains presentation-only.

Operational reasoning must not depend on Persona.

Conceptually:

```text
Operational Reasoning
→ semantic draft
→ Persona presentation formatting
→ user
```

Do not add Persona to V1 core runtime decisions.

---

## 13.8 Software / Hardware Adapters

Future:

```text
Capability
→ Adapter
→ Software World
```

or:

```text
Capability
→ Robot Adapter
→ Motion/Skill Runtime
→ Hardware
```

Harness Core must not distinguish these semantically.

Do not add robot-specific checks to core.

Robot safety may later be declared/enforced through generic policy interfaces.

---

## 13.9 Embodiment Context

Future robots may require:

```text
body model
available actions
sensor configuration
kinematic limits
current state
```

This should enter through ContextProvider / Capability / Policy extension points, not through a new robot-specific Harness.

---

# 14. Architectural Red Lines

Codex must NOT introduce the following patterns.

## 14.1 No God Harness

Bad:

```python
if task_type == "coding":
    ...
elif task_type == "research":
    ...
elif robot_type == "arm":
    ...
```

Harness must remain generic.

---

## 14.2 No role-specific Agent classes

Forbidden:

```text
CoderAgent
AuditorAgent
PlannerAgent
ReviewerAgent
```

---

## 14.3 No hard-coded workflow semantics in Harness

Forbidden:

```python
if workflow_id == "devflow":
    max_agents = 5
```

Workflow/policy configuration declares constraints. Harness validates/enforces generically.

---

## 14.4 No direct tool execution by Agent

Forbidden:

```python
agent.tools["write_file"].invoke(...)
```

All external actions pass through Harness.

---

## 14.5 No premature plugin ecosystem

Do not spend V1 time creating:

- marketplace
- plugin installer
- version resolver
- remote package registry
- sandbox plugin host

Only define clean interfaces.

---

## 14.6 No premature memory intelligence

Do not implement graph ranking, auto-reflection, personality formation, or core-memory promotion before the basic MemoryProvider contract is proven.

---

## 14.7 No prompt-defined architecture

Prompts may instruct models, but architectural guarantees must come from runtime structure.

Do not rely on prompts such as:

```text
"You are not allowed to write files."
```

If writing is forbidden, the capability must not be admitted.

---

# 15. Coding Guidelines for Codex

1. Prefer small typed dataclasses/protocols.
2. Keep dependency direction inward toward stable contracts.
3. Avoid global mutable registries.
4. Prefer constructor injection for providers/registries/policies.
5. Make side-effect paths explicit.
6. Keep fake implementations for deterministic tests.
7. Do not make tests depend on network or real LLM APIs.
8. Preserve backwards compatibility where existing repository design is already clean.
9. Refactor only when required by a core invariant.
10. Add docstrings explaining architectural intent, not obvious syntax.
11. Do not add abstractions with only speculative use unless required to preserve a critical boundary.
12. Prefer one obvious implementation over generalized factories in V1.
13. Do not add hidden autonomous loops.
14. Every lifecycle transition and external action should be observable/testable.
15. Keep Core stable and boring.

---

# 16. Definition of Done for SForge V1

V1 is complete when all of the following are true.

## Runtime

- Harness can create an Agent Process.
- Harness can terminate an Agent Process.
- Runtime state is explicit and testable.
- Agent Process does not own privileged runtime resources.

## Reasoning

- A ReasoningEngine receives structured OperationalContext.
- It can return FinalAnswer or ActionRequest.
- Raw model output cannot directly create side effects.

## Capability

- Capabilities register through one generic interface.
- Harness validates access before invocation.
- At least one read-only and one side-effecting test capability work.
- Adding a new Capability does not require modifying Harness Core.

## Memory

- Memory can be written and retrieved through MemoryProvider.
- Backend implementation can be replaced without changing Agent Process.
- Memory is injected through context boundaries.

## Workflow

- Direct Mode works.
- Minimal Workflow Mode works.
- Workflow can alter allowed capability visibility.
- Harness contains no workflow-specific branches.

## Reliability

- Deterministic E2E test passes.
- Capability failure does not crash/corrupt Harness.
- Invalid Agent/Capability/Action states are rejected explicitly.

## Architecture

- No role-specific Agent types.
- No parent/child Agent ontology.
- No business logic in Harness Core.
- No direct Agent-to-world execution path.
- Future Memory Lens / Graph / checkpoint / multi-agent / adapter layers can be added without redesigning core contracts.

---

# 17. Recommended Codex Execution Order

Codex should follow this exact priority unless repository reality requires a justified deviation:

```text
1. Audit existing code
2. Run existing tests
3. Normalize core models
4. Implement lifecycle/state
5. Implement Capability contract/registry
6. Implement admission + execute_action
7. Implement structured context
8. Implement basic memory provider
9. Implement minimal workflow
10. Build deterministic E2E loop
11. Add failure-path tests
12. Only then connect a real LLM adapter
13. Update README / ARCHITECTURE
```

Do not implement future roadmap features during V1.

If an existing feature conflicts with the invariants, prefer isolating it behind an adapter rather than rewriting the entire repository.

---

# 18. Architectural North Star

When uncertain, use this test:

> **Does this change add a generic runtime mechanism, or does it teach Harness the meaning of a specific task/domain?**

If it adds generic mechanism, it may belong in SForge Core.

If it adds semantic/domain understanding, place it behind a provider, policy, workflow, capability, or future plugin.

Another useful test:

> **If tomorrow the LLM, memory backend, workflow, tool ecosystem, or physical body changes, should Harness Core need to change?**

The ideal answer is:

> **No. Only adapters/providers/policies/modules should change.**

The long-term purpose of SForge is therefore not to contain every agent feature.

It is to provide a small, durable runtime contract on top of which those features can evolve independently.

---

# 19. V1 Philosophy in One Sentence

> **Build the runtime boundary first; add intelligence modules later.**

Or, using the project name:

> **Build the forge before building what it will forge.**
