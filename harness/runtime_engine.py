"""Deterministic runtime mechanisms behind the thin Harness interface."""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Iterable, Literal

from harness.agent_manager import AgentManager
from harness.capability import AdmissionPolicy, CapabilityRegistry
from harness.context_manager import ContextBundle, ContextManager
from harness.errors import (
    CapabilityNotFoundError,
    DecisionProtocolError,
    InvalidActionArgumentsError,
    InvalidAgentStateError,
    InvalidResourceBindingError,
    InvalidWorkAssignmentError,
    InvalidWorkflowStateError,
    WorkRoleNotFoundError,
    WorkflowNotFoundError,
)
from harness.events import EventLogger, EventType, RuntimeEvent
from harness.identity import Identity
from harness.inspector import RuntimeInspector, RuntimeSnapshot
from harness.memory_manager import MemoryProvider
from harness.models import (
    ActionRequest,
    ActionResult,
    AgentProcess,
    AgentStatus,
    FinalAnswer,
    MemoryRecord,
    OperationalContext,
    ResourceBindingAdmission,
    ResourceBindingRequest,
    RuntimeState,
    TaskSpec,
    WorkAssignment,
    WorkAssignmentAdmission,
    WorkAssignmentRequest,
    WorkflowAdmission,
    WorkflowRequest,
)
from harness.persona import Persona
from harness.cognitive_policy import CognitivePolicyRegistry
from harness.profession import ProfessionRegistry
from harness.skill import SkillRegistry
from harness.workflow_manager import WorkflowRegistry
from harness.work_role import WorkRoleRegistry
from harness.workspace import Workspace


class RuntimeEngine:
    """Own the loop boundary; never decide how a domain task should be solved."""

    def __init__(
        self,
        agents: AgentManager,
        contexts: ContextManager,
        memory: MemoryProvider,
        capabilities: CapabilityRegistry,
        workflows: WorkflowRegistry,
        roles: WorkRoleRegistry,
        policies: CognitivePolicyRegistry,
        professions: ProfessionRegistry,
        skills: SkillRegistry,
        workspace: Workspace,
        admission: AdmissionPolicy,
        persona: Persona,
        *,
        identity: Identity,
        default_work_role_id: str,
        events: EventLogger | None = None,
        basic_capabilities: frozenset[str] = frozenset(),
        max_steps: int = 8,
    ):
        if max_steps < 1:
            raise ValueError("max_steps 必须大于零")
        for capability_id in basic_capabilities:
            capabilities.get(capability_id)
        if identity.persona_reference != persona.reference:
            raise ValueError(
                "Identity persona_reference 与已装载 Persona 不一致"
            )
        roles.get(default_work_role_id)
        if identity.default_cognitive_policy_id:
            policies.get(identity.default_cognitive_policy_id)
        for summary in professions.available():
            profession = professions.get(summary["id"])
            for skill_id in profession.preferred_skills:
                skills.get(skill_id)
        for skill_id in workspace.local_skills:
            skills.get(skill_id)
        for capability_id in basic_capabilities:
            if capabilities.get(capability_id).descriptor.side_effects:
                raise ValueError(
                    "Base Agent 不能默认持有具备外部副作用的 Capability"
                )
        self._agents = agents
        self._contexts = contexts
        self._memory = memory
        self._capabilities = capabilities
        self._workflows = workflows
        self._roles = roles
        self._policies = policies
        self._professions = professions
        self._skills = skills
        self._workspace = workspace
        self._admission = admission
        self._persona = persona
        self._identity = identity
        self._events = events or EventLogger()
        self._inspector = RuntimeInspector(
            agents, memory, capabilities, self._events
        )
        self._basic_capabilities = frozenset(basic_capabilities)
        self._workspace_id = workspace.id
        self._default_work_role_id = default_work_role_id
        self._max_steps = max_steps
        self._observations: dict[
            str,
            ActionResult
            | WorkflowAdmission
            | WorkAssignmentAdmission
            | ResourceBindingAdmission,
        ] = {}
        self._binding_request_results: dict[
            tuple[str, str],
            tuple[tuple[str, str, str | None], ResourceBindingAdmission],
        ] = {}

    def create_agent(
        self,
        task: TaskSpec | str,
        workflow_id: str | None = None,
    ) -> AgentProcess:
        task_spec = task if isinstance(task, TaskSpec) else TaskSpec(str(task))
        if not task_spec.request.strip():
            raise ValueError("Task request 不能为空")

        try:
            if workflow_id is not None:
                self._workflows.get(workflow_id)
                task_spec = TaskSpec(
                    request=task_spec.request,
                    id=task_spec.id,
                    context={
                        **task_spec.context,
                        "requested_workflow_id": workflow_id,
                    },
                )
            for capability_id in self._basic_capabilities:
                self._capabilities.get(capability_id)
            process = self._agents.create(
                task_spec,
                identity=self._identity,
                persona=self._persona,
                mode="direct",
                allowed_capabilities=self._basic_capabilities,
                memory_scope="core",
                memory_scopes=("core",),
            )
            self._restore_identity_continuity(process.id)
        except Exception as exc:
            self._emit(
                EventType.ERROR,
                trace_id=task_spec.id,
                data={
                    "stage": "agent_creation",
                    "error_type": type(exc).__name__,
                },
            )
            raise

        self._emit(
            EventType.AGENT_CREATED,
            agent_id=process.id,
            trace_id=self._agents.runtime_state(process.id).task_id,
            data={
                "mode": "direct",
                "status": process.status.value,
                "workspace_catalog_id": self._workspace_id,
                "identity_id": self._identity.id,
                "cognitive_policy_id": self._agents.runtime_state(
                    process.id
                ).cognitive_policy_id,
                "profession_ids": list(
                    self._agents.runtime_state(process.id).profession_ids
                ),
                "resumed_assignment_id": self._agents.runtime_state(
                    process.id
                ).assignment_id,
            },
        )
        return process

    def terminate_agent(
        self, agent_id: str, reason: str | None = None
    ) -> AgentProcess:
        self._observations.pop(agent_id, None)
        for key in tuple(self._binding_request_results):
            if key[0] == agent_id:
                self._binding_request_results.pop(key, None)
        return self._agents.terminate(agent_id)

    def build_context(self, agent_id: str) -> OperationalContext:
        return self._bundle(agent_id).operational

    def retrieval_trace(self, agent_id: str) -> dict[str, Any]:
        """Return read-only retrieval instrumentation for the current state."""

        return self._bundle(agent_id).retrieval_trace

    def request_workflow(
        self, agent_id: str, request: WorkflowRequest
    ) -> WorkflowAdmission:
        """Validate one Agent proposal and atomically mount the declared state."""

        self._agents.require_active(agent_id)
        current = self._agents.runtime_state(agent_id)
        self._emit(
            EventType.WORKFLOW_REQUESTED,
            agent_id=agent_id,
            request_id=request.request_id,
            data={
                "workflow_id": request.workflow_id,
                "target_state_id": request.target_state_id,
            },
        )
        current_assignment = self._agents.current_assignment(agent_id)
        if current_assignment is None or current.workflow_id is None:
            result = WorkflowAdmission(
                request_id=request.request_id,
                status="rejected",
                workflow_id=request.workflow_id,
                previous_state_id=current.workflow_state_id,
                workflow_state_id=current.workflow_state_id,
                error=(
                    "Workflow 初始进入必须通过包含 role_id、workspace_id "
                    "和 workflow_id 的 WorkAssignmentRequest；"
                    "WorkflowRequest 只用于活动 Assignment 的状态迁移"
                ),
            )
            self._emit_workflow_admission(agent_id, result)
            return result
        try:
            definition = self._workflows.get(request.workflow_id)
            if current.workflow_id != request.workflow_id:
                raise InvalidWorkflowStateError(
                    "一个 WorkAssignment 内不能切换到另一个 Workflow"
                )
            previous_state_id = current.workflow_state_id
            if request.target_state_id is None:
                raise InvalidWorkflowStateError(
                    "Workflow 状态迁移必须声明 target_state_id"
                )
            if request.transition_condition is None:
                raise InvalidWorkflowStateError(
                    "Workflow 状态迁移必须声明 transition_condition"
                )
            target_state_id = request.target_state_id
            matches = [
                edge
                for edge in definition.outgoing(previous_state_id or "")
                if edge.target == target_state_id
                and edge.condition == request.transition_condition
            ]
            if not matches:
                raise InvalidWorkflowStateError(
                    "请求的迁移边未在当前 Workflow State 中声明"
                )

            state = definition.states[target_state_id]
            assignment_grants = state.allowed_capabilities.intersection(
                current_assignment.grants
            )
            allowed = self._basic_capabilities.union(assignment_grants)
            for capability_id in allowed:
                self._capabilities.get(capability_id)
            memory_scopes = self._assignment_memory_scopes(
                state.memory_scopes,
                current.task_id,
                definition.id,
                current.identity_id,
                self._workspace_id,
                current.profession_ids,
            )
            memory_scope = self._resolve_memory_scope(
                state.memory_write_scope, current.task_id, definition.id
            )
            mounted = self._agents.mount_workflow_state(
                agent_id,
                workflow_id=definition.id,
                workflow_state_id=state.id,
                allowed_capabilities=allowed,
                memory_scope=memory_scope,
                memory_scopes=memory_scopes,
            )
        except (
            WorkflowNotFoundError,
            InvalidWorkflowStateError,
            CapabilityNotFoundError,
        ) as exc:
            result = WorkflowAdmission(
                request_id=request.request_id,
                status="rejected",
                workflow_id=request.workflow_id,
                previous_state_id=current.workflow_state_id,
                workflow_state_id=current.workflow_state_id,
                error=str(exc),
            )
            self._emit_workflow_admission(agent_id, result)
            return result

        result = WorkflowAdmission(
            request_id=request.request_id,
            status="success",
            workflow_id=definition.id,
            previous_state_id=previous_state_id,
            workflow_state_id=mounted.workflow_state_id,
            memory_scope=mounted.memory_scope,
            memory_scopes=mounted.memory_scopes,
            allowed_capabilities=mounted.allowed_capabilities,
        )
        self._memory.write(
            MemoryRecord(
                scope=mounted.memory_scope,
                kind="runtime.workflow_admission",
                content=json.dumps(result.as_dict(), ensure_ascii=False),
                metadata={"request_id": request.request_id},
            )
        )
        self._emit_workflow_admission(agent_id, result)
        return result

    def request_work_assignment(
        self, agent_id: str, request: WorkAssignmentRequest
    ) -> WorkAssignmentAdmission:
        """Validate and mount one explicit process/work relationship."""

        self._agents.require_active(agent_id)
        current = self._agents.runtime_state(agent_id)
        previous = self._agents.current_assignment(agent_id)
        requested_workspace = request.workspace_id or self._workspace_id
        requested_task = request.task_id or current.task_id
        self._emit(
            EventType.WORK_ASSIGNMENT_REQUESTED,
            agent_id=agent_id,
            request_id=request.request_id,
            data={
                "role_id": request.role_id,
                "workspace_id": requested_workspace,
                "workflow_id": request.workflow_id,
            },
        )
        try:
            role = self._roles.get(request.role_id)
            if requested_workspace != self._workspace_id:
                raise InvalidWorkAssignmentError(
                    "WorkAssignment 只能进入当前 Runtime 挂载的 Workspace"
                )
            if requested_task != current.task_id:
                raise InvalidWorkAssignmentError(
                    "WorkAssignment task_id 必须匹配当前 Task"
                )
            (
                workflow_id,
                workflow_state_id,
                offered,
                memory_scope,
                memory_scopes,
            ) = self._assignment_resources(current, request)
            requested = frozenset(request.requested_capabilities)
            if len(requested) != len(request.requested_capabilities):
                raise InvalidWorkAssignmentError(
                    "requested_capabilities 不允许重复项"
                )
            unsupported = requested.difference(offered)
            if unsupported:
                raise InvalidWorkAssignmentError(
                    "请求了当前工作环境未提供的 Capability: "
                    + ", ".join(sorted(unsupported))
                )
            grants = (
                offered
                if not requested
                else requested
            )
            for capability_id in grants:
                self._capabilities.get(capability_id)
            assignment = self._agents.bind_work_assignment(
                agent_id,
                role_id=role.id,
                workspace_id=requested_workspace,
                task_id=requested_task,
                workflow_id=workflow_id,
                grants=grants,
                effective_capabilities=self._basic_capabilities.union(grants),
                workflow_state_id=workflow_state_id,
                memory_scope=memory_scope,
                memory_scopes=memory_scopes,
            )
        except (
            WorkRoleNotFoundError,
            WorkflowNotFoundError,
            InvalidWorkflowStateError,
            InvalidWorkAssignmentError,
            CapabilityNotFoundError,
        ) as exc:
            result = WorkAssignmentAdmission(
                request_id=request.request_id,
                status="rejected",
                role_id=request.role_id,
                workspace_id=requested_workspace,
                task_id=requested_task,
                previous_assignment_id=previous.id if previous else None,
                error=str(exc),
            )
            self._emit_work_assignment_admission(agent_id, result)
            return result

        mounted = self._agents.runtime_state(agent_id)
        if previous is None or assignment.id != previous.id:
            if previous is not None:
                self._record_assignment_ended(
                    previous, "WorkAssignment replaced"
                )
                self._emit(
                    EventType.WORK_ASSIGNMENT_ENDED,
                    agent_id=agent_id,
                    data={
                        "assignment_id": previous.id,
                        "role_id": previous.role_id,
                        "workspace_id": previous.workspace_id,
                    },
                )
            self._record_assignment_started(assignment)
        result = WorkAssignmentAdmission(
            request_id=request.request_id,
            status="success",
            assignment_id=assignment.id,
            previous_assignment_id=previous.id if previous else None,
            role_id=assignment.role_id,
            workspace_id=assignment.workspace_id,
            task_id=assignment.task_id,
            workflow_id=assignment.workflow_id,
            workflow_state_id=mounted.workflow_state_id,
            memory_scope=mounted.memory_scope,
            memory_scopes=mounted.memory_scopes,
            grants=assignment.grants,
        )
        self._emit_work_assignment_admission(agent_id, result)
        return result

    def end_work_assignment(
        self, agent_id: str, reason: str | None = None
    ) -> WorkAssignment | None:
        current = self._agents.current_assignment(agent_id)
        if current is None:
            return None
        state = self._agents.runtime_state(agent_id)
        post_assignment_scopes = self._binding_memory_scopes(
            state,
            profession_ids=state.profession_ids,
            include_workflow=False,
        )
        ended = self._agents.end_work_assignment(
            agent_id,
            basic_capabilities=self._basic_capabilities,
            memory_scope="core",
            memory_scopes=post_assignment_scopes,
        )
        assert ended is not None
        self._record_assignment_ended(
            ended, reason or "WorkAssignment ended"
        )
        self._emit(
            EventType.WORK_ASSIGNMENT_ENDED,
            agent_id=agent_id,
            data={
                "assignment_id": ended.id,
                "role_id": ended.role_id,
                "workspace_id": ended.workspace_id,
            },
        )
        return ended

    def record_work_experience(
        self,
        agent_id: str,
        lesson: str,
        *,
        objective_outcome: str | None = None,
        self_reflection: str | None = None,
        external_feedback: str | None = None,
        professional_tags: tuple[str, ...] = (),
        artifact_refs: tuple[str, ...] = (),
    ) -> MemoryRecord:
        """Persist a grounded Identity-owned lesson with Workspace provenance."""

        self._agents.require_active(agent_id)
        assignment = self._agents.current_assignment(agent_id)
        if assignment is None:
            raise InvalidWorkAssignmentError(
                "记录工作经验前必须存在活动 WorkAssignment"
            )
        if not lesson.strip():
            raise ValueError("work experience lesson 不能为空")
        if not objective_outcome and not external_feedback:
            raise ValueError(
                "工作经验至少需要 objective_outcome 或 external_feedback 作为依据"
            )
        return self._memory.write(
            MemoryRecord(
                scope=f"identity:{assignment.identity_id}",
                kind="identity.work_experience",
                content=lesson.strip(),
                metadata={
                    "derived_from_workspace": assignment.workspace_id,
                    "performed_as_role": assignment.role_id,
                    "task_id": assignment.task_id,
                    "objective_outcome": objective_outcome,
                    "self_reflection": self_reflection,
                    "external_feedback": external_feedback,
                    "professional_tags": list(professional_tags),
                    "artifact_refs": list(artifact_refs),
                },
            )
        )

    def request_binding(
        self, agent_id: str, request: ResourceBindingRequest
    ) -> ResourceBindingAdmission:
        """Validate and atomically mount one non-authoritative resource."""

        self._agents.require_active(agent_id)
        current = self._agents.runtime_state(agent_id)
        self._emit(
            EventType.RESOURCE_BINDING_REQUESTED,
            agent_id=agent_id,
            request_id=request.request_id,
            data={
                "resource_type": request.resource_type,
                "operation": request.operation,
                "resource_id": request.resource_id,
            },
        )
        signature = (
            request.resource_type,
            request.operation,
            request.resource_id,
        )
        cache_key = (agent_id, request.request_id)
        cached = self._binding_request_results.get(cache_key)
        if cached is not None:
            cached_signature, cached_result = cached
            if cached_signature != signature:
                result = ResourceBindingAdmission(
                    request_id=request.request_id,
                    status="rejected",
                    resource_type=request.resource_type,
                    operation=request.operation,
                    resource_id=request.resource_id,
                    cognitive_policy_id=current.cognitive_policy_id,
                    profession_ids=current.profession_ids,
                    changed=False,
                    replayed=True,
                    error="同一 request_id 不能用于不同 ResourceBindingRequest",
                )
            else:
                result = ResourceBindingAdmission(
                    request_id=cached_result.request_id,
                    status=cached_result.status,
                    resource_type=cached_result.resource_type,
                    operation=cached_result.operation,
                    resource_id=cached_result.resource_id,
                    cognitive_policy_id=current.cognitive_policy_id,
                    profession_ids=current.profession_ids,
                    changed=False,
                    replayed=True,
                    error=cached_result.error,
                )
            self._emit_binding_admission(agent_id, result)
            return result
        try:
            if request.operation not in {"activate", "deactivate"}:
                raise InvalidResourceBindingError("未知 binding operation")
            policy_id = current.cognitive_policy_id
            profession_ids = list(current.profession_ids)
            changed = False
            resource_id = request.resource_id.strip() if request.resource_id else None
            if request.resource_type == "cognitive_policy":
                if request.operation == "activate":
                    if not resource_id:
                        raise InvalidResourceBindingError(
                            "激活 CognitivePolicy 必须提供 resource_id"
                        )
                    resolved_policy = self._policies.get(resource_id).id
                    changed = resolved_policy != policy_id
                    policy_id = resolved_policy
                else:
                    if resource_id and resource_id.upper() != policy_id:
                        raise InvalidResourceBindingError(
                            "待解绑 CognitivePolicy 不是当前活动资源"
                        )
                    changed = policy_id is not None
                    policy_id = None
            elif request.resource_type == "profession":
                if request.operation == "activate":
                    if not resource_id:
                        raise InvalidResourceBindingError(
                            "激活 Profession 必须提供 resource_id"
                        )
                    profession_id = self._professions.get(resource_id).id
                    if profession_id not in profession_ids:
                        profession_ids.append(profession_id)
                        changed = True
                else:
                    if not resource_id or resource_id not in profession_ids:
                        raise InvalidResourceBindingError(
                            "待解绑 Profession 不是当前活动资源"
                        )
                    profession_ids.remove(resource_id)
                    changed = True
            else:
                raise InvalidResourceBindingError("未知 resource_type")
            if changed:
                memory_scopes = self._binding_memory_scopes(
                    current,
                    profession_ids=tuple(profession_ids),
                )
                mounted = self._agents.mount_resources(
                    agent_id,
                    cognitive_policy_id=policy_id,
                    profession_ids=tuple(profession_ids),
                    memory_scope=current.memory_scope,
                    memory_scopes=memory_scopes,
                )
            else:
                mounted = current
        except (InvalidResourceBindingError, KeyError) as exc:
            result = ResourceBindingAdmission(
                request_id=request.request_id,
                status="rejected",
                resource_type=request.resource_type,
                operation=request.operation,
                resource_id=request.resource_id,
                cognitive_policy_id=current.cognitive_policy_id,
                profession_ids=current.profession_ids,
                changed=False,
                error=str(exc),
            )
            self._binding_request_results[cache_key] = (
                signature,
                result,
            )
            self._emit_binding_admission(agent_id, result)
            return result

        result = ResourceBindingAdmission(
            request_id=request.request_id,
            status="success",
            resource_type=request.resource_type,
            operation=request.operation,
            resource_id=request.resource_id,
            cognitive_policy_id=mounted.cognitive_policy_id,
            profession_ids=mounted.profession_ids,
            changed=changed,
        )
        self._binding_request_results[cache_key] = (signature, result)
        self._emit_binding_admission(agent_id, result)
        return result

    def execute_action(
        self, agent_id: str, request: ActionRequest
    ) -> ActionResult:
        self._agents.require_active(agent_id)
        runtime = self._agents.runtime_state(agent_id)
        self._emit(
            EventType.CAPABILITY_REQUESTED,
            agent_id=agent_id,
            request_id=request.request_id,
            data={"capability_id": request.capability_id},
        )

        try:
            capability = self._capabilities.get(request.capability_id)
        except CapabilityNotFoundError as exc:
            return self._result(
                agent_id, request, "rejected", error=str(exc), stage="resolve"
            )
        try:
            self._capabilities.validate_input(request)
        except InvalidActionArgumentsError as exc:
            return self._result(
                agent_id,
                request,
                "rejected",
                error=str(exc),
                stage="validation",
            )

        decision = self._admission.authorize(
            runtime, request, capability.descriptor
        )
        if not decision.allowed:
            return self._result(
                agent_id,
                request,
                "rejected",
                error=decision.reason or "Action 被拒绝",
                stage="admission",
            )

        event_data = {"capability_id": request.capability_id}
        try:
            output = capability.invoke(request.arguments)
            self._capabilities.validate_output(request.capability_id, output)
            status: Literal["success", "failed"] = "success"
            error = None
        except Exception as exc:
            output = None
            status = "failed"
            error = str(exc)
            event_data["error_type"] = type(exc).__name__
        event_data["status"] = status
        self._emit(
            EventType.CAPABILITY_COMPLETED,
            agent_id=agent_id,
            request_id=request.request_id,
            data=event_data,
        )
        return self._result(
            agent_id,
            request,
            status,
            output=output,
            error=error,
            stage="execution",
        )

    def step(
        self, agent_id: str
    ) -> (
        FinalAnswer
        | ActionResult
        | WorkflowAdmission
        | WorkAssignmentAdmission
        | ResourceBindingAdmission
    ):
        identity = self._agents.require_active(agent_id)
        if identity.status is AgentStatus.WAITING:
            self._agents.mark_running(agent_id)

        stage = "context"
        try:
            bundle = self._bundle(agent_id)
            agent = self._agents.agent(agent_id)
            stage = "reasoning"
            self._emit(
                EventType.REASONING_STARTED,
                agent_id=agent_id,
                data={"phase": "decision"},
            )
            try:
                turn = agent.run(
                    bundle.operational,
                    self._agents.task(agent_id).request,
                    observation=self._observations.get(agent_id),
                )
            except DecisionProtocolError as exc:
                self._emit(
                    EventType.REASONING_COMPLETED,
                    agent_id=agent_id,
                    data={
                        "phase": "decision",
                        "decision_protocol": dict(exc.protocol),
                        "protocol_success": False,
                    },
                )
                raise
            self._emit(
                EventType.REASONING_COMPLETED,
                agent_id=agent_id,
                data={
                    "phase": "decision",
                    "usage": turn.usage.as_dict(),
                    "decision_protocol": dict(turn.decision_protocol),
                    "protocol_success": bool(
                        turn.decision_protocol.get(
                            "structured_decision_valid", False
                        )
                        and turn.decision_protocol.get(
                            "decision_schema_valid", False
                        )
                    ),
                },
            )
            decision = turn.decision
            if isinstance(decision, FinalAnswer):
                stage = "finalization"
                self._write_runtime_memory(
                    agent_id, "runtime.final_answer", decision.content
                )
                self._emit(
                    EventType.REASONING_STARTED,
                    agent_id=agent_id,
                    data={"phase": "presentation"},
                )
                rendered = agent.format_response(
                    bundle.response_rendering_for(decision.content)
                )
                self._emit(
                    EventType.REASONING_COMPLETED,
                    agent_id=agent_id,
                    data={
                        "phase": "presentation",
                        "usage": rendered.usage.as_dict(),
                    },
                )
                self._agents.complete(agent_id)
                self._observations.pop(agent_id, None)
                return FinalAnswer(
                    rendered.content,
                    decision.primary_evidence_id,
                    decision.secondary_evidence_ids,
                    decision.final_choice,
                )

            if isinstance(decision, WorkAssignmentRequest):
                stage = "work_assignment_admission"
                result = self.request_work_assignment(agent_id, decision)
            elif isinstance(decision, ResourceBindingRequest):
                stage = "resource_binding_admission"
                result = self.request_binding(agent_id, decision)
            elif isinstance(decision, WorkflowRequest):
                stage = "workflow_admission"
                result = self.request_workflow(agent_id, decision)
            else:
                stage = "action"
                result = self.execute_action(agent_id, decision)
            self._observations[agent_id] = result
            self._agents.mark_waiting(agent_id)
            return result
        except Exception as exc:
            self._fail_with_event(agent_id, stage, exc)
            raise

    def run(self, agent_id: str) -> str:
        for _ in range(self._max_steps):
            outcome = self.step(agent_id)
            if isinstance(outcome, FinalAnswer):
                return outcome.content
        error = InvalidAgentStateError(
            "Agent 超过单次运行允许的最大步骤数"
        )
        self._fail_with_event(agent_id, "step_budget", error)
        raise error

    def process(self, agent_id: str) -> AgentProcess:
        return self._agents.process(agent_id)

    def runtime_state(self, agent_id: str) -> RuntimeState:
        return self._agents.runtime_state(agent_id)

    def available_workflows(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._workflows.available())

    def available_workspaces(self) -> tuple[dict[str, str], ...]:
        return (
            {
                "id": self._workspace.id,
                "description": self._workspace.description,
            },
        )

    def available_cognitive_policies(self) -> tuple[dict[str, Any], ...]:
        return self._policies.available()

    def available_professions(self) -> tuple[dict[str, Any], ...]:
        return self._professions.available()

    def available_skills(self) -> tuple[dict[str, Any], ...]:
        return self._skills.available()

    def available_work_roles(self) -> tuple[dict[str, Any], ...]:
        return self._roles.available()

    def work_assignment(self, agent_id: str) -> WorkAssignment | None:
        return self._agents.current_assignment(agent_id)

    def recent_events(
        self,
        limit: int = 20,
        *,
        agent_id: str | None = None,
        trace_id: str | None = None,
        event_types: Iterable[EventType] | None = None,
    ) -> tuple[RuntimeEvent, ...]:
        return self._events.recent(
            limit,
            agent_id=agent_id,
            trace_id=trace_id,
            event_types=event_types,
        )

    def inspect(
        self,
        agent_id: str | None = None,
        *,
        event_limit: int = 20,
        memory_limit: int = 20,
    ) -> RuntimeSnapshot:
        return self._inspector.capture(
            agent_id,
            event_limit=event_limit,
            memory_limit=memory_limit,
        )

    def write_memory(
        self,
        agent_id: str,
        kind: str,
        content: str,
        *,
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        self._agents.require_active(agent_id)
        state = self._agents.runtime_state(agent_id)
        return self._memory.write(
            MemoryRecord(
                scope=state.memory_scope,
                kind=kind,
                content=content,
                importance=importance,
                metadata=dict(metadata or {}),
            )
        )

    def close(self) -> None:
        self._agents.close()
        self._memory.close()

    def _restore_identity_continuity(self, agent_id: str) -> None:
        """Recompute process-local views from identity-level primitive facts."""

        state = self._agents.runtime_state(agent_id)
        assignment = self._agents.current_assignment(agent_id)
        if assignment is None:
            if state.cognitive_policy_id or state.profession_ids:
                self._agents.mount_resources(
                    agent_id,
                    cognitive_policy_id=state.cognitive_policy_id,
                    profession_ids=state.profession_ids,
                    memory_scope="core",
                    memory_scopes=self._binding_memory_scopes(
                        state,
                        profession_ids=state.profession_ids,
                        include_workflow=False,
                    ),
                )
            return

        if assignment.workspace_id != self._workspace_id:
            raise InvalidWorkAssignmentError(
                "活动 WorkAssignment 不属于当前 Runtime Workspace"
            )
        grants = assignment.grants
        for capability_id in grants:
            self._capabilities.get(capability_id)
        if assignment.workflow_id is None:
            effective = self._basic_capabilities.union(grants)
            memory_scopes = list(
                self._binding_memory_scopes(
                    state,
                    profession_ids=state.profession_ids,
                    include_workflow=False,
                )
            )
            workspace_scope = f"workspace:{assignment.workspace_id}"
            if workspace_scope not in memory_scopes:
                memory_scopes.append(workspace_scope)
            memory_scope = workspace_scope
        else:
            definition = self._workflows.get(assignment.workflow_id)
            workflow_state_id = (
                assignment.workflow_state_id or definition.initial_state
            )
            workflow_state = definition.states[workflow_state_id]
            effective = self._basic_capabilities.union(
                grants.intersection(
                    workflow_state.allowed_capabilities
                )
            )
            memory_scopes = list(
                self._assignment_memory_scopes(
                    workflow_state.memory_scopes,
                    assignment.task_id,
                    definition.id,
                    state.identity_id,
                    assignment.workspace_id,
                    state.profession_ids,
                )
            )
            memory_scope = self._resolve_memory_scope(
                workflow_state.memory_write_scope,
                assignment.task_id,
                definition.id,
            )
        self._agents.resume_work_assignment(
            agent_id,
            effective_capabilities=effective,
            memory_scope=memory_scope,
            memory_scopes=tuple(memory_scopes),
        )

    def _bundle(self, agent_id: str) -> ContextBundle:
        process = self._agents.require_active(agent_id)
        bundle = self._contexts.build(
            process,
            self._agents.runtime_state(agent_id),
            self._agents.task(agent_id),
            self._agents.agent(agent_id).identity,
            self._agents.current_assignment(agent_id),
            self._agents.agent(agent_id).persona,
        )
        self._emit(
            EventType.CONTEXT_BUILT,
            agent_id=agent_id,
            data={
                "memory_count": len(bundle.operational.memory),
                "capability_count": len(bundle.operational.capabilities),
                "has_workflow": bundle.operational.workflow is not None,
                "has_work_assignment": (
                    bundle.operational.work_assignment is not None
                ),
                "work_role_id": (
                    self._agents.current_assignment(agent_id).role_id
                    if self._agents.current_assignment(agent_id)
                    else None
                ),
                "candidate_memory_count": bundle.retrieval_trace.get(
                    "candidate_memory_count", 0
                ),
                "selected_memory_count": bundle.retrieval_trace.get(
                    "selected_memory_count", 0
                ),
                "deduplicated_memory_count": bundle.retrieval_trace.get(
                    "deduplicated_memory_count", 0
                ),
                "budget_dropped_memory_count": bundle.retrieval_trace.get(
                    "budget_dropped_memory_count", 0
                ),
                "estimated_context_tokens": bundle.retrieval_trace.get(
                    "estimated_context_tokens", 0
                ),
                "region_size_estimates": bundle.retrieval_trace.get(
                    "region_size_estimates", {}
                ),
            },
        )
        return bundle

    def _result(
        self,
        agent_id: str,
        request: ActionRequest,
        status: Literal["success", "rejected", "failed"],
        *,
        output: Any | None = None,
        error: str | None = None,
        stage: str,
    ) -> ActionResult:
        result = ActionResult(
            request_id=request.request_id,
            capability_id=request.capability_id,
            status=status,
            output=output,
            error=error,
            metadata={"stage": stage},
        )
        serialized_result = json.dumps(result.as_dict(), ensure_ascii=False)
        evidence_metadata = self._action_result_memory_metadata(
            request, result
        )
        state = self._agents.runtime_state(agent_id)
        if state.memory_scope != "core":
            self._memory.write(
                MemoryRecord(
                    scope=state.memory_scope,
                    kind="runtime.action_result",
                    content=serialized_result,
                    metadata=dict(evidence_metadata),
                )
            )
        assignment = self._agents.current_assignment(agent_id)
        workspace_scope = (
            f"workspace:{assignment.workspace_id}" if assignment else None
        )
        if workspace_scope and workspace_scope != state.memory_scope:
            self._memory.write(
                MemoryRecord(
                    scope=workspace_scope,
                    kind="workspace.action_result",
                    content=serialized_result,
                    metadata={
                        **evidence_metadata,
                        "assignment_id": assignment.id,
                        "role_id": assignment.role_id,
                        "task_id": assignment.task_id,
                    },
                )
            )
        self._emit(
            EventType.ACTION_COMPLETED,
            agent_id=agent_id,
            request_id=request.request_id,
            data={
                "capability_id": request.capability_id,
                "status": status,
                "stage": stage,
            },
        )
        return result

    @staticmethod
    def _action_result_memory_metadata(
        request: ActionRequest, result: ActionResult
    ) -> dict[str, Any]:
        output = result.output
        if isinstance(output, str):
            serialized_output = output
        else:
            serialized_output = json.dumps(
                output,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
        arguments = json.dumps(
            request.arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        resource_reference = next(
            (
                str(request.arguments[key])
                for key in (
                    "path",
                    "uri",
                    "url",
                    "resource",
                    "resource_id",
                    "artifact_id",
                )
                if key in request.arguments
                and isinstance(request.arguments[key], (str, int, float))
            ),
            None,
        )
        return {
            "request_id": request.request_id,
            "capability_id": request.capability_id,
            "status": result.status,
            "resource_reference": resource_reference,
            "arguments_sha256": sha256(arguments.encode("utf-8")).hexdigest(),
            "output_characters": len(serialized_output),
            "output_sha256": sha256(
                serialized_output.encode("utf-8")
            ).hexdigest(),
        }

    def _write_runtime_memory(
        self, agent_id: str, kind: str, content: str
    ) -> MemoryRecord | None:
        state = self._agents.runtime_state(agent_id)
        if state.memory_scope == "core":
            return None
        return self._memory.write(
            MemoryRecord(
                scope=state.memory_scope,
                kind=kind,
                content=content,
                metadata={"agent_id": agent_id},
            )
        )

    def _emit_workflow_admission(
        self, agent_id: str, result: WorkflowAdmission
    ) -> RuntimeEvent:
        return self._emit(
            EventType.WORKFLOW_ADMISSION_COMPLETED,
            agent_id=agent_id,
            request_id=result.request_id,
            data={
                "workflow_id": result.workflow_id,
                "previous_state_id": result.previous_state_id,
                "state_id": result.workflow_state_id,
                "status": result.status,
                "memory_scopes": list(result.memory_scopes),
                "capability_count": len(result.allowed_capabilities),
            },
        )

    def _emit_binding_admission(
        self, agent_id: str, result: ResourceBindingAdmission
    ) -> RuntimeEvent:
        return self._emit(
            EventType.RESOURCE_BINDING_COMPLETED,
            agent_id=agent_id,
            request_id=result.request_id,
            data={
                "resource_type": result.resource_type,
                "operation": result.operation,
                "resource_id": result.resource_id,
                "cognitive_policy_id": result.cognitive_policy_id,
                "profession_ids": list(result.profession_ids),
                "status": result.status,
                "changed": result.changed,
                "replayed": result.replayed,
            },
        )

    def _emit_work_assignment_admission(
        self, agent_id: str, result: WorkAssignmentAdmission
    ) -> RuntimeEvent:
        return self._emit(
            EventType.WORK_ASSIGNMENT_ADMISSION_COMPLETED,
            agent_id=agent_id,
            request_id=result.request_id,
            data={
                "assignment_id": result.assignment_id,
                "previous_assignment_id": result.previous_assignment_id,
                "role_id": result.role_id,
                "workspace_id": result.workspace_id,
                "workflow_id": result.workflow_id,
                "workflow_state_id": result.workflow_state_id,
                "status": result.status,
                "capability_count": len(result.grants),
            },
        )

    def _record_assignment_started(self, assignment: WorkAssignment) -> None:
        self._memory.write(
            MemoryRecord(
                scope=f"workspace:{assignment.workspace_id}",
                kind="workspace.assignment_started",
                content=json.dumps(
                    assignment.as_context(), ensure_ascii=False
                ),
                metadata={
                    "assignment_id": assignment.id,
                    "identity_id": assignment.identity_id,
                    "role_id": assignment.role_id,
                    "task_id": assignment.task_id,
                    "workflow_id": assignment.workflow_id,
                },
            )
        )

    def _record_assignment_ended(
        self, assignment: WorkAssignment, reason: str
    ) -> None:
        self._memory.write(
            MemoryRecord(
                scope=f"workspace:{assignment.workspace_id}",
                kind="workspace.assignment_ended",
                content=reason,
                metadata={
                    "assignment_id": assignment.id,
                    "identity_id": assignment.identity_id,
                    "role_id": assignment.role_id,
                    "task_id": assignment.task_id,
                    "workflow_id": assignment.workflow_id,
                },
            )
        )

    def _assignment_resources(
        self, current: RuntimeState, request: WorkAssignmentRequest
    ) -> tuple[
        str | None,
        str | None,
        frozenset[str],
        str,
        tuple[str, ...],
    ]:
        workspace_id = request.workspace_id or self._workspace_id
        base_scopes = list(
            self._binding_memory_scopes(
                current,
                profession_ids=current.profession_ids,
                include_workflow=False,
            )
        )
        base_scopes.append(f"workspace:{workspace_id}")
        if request.workflow_id is None:
            if request.target_state_id is not None:
                raise InvalidWorkAssignmentError(
                    "没有 workflow_id 时不能指定 target_state_id"
                )
            return (
                None,
                None,
                frozenset(),
                f"workspace:{workspace_id}",
                tuple(base_scopes),
            )
        definition = self._workflows.get(request.workflow_id)
        target_state_id = request.target_state_id or definition.initial_state
        if target_state_id != definition.initial_state and not (
            current.workflow_id == definition.id
            and current.workflow_state_id == target_state_id
        ):
            raise InvalidWorkflowStateError(
                "新 WorkAssignment 只能进入 initial_state；"
                "角色重绑定只能显式保留当前 Workflow State"
            )
        state = definition.states[target_state_id]
        offered = state.allowed_capabilities
        scopes = self._assignment_memory_scopes(
            state.memory_scopes,
            current.task_id,
            definition.id,
            current.identity_id,
            workspace_id,
            current.profession_ids,
        )
        return (
            definition.id,
            state.id,
            offered,
            self._resolve_memory_scope(
                state.memory_write_scope, current.task_id, definition.id
            ),
            scopes,
        )

    def _fail_with_event(
        self, agent_id: str, stage: str, error: Exception
    ) -> None:
        process = self._agents.process(agent_id)
        if process.status in {AgentStatus.RUNNING, AgentStatus.WAITING}:
            self.end_work_assignment(agent_id, reason=f"Runtime failed: {stage}")
            self._agents.fail(agent_id)
        self._emit(
            EventType.ERROR,
            agent_id=agent_id,
            data={"stage": stage, "error_type": type(error).__name__},
        )

    def _emit(
        self,
        event_type: EventType,
        *,
        agent_id: str | None = None,
        trace_id: str | None = None,
        request_id: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> RuntimeEvent:
        if agent_id is not None:
            trace_id = trace_id or self._agents.runtime_state(agent_id).task_id
        return self._events.emit(
            RuntimeEvent(
                type=event_type,
                trace_id=trace_id or "runtime",
                agent_id=agent_id,
                request_id=request_id,
                data=data or {},
            )
        )

    @staticmethod
    def _resolve_memory_scopes(
        declared: tuple[str, ...], task_id: str, workflow_id: str
    ) -> tuple[str, ...]:
        resolved = ["core"]
        for scope in declared:
            value = RuntimeEngine._resolve_memory_scope(
                scope, task_id, workflow_id
            )
            if value not in resolved:
                resolved.append(value)
        return tuple(resolved)

    @staticmethod
    def _assignment_memory_scopes(
        declared: tuple[str, ...],
        task_id: str,
        workflow_id: str,
        identity_id: str,
        workspace_id: str,
        profession_ids: tuple[str, ...],
    ) -> tuple[str, ...]:
        resolved = list(
            RuntimeEngine._resolve_memory_scopes(
                declared, task_id, workflow_id
            )
        )
        mounted_scopes = [f"workspace:{workspace_id}"]
        if profession_ids:
            mounted_scopes.append(f"identity:{identity_id}")
        for scope in mounted_scopes:
            if scope not in resolved:
                resolved.append(scope)
        return tuple(resolved)

    def _binding_memory_scopes(
        self,
        current: RuntimeState,
        *,
        profession_ids: tuple[str, ...],
        include_workflow: bool = True,
    ) -> tuple[str, ...]:
        """Compose read scopes without consulting cognitive ranking policy."""

        scopes = ["core"]
        if profession_ids:
            scopes.append(f"identity:{current.identity_id}")
        if include_workflow and current.assignment_id:
            scopes.append(f"workspace:{self._workspace_id}")
        if include_workflow and current.assignment_id and current.workflow_id:
            for scope in current.memory_scopes:
                if scope == "core" or scope.startswith(
                    ("identity:", "workspace:")
                ):
                    continue
                if scope not in scopes:
                    scopes.append(scope)
        return tuple(scopes)

    @staticmethod
    def _resolve_memory_scope(
        declared: str, task_id: str, workflow_id: str
    ) -> str:
        if declared == "core":
            return "core"
        if declared == "task":
            return f"task:{task_id}"
        if declared == "workflow":
            return f"workflow:{workflow_id}"
        return f"workflow:{workflow_id}:{declared}"
