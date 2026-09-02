"""Assemble the complete Agent Context from runtime-owned resources."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from hashlib import sha256
import json
from math import ceil
from typing import Any, Callable, Mapping, Protocol

from harness.capability import CapabilityRegistry
from harness.cognitive_policy import CognitivePolicy, CognitivePolicyRegistry
from harness.identity import Identity
from harness.memory_manager import MemoryProvider
from harness.models import (
    AgentProcess,
    MemoryRecord,
    OperationalContext,
    RuntimeState,
    TaskSpec,
    WorkAssignment,
)
from harness.persona import Persona
from harness.profession import Profession, ProfessionRegistry
from harness.skill import SkillRegistry
from harness.workflow_manager import WorkflowRegistry
from harness.work_role import WorkRoleRegistry
from harness.workspace import Workspace


class ContextProvider(Protocol):
    def provide(self, runtime_state: RuntimeState) -> dict[str, Any]: ...


ModelProjectionOverride = Callable[
    [
        dict[str, Any],
        OperationalContext,
        TaskSpec,
        list[tuple[MemoryRecord, float, dict[str, float]]],
        dict[str, str] | None,
    ],
    tuple[dict[str, Any], dict[str, Any]],
]


class ContextManager:
    _REGIONS = ("runtime_envelope", "life", "profession", "work")

    def __init__(
        self,
        memory: MemoryProvider,
        workflows: WorkflowRegistry,
        roles: WorkRoleRegistry,
        capabilities: CapabilityRegistry,
        policies: CognitivePolicyRegistry,
        professions: ProfessionRegistry,
        skills: SkillRegistry,
        workspace: Workspace,
        *,
        providers: tuple[ContextProvider, ...] = (),
        policy_strength: float = 1.0,
        model_projection_override: ModelProjectionOverride | None = None,
        total_context_budget: int = 12_000,
        region_context_budgets: Mapping[str, int] | None = None,
        max_memory_records: int = 20,
        action_result_excerpt_characters: int = 1_200,
    ):
        if (
            not isinstance(policy_strength, (int, float))
            or isinstance(policy_strength, bool)
            or not 0.0 <= float(policy_strength) <= 1.0
        ):
            raise ValueError("policy_strength 必须在 0 到 1 之间")
        if total_context_budget < 1:
            raise ValueError("total_context_budget must be positive")
        configured_regions = dict(
            region_context_budgets
            or {
                "runtime_envelope": 2_000,
                "life": 2_000,
                "profession": 2_500,
                "work": 5_500,
            }
        )
        if set(configured_regions) != set(self._REGIONS):
            raise ValueError("region_context_budgets must cover all four regions")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in configured_regions.values()
        ):
            raise ValueError("region_context_budgets must contain positive integers")
        if sum(configured_regions.values()) > total_context_budget:
            raise ValueError("region budgets cannot exceed total_context_budget")
        if max_memory_records < 1:
            raise ValueError("max_memory_records must be positive")
        if action_result_excerpt_characters < 1:
            raise ValueError("action_result_excerpt_characters must be positive")
        self.memory = memory
        self.workflows = workflows
        self.roles = roles
        self.capabilities = capabilities
        self.policies = policies
        self.professions = professions
        self.skills = skills
        self.workspace = workspace
        self.providers = providers
        self.policy_strength = float(policy_strength)
        self.model_projection_override = model_projection_override
        self.total_context_budget = total_context_budget
        self.region_context_budgets = configured_regions
        self.max_memory_records = max_memory_records
        self.action_result_excerpt_characters = (
            action_result_excerpt_characters
        )

    def build(
        self,
        process: AgentProcess,
        runtime: RuntimeState,
        task: TaskSpec,
        identity: Identity,
        assignment: WorkAssignment | None,
        persona: Persona,
    ) -> "ContextBundle":
        policy = (
            self.policies.get(runtime.cognitive_policy_id)
            if runtime.cognitive_policy_id
            else None
        )
        active_professions = tuple(
            self.professions.get(item) for item in runtime.profession_ids
        )
        workflow_context = None
        memory_hints: tuple[str, ...] = ()
        if runtime.workflow_id:
            definition = self.workflows.get(runtime.workflow_id)
            state = definition.states[runtime.workflow_state_id or ""]
            memory_hints = state.memory_hints
            workflow_context = {
                "id": definition.id,
                "description": definition.description,
                "instruction": definition.instruction,
                "current_state": state.as_context(),
                "outgoing_transitions": [
                    edge.as_context()
                    for edge in definition.outgoing(state.id)
                ],
                "state_space": {
                    "initial_state": definition.initial_state,
                    "states": sorted(definition.states),
                    "transitions": {
                        source: [edge.as_context() for edge in edges]
                        for source, edges in definition.transitions.items()
                    },
                },
            }

        role_context = (
            self.roles.get(assignment.role_id).as_context()
            if assignment
            else None
        )
        profession_hints = tuple(
            tag for item in active_professions for tag in item.memory_tags
        )
        workspace_hints = (
            self.workspace.retrieval_preferences
            if assignment and assignment.workspace_id == self.workspace.id
            else ()
        )
        all_hints = tuple(
            dict.fromkeys((*memory_hints, *profession_hints, *workspace_hints))
        )
        legal_memory_records = self._legal_memory_candidates(
            runtime, active_professions, all_hints
        )
        task_relevant_records = self._task_relevant_candidates(
            legal_memory_records, task
        )
        resource_relevant_records = self._resource_relevant_candidates(
            task_relevant_records, active_professions, assignment
        )
        ranked_with_scores = self._rank_memory_with_scores(
            resource_relevant_records,
            task,
            active_professions,
            policy,
            assignment,
        )
        deduplicated_ranked, duplicate_records = self._deduplicate_ranked(
            ranked_with_scores
        )
        _, preferences, history, communication = self._partition_memory(
            resource_relevant_records
        )
        extensions = {}
        for provider in self.providers:
            extensions.update(provider.provide(runtime))
        operational_base = OperationalContext(
            system={
                "runtime": "SForge V1.6",
                "agent_id": process.id,
                "status": process.status.value,
                "workspace_catalog": [
                    {
                        "id": self.workspace.id,
                        "description": self.workspace.description,
                    }
                ],
                "workflow_catalog": self.workflows.available(),
                "work_role_catalog": self.roles.available(),
                "cognitive_policy_catalog": self.policies.available(),
                "profession_catalog": self.professions.available(),
                "skill_catalog": self.skills.available(),
                "rules": [
                    "Agent 只返回 FinalAnswer、ActionRequest、ResourceBindingRequest、WorkflowRequest 或 WorkAssignmentRequest",
                    "初始运行时只有 Core Memory 与基础 Capability",
                    "CognitivePolicy 与 Profession 可独立申请绑定或解绑，且不产生 Capability",
                    "Workspace 只可发现基本 metadata；项目上下文必须通过 WorkAssignment 进入",
                    "WorkRole 只存在于 WorkAssignment 中，不是独立运行时状态",
                    "WorkAssignment 是 Workspace、Role、Task、Workflow 与 Capability Grants 的唯一工作关系",
                    "Profession 提供专业资源；Skill 只是声明式方法知识，二者不能执行外部动作",
                    "CognitivePolicy 只能排序已经合法可见的候选信息，不能扩展 Memory scope",
                    "外部操作必须经过 Harness Capability boundary，且只能请求上下文列出的 Capability",
                ],
            },
            task={
                "id": task.id,
                "request": task.request,
                "context": deepcopy(task.context),
            },
            runtime={**runtime.as_context(), "extensions": extensions},
            identity=identity.as_context(),
            cognitive_policy=(self._policy_context(policy) if policy else None),
            professions=tuple(item.as_context() for item in active_professions),
            skills=self._skill_contexts(active_professions, assignment),
            workspace=(
                self.workspace.as_context()
                if assignment and assignment.workspace_id == self.workspace.id
                else None
            ),
            work_assignment=(assignment.as_context() if assignment else None),
            work_role=role_context,
            workflow=workflow_context,
            capabilities=self.capabilities.descriptors(
                runtime.allowed_capabilities
            ),
            memory=(),
        )
        cognitive_projection = (
            policy.compile_model_projection() if policy else None
        )
        ranked_memory_records, budget_dropped = self._select_memory_with_budget(
            deduplicated_ranked,
            operational_base,
            cognitive_projection,
        )
        operational_memory, _, _, _ = self._partition_memory(
            ranked_memory_records
        )
        operational = replace(
            operational_base,
            memory=tuple(operational_memory),
        )
        model_projection = self._model_context(
            operational, cognitive_projection
        )
        pre_override_region_sizes = self._region_size_estimates(model_projection)
        pre_override_total_size = self._estimate_tokens(model_projection)
        unbounded_operational = replace(
            operational_base,
            memory=tuple(
                item[0] for item in ranked_with_scores[: self.max_memory_records]
            ),
        )
        unbounded_context_size = self._estimate_tokens(
            self._model_context(
                unbounded_operational,
                cognitive_projection,
                bound_action_results=False,
            )
        )
        projection_trace: dict[str, Any] = {
            "model_context_regions": [
                "runtime_envelope",
                "life",
                "profession",
                "work",
            ],
            "cognitive_projection_direction": (
                cognitive_projection.get("direction")
                if cognitive_projection
                else None
            ),
            "estimated_context_tokens": pre_override_total_size,
            "region_size_estimates": pre_override_region_sizes,
            "unbounded_context_tokens_estimate": unbounded_context_size,
        }
        if self.model_projection_override is not None:
            model_projection, override_trace = self.model_projection_override(
                model_projection,
                operational,
                task,
                ranked_with_scores,
                cognitive_projection,
            )
            projection_trace.update(override_trace)
        operational = replace(
            operational, model_projection=model_projection
        )
        response_rendering = {
            "persona": persona.as_context(),
            "user_request": task.request,
            "user_preferences": [self._memory_context(item) for item in preferences],
            "interaction_history": [self._memory_context(item) for item in history],
            "communication_memory": [
                self._memory_context(item) for item in communication
            ],
        }
        retrieval_trace = {
            "pipeline": [
                "legal_access_scope",
                "task_relevance",
                "profession_workspace_relevance",
                "cognitive_policy_ranking",
                "context_budget",
            ],
            "fixture_scope": task.context.get("fixture_scope"),
            "policy_id": policy.id if policy else None,
            "policy_strength": self.policy_strength if policy else 0.0,
            "legal_memory_ids": [item.id for item in legal_memory_records],
            "task_relevant_memory_ids": [
                item.id for item in task_relevant_records
            ],
            "resource_relevant_memory_ids": [
                item.id for item in resource_relevant_records
            ],
            "ranked_memory_ids": [item[0].id for item in ranked_with_scores],
            "context_memory_ids": [item.id for item in ranked_memory_records],
            "deduplicated_memory_ids": [
                item.id for item in duplicate_records
            ],
            "budget_dropped_memory_ids": [
                item.id for item in budget_dropped
            ],
            "retrieval_scores": [
                {
                    "id": record.id,
                    "rank": rank,
                    "score": round(score, 8),
                    "components": components,
                }
                for rank, (record, score, components) in enumerate(
                    ranked_with_scores, start=1
                )
            ],
            "context_budget": self.max_memory_records,
            "context_budget_tokens": self.total_context_budget,
            "context_region_budgets": dict(self.region_context_budgets),
            "candidate_memory_count": len(resource_relevant_records),
            "deduplicated_memory_count": len(duplicate_records),
            "selected_memory_count": len(ranked_memory_records),
            "budget_dropped_memory_count": len(budget_dropped),
            **projection_trace,
        }
        return ContextBundle(
            operational, response_rendering, retrieval_trace
        )

    def _model_context(
        self,
        operational: OperationalContext,
        cognitive_projection: dict[str, str] | None,
        *,
        bound_action_results: bool = True,
    ) -> dict[str, Any]:
        """Project fine-grained runtime ontology into four stable model regions."""

        raw = operational.as_dict()
        memories = [
            (
                self._memory_context_for_model(record)
                if bound_action_results
                else self._memory_context(record)
            )
            for record in operational.memory
        ]
        core_memory = [item for item in memories if item["scope"] == "core"]
        professional_memory = [
            item
            for item in memories
            if str(item["scope"]).startswith("identity:")
        ]
        recent_observations = [
            item
            for item in memories
            if item["kind"] in {
                "runtime.action_result",
                "runtime.final_answer",
            }
        ]
        work_memory = [
            item
            for item in memories
            if item not in core_memory
            and item not in professional_memory
            and item not in recent_observations
        ]
        profession_skills = [
            item
            for item in raw["skills"]
            if any(
                str(source).startswith("profession:")
                for source in item.get("sources", [])
            )
        ]
        local_skills = [
            item
            for item in raw["skills"]
            if any(
                str(source).startswith("workspace:")
                for source in item.get("sources", [])
            )
        ]
        task = deepcopy(raw["task"])
        task["context"] = {
            key: value
            for key, value in task.get("context", {}).items()
            if not str(key).startswith("_")
        }
        active_policy = None
        if raw["cognitive_policy"] is not None:
            active_policy = {
                "id": raw["cognitive_policy"]["id"],
                "orientation": deepcopy(cognitive_projection),
            }
        return {
            "runtime_envelope": {
                "protocol": {
                    "rules": deepcopy(raw["system"].get("rules", [])),
                    "external_action_path": (
                        "ActionRequest -> Admission -> Capability -> ActionResult"
                    ),
                },
                "token_budget": {
                    "estimated_total_context": self.total_context_budget,
                    "regions": dict(self.region_context_budgets),
                    "memory_records": self.max_memory_records,
                    "large_action_result_excerpt_characters": (
                        self.action_result_excerpt_characters
                    ),
                    "model_output_tokens": "provider-configured",
                },
                "runtime": {
                    "name": raw["system"].get("runtime"),
                    "status": raw["system"].get("status"),
                    "mode": raw["runtime"].get("mode"),
                    "state_version": raw["runtime"].get("version"),
                    "extensions": deepcopy(
                        raw["runtime"].get("extensions", {})
                    ),
                },
            },
            "life": {
                "identity": deepcopy(raw["identity"]),
                "core_memory": core_memory,
                "cognitive_configuration": active_policy,
                "available_cognitive_configurations": deepcopy(
                    raw["system"].get("cognitive_policy_catalog", [])
                ),
            },
            "profession": {
                "active_resources": deepcopy(raw["professions"]),
                "professional_memory": professional_memory,
                "methods_and_skills": profession_skills,
                "available_professions": deepcopy(
                    raw["system"].get("profession_catalog", [])
                ),
                "available_skills": deepcopy(
                    raw["system"].get("skill_catalog", [])
                ),
            },
            "work": {
                "assignment": deepcopy(raw["work_assignment"]),
                "workspace": deepcopy(raw["workspace"]),
                "role": deepcopy(raw["work_role"]),
                "task": task,
                "workflow": deepcopy(raw["workflow"]),
                "relevant_archive_and_artifacts": work_memory,
                "local_skills": local_skills,
                "recent_observations": recent_observations,
                "capability_boundary": {
                    "available": deepcopy(raw["capabilities"]),
                    "authority_source": (
                        "base grants plus active WorkAssignment grants"
                    ),
                },
                "available_workspaces": deepcopy(
                    raw["system"].get("workspace_catalog", [])
                ),
                "available_workflows": deepcopy(
                    raw["system"].get("workflow_catalog", [])
                ),
                "available_roles": deepcopy(
                    raw["system"].get("work_role_catalog", [])
                ),
            },
        }

    def _select_memory_with_budget(
        self,
        ranked: list[tuple[MemoryRecord, float, dict[str, float]]],
        operational_base: OperationalContext,
        cognitive_projection: dict[str, str] | None,
    ) -> tuple[list[MemoryRecord], list[MemoryRecord]]:
        """Select ranked records by trial-projecting the real model context."""

        selected: list[MemoryRecord] = []
        dropped: list[MemoryRecord] = []
        for record, _, _ in ranked:
            if len(selected) >= self.max_memory_records:
                dropped.append(record)
                continue
            trial = [*selected, record]
            operational = replace(operational_base, memory=tuple(trial))
            projection = self._model_context(
                operational, cognitive_projection
            )
            region_sizes = self._region_size_estimates(projection)
            if (
                self._estimate_tokens(projection)
                > self.total_context_budget
                or any(
                    region_sizes[region] > budget
                    for region, budget in self.region_context_budgets.items()
                )
            ):
                dropped.append(record)
                continue
            selected.append(record)
        return selected, dropped

    @classmethod
    def _deduplicate_ranked(
        cls,
        ranked: list[tuple[MemoryRecord, float, dict[str, float]]],
    ) -> tuple[
        list[tuple[MemoryRecord, float, dict[str, float]]],
        list[MemoryRecord],
    ]:
        selected: list[tuple[MemoryRecord, float, dict[str, float]]] = []
        duplicates: list[MemoryRecord] = []
        seen: set[tuple[str, ...]] = set()
        for item in ranked:
            record = item[0]
            key = cls._memory_dedup_key(record)
            if key in seen:
                duplicates.append(record)
                continue
            seen.add(key)
            selected.append(item)
        return selected, duplicates

    @classmethod
    def _memory_dedup_key(cls, record: MemoryRecord) -> tuple[str, ...]:
        family = (
            "action_result"
            if record.kind.endswith(".action_result")
            else record.kind
        )
        metadata = record.metadata
        resource = str(
            metadata.get("resource_reference")
            or metadata.get("source")
            or metadata.get("artifact_ref")
            or ""
        )
        if family == "action_result":
            try:
                payload = json.loads(record.content)
            except (TypeError, json.JSONDecodeError):
                payload = {}
            capability = str(
                metadata.get("capability_id")
                or payload.get("capability_id")
                or ""
            )
            status = str(
                metadata.get("status") or payload.get("status") or ""
            )
            output_hash = str(metadata.get("output_sha256") or "")
            if not output_hash and "output" in payload:
                output_hash = cls._content_hash(
                    json.dumps(
                        payload.get("output"),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
            if output_hash:
                return family, capability, status, resource, output_hash
            request_id = str(metadata.get("request_id") or "")
            if request_id:
                return family, request_id
        normalized = " ".join(record.content.split())
        return family, resource, cls._content_hash(normalized)

    def _memory_context_for_model(
        self, record: MemoryRecord
    ) -> dict[str, Any]:
        context = self._memory_context(record)
        if not record.kind.endswith(".action_result"):
            return context
        try:
            payload = json.loads(record.content)
        except (TypeError, json.JSONDecodeError):
            payload = {"output": record.content}
        output = payload.get("output")
        serialized = self._serialized_output(output)
        if len(serialized) <= self.action_result_excerpt_characters:
            return context
        payload["output"] = {
            "projection": "bounded_excerpt",
            "type": type(output).__name__,
            "size_characters": len(serialized),
            "sha256": self._content_hash(serialized),
            "excerpt": serialized[: self.action_result_excerpt_characters],
        }
        context["content"] = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        context["metadata"]["context_projection"] = {
            "full_payload_persisted": True,
            "truncated": True,
            "original_characters": len(serialized),
            "sha256": self._content_hash(serialized),
        }
        return context

    @staticmethod
    def _serialized_output(value: Any) -> str:
        if isinstance(value, str):
            return value
        try:
            return json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _content_hash(value: str) -> str:
        return sha256(value.encode("utf-8")).hexdigest()

    @classmethod
    def _estimate_tokens(cls, value: Any) -> int:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return max(1, ceil(len(serialized.encode("utf-8")) / 4))

    @classmethod
    def _region_size_estimates(
        cls, model_context: dict[str, Any]
    ) -> dict[str, int]:
        return {
            region: cls._estimate_tokens(model_context.get(region, {}))
            for region in cls._REGIONS
        }

    def _legal_memory_candidates(
        self,
        runtime: RuntimeState,
        professions: tuple[Profession, ...],
        hints: tuple[str, ...],
    ) -> list[MemoryRecord]:
        """Retrieve only mounted scopes before policy-based ranking."""

        records: list[MemoryRecord] = []
        seen: set[str] = set()
        for scope in runtime.memory_scopes:
            scoped = self.memory.retrieve(scope=scope, limit=50)
            for hint in hints:
                scoped.extend(
                    self.memory.retrieve(scope=scope, query=hint, limit=20)
                )
            for record in scoped:
                if record.id in seen:
                    continue
                seen.add(record.id)
                records.append(record)
        return records

    @classmethod
    def _task_relevant_candidates(
        cls, records: list[MemoryRecord], task: TaskSpec
    ) -> list[MemoryRecord]:
        """Apply explicit task/fixture constraints before resource ranking."""

        active_fixture = task.context.get("fixture_scope")
        result: list[MemoryRecord] = []
        for record in records:
            metadata = record.metadata
            # ``task_id`` is established provenance on Workspace Archive
            # records. Only retrieval-specific keys are hard constraints.
            declared_task = metadata.get("retrieval_task_id")
            if declared_task is not None and declared_task != task.id:
                continue
            declared_tasks = cls._metadata_tags(
                metadata.get("retrieval_task_ids")
            )
            if declared_tasks and task.id not in declared_tasks:
                continue
            declared_fixtures = cls._metadata_tags(
                metadata.get("fixture_scope")
            )
            if declared_fixtures and active_fixture not in declared_fixtures:
                continue
            result.append(record)
        return result

    @classmethod
    def _resource_relevant_candidates(
        cls,
        records: list[MemoryRecord],
        professions: tuple[Profession, ...],
        assignment: WorkAssignment | None,
    ) -> list[MemoryRecord]:
        """Apply Profession/Workspace relevance without granting access."""

        profession_tags = {
            tag.casefold() for item in professions for tag in item.memory_tags
        }
        result: list[MemoryRecord] = []
        for record in records:
            if record.scope.startswith("identity:") and not cls._profession_memory_match(
                record, profession_tags
            ):
                continue
            workspace_id = record.metadata.get("workspace_id")
            if workspace_id is not None and (
                assignment is None or assignment.workspace_id != workspace_id
            ):
                continue
            result.append(record)
        return result

    @classmethod
    def _profession_memory_match(
        cls, record: MemoryRecord, profession_tags: set[str]
    ) -> bool:
        if not profession_tags:
            return False
        record_tags = cls._metadata_tags(
            record.metadata.get("professional_tags")
        ).union(
            cls._metadata_tags(record.metadata.get("related_to_profession"))
        )
        return bool(
            {item.casefold() for item in record_tags}.intersection(
                profession_tags
            )
        )

    def _rank_memory(
        self,
        records: list[MemoryRecord],
        task: TaskSpec,
        professions: tuple[Profession, ...],
        policy: CognitivePolicy | None,
        assignment: WorkAssignment | None,
    ) -> list[MemoryRecord]:
        return [
            item[0]
            for item in self._rank_memory_with_scores(
                records, task, professions, policy, assignment
            )[:20]
        ]

    def _rank_memory_with_scores(
        self,
        records: list[MemoryRecord],
        task: TaskSpec,
        professions: tuple[Profession, ...],
        policy: CognitivePolicy | None,
        assignment: WorkAssignment | None,
    ) -> list[tuple[MemoryRecord, float, dict[str, float]]]:
        task_terms = self._terms(task.request)
        profession_tags = {
            tag.casefold() for item in professions for tag in item.memory_tags
        }

        def scored(
            record: MemoryRecord,
        ) -> tuple[MemoryRecord, float, dict[str, float]]:
            text = f"{record.kind} {record.content}".casefold()
            tags = {
                item.casefold()
                for item in self._metadata_tags(
                    record.metadata.get("professional_tags")
                )
            }
            importance = (
                record.importance if record.importance is not None else 0.5
            )
            explicit_relevance = record.metadata.get("task_relevance")
            if isinstance(explicit_relevance, (int, float)) and not isinstance(
                explicit_relevance, bool
            ):
                task_relevance = 0.08 * float(explicit_relevance)
            else:
                task_relevance = 0.08 * len(
                    task_terms.intersection(self._terms(text))
                )
            profession_relevance = 0.12 * len(
                tags.intersection(profession_tags)
            )
            assignment_relevance = (
                0.15
                if assignment
                and record.metadata.get("task_id") == assignment.task_id
                else 0.0
            )
            policy_bias = (
                self.policy_strength * self._policy_score(policy, text, tags)
                if policy
                else 0.0
            )
            components = {
                "importance": round(float(importance), 8),
                "task_relevance": round(task_relevance, 8),
                "profession_relevance": round(profession_relevance, 8),
                "assignment_relevance": round(assignment_relevance, 8),
                "policy_bias": round(policy_bias, 8),
            }
            return record, sum(components.values()), components

        scored_records = [scored(record) for record in records]
        return sorted(
            scored_records,
            key=lambda item: (
                -item[1],
                -item[0].created_at.timestamp(),
                item[0].id,
            ),
        )

    def _policy_context(self, policy: CognitivePolicy) -> dict[str, Any]:
        context = policy.as_context()
        context["policy_strength"] = self.policy_strength
        context["effective_parameters"] = {
            section: {
                name: round(float(value) * self.policy_strength, 8)
                for name, value in values.items()
            }
            for section, values in context["parameters"].items()
        }
        context["strength_boundary"] = (
            "Only effective_parameters express active bias strength; "
            "strength never changes access or authority."
        )
        return context

    @staticmethod
    def _policy_score(
        policy: CognitivePolicy, text: str, tags: set[str]
    ) -> float:
        score = 0.0
        if any(
            word in text
            for word in ("contradiction", "conflict", "risk", "反例", "风险")
        ):
            score += 0.16 * policy.value("memory", "contradiction_weight")
        if any(
            word in text
            for word in ("novel", "idea", "new", "创新", "新颖")
        ):
            score += 0.16 * policy.value("memory", "novelty_weight")
        if any(
            word in text
            for word in ("precedent", "history", "prior", "历史", "先例")
        ):
            score += 0.16 * policy.value("memory", "precedent_weight")
        if len(tags) > 1:
            score += 0.12 * policy.value("memory", "cross_domain_weight")
        if any(
            word in text
            for word in ("evidence", "test", "verified", "证据", "测试")
        ):
            score += 0.12 * policy.value("cognition", "verification_weight")
        return score

    def _skill_contexts(
        self,
        professions: tuple[Profession, ...],
        assignment: WorkAssignment | None,
    ) -> tuple[dict[str, Any], ...]:
        profession_sources: dict[str, list[str]] = {}
        for profession in professions:
            for skill_id in profession.preferred_skills:
                profession_sources.setdefault(skill_id, []).append(
                    f"profession:{profession.id}"
                )
        workspace_skills = (
            set(self.workspace.local_skills)
            if assignment and assignment.workspace_id == self.workspace.id
            else set()
        )
        result: list[dict[str, Any]] = []
        for summary in self.skills.available():
            skill_id = summary["id"]
            sources = ["global", *profession_sources.get(skill_id, ())]
            if skill_id in workspace_skills:
                sources.append(f"workspace:{self.workspace.id}")
            result.append(
                self.skills.get(skill_id).as_context(sources=tuple(sources))
            )
        return tuple(result)

    @staticmethod
    def _terms(value: str) -> set[str]:
        return {
            item.strip(".,:;!?()[]{}\"'")
            for item in value.casefold().split()
            if item.strip(".,:;!?()[]{}\"'")
        }

    @staticmethod
    def _metadata_tags(value: Any) -> set[str]:
        if isinstance(value, str):
            return {value}
        if isinstance(value, (list, tuple, set)):
            return {item for item in value if isinstance(item, str)}
        return set()

    @staticmethod
    def _memory_context(record: MemoryRecord) -> dict[str, Any]:
        return {
            "id": record.id,
            "scope": record.scope,
            "kind": record.kind,
            "content": record.content,
            "metadata": deepcopy(record.metadata),
        }

    @staticmethod
    def _partition_memory(records: list[MemoryRecord]) -> tuple[
        list[MemoryRecord],
        list[MemoryRecord],
        list[MemoryRecord],
        list[MemoryRecord],
    ]:
        operational: list[MemoryRecord] = []
        preferences: list[MemoryRecord] = []
        history: list[MemoryRecord] = []
        communication: list[MemoryRecord] = []
        for record in records:
            kind = record.kind.casefold()
            if kind == "core.style" or kind.startswith(
                ("communication.preference", "user.communication.preference")
            ):
                preferences.append(record)
            elif kind.startswith(("communication.history", "interaction.")):
                history.append(record)
            elif kind.startswith("communication."):
                communication.append(record)
            else:
                operational.append(record)
        return operational, preferences, history, communication


@dataclass(frozen=True)
class ContextBundle:
    """Two independently assembled views with no shared control authority."""

    operational: OperationalContext
    response_rendering: dict[str, Any]
    retrieval_trace: dict[str, Any] = field(default_factory=dict)

    def response_rendering_for(self, draft: str) -> dict[str, Any]:
        context = deepcopy(self.response_rendering)
        context["draft_answer"] = draft
        return context

    @property
    def presentation(self) -> dict[str, Any]:
        """Compatibility alias; new code should use response_rendering."""

        return deepcopy(self.response_rendering)

    def presentation_for(self, draft: str) -> dict[str, Any]:
        """Compatibility alias for pre-V6 experiment integrations."""

        return self.response_rendering_for(draft)
