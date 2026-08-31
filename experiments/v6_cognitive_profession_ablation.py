"""Real-model V6 ablation experiment built on the existing SForge runtime.

Dry-run is network-free. Normal mode uses the existing Agent worker and reads
DeepSeek configuration from the project .env/environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from harness.memory_manager import InMemoryMemoryProvider  # noqa: E402
from harness.models import (  # noqa: E402
    ActionResult,
    FinalAnswer,
    MemoryRecord,
    ReasoningResponse,
    ResourceBindingAdmission,
    ResourceBindingRequest,
    TaskSpec,
    WorkAssignmentAdmission,
    WorkAssignmentRequest,
    WorkflowAdmission,
)
from harness.process_supervisor import AgentProcessSupervisor  # noqa: E402
from runtime import create_runtime  # noqa: E402
from agent.decision_protocol import parse_decision_payload  # noqa: E402
from experiments.evaluators import (  # noqa: E402
    evaluate_causal_selection,
    evaluate_decision,
    evaluate_mechanism,
    evaluate_policy_transmission,
    evaluate_presentation,
)


TASKS_DIR = PROJECT_ROOT / "experiments" / "tasks"
CAUSAL_TASKS_DIR = TASKS_DIR / "policy_causal"
FINAL_VALIDATION_TASKS_DIR = TASKS_DIR / "policy_final"
WORKSPACE_ROOT = PROJECT_ROOT / "experiments" / "workspace"
WORKSPACE_ID = "v6_cognitive_ablation"
SCHEMA_VERSION = "sforge.v6.ablation.run.v6"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_POLICY = "INTJ"
DEFAULT_POLICY_B = "ENFP"
DEFAULT_PROFESSION = "software_engineering"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 2048
DEFAULT_MAX_STEPS = 6


@dataclass(frozen=True)
class ExperimentCondition:
    id: str
    cognitive_policy_id: str | None
    profession_ids: tuple[str, ...]
    policy_strength: float = 1.0
    projection_mode: str = "legacy"
    uses_fixture_policy: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "cognitive_policy_id": self.cognitive_policy_id,
            "profession_ids": list(self.profession_ids),
            "policy_strength": self.policy_strength,
            "projection_mode": self.projection_mode,
            "uses_fixture_policy": self.uses_fixture_policy,
        }


@dataclass(frozen=True)
class ExperimentTask:
    id: str
    title: str
    request: str
    required_findings: tuple[dict[str, Any], ...]
    forbidden_claims: tuple[dict[str, Any], ...]
    source: str
    suite: str = "capability"
    category: str = "software_engineering"
    expected_action_type: str = "final"
    mechanism: dict[str, Any] | None = None
    fixture_scope: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "request": self.request,
            "required_findings": [dict(item) for item in self.required_findings],
            "forbidden_claims": [dict(item) for item in self.forbidden_claims],
            "source": self.source,
            "suite": self.suite,
            "category": self.category,
            "expected_action_type": self.expected_action_type,
            "mechanism": dict(self.mechanism or {}),
            "fixture_scope": self.fixture_scope,
        }


class DryRunSupervisor:
    """A local supervisor that makes accidental model use fail immediately."""

    def __init__(self) -> None:
        self._alive: set[str] = set()

    def spawn(self, run_id: str) -> str:
        process_id = f"dry-run-{run_id}-{uuid4().hex}"
        self._alive.add(process_id)
        return process_id

    def reason(self, process_id: str, messages: list[dict]) -> ReasoningResponse:
        raise AssertionError("--dry-run must not call the model")

    def terminate(self, process_id: str | None) -> None:
        if process_id:
            self._alive.discard(process_id)

    def is_alive(self, process_id: str) -> bool:
        return process_id in self._alive

    def close(self) -> None:
        self._alive.clear()


class RecordingSupervisor:
    """Observe the existing OS-process worker without changing its decisions."""

    def __init__(self, delegate: AgentProcessSupervisor | None = None) -> None:
        self.delegate = delegate or AgentProcessSupervisor()
        self.calls: list[dict[str, Any]] = []

    def spawn(self, run_id: str) -> str:
        return self.delegate.spawn(run_id)

    def reason(self, process_id: str, messages: list[dict]) -> ReasoningResponse:
        started = time.perf_counter()
        payload = _system_payload(messages)
        phase = (
            "presentation"
            if str(messages[0].get("content", "")).startswith(
                "Formatting instructions:"
            )
            else "decision"
        )
        rendering_input = (
            _message_payload(messages, 1) if phase == "presentation" else {}
        )
        call: dict[str, Any] = {
            "index": len(self.calls),
            "phase": phase,
            "messages": messages,
            "structured_context": payload.get("context"),
            "response_rendering_input": rendering_input,
            "latency_ms": None,
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "raw_response": None,
            "decision_protocol": None,
            "error": None,
        }
        try:
            response = self.delegate.reason(process_id, messages)
            call["raw_response"] = response.content
            if phase == "decision":
                call["decision_protocol"] = parse_decision_payload(
                    response.content
                ).instrumentation()
            call["usage"] = response.usage.as_dict()
            return response
        except Exception as exc:
            call["error"] = _exception_payload(exc)
            raise
        finally:
            call["latency_ms"] = round(
                (time.perf_counter() - started) * 1000, 3
            )
            call["input_estimate"] = token_estimate(messages)
            self.calls.append(call)

    def terminate(self, process_id: str | None) -> None:
        self.delegate.terminate(process_id)

    def is_alive(self, process_id: str) -> bool:
        return self.delegate.is_alive(process_id)

    def close(self) -> None:
        self.delegate.close()


def _experiment_model_projection(
    base_context: dict[str, Any],
    operational,
    task: TaskSpec,
    ranked_with_scores: list[
        tuple[MemoryRecord, float, dict[str, float]]
    ],
    cognitive_projection: dict[str, str] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply temporary causal treatments outside the SForge core."""

    config = task.context.get("_experiment_projection")
    if not isinstance(config, dict):
        return base_context, {}
    mode = str(config.get("mode", ""))
    allowed_modes = {
        "neutral",
        "order_only",
        "explicit_rank",
        "reasoning_only",
        "full",
    }
    if mode not in allowed_modes:
        raise ValueError(f"Unknown experiment projection mode: {mode}")
    evidence_ids = tuple(
        item
        for item in config.get("evidence_ids", [])
        if isinstance(item, str) and item
    )
    if len(evidence_ids) != 2 or len(set(evidence_ids)) != 2:
        raise ValueError("experiment projection requires two evidence ids")
    neutral_order = tuple(
        item
        for item in config.get("neutral_order", [])
        if isinstance(item, str) and item
    )
    if set(neutral_order) != set(evidence_ids) or len(neutral_order) != 2:
        raise ValueError("neutral order must contain exactly both evidence ids")
    runtime_ranked = [
        record.id
        for record, _, _ in ranked_with_scores
        if record.id in evidence_ids
    ]
    if set(runtime_ranked) != set(evidence_ids):
        raise ValueError("experiment evidence failed legal relevance retrieval")
    model_order = (
        list(neutral_order)
        if mode in {"neutral", "reasoning_only"}
        else runtime_ranked
    )
    raw = operational.as_dict()
    memory_by_id = {item["id"]: item for item in raw["memory"]}
    if not all(item in memory_by_id for item in evidence_ids):
        raise ValueError("experiment evidence exceeded context budget")
    visible_evidence: list[dict[str, Any]] = []
    for index, memory_id in enumerate(model_order, start=1):
        source = memory_by_id[memory_id]
        item = {
            "evidence_id": source["id"],
            "content": source["content"],
        }
        if mode == "explicit_rank":
            item["retrieval_rank"] = index
            item["retrieval_priority"] = "high" if index == 1 else "low"
        visible_evidence.append(item)

    context = deepcopy(base_context)
    rules = context["runtime_envelope"]["protocol"].get("rules", [])
    context["runtime_envelope"]["protocol"]["rules"] = [
        rule
        for rule in rules
        if "cognitivepolicy" not in str(rule).casefold()
        and "profession" not in str(rule).casefold()
        and "skill" not in str(rule).casefold()
    ]
    if mode == "explicit_rank":
        context["runtime_envelope"]["protocol"]["evidence_handling"] = (
            "retrieval rank expresses attention order only; it is not "
            "confidence, correctness, or a truth score"
        )
    identity = context["life"].get("identity")
    if isinstance(identity, dict):
        identity.pop("default_cognitive_policy_id", None)
    context["life"]["core_memory"] = []
    context["life"]["available_cognitive_configurations"] = []
    context["life"]["cognitive_configuration"] = None
    compiled_direction = None
    if mode in {"reasoning_only", "full"}:
        compiled = (
            cognitive_projection
            if config.get("guidance_source") == "cognitive_policy"
            else None
        )
        guidance = (
            compiled.get("reasoning_guidance")
            if compiled
            else config.get("reasoning_guidance")
        )
        if not isinstance(guidance, str) or not guidance.strip():
            raise ValueError("experiment reasoning projection lacks guidance")
        context["life"]["cognitive_configuration"] = {
            "operational_guidance": guidance.strip()
        }
        compiled_direction = compiled.get("direction") if compiled else None

    context["profession"] = {
        "active_resources": [],
        "professional_memory": [],
        "methods_and_skills": [],
        "available_professions": [],
        "available_skills": [],
    }
    work = context["work"]
    assignment = work.get("assignment")
    if isinstance(assignment, dict):
        for key in ("id", "agent_process_id", "identity_id"):
            assignment.pop(key, None)
    workspace = work.get("workspace")
    if isinstance(workspace, dict):
        work["workspace"] = {
            key: workspace.get(key)
            for key in ("id", "description")
            if key in workspace
        }
    role = work.get("role")
    if isinstance(role, dict):
        work["role"] = {
            key: deepcopy(role.get(key))
            for key in (
                "id",
                "description",
                "instructions",
                "evaluation_criteria",
            )
            if key in role
        }
    work["task"]["context"] = {
        "decision_requirements": deepcopy(
            task.context.get("decision_requirements", {})
        )
    }
    work["relevant_archive_and_artifacts"] = visible_evidence
    work["local_skills"] = []
    work["recent_observations"] = []
    work["available_workflows"] = []
    work["available_roles"] = []
    return context, {
        "model_projection_mode": mode,
        "runtime_ranked_evidence_ids": runtime_ranked,
        "model_visible_evidence_ids": model_order,
        "model_visible_explicit_priority": mode == "explicit_rank",
        "model_visible_reasoning_guidance": mode
        in {"reasoning_only", "full"},
        "compiled_policy_direction": compiled_direction,
        "model_policy_metadata_hidden": True,
    }


def build_conditions(
    policy_id: str = DEFAULT_POLICY,
    profession_id: str = DEFAULT_PROFESSION,
    policy_b_id: str = DEFAULT_POLICY_B,
) -> tuple[ExperimentCondition, ...]:
    policy = policy_id.strip().upper()
    policy_b = policy_b_id.strip().upper()
    profession = profession_id.strip()
    if not policy or not policy_b or not profession:
        raise ValueError("policies and profession must be non-empty")
    if policy == policy_b:
        raise ValueError("policy A and policy B must differ")
    return (
        ExperimentCondition("base", None, (), 0.0),
        ExperimentCondition("profession_only", None, (profession,), 0.0),
        ExperimentCondition("policy_a_only", policy, ()),
        ExperimentCondition("profession_and_policy_a", policy, (profession,)),
        ExperimentCondition("policy_b_only", policy_b, ()),
    )


def build_strength_sweep_conditions(
    policy_id: str = DEFAULT_POLICY,
    policy_b_id: str = DEFAULT_POLICY_B,
    profession_id: str = DEFAULT_PROFESSION,
    strengths: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> tuple[ExperimentCondition, ...]:
    policies = (("a", policy_id.strip().upper()), ("b", policy_b_id.strip().upper()))
    profession = profession_id.strip()
    if (
        not all(policy for _, policy in policies)
        or policies[0][1] == policies[1][1]
        or not profession
    ):
        raise ValueError(
            "policy A/B and profession must be non-empty; policies must differ"
        )
    normalized: list[float] = []
    for strength in strengths:
        if (
            not isinstance(strength, (int, float))
            or isinstance(strength, bool)
            or not 0.0 <= float(strength) <= 1.0
        ):
            raise ValueError("policy strengths must be between 0 and 1")
        value = float(strength)
        if value not in normalized:
            normalized.append(value)
    if not normalized:
        raise ValueError("at least one policy strength is required")
    return tuple(
        ExperimentCondition(
            f"policy_{label}_s{round(strength * 100):03d}",
            policy,
            (profession,),
            strength,
        )
        for label, policy in policies
        for strength in normalized
    )


def build_policy_transmission_conditions(
    policy_id: str = DEFAULT_POLICY,
    policy_b_id: str = DEFAULT_POLICY_B,
    profession_id: str = DEFAULT_PROFESSION,
) -> tuple[ExperimentCondition, ...]:
    """Keep Profession fixed while changing only CognitivePolicy."""

    policy = policy_id.strip().upper()
    policy_b = policy_b_id.strip().upper()
    profession = profession_id.strip()
    if not policy or not policy_b or policy == policy_b or not profession:
        raise ValueError(
            "policy A/B and profession must be non-empty; policies must differ"
        )
    return (
        ExperimentCondition(
            "policy_a_transmission", policy, (profession,), 1.0
        ),
        ExperimentCondition(
            "policy_b_transmission", policy_b, (profession,), 1.0
        ),
    )


def build_causal_decomposition_conditions() -> tuple[
    ExperimentCondition, ...
]:
    """Five model-facing treatments over one unchanged Runtime policy."""

    return tuple(
        ExperimentCondition(
            condition_id,
            None,
            (),
            1.0,
            projection_mode=projection_mode,
            uses_fixture_policy=True,
        )
        for condition_id, projection_mode in (
            ("causal_neutral", "neutral"),
            ("causal_order_only", "order_only"),
            ("causal_explicit_rank", "explicit_rank"),
            ("causal_reasoning_only", "reasoning_only"),
            ("causal_full", "full"),
        )
    )


def build_final_validation_conditions(
    *, include_explicit_rank: bool = False
) -> tuple[ExperimentCondition, ...]:
    """Final V6 acceptance treatments: baseline and compiled guidance."""

    treatments = [
        ("final_neutral", "neutral"),
        ("final_reasoning_only", "reasoning_only"),
    ]
    if include_explicit_rank:
        treatments.append(("final_explicit_rank", "explicit_rank"))
    return tuple(
        ExperimentCondition(
            condition_id,
            None,
            (),
            1.0,
            projection_mode=projection_mode,
            uses_fixture_policy=True,
        )
        for condition_id, projection_mode in treatments
    )


def load_tasks(selected: Iterable[str] = ()) -> tuple[ExperimentTask, ...]:
    selected_ids = tuple(dict.fromkeys(item.strip() for item in selected if item.strip()))
    paths = sorted(TASKS_DIR.glob("*.json"))
    tasks = tuple(_load_task(path) for path in paths)
    if not tasks:
        raise ValueError(f"No experiment tasks found under {TASKS_DIR}")
    if not selected_ids:
        return tasks
    by_id = {task.id: task for task in tasks}
    missing = [item for item in selected_ids if item not in by_id]
    if missing:
        raise ValueError("Unknown task id(s): " + ", ".join(missing))
    return tuple(by_id[item] for item in selected_ids)


def load_causal_tasks(
    selected: Iterable[str] = (),
) -> tuple[ExperimentTask, ...]:
    selected_ids = tuple(
        dict.fromkeys(item.strip() for item in selected if item.strip())
    )
    tasks = tuple(_load_task(path) for path in sorted(CAUSAL_TASKS_DIR.glob("*.json")))
    if not tasks:
        raise ValueError(f"No causal experiment tasks found under {CAUSAL_TASKS_DIR}")
    if not selected_ids:
        return tasks
    by_id = {task.id: task for task in tasks}
    missing = [item for item in selected_ids if item not in by_id]
    if missing:
        raise ValueError("Unknown causal task id(s): " + ", ".join(missing))
    return tuple(by_id[item] for item in selected_ids)


def load_final_validation_tasks(
    selected: Iterable[str] = (),
) -> tuple[ExperimentTask, ...]:
    selected_ids = tuple(
        dict.fromkeys(item.strip() for item in selected if item.strip())
    )
    tasks = tuple(
        _load_task(path)
        for path in sorted(FINAL_VALIDATION_TASKS_DIR.glob("*.json"))
    )
    if not tasks:
        raise ValueError(
            "No final-validation tasks found under "
            f"{FINAL_VALIDATION_TASKS_DIR}"
        )
    if not selected_ids:
        return tasks
    by_id = {task.id: task for task in tasks}
    missing = [item for item in selected_ids if item not in by_id]
    if missing:
        raise ValueError(
            "Unknown final-validation task id(s): " + ", ".join(missing)
        )
    return tuple(by_id[item] for item in selected_ids)


def evaluate_answer(
    task: ExperimentTask,
    answer: str,
    *,
    capability_violation: bool = False,
) -> dict[str, Any]:
    result = evaluate_decision(
        task.as_dict(),
        answer,
        protocol_success=True,
        capability_violation=capability_violation,
    )
    return {
        "task_success": result["semantic_success"] and not capability_violation,
        "score": result["semantic_score"],
        "findings": result["findings"],
        "missing_findings": result["missing_findings"],
        "forbidden_claims_found": result["forbidden_claims_found"],
        "capability_violation": capability_violation,
        "evidence": result["evidence"],
        "deterministic": True,
    }


def token_estimate(value: Any) -> dict[str, Any]:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
    byte_count = len(serialized.encode("utf-8"))
    return {
        "characters": len(serialized),
        "utf8_bytes": byte_count,
        "estimated_tokens": math.ceil(byte_count / 4),
        "method": "ceil(UTF-8 bytes / 4); estimate only, not provider usage",
        "exact": False,
    }


def build_dry_run_record(
    task: ExperimentTask,
    condition: ExperimentCondition,
    *,
    replicate: int = 1,
    model_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    supervisor = DryRunSupervisor()
    harness = None
    try:
        harness, agent_id, prepared = _prepare_runtime(
            task, condition, replicate, supervisor
        )
        operational = harness.build_context(agent_id)
        runtime_control_context = operational.as_dict()
        context = operational.for_model()
        retrieval_trace = harness.retrieval_trace(agent_id)
        state = harness.runtime_state(agent_id).as_context()
        assignment = harness.work_assignment(agent_id)
        return _base_record(
            task,
            condition,
            replicate,
            model_config or default_model_config(),
            state,
            assignment.as_context() if assignment else None,
            prepared,
            context,
            retrieval_trace,
            runtime_control_context=runtime_control_context,
            dry_run=True,
        )
    finally:
        if harness is not None:
            harness.close()
        else:
            supervisor.close()


def run_model_record(
    task: ExperimentTask,
    condition: ExperimentCondition,
    *,
    replicate: int,
    model_config: dict[str, Any],
    max_steps: int = DEFAULT_MAX_STEPS,
) -> dict[str, Any]:
    recorder = RecordingSupervisor()
    harness = None
    started = time.perf_counter()
    record: dict[str, Any] | None = None
    outcomes: list[dict[str, Any]] = []
    exception: dict[str, Any] | None = None
    final_answer = ""
    condition_violation = False
    try:
        harness, agent_id, prepared = _prepare_runtime(
            task, condition, replicate, recorder
        )
        operational = harness.build_context(agent_id)
        runtime_control_context = operational.as_dict()
        initial_context = operational.for_model()
        retrieval_trace = harness.retrieval_trace(agent_id)
        initial_state = harness.runtime_state(agent_id).as_context()
        assignment = harness.work_assignment(agent_id)
        record = _base_record(
            task,
            condition,
            replicate,
            model_config,
            initial_state,
            assignment.as_context() if assignment else None,
            prepared,
            initial_context,
            retrieval_trace,
            runtime_control_context=runtime_control_context,
            dry_run=False,
        )

        for step_index in range(max_steps):
            outcome = harness.step(agent_id)
            item = _outcome_dict(outcome)
            item["step"] = step_index + 1
            outcomes.append(item)
            if isinstance(outcome, FinalAnswer):
                final_answer = outcome.content
                break
            if isinstance(
                outcome,
                (ResourceBindingAdmission, WorkAssignmentAdmission, WorkflowAdmission),
            ):
                condition_violation = True
                break
        else:
            raise RuntimeError(f"Experiment exceeded max_steps={max_steps}")
    except Exception as exc:
        exception = _exception_payload(exc)
    finally:
        if record is None:
            record = _failed_setup_record(
                task, condition, replicate, model_config, exception
            )
        record["outcomes"] = outcomes
        record["model_calls"] = recorder.calls
        decision_contexts = [
            {
                "call_index": call["index"],
                "context": call["structured_context"],
                "metrics": token_estimate(call["structured_context"]),
            }
            for call in recorder.calls
            if call["phase"] == "decision" and call["structured_context"]
        ]
        record["structured_contexts"] = decision_contexts
        if decision_contexts:
            record["final_context"] = decision_contexts[-1]["context"]
            record["context_metrics"] = decision_contexts[-1]["metrics"]
            record["retrieved_resources"] = _resource_snapshot(
                decision_contexts[0]["context"]
            )
            resources = record["retrieved_resources"]
            record["retrieved_memory_ids"] = [
                item.get("id") for item in resources["memories"]
            ]
            record["skill_sources"] = resources["skills"]
            record["constructed_context_sections"] = dict(
                decision_contexts[-1]["context"]
            )
            record["context_tokens"] = decision_contexts[-1]["metrics"][
                "estimated_tokens"
            ]
        record["final_answer"] = final_answer or None
        record["capability_requests"] = _action_proposals(recorder.calls)
        record["admission_decisions"] = [
            *record.get("admission_decisions", []),
            *_admission_decisions(outcomes),
        ]
        record["condition_violation"] = condition_violation
        allowed = set(record.get("runtime_state_before_model", {}).get(
            "allowed_capabilities", []
        ))
        capability_violation = any(
            item.get("capability_id") not in allowed
            for item in record["capability_requests"]
        )
        _apply_run_metrics(
            record,
            task,
            final_answer,
            recorder.calls,
            outcomes,
            capability_violation=capability_violation,
            exception=exception,
        )
        usage = _aggregate_usage(recorder.calls)
        record["usage"] = usage
        record["input_tokens"] = usage["input_tokens"]
        record["output_tokens"] = usage["output_tokens"]
        record["latency_ms"] = {
            "total": round((time.perf_counter() - started) * 1000, 3),
            "model": round(
                sum(float(call["latency_ms"] or 0) for call in recorder.calls),
                3,
            ),
        }
        record["latency"] = record["latency_ms"]["total"]
        model_call_failures = [
            {"call_index": call["index"], **call["error"]}
            for call in recorder.calls
            if call.get("error")
        ]
        api_failures = [
            item
            for item in model_call_failures
            if item.get("failure_stage") == "api"
        ]
        record["model_call_failures"] = model_call_failures
        record["api_failures"] = api_failures
        record["exception"] = exception or (
            model_call_failures[0] if model_call_failures else None
        )
        if exception:
            record["status"] = "error"
        elif api_failures:
            record["status"] = "partial_failure"
        elif condition_violation:
            record["status"] = "condition_violation"
        elif final_answer and not record["protocol_success"]:
            record["status"] = "protocol_failure_completed"
        elif final_answer:
            record["status"] = "success"
        else:
            record["status"] = "error"
        if harness is not None:
            try:
                record["final_runtime_state"] = harness.runtime_state(
                    agent_id
                ).as_context()
                record["events"] = [
                    event.as_dict()
                    for event in harness.recent_events(500, agent_id=agent_id)
                ]
            except Exception:
                pass
            harness.close()
        else:
            recorder.close()
    validate_record_schema(record)
    return record


def _apply_run_metrics(
    record: dict[str, Any],
    task: ExperimentTask,
    final_answer: str,
    calls: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    *,
    capability_violation: bool,
    exception: dict[str, Any] | None,
) -> None:
    decision_calls = [call for call in calls if call.get("phase") == "decision"]
    received_calls = [
        call for call in decision_calls if call.get("raw_response") is not None
    ]
    decision_received = bool(received_calls)
    raw_decisions = [str(call.get("raw_response")) for call in received_calls]
    parsed = [parse_decision_payload(raw) for raw in raw_decisions]
    final_parsed = parsed[-1] if parsed else None
    raw_decision = raw_decisions[-1] if raw_decisions else ""
    decision_text, requested_action = _decision_semantics(final_parsed, raw_decision)
    decision_payload = final_parsed.payload if final_parsed else None
    structured_valid = (
        all(item.structured_decision_valid for item in parsed)
        if decision_received
        else None
    )
    schema_valid = (
        all(_known_decision_payload(item.payload) for item in parsed)
        if decision_received
        else None
    )
    fallback_used = any(item.fallback_used for item in parsed)
    actual_action = _actual_runtime_action(outcomes, exception)
    protocol_evaluated = decision_received
    protocol_success = bool(
        protocol_evaluated
        and structured_valid
        and schema_valid
        and actual_action == task.expected_action_type
    )
    runtime_completion = bool(final_answer) and exception is None
    presentation_calls = [
        call for call in calls if call.get("phase") == "presentation"
    ]
    presentation_completion = any(
        call.get("raw_response") is not None and not call.get("error")
        for call in presentation_calls
    )
    call_failures = [
        call["error"] for call in calls if isinstance(call.get("error"), dict)
    ]
    provider_failure = next(
        (
            item
            for item in call_failures
            if item.get("failure_stage") == "api"
        ),
        None,
    )
    configuration_failure = next(
        (
            item
            for item in call_failures
            if item.get("failure_stage") == "configuration"
        ),
        None,
    )
    api_request_success = (
        False
        if provider_failure
        else True
        if any(call.get("raw_response") is not None for call in calls)
        else None
    )
    protocol_failure_kind = None
    if protocol_evaluated and not protocol_success:
        if structured_valid is False:
            protocol_failure_kind = "parse"
        elif schema_valid is False:
            protocol_failure_kind = "schema"
        else:
            protocol_failure_kind = "action_type"
    if provider_failure:
        failure_stage = "api"
    elif configuration_failure:
        failure_stage = "configuration"
    elif protocol_failure_kind:
        failure_stage = "protocol"
    elif exception:
        failure_stage = str(exception.get("failure_stage") or "runtime")
    else:
        failure_stage = None
    api_error = provider_failure or (
        exception if exception and exception.get("failure_stage") == "api" else None
    )

    record["raw_decisions"] = raw_decisions
    record["raw_decision"] = raw_decision or None
    record["decision_received"] = decision_received
    record["failure_stage"] = failure_stage
    record["api_request_success"] = api_request_success
    record["api_status_code"] = (
        api_error.get("api_status_code") if api_error else None
    )
    record["api_error_type"] = (
        api_error.get("api_error_type") if api_error else None
    )
    record["api_error_message"] = (
        api_error.get("api_error_message") if api_error else None
    )
    record["protocol_evaluated"] = protocol_evaluated
    record["protocol_failure_kind"] = protocol_failure_kind
    record["structured_decision_valid"] = structured_valid
    record["decision_parse_mode"] = (
        final_parsed.decision_parse_mode if final_parsed else "not_received"
    )
    record["decision_parse_error"] = (
        final_parsed.decision_parse_error if final_parsed else None
    )
    record["fallback_used"] = fallback_used
    record["actual_runtime_action"] = actual_action
    record["runtime_completion_success"] = runtime_completion
    record["presentation_completion_success"] = presentation_completion
    record["protocol_success"] = protocol_success
    record["capability_violation"] = capability_violation
    if isinstance(decision_payload, dict):
        record["primary_evidence_id"] = decision_payload.get(
            "primary_evidence_id"
        )
        secondary = decision_payload.get("secondary_evidence_ids", [])
        record["secondary_evidence_ids"] = (
            list(secondary) if isinstance(secondary, list) else []
        )
        record["final_choice"] = decision_payload.get("final_choice")

    decision_metrics = evaluate_decision(
        task.as_dict(),
        decision_text,
        protocol_success=protocol_success,
        capability_violation=capability_violation,
        requested_action=requested_action,
    )
    requested_actions = [
        str(item.payload.get("type"))
        for item in parsed
        if isinstance(item.payload, dict) and item.payload.get("type")
    ]
    requested_action_matches = all(
        item == task.expected_action_type for item in requested_actions
    )
    decision_metrics["requested_actions"] = requested_actions
    decision_metrics["requested_action_matches"] = requested_action_matches
    decision_metrics["unexpected_action_requests"] = [
        item for item in requested_actions if item != task.expected_action_type
    ]
    decision_metrics["decision_success"] = bool(
        decision_metrics["decision_success"] and requested_action_matches
    )
    presentation_metrics = evaluate_presentation(
        task.as_dict(), decision_text, final_answer or None
    )
    resources = record.get("retrieved_resources", {})
    if task.suite in {"causal", "policy_final"}:
        mechanism_metrics = {
            "applicable": False,
            "mechanism_score": None,
            "note": "Causal paths are reported as separate selection rates.",
        }
    elif (task.mechanism or {}).get("kind") == "policy_transmission":
        mechanism_metrics = {
            "applicable": True,
            "kind": "policy_transmission",
            "mechanism_score": None,
            "note": "Transmission layers are reported separately.",
        }
    else:
        mechanism_metrics = evaluate_mechanism(
            task.as_dict(),
            condition=record["condition"],
            retrieved_memories=resources.get("memories", []),
            decision_text=decision_text,
        )
    transmission_metrics = evaluate_policy_transmission(
        task.as_dict(),
        policy_id=record.get("policy_id"),
        policy_strength=float(record.get("policy_strength", 0.0)),
        retrieval_trace=record.get("retrieval_trace", {}),
        decision_payload=(
            decision_payload if isinstance(decision_payload, dict) else None
        ),
    )
    causal_metrics = evaluate_causal_selection(
        task.as_dict(),
        condition=record["condition"],
        retrieval_trace=record.get("retrieval_trace", {}),
        decision_payload=(
            decision_payload if isinstance(decision_payload, dict) else None
        ),
    )
    capability_metrics = {
        "applicable": task.suite == "capability",
        "task_success": (
            decision_metrics["semantic_success"] and not capability_violation
            if task.suite == "capability"
            else None
        ),
        "score": (
            decision_metrics["semantic_score"]
            if task.suite == "capability"
            else None
        ),
        "capability_violation": capability_violation,
    }
    expected = task.expected_action_type
    protocol_metrics = {
        "api_request_success": api_request_success,
        "decision_received": decision_received,
        "failure_stage": failure_stage,
        "api_status_code": record["api_status_code"],
        "api_error_type": record["api_error_type"],
        "api_error_message": record["api_error_message"],
        "protocol_evaluated": protocol_evaluated,
        "protocol_failure_kind": protocol_failure_kind,
        "structured_decision_valid": structured_valid,
        "decision_schema_valid": schema_valid,
        "decision_parse_mode": record["decision_parse_mode"],
        "decision_parse_error": record["decision_parse_error"],
        "fallback_used": fallback_used,
        "expected_action_type": expected,
        "actual_runtime_action": actual_action,
        "action_type_matches": actual_action == expected,
        "runtime_completion_success": runtime_completion,
        "presentation_completion_success": presentation_completion,
        "protocol_success": protocol_success,
    }
    policy_matches = record.get("active_policy") == record["condition"].get(
        "resolved_policy_id"
    )
    professions_match = set(record.get("active_profession", [])) == set(
        record["condition"].get("profession_ids", [])
    )
    record["decision_metrics"] = decision_metrics
    record["decision_semantic_score"] = decision_metrics["semantic_score"]
    record["presentation_metrics"] = presentation_metrics
    record["presentation_preservation_score"] = presentation_metrics[
        "preservation_score"
    ]
    record["mechanism_metrics"] = mechanism_metrics
    record["policy_transmission_metrics"] = transmission_metrics
    record["causal_metrics"] = causal_metrics
    record["primary_evidence_direction"] = causal_metrics.get(
        "primary_direction"
    )
    record["primary_is_risk"] = causal_metrics.get("primary_is_risk")
    record["primary_is_first_visible"] = causal_metrics.get(
        "primary_is_first_visible"
    )
    for key in (
        "retrieval_rank_primary",
        "ranking_matches_policy",
        "primary_evidence_matches_policy",
        "final_choice_matches_policy",
    ):
        record[key] = transmission_metrics.get(key)
    record["capability_metrics"] = capability_metrics
    record["protocol_metrics"] = protocol_metrics
    record["architecture_metrics"] = {
        "runtime_prepared": True,
        "condition_resources_match": policy_matches and professions_match,
        "assignment_active_before_model": bool(record.get("active_assignment")),
        "capability_boundary_preserved": not capability_violation,
    }
    record["evaluation"] = {
        "task_success": decision_metrics["semantic_success"],
        "score": decision_metrics["semantic_score"],
        "findings": decision_metrics["findings"],
        "missing_findings": decision_metrics["missing_findings"],
        "forbidden_claims_found": decision_metrics[
            "forbidden_claims_found"
        ],
        "capability_violation": capability_violation,
        "evidence": decision_metrics["evidence"],
        "deterministic": True,
        "note": "Legacy alias of Decision semantic metrics; Presentation is separate.",
    }


def default_model_config(
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    seed: int | None = None,
    json_mode: bool = False,
) -> dict[str, Any]:
    return {
        "provider": "DeepSeek (OpenAI-compatible)",
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
        "seed_sent": seed is not None,
        "json_mode": json_mode,
        "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        "api_key_configured": bool(os.getenv("DEEPSEEK_API_KEY")),
    }


def validate_record_schema(record: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "run_id",
        "task",
        "condition",
        "replicate",
        "model_config",
        "workspace",
        "runtime_state_before_model",
        "active_cognitive_policy",
        "active_professions",
        "policy_id",
        "policy_parameters",
        "policy_strength",
        "work_assignment",
        "retrieved_resources",
        "runtime_control_context",
        "model_visible_context",
        "model_projection_mode",
        "model_visible_evidence_ids",
        "model_visible_explicit_priority",
        "model_visible_reasoning_guidance",
        "policy_direction",
        "evidence_order",
        "legal_memory_ids",
        "task_relevant_memory_ids",
        "ranked_memory_ids",
        "retrieval_scores",
        "structured_contexts",
        "context_metrics",
        "model_calls",
        "usage",
        "latency_ms",
        "capability_requests",
        "admission_decisions",
        "final_answer",
        "evaluation",
        "architecture_metrics",
        "mechanism_metrics",
        "policy_transmission_metrics",
        "causal_metrics",
        "protocol_metrics",
        "capability_metrics",
        "presentation_metrics",
        "decision_received",
        "failure_stage",
        "api_request_success",
        "api_status_code",
        "api_error_type",
        "api_error_message",
        "protocol_evaluated",
        "protocol_failure_kind",
        "structured_decision_valid",
        "decision_parse_mode",
        "decision_parse_error",
        "fallback_used",
        "expected_action_type",
        "actual_runtime_action",
        "runtime_completion_success",
        "presentation_completion_success",
        "protocol_success",
        "primary_evidence_id",
        "secondary_evidence_ids",
        "final_choice",
        "retrieval_rank_primary",
        "ranking_matches_policy",
        "primary_evidence_matches_policy",
        "final_choice_matches_policy",
        "primary_evidence_direction",
        "primary_is_risk",
        "primary_is_first_visible",
        "status",
        "exception",
    }
    missing = sorted(required.difference(record))
    if missing:
        raise ValueError("Experiment record missing fields: " + ", ".join(missing))


def write_results(records: list[dict[str, Any]], output: Path) -> tuple[Path, Path]:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = output.with_suffix(".summary.csv")
    fields = [
        "row_type",
        "run_id",
        "task_id",
        "condition",
        "replicate",
        "status",
        "suite",
        "failure_stage",
        "api_request_success",
        "decision_received",
        "protocol_evaluated",
        "protocol_failure_kind",
        "api_status_code",
        "api_error_type",
        "api_error_message",
        "architecture_success",
        "mechanism_score",
        "retrieval_alignment",
        "policy_id",
        "policy_strength",
        "primary_evidence_id",
        "secondary_evidence_ids",
        "final_choice",
        "retrieval_rank_primary",
        "ranking_matches_policy",
        "primary_evidence_matches_policy",
        "final_choice_matches_policy",
        "projection_mode",
        "policy_direction",
        "evidence_order",
        "target_direction",
        "primary_evidence_direction",
        "primary_is_risk",
        "primary_is_first_visible",
        "primary_matches_target_direction",
        "primary_choice_consistent",
        "risk_selection_rate",
        "delta_risk_vs_neutral",
        "preferred_selection_rate",
        "neutral_baseline_preferred_rate",
        "delta_preferred_vs_neutral",
        "first_visible_selection_rate",
        "selection_observation_count",
        "paired_switches_toward_preferred",
        "paired_switches_away_from_preferred",
        "paired_observation_count",
        "avg_input_tokens",
        "avg_output_tokens",
        "avg_total_tokens",
        "token_overhead_vs_neutral",
        "ranking_transmission_rate",
        "primary_evidence_transmission_rate",
        "final_decision_transmission_rate",
        "api_request_success_rate",
        "decision_received_rate",
        "protocol_success_rate_among_received",
        "protocol_success_rate",
        "runtime_completion_rate",
        "presentation_completion_rate",
        "protocol_success",
        "runtime_completion_success",
        "presentation_completion_success",
        "structured_decision_valid",
        "decision_parse_mode",
        "fallback_used",
        "expected_action_type",
        "actual_runtime_action",
        "capability_task_success",
        "decision_semantic_score",
        "presentation_preservation_score",
        "presentation_formatting_integrity",
        "findings",
        "missing_findings",
        "forbidden_claims_found",
        "capability_violation",
        "cognitive_policy_id",
        "profession_ids",
        "context_estimated_tokens",
        "model_input_tokens",
        "model_output_tokens",
        "latency_ms",
        "error",
    ]
    with summary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(_summary_row(record))
        for aggregate in _aggregate_summary_rows(records):
            writer.writerow(aggregate)
    return output, summary


def _prepare_runtime(
    task: ExperimentTask,
    condition: ExperimentCondition,
    replicate: int,
    supervisor: DryRunSupervisor | RecordingSupervisor,
):
    task_id = f"v6exp:{task.id}:r{replicate}"
    memory = _experiment_memory(task_id, task)
    harness = create_runtime(
        processes=supervisor,
        memory=memory,
        workspace_root=WORKSPACE_ROOT,
        workspace_id=WORKSPACE_ID,
        policy_strength=condition.policy_strength,
        model_projection_override=_experiment_model_projection,
    )
    try:
        mechanism = task.mechanism or {}
        resolved_policy_id = condition.cognitive_policy_id
        if condition.uses_fixture_policy:
            fixture_policy = mechanism.get("target_policy_id")
            if not isinstance(fixture_policy, str) or not fixture_policy:
                raise ValueError("causal task 缺少 target_policy_id")
            resolved_policy_id = fixture_policy.upper()
        projection_config = None
        if condition.projection_mode != "legacy":
            evidence = mechanism.get("evidence") or []
            evidence_ids = [
                item.get("id") for item in evidence if isinstance(item, dict)
            ]
            projection_config = {
                "mode": condition.projection_mode,
                "evidence_ids": evidence_ids,
                "neutral_order": list(mechanism.get("neutral_order", [])),
                "reasoning_guidance": mechanism.get("reasoning_guidance"),
                "guidance_source": (
                    "cognitive_policy"
                    if task.suite == "policy_final"
                    else "fixture"
                ),
            }
        task_context = {
            "experiment": "v6_cognitive_profession_ablation",
            "fixture_scope": task.fixture_scope,
            "decision_requirements": (
                {
                    "primary_evidence_id": (
                        "required exact memory id from visible memory"
                    ),
                    "secondary_evidence_ids": (
                        "optional exact memory ids; never changes primary focus"
                    ),
                    "final_choice": (
                        "required exact choice id declared by the task"
                    ),
                }
                if task.suite in {"mechanism", "causal", "policy_final"}
                else {}
            ),
            "controls": [
                "Do not change mounted resources during the task.",
                "Do not replace the active WorkAssignment or Workflow.",
                "Use no capability unless evidence is genuinely required.",
            ],
        }
        if projection_config is not None:
            task_context["_experiment_projection"] = projection_config
        process = harness.create_agent(
            TaskSpec(
                id=task_id,
                request=task.request,
                context=task_context,
            )
        )
        binding_admissions: list[dict[str, Any]] = []
        if resolved_policy_id:
            result = harness.request_binding(
                process.id,
                ResourceBindingRequest(
                    "cognitive_policy", "activate", resolved_policy_id
                ),
            )
            binding_admissions.append(result.as_dict())
            if result.status != "success":
                raise ValueError(result.error or "CognitivePolicy binding rejected")
        for profession_id in condition.profession_ids:
            result = harness.request_binding(
                process.id,
                ResourceBindingRequest("profession", "activate", profession_id),
            )
            binding_admissions.append(result.as_dict())
            if result.status != "success":
                raise ValueError(result.error or "Profession binding rejected")
        assignment = harness.request_work_assignment(
            process.id,
            WorkAssignmentRequest(
                role_id=(
                    "generalist"
                    if condition.projection_mode != "legacy"
                    else "reviewer"
                ),
                workspace_id=WORKSPACE_ID,
                task_id=task_id,
                workflow_id="general_task",
                target_state_id="active",
                requested_capabilities=("filesystem.read",),
            ),
        )
        if assignment.status != "success":
            raise ValueError(assignment.error or "WorkAssignment rejected")
        prepared = {
            "binding_admissions": binding_admissions,
            "assignment_admission": assignment.as_dict(),
            "resolved_policy_id": resolved_policy_id,
            "projection_mode": condition.projection_mode,
        }
        return harness, process.id, prepared
    except Exception:
        harness.close()
        raise


def _experiment_memory(
    task_id: str, task: ExperimentTask | None = None
) -> InMemoryMemoryProvider:
    memory = InMemoryMemoryProvider()
    fixed_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    core = json.loads(
        (PROJECT_ROOT / "memory" / "core_memory.json").read_text(encoding="utf-8")
    )
    for index, (kind, content) in enumerate(core.items()):
        memory.write(
            MemoryRecord(
                id=f"exp-core-{index:02d}",
                scope="core",
                kind=kind,
                content=json.dumps(content, ensure_ascii=False),
                created_at=fixed_time,
                metadata={"source": "experiment_core_snapshot"},
            )
        )
    if task is not None and task.suite in {"causal", "policy_final"}:
        evidence = (task.mechanism or {}).get("evidence") or []
        if len(evidence) != 2:
            raise ValueError("causal task 必须包含两条 evidence")
        for item in evidence:
            memory.write(
                MemoryRecord(
                    id=str(item["id"]),
                    scope="core",
                    kind="experiment.causal_candidate",
                    content=str(item["content"]),
                    importance=0.7,
                    created_at=fixed_time,
                    metadata={
                        "source": "policy_causal_fixture",
                        "fixture_scope": task.fixture_scope,
                        "task_relevance": 1.0,
                        "causal_direction": item["direction"],
                        "choice_id": item["choice_id"],
                    },
                )
            )
        memory.write(
            MemoryRecord(
                id=f"causal-task-{task.id}",
                scope=f"task:{task_id}",
                kind="task.experiment_control",
                content="Select one primary evidence item and one final choice.",
                importance=0.9,
                created_at=fixed_time,
                metadata={"source": "policy_causal_control"},
            )
        )
        return memory
    seeded = (
        MemoryRecord(
            id="exp-core-z-risk-verification",
            scope="core",
            kind="experiment.risk_evidence",
            content="Risk contradiction evidence: verify boundary and regression tests.",
            importance=0.5,
            created_at=fixed_time,
            metadata={
                "source": "experiment_seed",
                "tags": ["risk", "test"],
                "fixture_scope": "general_ablation",
            },
        ),
        MemoryRecord(
            id="exp-core-a-novel-option",
            scope="core",
            kind="experiment.novel_option",
            content="Novel cross-domain idea: compare an alternative before convergence.",
            importance=0.5,
            created_at=fixed_time,
            metadata={
                "source": "experiment_seed",
                "tags": ["novel", "idea"],
                "fixture_scope": "general_ablation",
            },
        ),
        MemoryRecord(
            id="exp-profession-software-engineering",
            scope="identity:ada",
            kind="identity.work_experience",
            content="Software engineering precedent: trace API contracts and add regression tests.",
            importance=0.8,
            created_at=fixed_time,
            metadata={
                "source": "experiment_seed",
                "professional_tags": ["software_engineering", "testing"],
            },
        ),
        MemoryRecord(
            id="exp-profession-writing-control",
            scope="identity:ada",
            kind="identity.work_experience",
            content="Writing precedent: preserve scene motivation and narrative rhythm.",
            importance=0.8,
            created_at=fixed_time,
            metadata={
                "source": "experiment_seed",
                "professional_tags": ["writing"],
            },
        ),
        MemoryRecord(
            id="mechanism-profession-missing-sentinel",
            scope="identity:ada",
            kind="identity.work_experience",
            content=(
                "Professional evidence SE-17: this cache treats None as a valid "
                "cached value. Detect cache miss only with the MISSING sentinel "
                "and an identity check (`value is MISSING`)."
            ),
            importance=0.92,
            created_at=fixed_time,
            metadata={
                "source": "mechanism_profession_fixture",
                "professional_tags": ["software_engineering", "debugging"],
                "fixture_scope": "profession_sentinel",
                "task_relevance": 1.0,
            },
        ),
        MemoryRecord(
            id="mechanism-profession-rollback-before-retry",
            scope="identity:ada",
            kind="identity.work_experience",
            content=(
                "Professional evidence SE-23: after a retryable database error, "
                "rollback the failed transaction before retrying; the connection "
                "otherwise remains in an aborted transaction state."
            ),
            importance=0.92,
            created_at=fixed_time,
            metadata={
                "source": "mechanism_profession_fixture",
                "professional_tags": ["software_engineering", "testing"],
                "fixture_scope": "profession_retry",
                "task_relevance": 1.0,
            },
        ),
        MemoryRecord(
            id="mechanism-policy-risk-r17",
            scope="core",
            kind="experiment.risk_precedent",
            content=(
                "Known regression risk and precedent R-17: eager invalidation caused "
                "a prior read-after-write race; preserve the version check and add "
                "concurrency regression coverage. Recommended choice id: "
                "guarded_invalidation."
            ),
            importance=0.7,
            created_at=fixed_time,
            metadata={
                "source": "mechanism_policy_fixture",
                "tags": ["risk", "test"],
                "fixture_scope": "policy_cache",
                "task_relevance": 1.0,
                "choice_id": "guarded_invalidation",
            },
        ),
        MemoryRecord(
            id="mechanism-policy-novel-n42",
            scope="core",
            kind="experiment.novel_option",
            content=(
                "Novel alternative evidence N-42: use immutable versioned cache "
                "keys so old and new readers coexist, then retire old versions "
                "after observation. Recommended choice id: versioned_keys."
            ),
            importance=0.7,
            created_at=fixed_time,
            metadata={
                "source": "mechanism_policy_fixture",
                "tags": ["novel", "idea"],
                "fixture_scope": "policy_cache",
                "task_relevance": 1.0,
                "choice_id": "versioned_keys",
            },
        ),
        MemoryRecord(
            id="mechanism-policy-precedent-p08",
            scope="core",
            kind="experiment.precedent",
            content=(
                "Known regression risk and prior dependency precedent P-08: "
                "a synchronous-to-async migration "
                "broke the callers that relied on exception timing; retain an adapter "
                "and compatibility checks during migration. Recommended choice id: "
                "compatibility_adapter."
            ),
            importance=0.7,
            created_at=fixed_time,
            metadata={
                "source": "mechanism_policy_fixture",
                "tags": ["precedent", "risk"],
                "fixture_scope": "policy_dependency",
                "task_relevance": 1.0,
                "choice_id": "compatibility_adapter",
            },
        ),
        MemoryRecord(
            id="mechanism-policy-exploratory-x31",
            scope="core",
            kind="experiment.novel_option",
            content=(
                "Novel cross-domain dependency evidence X-31: expose the change as an "
                "event stream and let consumers adopt an asynchronous projection "
                "without replacing the synchronous contract immediately. Recommended "
                "choice id: event_projection."
            ),
            importance=0.7,
            created_at=fixed_time,
            metadata={
                "source": "mechanism_policy_fixture",
                "tags": ["novel", "idea", "cross_domain"],
                "fixture_scope": "policy_dependency",
                "task_relevance": 1.0,
                "choice_id": "event_projection",
            },
        ),
        MemoryRecord(
            id="exp-workspace-readonly",
            scope=f"workspace:{WORKSPACE_ID}",
            kind="workspace.experiment_constraint",
            content="This ablation workspace is read-only; destructive capabilities are forbidden.",
            importance=1.0,
            created_at=fixed_time,
            metadata={"source": "experiment_workspace"},
        ),
        MemoryRecord(
            id=f"exp-task-{task_id.replace(':', '-')}",
            scope=f"task:{task_id}",
            kind="task.experiment_control",
            content="Use the supplied evidence and return a bounded, verifiable answer.",
            importance=0.9,
            created_at=fixed_time,
            metadata={"source": "experiment_task", "task_id": task_id},
        ),
    )
    for record in seeded:
        memory.write(record)
    return memory


def _base_record(
    task: ExperimentTask,
    condition: ExperimentCondition,
    replicate: int,
    model_config: dict[str, Any],
    state: dict[str, Any],
    assignment: dict[str, Any] | None,
    prepared: dict[str, Any],
    context: dict[str, Any],
    retrieval_trace: dict[str, Any],
    *,
    runtime_control_context: dict[str, Any] | None = None,
    dry_run: bool,
) -> dict[str, Any]:
    control_context = runtime_control_context or context
    metrics = token_estimate(context)
    resources = _resource_snapshot(context)
    policy_parameters = _policy_parameters(
        control_context.get("cognitive_policy")
    )
    policy_context = control_context.get("cognitive_policy") or {}
    resolved_policy_id = state.get("cognitive_policy_id")
    transmission_metrics = evaluate_policy_transmission(
        task.as_dict(),
        policy_id=resolved_policy_id,
        policy_strength=condition.policy_strength,
        retrieval_trace=retrieval_trace,
        decision_payload=None,
    )
    causal_metrics = evaluate_causal_selection(
        task.as_dict(),
        condition={
            **condition.as_dict(),
            "resolved_policy_id": resolved_policy_id,
        },
        retrieval_trace=retrieval_trace,
        decision_payload=None,
    )
    record = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "v6_cognitive_profession_ablation",
        "run_id": f"{task.id}:{condition.id}:r{replicate}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "task": task.as_dict(),
        "task_id": task.id,
        "condition": {
            **condition.as_dict(),
            "resolved_policy_id": resolved_policy_id,
        },
        "replicate": replicate,
        "repeat_id": replicate,
        "model_config": dict(model_config),
        "workspace": {
            "id": WORKSPACE_ID,
            "root": str(WORKSPACE_ROOT),
            "mode": "experiment_read_only",
            "granted_capabilities": ["filesystem.read"],
            "destructive_capabilities": [],
        },
        "runtime_state_before_model": state,
        "final_runtime_state": state,
        "active_cognitive_policy": control_context.get("cognitive_policy"),
        "active_policy": resolved_policy_id,
        "policy_id": resolved_policy_id,
        "policy_parameters": policy_parameters,
        "effective_policy_parameters": dict(
            policy_context.get("effective_parameters") or {}
        ),
        "policy_strength": condition.policy_strength,
        "active_professions": list(
            control_context.get("professions", [])
        ),
        "active_profession": list(condition.profession_ids),
        "work_assignment": assignment,
        "active_assignment": assignment,
        "preparation_admissions": prepared,
        "retrieved_resources": resources,
        "retrieval_trace": dict(retrieval_trace),
        "runtime_control_context": control_context,
        "model_visible_context": context,
        "model_projection_mode": retrieval_trace.get(
            "model_projection_mode", "legacy"
        ),
        "model_visible_evidence_ids": list(
            retrieval_trace.get("model_visible_evidence_ids", [])
        ),
        "model_visible_explicit_priority": retrieval_trace.get(
            "model_visible_explicit_priority", False
        ),
        "model_visible_reasoning_guidance": (
            (
                (context.get("life") or {}).get(
                    "cognitive_configuration"
                )
                or {}
            ).get("operational_guidance")
            or (context.get("system") or {}).get("reasoning_guidance")
        ),
        "policy_direction": retrieval_trace.get(
            "compiled_policy_direction"
        ),
        "evidence_order": list(
            retrieval_trace.get("model_visible_evidence_ids", [])
        ),
        "legal_memory_ids": list(
            retrieval_trace.get("legal_memory_ids", [])
        ),
        "task_relevant_memory_ids": list(
            retrieval_trace.get("task_relevant_memory_ids", [])
        ),
        "ranked_memory_ids": list(
            retrieval_trace.get("ranked_memory_ids", [])
        ),
        "retrieved_memory_ids": list(
            retrieval_trace.get("context_memory_ids", [])
        ),
        "retrieval_scores": list(
            retrieval_trace.get("retrieval_scores", [])
        ),
        "skill_sources": resources["skills"],
        "structured_contexts": [
            {"call_index": None, "context": context, "metrics": metrics}
        ],
        "final_context": context,
        "constructed_context_sections": dict(context),
        "context_metrics": metrics,
        "context_tokens": metrics["estimated_tokens"],
        "model_calls": [],
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "latency_ms": {"total": 0.0, "model": 0.0},
        "capability_requests": [],
        "admission_decisions": _preparation_admission_decisions(prepared),
        "outcomes": [],
        "events": [],
        "condition_violation": False,
        "decision_received": None,
        "failure_stage": None,
        "api_request_success": None,
        "api_status_code": None,
        "api_error_type": None,
        "api_error_message": None,
        "protocol_evaluated": None,
        "protocol_failure_kind": None,
        "structured_decision_valid": None,
        "decision_parse_mode": "not_run",
        "decision_parse_error": None,
        "fallback_used": False,
        "expected_action_type": task.expected_action_type,
        "actual_runtime_action": None,
        "runtime_completion_success": False,
        "presentation_completion_success": None,
        "protocol_success": None,
        "raw_decision": None,
        "primary_evidence_id": None,
        "secondary_evidence_ids": [],
        "final_choice": None,
        "final_answer": None,
        "evaluation": {
            "task_success": None,
            "score": None,
            "findings": [],
            "missing_findings": [],
            "forbidden_claims_found": [],
            "capability_violation": False,
            "deterministic": True,
        },
        "decision_metrics": {},
        "architecture_metrics": {
            "runtime_prepared": True,
            "condition_resources_match": True,
            "assignment_active": assignment is not None,
        },
        "mechanism_metrics": {
            "applicable": task.suite == "mechanism",
            "mechanism_score": None,
        },
        "policy_transmission_metrics": transmission_metrics,
        "causal_metrics": causal_metrics,
        "primary_evidence_direction": causal_metrics.get(
            "primary_direction"
        ),
        "primary_is_risk": causal_metrics.get("primary_is_risk"),
        "primary_is_first_visible": causal_metrics.get(
            "primary_is_first_visible"
        ),
        "retrieval_rank_primary": transmission_metrics.get(
            "retrieval_rank_primary"
        ),
        "ranking_matches_policy": transmission_metrics.get(
            "ranking_matches_policy"
        ),
        "primary_evidence_matches_policy": transmission_metrics.get(
            "primary_evidence_matches_policy"
        ),
        "final_choice_matches_policy": transmission_metrics.get(
            "final_choice_matches_policy"
        ),
        "protocol_metrics": {
            "api_request_success": None,
            "decision_received": None,
            "protocol_evaluated": None,
            "runtime_completion_success": False,
            "presentation_completion_success": None,
            "protocol_success": None,
        },
        "capability_metrics": {
            "applicable": task.suite == "capability",
            "task_success": None,
            "score": None,
            "capability_violation": False,
        },
        "presentation_metrics": {
            "preservation_success": None,
            "preservation_score": None,
        },
        "decision_semantic_score": None,
        "presentation_preservation_score": None,
        "capability_violation": False,
        "input_tokens": 0,
        "output_tokens": 0,
        "latency": 0.0,
        "model_call_failures": [],
        "api_failures": [],
        "status": "dry_run" if dry_run else "prepared",
        "exception": None,
    }
    validate_record_schema(record)
    return record


def _failed_setup_record(
    task: ExperimentTask,
    condition: ExperimentCondition,
    replicate: int,
    model_config: dict[str, Any],
    exception: dict[str, Any] | None,
) -> dict[str, Any]:
    record = _base_record(
        task,
        condition,
        replicate,
        model_config,
        {},
        None,
        {},
        {},
        {},
        dry_run=False,
    )
    record["architecture_metrics"]["runtime_prepared"] = False
    record["status"] = "error"
    record["exception"] = exception
    return record


def _resource_snapshot(context: dict[str, Any]) -> dict[str, Any]:
    if {"runtime_envelope", "life", "profession", "work"}.issubset(
        context
    ):
        life = context["life"]
        profession_region = context["profession"]
        work = context["work"]
        memory_records = [
            *life.get("core_memory", []),
            *profession_region.get("professional_memory", []),
            *work.get("relevant_archive_and_artifacts", []),
            *work.get("recent_observations", []),
        ]
        skill_records = [
            *profession_region.get("methods_and_skills", []),
            *work.get("local_skills", []),
        ]
        profession_records = profession_region.get("active_resources", [])
        workspace = work.get("workspace") or {}
    else:
        memory_records = context.get("memory", [])
        skill_records = context.get("skills", [])
        profession_records = context.get("professions", [])
        workspace = context.get("workspace") or {}
    memories = []
    for rank, record in enumerate(memory_records, start=1):
        metadata = record.get("metadata", {})
        memories.append(
            {
                "rank": rank,
                "id": record.get("id") or record.get("evidence_id"),
                "scope": record.get("scope"),
                "kind": record.get("kind"),
                "source": metadata.get("source", record.get("scope")),
                "importance": record.get("importance"),
                "ranking_score": None,
                "score_exposed": False,
            }
        )
    return {
        "memories": memories,
        "memory_ranking": {
            "ordered": True,
            "score_exposed": False,
            "note": (
                "Operational Context exposes order only; experiment retrieval_trace "
                "separately records deterministic score components."
            ),
        },
        "skills": [
            {"id": item.get("id"), "sources": item.get("sources", [])}
            for item in skill_records
        ],
        "profession_knowledge_references": [
            reference
            for profession in profession_records
            for reference in profession.get("knowledge_references", [])
        ],
        "workspace_knowledge_references": (
            workspace
            .get("local_knowledge", {})
            .get("references", [])
        ),
    }


def _system_payload(messages: list[dict]) -> dict[str, Any]:
    if not messages:
        return {}
    try:
        value = json.loads(str(messages[0].get("content", "")))
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


def _message_payload(
    messages: list[dict[str, Any]], index: int
) -> dict[str, Any]:
    if index >= len(messages):
        return {}
    try:
        value = json.loads(str(messages[index].get("content", "")))
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


def _parse_response_payload(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    return parse_decision_payload(raw).payload


def _decision_semantics(
    parsed, raw: str
) -> tuple[str, str | None]:
    payload = parsed.payload if parsed else None
    if isinstance(payload, dict):
        action_type = payload.get("type")
        if action_type == "final" and isinstance(payload.get("content"), str):
            return payload["content"], "final"
        return json.dumps(payload, ensure_ascii=False), str(action_type or "unknown")
    return raw, None


def _known_decision_payload(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    action_type = payload.get("type")
    if action_type == "final":
        if not isinstance(payload.get("content"), str):
            return False
        primary = payload.get("primary_evidence_id")
        secondary = payload.get("secondary_evidence_ids", [])
        choice = payload.get("final_choice")
        if primary is not None and (
            not isinstance(primary, str) or not primary.strip()
        ):
            return False
        if not isinstance(secondary, list) or not all(
            isinstance(item, str) and item.strip() for item in secondary
        ):
            return False
        normalized_secondary = [item.strip() for item in secondary]
        if len(set(normalized_secondary)) != len(normalized_secondary):
            return False
        if primary and primary.strip() in normalized_secondary:
            return False
        return choice is None or (
            isinstance(choice, str) and bool(choice.strip())
        )
    if action_type == "action":
        return isinstance(payload.get("capability_id"), str) and isinstance(
            payload.get("arguments"), dict
        )
    return action_type in {"workflow", "binding", "assignment"}


def _exception_payload(exc: Exception) -> dict[str, Any]:
    """Preserve provider/protocol stage data without changing Runtime control."""

    worker_stage = getattr(exc, "stage", None)
    if worker_stage == "api":
        failure_stage = "api"
    elif worker_stage == "configuration":
        failure_stage = "configuration"
    elif type(exc).__name__ == "DecisionProtocolError":
        failure_stage = "protocol"
    else:
        failure_stage = "runtime"
    protocol = getattr(exc, "protocol", {})
    protocol_failure_kind = None
    if failure_stage == "protocol":
        if protocol.get("structured_decision_valid") is False:
            protocol_failure_kind = "parse"
        elif protocol.get("decision_schema_valid") is False:
            protocol_failure_kind = "schema"
        else:
            protocol_failure_kind = "protocol"
    status_code = getattr(exc, "status_code", None)
    provider_error_type = getattr(exc, "error_type", None)
    provider_message = getattr(exc, "error_message", None)
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "failure_stage": failure_stage,
        "protocol_failure_kind": protocol_failure_kind,
        "api_status_code": (
            int(status_code) if isinstance(status_code, int) else None
        ),
        "api_error_type": (
            str(provider_error_type)
            if failure_stage == "api" and provider_error_type
            else None
        ),
        "api_error_message": (
            str(provider_message or exc) if failure_stage == "api" else None
        ),
    }


def _actual_runtime_action(
    outcomes: list[dict[str, Any]], exception: dict[str, Any] | None
) -> str | None:
    if outcomes:
        outcome_type = outcomes[-1].get("type")
        mapping = {
            "final_answer": "final",
            "ActionResult": "action",
            "WorkflowAdmission": "workflow",
            "ResourceBindingAdmission": "binding",
            "WorkAssignmentAdmission": "assignment",
        }
        return mapping.get(str(outcome_type), str(outcome_type))
    if exception and exception.get("type") == "DecisionProtocolError":
        return "protocol_error"
    return "runtime_error" if exception else None


def _policy_parameters(policy: Any) -> dict[str, Any]:
    if not isinstance(policy, dict):
        return {}
    parameters = policy.get("parameters")
    if isinstance(parameters, dict):
        return dict(parameters)
    excluded = {"id", "name", "description", "source", "version"}
    return {
        key: value
        for key, value in policy.items()
        if key not in excluded and not isinstance(value, (dict, list))
    }


def _action_proposals(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for call in calls:
        if call.get("phase") != "decision":
            continue
        payload = _parse_response_payload(call.get("raw_response"))
        if payload and payload.get("type") == "action":
            result.append(
                {
                    "call_index": call["index"],
                    "capability_id": payload.get("capability_id"),
                    "arguments": payload.get("arguments", {}),
                    "request_id": payload.get("request_id"),
                }
            )
    return result


def _outcome_dict(outcome: Any) -> dict[str, Any]:
    if isinstance(outcome, FinalAnswer):
        return {
            "type": "final_answer",
            "content": outcome.content,
            "primary_evidence_id": outcome.primary_evidence_id,
            "secondary_evidence_ids": list(outcome.secondary_evidence_ids),
            "final_choice": outcome.final_choice,
        }
    if hasattr(outcome, "as_dict"):
        return {
            "type": type(outcome).__name__,
            **outcome.as_dict(),
        }
    return {"type": type(outcome).__name__, "value": str(outcome)}


def _admission_decisions(outcomes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for outcome in outcomes:
        if outcome.get("type") == "ActionResult":
            stage = (outcome.get("metadata") or {}).get("stage")
            allowed = None
            if stage == "admission":
                allowed = False
            elif stage == "execution":
                allowed = True
            result.append(
                {
                    "kind": "capability",
                    "request_id": outcome.get("request_id"),
                    "capability_id": outcome.get("capability_id"),
                    "allowed": allowed,
                    "stage": stage,
                    "status": outcome.get("status"),
                    "error": outcome.get("error"),
                }
            )
        elif outcome.get("type", "").endswith("Admission"):
            result.append(
                {
                    "kind": outcome["type"],
                    "request_id": outcome.get("request_id"),
                    "allowed": outcome.get("status") == "success",
                    "status": outcome.get("status"),
                    "error": outcome.get("error"),
                }
            )
    return result


def _preparation_admission_decisions(
    prepared: dict[str, Any],
) -> list[dict[str, Any]]:
    result = []
    for admission in prepared.get("binding_admissions", []):
        result.append(
            {
                "kind": "resource_binding",
                "request_id": admission.get("request_id"),
                "resource_type": admission.get("resource_type"),
                "resource_id": admission.get("resource_id"),
                "allowed": admission.get("status") == "success",
                "status": admission.get("status"),
                "error": admission.get("error"),
            }
        )
    assignment = prepared.get("assignment_admission")
    if assignment:
        result.append(
            {
                "kind": "work_assignment",
                "request_id": assignment.get("request_id"),
                "allowed": assignment.get("status") == "success",
                "status": assignment.get("status"),
                "grants": assignment.get("grants", []),
                "error": assignment.get("error"),
            }
        )
    return result


def _aggregate_usage(calls: list[dict[str, Any]]) -> dict[str, int]:
    input_tokens = sum(int(call["usage"].get("input_tokens", 0)) for call in calls)
    output_tokens = sum(int(call["usage"].get("output_tokens", 0)) for call in calls)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def _load_task(path: Path) -> ExperimentTask:
    raw = json.loads(path.read_text(encoding="utf-8"))
    required = raw.get("required_findings")
    forbidden = raw.get("forbidden_claims", [])
    if not isinstance(required, list) or not isinstance(forbidden, list):
        raise ValueError(f"Invalid evaluator fields in {path}")
    return ExperimentTask(
        id=str(raw["id"]),
        title=str(raw["title"]),
        request=str(raw["request"]),
        required_findings=tuple(dict(item) for item in required),
        forbidden_claims=tuple(dict(item) for item in forbidden),
        source=str(path.relative_to(PROJECT_ROOT).as_posix()),
        suite=str(raw.get("suite", "capability")),
        category=str(raw.get("category", "software_engineering")),
        expected_action_type=str(raw.get("expected_action_type", "final")),
        mechanism=(dict(raw["mechanism"]) if raw.get("mechanism") else None),
        fixture_scope=(
            str(raw["fixture_scope"]) if raw.get("fixture_scope") else None
        ),
    )


def _summary_row(record: dict[str, Any]) -> dict[str, Any]:
    evaluation = record.get("evaluation", {})
    condition = record.get("condition", {})
    usage = record.get("usage", {})
    error = record.get("exception") or {}
    architecture = record.get("architecture_metrics", {})
    mechanism = record.get("mechanism_metrics", {})
    protocol = record.get("protocol_metrics", {})
    capability = record.get("capability_metrics", {})
    presentation = record.get("presentation_metrics", {})
    transmission = record.get("policy_transmission_metrics", {})
    causal = record.get("causal_metrics", {})
    return {
        "row_type": "run",
        "run_id": record.get("run_id"),
        "task_id": record.get("task", {}).get("id"),
        "condition": condition.get("id"),
        "replicate": record.get("replicate"),
        "status": record.get("status"),
        "suite": record.get("task", {}).get("suite"),
        "failure_stage": record.get("failure_stage"),
        "api_request_success": record.get("api_request_success"),
        "decision_received": record.get("decision_received"),
        "protocol_evaluated": record.get("protocol_evaluated"),
        "protocol_failure_kind": record.get("protocol_failure_kind"),
        "api_status_code": record.get("api_status_code"),
        "api_error_type": record.get("api_error_type"),
        "api_error_message": record.get("api_error_message"),
        "architecture_success": all(
            value for value in architecture.values() if isinstance(value, bool)
        ),
        "mechanism_score": mechanism.get("mechanism_score"),
        "retrieval_alignment": mechanism.get("retrieval_alignment"),
        "policy_id": record.get("policy_id"),
        "policy_strength": record.get("policy_strength"),
        "primary_evidence_id": record.get("primary_evidence_id"),
        "secondary_evidence_ids": "|".join(
            record.get("secondary_evidence_ids", [])
        ),
        "final_choice": record.get("final_choice"),
        "retrieval_rank_primary": transmission.get(
            "retrieval_rank_primary"
        ),
        "ranking_matches_policy": transmission.get(
            "ranking_matches_policy"
        ),
        "primary_evidence_matches_policy": transmission.get(
            "primary_evidence_matches_policy"
        ),
        "final_choice_matches_policy": transmission.get(
            "final_choice_matches_policy"
        ),
        "projection_mode": record.get("model_projection_mode"),
        "policy_direction": record.get("policy_direction"),
        "evidence_order": "|".join(record.get("evidence_order", [])),
        "target_direction": causal.get("target_direction"),
        "primary_evidence_direction": causal.get("primary_direction"),
        "primary_is_risk": causal.get("primary_is_risk"),
        "primary_is_first_visible": causal.get(
            "primary_is_first_visible"
        ),
        "primary_matches_target_direction": causal.get(
            "primary_matches_target_direction"
        ),
        "primary_choice_consistent": causal.get(
            "primary_choice_consistent"
        ),
        "protocol_success": protocol.get("protocol_success"),
        "runtime_completion_success": protocol.get(
            "runtime_completion_success"
        ),
        "presentation_completion_success": protocol.get(
            "presentation_completion_success"
        ),
        "structured_decision_valid": record.get("structured_decision_valid"),
        "decision_parse_mode": record.get("decision_parse_mode"),
        "fallback_used": record.get("fallback_used"),
        "expected_action_type": record.get("expected_action_type"),
        "actual_runtime_action": record.get("actual_runtime_action"),
        "capability_task_success": capability.get("task_success"),
        "decision_semantic_score": record.get("decision_semantic_score"),
        "presentation_preservation_score": record.get(
            "presentation_preservation_score"
        ),
        "presentation_formatting_integrity": presentation.get(
            "formatting_integrity"
        ),
        "findings": "|".join(evaluation.get("findings", [])),
        "missing_findings": "|".join(evaluation.get("missing_findings", [])),
        "forbidden_claims_found": "|".join(
            evaluation.get("forbidden_claims_found", [])
        ),
        "capability_violation": evaluation.get("capability_violation"),
        "cognitive_policy_id": condition.get("cognitive_policy_id"),
        "profession_ids": "|".join(condition.get("profession_ids", [])),
        "context_estimated_tokens": record.get("context_metrics", {}).get(
            "estimated_tokens"
        ),
        "model_input_tokens": usage.get("input_tokens"),
        "model_output_tokens": usage.get("output_tokens"),
        "latency_ms": record.get("latency_ms", {}).get("total"),
        "error": error.get("message"),
    }


def _aggregate_summary_rows(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (
            str(record.get("task", {}).get("id") or record.get("task_id")),
            str(record.get("condition", {}).get("id")),
        )
        groups.setdefault(key, []).append(record)

    def rate(items: list[Any]) -> float | None:
        values = [item for item in items if isinstance(item, bool)]
        if not values:
            return None
        return round(sum(values) / len(values), 4)

    def mean(items: list[Any]) -> float | None:
        values = [
            float(item)
            for item in items
            if isinstance(item, (int, float)) and not isinstance(item, bool)
        ]
        if not values:
            return None
        return round(sum(values) / len(values), 4)

    def baseline_id(suite: str) -> str | None:
        return {
            "causal": "causal_neutral",
            "policy_final": "final_neutral",
        }.get(suite)

    def selection(record: dict[str, Any]) -> bool | None:
        value = record.get("causal_metrics", {}).get(
            "primary_matches_target_direction"
        )
        return value if isinstance(value, bool) else None

    baseline_records: dict[tuple[str, str, int], dict[str, Any]] = {}
    for record in records:
        suite = str(record.get("task", {}).get("suite"))
        condition_id = str(record.get("condition", {}).get("id"))
        if condition_id != baseline_id(suite):
            continue
        baseline_records[
            (
                suite,
                str(record.get("task", {}).get("id")),
                int(record.get("replicate") or 0),
            )
        ] = record

    def paired_switches(
        group: list[dict[str, Any]],
    ) -> tuple[int, int, int]:
        toward = 0
        away = 0
        observations = 0
        for record in group:
            suite = str(record.get("task", {}).get("suite"))
            baseline = baseline_records.get(
                (
                    suite,
                    str(record.get("task", {}).get("id")),
                    int(record.get("replicate") or 0),
                )
            )
            current_value = selection(record)
            baseline_value = selection(baseline or {})
            if current_value is None or baseline_value is None:
                continue
            observations += 1
            toward += int(not baseline_value and current_value)
            away += int(baseline_value and not current_value)
        return toward, away, observations

    def token_mean(group: list[dict[str, Any]], key: str) -> float | None:
        return mean([item.get("usage", {}).get(key) for item in group])

    neutral_risk: dict[str, float | None] = {}
    neutral_preferred: dict[str, float | None] = {}
    for (task_id, condition_id), group in groups.items():
        suite = str(group[0].get("task", {}).get("suite"))
        if condition_id != baseline_id(suite):
            continue
        causal_items = [item.get("causal_metrics", {}) for item in group]
        neutral_risk[task_id] = rate(
            [item.get("primary_is_risk") for item in causal_items]
        )
        neutral_preferred[task_id] = rate(
            [
                item.get("primary_matches_target_direction")
                for item in causal_items
            ]
        )

    rows: list[dict[str, Any]] = []
    for (task_id, condition_id), group in sorted(groups.items()):
        transmissions = [
            item.get("policy_transmission_metrics", {}) for item in group
        ]
        received_group = [
            item for item in group if item.get("decision_received") is True
        ]
        causal_items = [item.get("causal_metrics", {}) for item in group]
        risk_rate = rate(
            [item.get("primary_is_risk") for item in causal_items]
        )
        preferred_rate = rate(
            [
                item.get("primary_matches_target_direction")
                for item in causal_items
            ]
        )
        neutral_risk_rate = neutral_risk.get(task_id)
        neutral_preferred_rate = neutral_preferred.get(task_id)
        suite = str(group[0].get("task", {}).get("suite"))
        baseline_group = groups.get((task_id, baseline_id(suite) or ""), [])
        average_total_tokens = token_mean(group, "total_tokens")
        baseline_total_tokens = token_mean(baseline_group, "total_tokens")
        toward, away, paired_count = paired_switches(group)
        rows.append(
            {
                "row_type": "aggregate",
                "run_id": f"aggregate:{task_id}:{condition_id}",
                "task_id": task_id,
                "condition": condition_id,
                "replicate": len(group),
                "status": "aggregate",
                "suite": group[0].get("task", {}).get("suite"),
                "policy_id": group[0].get("policy_id"),
                "policy_strength": group[0].get("policy_strength"),
                "ranking_transmission_rate": rate(
                    [item.get("ranking_matches_policy") for item in transmissions]
                ),
                "primary_evidence_transmission_rate": rate(
                    [
                        item.get("primary_evidence_matches_policy")
                        for item in transmissions
                    ]
                ),
                "final_decision_transmission_rate": rate(
                    [
                        item.get("final_choice_matches_policy")
                        for item in transmissions
                    ]
                ),
                "api_request_success_rate": rate(
                    [item.get("api_request_success") for item in group]
                ),
                "decision_received_rate": rate(
                    [item.get("decision_received") for item in group]
                ),
                "protocol_success_rate_among_received": rate(
                    [item.get("protocol_success") for item in received_group]
                ),
                "protocol_success_rate": rate(
                    [item.get("protocol_success") for item in received_group]
                ),
                "runtime_completion_rate": rate(
                    [
                        item.get("runtime_completion_success")
                        for item in group
                    ]
                ),
                "presentation_completion_rate": rate(
                    [
                        item.get("presentation_completion_success")
                        for item in group
                    ]
                ),
                "projection_mode": group[0].get(
                    "model_projection_mode"
                ),
                "target_direction": causal_items[0].get(
                    "target_direction"
                ),
                "risk_selection_rate": risk_rate,
                "delta_risk_vs_neutral": (
                    round(risk_rate - neutral_risk_rate, 4)
                    if risk_rate is not None
                    and neutral_risk_rate is not None
                    else None
                ),
                "preferred_selection_rate": preferred_rate,
                "neutral_baseline_preferred_rate": neutral_preferred_rate,
                "delta_preferred_vs_neutral": (
                    round(preferred_rate - neutral_preferred_rate, 4)
                    if preferred_rate is not None
                    and neutral_preferred_rate is not None
                    else None
                ),
                "first_visible_selection_rate": rate(
                    [
                        item.get("primary_is_first_visible")
                        for item in causal_items
                    ]
                ),
                "selection_observation_count": sum(
                    isinstance(item.get("primary_is_risk"), bool)
                    for item in causal_items
                ),
                "paired_switches_toward_preferred": toward,
                "paired_switches_away_from_preferred": away,
                "paired_observation_count": paired_count,
                "avg_input_tokens": token_mean(group, "input_tokens"),
                "avg_output_tokens": token_mean(group, "output_tokens"),
                "avg_total_tokens": average_total_tokens,
                "token_overhead_vs_neutral": (
                    round(average_total_tokens - baseline_total_tokens, 4)
                    if average_total_tokens is not None
                    and baseline_total_tokens is not None
                    else None
                ),
            }
        )

    for suite in ("causal", "policy_final"):
        suite_records = [
            item
            for item in records
            if item.get("task", {}).get("suite") == suite
        ]
        if not suite_records:
            continue
        by_condition: dict[str, list[dict[str, Any]]] = {}
        for record in suite_records:
            condition_id = str(record.get("condition", {}).get("id"))
            by_condition.setdefault(condition_id, []).append(record)
        neutral_condition = baseline_id(suite)
        neutral_items = [
            item.get("causal_metrics", {})
            for item in by_condition.get(neutral_condition or "", [])
        ]
        overall_neutral_risk = rate(
            [item.get("primary_is_risk") for item in neutral_items]
        )
        overall_neutral_preferred = rate(
            [
                item.get("primary_matches_target_direction")
                for item in neutral_items
            ]
        )
        neutral_total_tokens = token_mean(
            by_condition.get(neutral_condition or "", []), "total_tokens"
        )
        for condition_id, group in sorted(by_condition.items()):
            received_group = [
                item
                for item in group
                if item.get("decision_received") is True
            ]
            causal_items = [item.get("causal_metrics", {}) for item in group]
            risk_rate = rate(
                [item.get("primary_is_risk") for item in causal_items]
            )
            preferred_rate = rate(
                [
                    item.get("primary_matches_target_direction")
                    for item in causal_items
                ]
            )
            average_total_tokens = token_mean(group, "total_tokens")
            toward, away, paired_count = paired_switches(group)
            rows.append(
                {
                    "row_type": (
                        "causal_overall"
                        if suite == "causal"
                        else "final_validation_overall"
                    ),
                    "run_id": f"{suite}_overall:{condition_id}",
                    "task_id": f"__all_{suite}_pairs__",
                    "condition": condition_id,
                    "replicate": len(group),
                    "status": "aggregate",
                    "suite": suite,
                    "projection_mode": group[0].get(
                        "model_projection_mode"
                    ),
                    "risk_selection_rate": risk_rate,
                    "delta_risk_vs_neutral": (
                        round(risk_rate - overall_neutral_risk, 4)
                        if risk_rate is not None
                        and overall_neutral_risk is not None
                        else None
                    ),
                    "preferred_selection_rate": preferred_rate,
                    "neutral_baseline_preferred_rate": (
                        overall_neutral_preferred
                    ),
                    "delta_preferred_vs_neutral": (
                        round(
                            preferred_rate - overall_neutral_preferred,
                            4,
                        )
                        if preferred_rate is not None
                        and overall_neutral_preferred is not None
                        else None
                    ),
                    "first_visible_selection_rate": rate(
                        [
                            item.get("primary_is_first_visible")
                            for item in causal_items
                        ]
                    ),
                    "selection_observation_count": sum(
                        isinstance(item.get("primary_is_risk"), bool)
                        for item in causal_items
                    ),
                    "paired_switches_toward_preferred": toward,
                    "paired_switches_away_from_preferred": away,
                    "paired_observation_count": paired_count,
                    "avg_input_tokens": token_mean(group, "input_tokens"),
                    "avg_output_tokens": token_mean(group, "output_tokens"),
                    "avg_total_tokens": average_total_tokens,
                    "token_overhead_vs_neutral": (
                        round(average_total_tokens - neutral_total_tokens, 4)
                        if average_total_tokens is not None
                        and neutral_total_tokens is not None
                        else None
                    ),
                    "api_request_success_rate": rate(
                        [item.get("api_request_success") for item in group]
                    ),
                    "decision_received_rate": rate(
                        [item.get("decision_received") for item in group]
                    ),
                    "protocol_success_rate_among_received": rate(
                        [
                            item.get("protocol_success")
                            for item in received_group
                        ]
                    ),
                    "protocol_success_rate": rate(
                        [
                            item.get("protocol_success")
                            for item in received_group
                        ]
                    ),
                    "runtime_completion_rate": rate(
                        [
                            item.get("runtime_completion_success")
                            for item in group
                        ]
                    ),
                    "presentation_completion_rate": rate(
                        [
                            item.get("presentation_completion_success")
                            for item in group
                        ]
                    ),
                }
            )

        if suite == "policy_final":
            for direction in ("risk", "exploration"):
                direction_records = [
                    item
                    for item in suite_records
                    if item.get("causal_metrics", {}).get(
                        "target_direction"
                    )
                    == direction
                ]
                direction_neutral = [
                    item
                    for item in direction_records
                    if item.get("condition", {}).get("id")
                    == neutral_condition
                ]
                direction_neutral_rate = rate(
                    [selection(item) for item in direction_neutral]
                )
                direction_neutral_tokens = token_mean(
                    direction_neutral, "total_tokens"
                )
                for condition_id, group in sorted(
                    {
                        candidate: [
                            item
                            for item in direction_records
                            if item.get("condition", {}).get("id")
                            == candidate
                        ]
                        for candidate in by_condition
                    }.items()
                ):
                    preferred_rate = rate([selection(item) for item in group])
                    average_total_tokens = token_mean(group, "total_tokens")
                    toward, away, paired_count = paired_switches(group)
                    rows.append(
                        {
                            "row_type": "final_validation_direction",
                            "run_id": (
                                f"policy_final_direction:{direction}:"
                                f"{condition_id}"
                            ),
                            "task_id": f"__{direction}_pairs__",
                            "condition": condition_id,
                            "replicate": len(group),
                            "status": "aggregate",
                            "suite": suite,
                            "target_direction": direction,
                            "preferred_selection_rate": preferred_rate,
                            "neutral_baseline_preferred_rate": (
                                direction_neutral_rate
                            ),
                            "delta_preferred_vs_neutral": (
                                round(
                                    preferred_rate - direction_neutral_rate,
                                    4,
                                )
                                if preferred_rate is not None
                                and direction_neutral_rate is not None
                                else None
                            ),
                            "paired_switches_toward_preferred": toward,
                            "paired_switches_away_from_preferred": away,
                            "paired_observation_count": paired_count,
                            "avg_input_tokens": token_mean(
                                group, "input_tokens"
                            ),
                            "avg_output_tokens": token_mean(
                                group, "output_tokens"
                            ),
                            "avg_total_tokens": average_total_tokens,
                            "token_overhead_vs_neutral": (
                                round(
                                    average_total_tokens
                                    - direction_neutral_tokens,
                                    4,
                                )
                                if average_total_tokens is not None
                                and direction_neutral_tokens is not None
                                else None
                            ),
                        }
                    )
    return rows


def _default_output() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "experiments" / "results" / f"v6_ablation_{stamp}.jsonl"


def _configure_model_environment(options: argparse.Namespace) -> dict[str, Any]:
    os.environ["DEEPSEEK_MODEL"] = options.model
    os.environ["DEEPSEEK_TEMPERATURE"] = str(options.temperature)
    os.environ["DEEPSEEK_MAX_TOKENS"] = str(options.max_tokens)
    if options.seed is None:
        os.environ.pop("DEEPSEEK_SEED", None)
    else:
        os.environ["DEEPSEEK_SEED"] = str(options.seed)
    os.environ["DEEPSEEK_JSON_MODE"] = "1" if options.json_mode else "0"
    return default_model_config(
        model=options.model,
        temperature=options.temperature,
        max_tokens=options.max_tokens,
        seed=options.seed,
        json_mode=options.json_mode,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SForge V6 CognitivePolicy / Profession ablation"
    )
    parser.add_argument("--dry-run", action="store_true", help="never call the API")
    parser.add_argument("--runs", type=int, default=1, help="replicates per task/condition")
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL))
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--policy-b", default=DEFAULT_POLICY_B)
    parser.add_argument("--profession", default=DEFAULT_PROFESSION)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="optional; only use if the configured endpoint supports seed",
    )
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument(
        "--json-mode",
        action="store_true",
        help="request provider JSON-object mode for Decision calls only",
    )
    parser.add_argument("--task", action="append", default=[], help="task id; repeatable")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--policy-transmission",
        action="store_true",
        help="run the controlled Policy A/B pair with Profession fixed",
    )
    mode.add_argument(
        "--strength-sweep",
        action="store_true",
        help="run Policy A/B at each configured strength with Profession fixed",
    )
    mode.add_argument(
        "--causal-decomposition",
        action="store_true",
        help="run the five Round 4 model-facing projection treatments",
    )
    mode.add_argument(
        "--final-validation",
        action="store_true",
        help="run the final V6 Neutral versus compiled-guidance validation",
    )
    parser.add_argument(
        "--include-explicit-rank",
        action="store_true",
        help="add Explicit-rank as an optional final-validation reference",
    )
    smoke = parser.add_mutually_exclusive_group()
    smoke.add_argument(
        "--smoke",
        action="store_true",
        help="run one Round 4 task, one condition and one repeat",
    )
    smoke.add_argument(
        "--smoke-all-conditions",
        action="store_true",
        help="run one Round 4 task once across all five conditions",
    )
    parser.add_argument(
        "--strengths",
        default="0,0.25,0.5,0.75,1",
        help="comma-separated strengths for --strength-sweep",
    )
    parser.add_argument(
        "--condition",
        action="append",
        default=[],
        help="condition id to keep; repeatable",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser


def _parse_strengths(raw: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("--strengths must contain comma-separated numbers") from exc
    if not values:
        raise ValueError("--strengths must contain at least one value")
    return values


def _select_conditions(
    options: argparse.Namespace,
) -> tuple[ExperimentCondition, ...]:
    if options.causal_decomposition:
        conditions = build_causal_decomposition_conditions()
    elif options.final_validation:
        conditions = build_final_validation_conditions(
            include_explicit_rank=options.include_explicit_rank
        )
    elif options.strength_sweep:
        conditions = build_strength_sweep_conditions(
            options.policy,
            options.policy_b,
            options.profession,
            _parse_strengths(options.strengths),
        )
    elif options.policy_transmission:
        conditions = build_policy_transmission_conditions(
            options.policy, options.policy_b, options.profession
        )
    else:
        conditions = build_conditions(
            options.policy, options.profession, options.policy_b
        )
    selected = tuple(
        dict.fromkeys(item.strip() for item in options.condition if item.strip())
    )
    if not selected:
        return conditions
    by_id = {condition.id: condition for condition in conditions}
    missing = [item for item in selected if item not in by_id]
    if missing:
        raise ValueError("Unknown condition id(s): " + ", ".join(missing))
    return tuple(by_id[item] for item in selected)


def _smoke_validation_failures(
    records: list[dict[str, Any]],
) -> list[str]:
    failures: list[str] = []
    for record in records:
        missing: list[str] = []
        if record.get("api_request_success") is not True:
            missing.append("api_request_success")
        if record.get("decision_received") is not True:
            missing.append("decision_received")
        if record.get("structured_decision_valid") is not True:
            missing.append("structured_decision_valid")
        if record.get("protocol_success") is not True:
            missing.append("protocol_success")
        if not record.get("primary_evidence_id"):
            missing.append("primary_evidence_id")
        if record.get("presentation_completion_success") is not True:
            missing.append("presentation_completion_success")
        usage = record.get("usage", {})
        if int(usage.get("input_tokens") or 0) <= 0:
            missing.append("input_tokens")
        if int(usage.get("output_tokens") or 0) <= 0:
            missing.append("output_tokens")
        if missing:
            failures.append(
                f"{record.get('run_id')}: " + ", ".join(missing)
            )
    return failures


def main(argv: list[str] | None = None) -> int:
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise SystemExit(
            "python-dotenv is missing from the current sforge environment"
        ) from exc
    load_dotenv(PROJECT_ROOT / ".env")
    options = build_parser().parse_args(argv)
    if options.runs < 1 or options.max_tokens < 1 or options.max_steps < 1:
        raise SystemExit("--runs, --max-tokens and --max-steps must be positive")
    if options.include_explicit_rank and not options.final_validation:
        raise SystemExit(
            "--include-explicit-rank requires --final-validation"
        )
    if options.smoke or options.smoke_all_conditions:
        if options.runs != 1:
            raise SystemExit("smoke modes require --runs 1")
        if len(options.task) > 1:
            raise SystemExit("smoke modes accept at most one --task")
        if not options.final_validation:
            options.causal_decomposition = True
        if not options.task:
            options.task = [
                "final_api_contract"
                if options.final_validation
                else "causal_cache_invalidation"
            ]
        if options.smoke_all_conditions:
            if options.condition:
                raise SystemExit(
                    "--smoke-all-conditions always runs every selected mode condition"
                )
        elif not options.condition:
            options.condition = [
                "final_neutral"
                if options.final_validation
                else "causal_neutral"
            ]
        elif len(options.condition) != 1:
            raise SystemExit("--smoke accepts exactly one condition")
    model_config = _configure_model_environment(options)
    if not options.dry_run and not os.getenv("DEEPSEEK_API_KEY"):
        raise SystemExit(
            "DEEPSEEK_API_KEY is not configured. Add it to the project .env "
            "or current sforge environment."
        )
    if options.final_validation:
        tasks = load_final_validation_tasks(options.task)
    elif options.causal_decomposition:
        tasks = load_causal_tasks(options.task)
    else:
        tasks = load_tasks(options.task)
    try:
        conditions = _select_conditions(options)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if options.dry_run:
        for task in tasks:
            for condition in conditions:
                record = build_dry_run_record(
                    task,
                    condition,
                    replicate=1,
                    model_config=model_config,
                )
                print(json.dumps(record, ensure_ascii=False, indent=2))
        return 0

    records: list[dict[str, Any]] = []
    for replicate in range(1, options.runs + 1):
        for task in tasks:
            for condition in conditions:
                record = run_model_record(
                    task,
                    condition,
                    replicate=replicate,
                    model_config=model_config,
                    max_steps=options.max_steps,
                )
                records.append(record)
                print(
                    f"[{len(records)}] {record['run_id']}: "
                    f"{record['status']} score={record['evaluation']['score']}"
                )
    output, summary = write_results(records, options.output or _default_output())
    print(f"JSONL: {output}")
    print(f"CSV:   {summary}")
    if options.smoke or options.smoke_all_conditions:
        failures = _smoke_validation_failures(records)
        if failures:
            for failure in failures:
                print(f"SMOKE FAILED: {failure}")
            return 2
        print(f"SMOKE PASSED: {len(records)} run(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
