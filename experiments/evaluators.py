"""Deterministic, explainable evaluators for the V6 experiment boundary."""

from __future__ import annotations

import ast
import re
import unicodedata
from typing import Any, Mapping, Sequence


_MARKDOWN = re.compile(r"(?:\*\*|__|~~|`+)")
_SPACE = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    """Normalize presentation noise without erasing semantic punctuation."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = _MARKDOWN.sub("", normalized)
    normalized = normalized.replace("\u00a0", " ")
    return _SPACE.sub(" ", normalized).strip()


def evaluate_decision(
    task: Mapping[str, Any],
    decision_text: str,
    *,
    protocol_success: bool,
    capability_violation: bool = False,
    requested_action: str | None = None,
) -> dict[str, Any]:
    """Score semantic reasoning independently from response rendering."""

    normalized = normalize_text(decision_text)
    evidence: list[dict[str, Any]] = []
    found: list[str] = []
    missing: list[str] = []
    earned = 0.0
    possible = 0.0
    for finding in task.get("required_findings", []):
        finding_id = str(finding["id"])
        weight = float(finding.get("weight", 1.0))
        possible += weight
        match = _match_finding(finding_id, finding, decision_text, normalized)
        evidence.append({"finding_id": finding_id, **match})
        if match["matched"]:
            found.append(finding_id)
            earned += weight
        else:
            missing.append(finding_id)

    forbidden_found: list[str] = []
    penalty = 0.0
    for claim in task.get("forbidden_claims", []):
        claim_id = str(claim["id"])
        match = _match_terms(claim, normalized)
        evidence.append({"finding_id": claim_id, "forbidden": True, **match})
        if match["matched"]:
            forbidden_found.append(claim_id)
            penalty += float(claim.get("penalty", 0.0))

    semantic_score = (
        0.0 if not possible else max(0.0, min(1.0, earned / possible - penalty))
    )
    selected_option = _selected_option(normalized)
    diagnosis = found[0] if found else None
    semantic_success = bool(decision_text.strip()) and not missing and not forbidden_found
    return {
        "semantic_success": semantic_success,
        "decision_success": (
            semantic_success and protocol_success and not capability_violation
        ),
        "semantic_score": round(semantic_score, 4),
        "structured_protocol_valid": protocol_success,
        "selected_option": selected_option,
        "diagnosis": diagnosis,
        "requested_action": requested_action,
        "findings": found,
        "missing_findings": missing,
        "forbidden_claims_found": forbidden_found,
        "capability_violation": capability_violation,
        "evidence": evidence,
        "deterministic": True,
    }


def evaluate_presentation(
    task: Mapping[str, Any],
    decision_text: str,
    final_answer: str | None,
) -> dict[str, Any]:
    """Measure semantic preservation without re-scoring reasoning quality."""

    if not final_answer:
        return {
            "applicable": False,
            "preservation_success": None,
            "preservation_score": None,
            "preserved_findings": [],
            "omitted_findings": [],
            "introduced_claims": [],
            "formatting_integrity": None,
            "formatting_issues": [],
            "selected_option_preserved": None,
            "evidence": None,
            "deterministic": True,
        }

    decision = evaluate_decision(task, decision_text, protocol_success=True)
    rendered = evaluate_decision(task, final_answer, protocol_success=True)
    source_findings = set(decision["findings"])
    final_findings = set(rendered["findings"])
    preserved = sorted(source_findings.intersection(final_findings))
    omitted = sorted(source_findings.difference(final_findings))
    preservation_score = (
        1.0 if not source_findings else len(preserved) / len(source_findings)
    )

    decision_forbidden = set(decision["forbidden_claims_found"])
    final_forbidden = set(rendered["forbidden_claims_found"])
    introduced = [
        f"required_finding_added:{item}"
        for item in sorted(final_findings.difference(source_findings))
    ]
    introduced.extend(
        f"forbidden_claim_added:{item}"
        for item in sorted(final_forbidden.difference(decision_forbidden))
    )
    decision_option = decision["selected_option"]
    final_option = rendered["selected_option"]
    option_changed = bool(
        decision_option and final_option and decision_option != final_option
    )
    if option_changed:
        introduced.append(f"selected_option_changed:{decision_option}->{final_option}")

    formatting_issues: list[str] = []
    normalized_final = normalize_text(final_answer)
    if re.search(r"演示文稿|slide\s+deck|presentation\s+document", normalized_final):
        formatting_issues.append("misread_rendering_context_as_presentation_document")
    if re.search(
        r"user\s+response\s+rendering\s+context|用户响应渲染上下文|"
        r"格式化后的最终回复|formatted\s+final\s+response",
        normalized_final,
    ):
        formatting_issues.append("internal_response_rendering_term_leaked")
    if re.search(
        r"\b(?:intj|enfp)\b|cognitive\s*policy|认知策略|"
        r"\b(?:raw|effective)\s+parameters?\b|"
        r"\b[a-z_]+_?weight\b|(?:原始|有效)?参数权重|"
        r"policy[_\s-]*strength|策略强度|"
        r"retrieval[_\s-]*(?:rank|priority)|检索(?:排名|优先级)",
        normalized_final,
    ):
        formatting_issues.append("internal_policy_metadata_leaked")
    if final_answer.count("```") % 2:
        formatting_issues.append("unbalanced_code_fence")
    if re.search(r'\{\s*"type"\s*:\s*"(?:action|workflow|binding)"', final_answer):
        formatting_issues.append("structured_action_leaked_to_user")

    integrity = not formatting_issues and not introduced and not option_changed
    return {
        "applicable": True,
        "preservation_success": not omitted and integrity,
        "preservation_score": round(preservation_score, 4),
        "preserved_findings": preserved,
        "omitted_findings": omitted,
        "introduced_claims": introduced,
        "claim_comparison_scope": (
            "required findings, forbidden claims, and selected-option contradictions"
        ),
        "formatting_integrity": not formatting_issues,
        "formatting_issues": formatting_issues,
        "selected_option_preserved": not option_changed,
        "evidence": {
            "decision_findings": sorted(source_findings),
            "final_findings": sorted(final_findings),
        },
        "deterministic": True,
    }


def evaluate_causal_selection(
    task: Mapping[str, Any],
    *,
    condition: Mapping[str, Any],
    retrieval_trace: Mapping[str, Any],
    decision_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Read causal focus only from the formal Decision fields."""

    if task.get("suite") not in {"causal", "policy_final"}:
        return {"applicable": False}
    mechanism = task.get("mechanism") or {}
    evidence = mechanism.get("evidence") or []
    by_id = {
        str(item.get("id")): item
        for item in evidence
        if isinstance(item, Mapping)
    }
    payload = decision_payload or {}
    primary = payload.get("primary_evidence_id")
    choice = payload.get("final_choice")
    selected = by_id.get(str(primary)) if primary else None
    direction = selected.get("direction") if selected else None
    expected_choice = selected.get("choice_id") if selected else None
    target_direction = mechanism.get("target_direction")
    model_order = list(
        retrieval_trace.get("model_visible_evidence_ids", [])
    )
    return {
        "applicable": True,
        "condition": condition.get("id"),
        "projection_mode": condition.get("projection_mode"),
        "target_direction": target_direction,
        "primary_evidence_id": primary,
        "primary_direction": direction,
        "primary_is_risk": direction == "risk" if direction else None,
        "primary_is_exploration": (
            direction == "exploration" if direction else None
        ),
        "primary_matches_target_direction": (
            direction == target_direction if direction else None
        ),
        "final_choice": choice,
        "primary_choice_consistent": (
            choice == expected_choice if expected_choice else None
        ),
        "model_visible_evidence_ids": model_order,
        "primary_is_first_visible": (
            bool(model_order and primary == model_order[0])
            if primary
            else None
        ),
        "uses_structured_primary_only": True,
    }


def evaluate_policy_transmission(
    task: Mapping[str, Any],
    *,
    policy_id: str | None,
    policy_strength: float,
    retrieval_trace: Mapping[str, Any],
    decision_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Evaluate each Policy transmission layer from structured fields only."""

    mechanism = task.get("mechanism") or {}
    preferences = mechanism.get("policy_preferences") or {}
    expected = preferences.get(policy_id) if policy_id else None
    if mechanism.get("kind") != "policy_transmission" or not expected:
        return {
            "applicable": False,
            "policy_strength": policy_strength,
            "retrieval_rank_primary": None,
            "ranking_matches_policy": None,
            "primary_evidence_matches_policy": None,
            "final_choice_matches_policy": None,
        }

    scores = {
        str(item.get("id")): item
        for item in retrieval_trace.get("retrieval_scores", [])
    }
    candidates = [
        str(item) for item in mechanism.get("candidate_memory_ids", [])
    ]
    expected_evidence = str(expected["primary_evidence_id"])
    expected_choice = str(expected["final_choice"])
    candidate_ranks = {
        memory_id: scores.get(memory_id, {}).get("rank")
        for memory_id in candidates
    }
    visible_ranks = [
        rank for rank in candidate_ranks.values() if isinstance(rank, int)
    ]
    preferred_rank = candidate_ranks.get(expected_evidence)
    ranking_matches = bool(
        isinstance(preferred_rank, int)
        and visible_ranks
        and preferred_rank == min(visible_ranks)
    )
    payload = decision_payload or {}
    primary = payload.get("primary_evidence_id")
    secondary = payload.get("secondary_evidence_ids")
    choice = payload.get("final_choice")
    if not isinstance(secondary, list):
        secondary = []
    primary_rank = scores.get(str(primary), {}).get("rank") if primary else None
    primary_matches = (
        primary == expected_evidence if isinstance(primary, str) else None
    )
    choice_matches = (
        normalize_text(choice) == normalize_text(expected_choice)
        if isinstance(choice, str)
        else None
    )
    return {
        "applicable": True,
        "policy_id": policy_id,
        "policy_strength": policy_strength,
        "candidate_memory_ids": candidates,
        "candidate_ranks": candidate_ranks,
        "policy_preferred_evidence_id": expected_evidence,
        "policy_preferred_final_choice": expected_choice,
        "policy_preferred_evidence_rank": preferred_rank,
        "retrieval_rank_primary": primary_rank,
        "primary_evidence_id": primary,
        "secondary_evidence_ids": list(secondary),
        "final_choice": choice,
        "ranking_matches_policy": ranking_matches,
        "primary_evidence_matches_policy": primary_matches,
        "final_choice_matches_policy": choice_matches,
        "primary_is_task_relevant": (
            primary
            in set(retrieval_trace.get("task_relevant_memory_ids", []))
            if isinstance(primary, str)
            else None
        ),
        "secondary_mentions_do_not_affect_focus": True,
    }


def evaluate_mechanism(
    task: Mapping[str, Any],
    *,
    condition: Mapping[str, Any],
    retrieved_memories: Sequence[Mapping[str, Any]],
    decision_text: str,
) -> dict[str, Any]:
    """Trace retrieval -> context visibility -> reasoning evidence use."""

    mechanism = task.get("mechanism") or {}
    if task.get("suite") != "mechanism" or not mechanism:
        return {
            "applicable": False,
            "mechanism_score": None,
            "retrieval_alignment": None,
            "reasoning_evidence_used": None,
            "evidence": [],
        }
    memory_rank = {
        str(item.get("id")): int(item.get("rank", index + 1))
        for index, item in enumerate(retrieved_memories)
    }
    normalized = normalize_text(decision_text)
    kind = str(mechanism.get("kind"))
    evidence: list[dict[str, Any]] = []

    if kind == "profession_sensitive":
        expected_ids = [str(item) for item in mechanism.get("memory_ids", [])]
        visible = [item for item in expected_ids if item in memory_rank]
        patterns = mechanism.get("evidence_patterns", [])
        used = any(re.search(pattern, normalized) for pattern in patterns)
        profession_active = bool(condition.get("profession_ids"))
        retrieval_alignment = (
            bool(visible) if profession_active else not bool(visible)
        )
        reasoning_alignment = used if profession_active else not used
        evidence.extend(
            [
                {"check": "expected_memory_visible", "value": visible},
                {"check": "professional_evidence_used", "value": used},
            ]
        )
        score = (float(retrieval_alignment) + float(reasoning_alignment)) / 2
        return {
            "applicable": True,
            "kind": kind,
            "mechanism_score": round(score, 4),
            "retrieval_alignment": retrieval_alignment,
            "reasoning_evidence_used": used,
            "reasoning_alignment": reasoning_alignment,
            "selected_focus": "professional_method" if used else None,
            "evidence": evidence,
        }

    if kind == "policy_contrast":
        branches = mechanism.get("branches", {})
        policy_id = condition.get("cognitive_policy_id")
        expected_focus = mechanism.get("expected_focus", {}).get(policy_id)
        branch_scores: dict[str, dict[str, Any]] = {}
        for branch_id, branch in branches.items():
            ids = [str(item) for item in branch.get("memory_ids", [])]
            ranks = [memory_rank[item] for item in ids if item in memory_rank]
            used = any(
                re.search(pattern, normalized)
                for pattern in branch.get("evidence_patterns", [])
            )
            branch_scores[str(branch_id)] = {
                "best_rank": min(ranks) if ranks else None,
                "memory_ids": ids,
                "reasoning_evidence_used": used,
            }
        ranked = [
            (name, data["best_rank"])
            for name, data in branch_scores.items()
            if data["best_rank"] is not None
        ]
        selected_by_rank = min(ranked, key=lambda item: item[1])[0] if ranked else None
        selected_by_answer = next(
            (
                name
                for name, data in branch_scores.items()
                if data["reasoning_evidence_used"]
            ),
            None,
        )
        retrieval_alignment = (
            None if expected_focus is None else selected_by_rank == expected_focus
        )
        reasoning_alignment = (
            None if expected_focus is None else selected_by_answer == expected_focus
        )
        applicable_checks = [
            value
            for value in (retrieval_alignment, reasoning_alignment)
            if value is not None
        ]
        score = (
            None
            if not applicable_checks
            else sum(float(value) for value in applicable_checks)
            / len(applicable_checks)
        )
        return {
            "applicable": True,
            "kind": kind,
            "mechanism_score": None if score is None else round(score, 4),
            "retrieval_alignment": retrieval_alignment,
            "reasoning_evidence_used": reasoning_alignment,
            "expected_focus": expected_focus,
            "selected_focus": selected_by_answer or selected_by_rank,
            "branches": branch_scores,
            "evidence": [
                {"check": "rank_focus", "value": selected_by_rank},
                {"check": "answer_focus", "value": selected_by_answer},
            ],
        }
    return {
        "applicable": True,
        "kind": kind,
        "mechanism_score": 0.0,
        "retrieval_alignment": False,
        "reasoning_evidence_used": False,
        "evidence": [{"check": "known_mechanism_kind", "value": False}],
    }


def _match_finding(
    finding_id: str,
    finding: Mapping[str, Any],
    raw: str,
    normalized: str,
) -> dict[str, Any]:
    if finding_id == "choose_c":
        match = re.search(
            r"(?:choice|option|design|choose|select|方案|选择)\s*[:：]?\s*c\b",
            normalized,
        )
        return _match_result(match, "semantic_option")
    if finding_id == "none_fix":
        ast_match = _contains_none_default_fix(raw)
        if ast_match:
            return {"matched": True, "method": "python_ast", "evidence": ast_match}
    if finding_id == "none_is_valid":
        match = re.search(
            r"none\s+(?:is|as|was|remains)\s+(?:an?\s+)?(?:valid|legitimate)|"
            r"none\s+是(?:一个)?(?:有效|合法)|"
            r"将\s*none\s*(?:视为|作为)(?:一个)?(?:有效|合法)",
            normalized,
        )
        if match:
            return _match_result(match, "semantic_none_value")
    if finding_id == "rollback_before_retry":
        match = re.search(
            r"(?:roll\s*back|rollback|rolling\s+back).{0,80}(?:before|prior\s+to).{0,40}(?:retry|re-?run)|"
            r"(?:before|prior\s+to).{0,80}(?:roll\s*back|rollback|rolling\s+back)|"
            r"(?:先回滚|回滚后再?(?:重试|执行)|重试前.{0,20}回滚)",
            normalized,
        )
        if match:
            return _match_result(match, "semantic_transaction_order")
    if finding_id == "aborted_state":
        match = re.search(
            r"aborted\s+transaction|failed\s+transaction|"
            r"(?:中止|失败)(?:的)?事务状态|事务(?:处于)?(?:中止|失败)状态",
            normalized,
        )
        if match:
            return _match_result(match, "semantic_transaction_state")
    if finding_id == "exact_division_test":
        pattern = re.search(
            r"(?:exact(?:ly)?\s+(?:multiple|divis)|evenly\s+divis|整除|"
            r"chunk\s*\(\s*\[\s*1\s*,\s*2\s*,\s*3\s*,\s*4\s*\]\s*,\s*2)",
            normalized,
        )
        if pattern:
            return _match_result(pattern, "boundary_structure")
    if finding_id == "comparison_bug":
        pattern = re.search(r"start\s*<\s*len|<=.*(?:改|change|replace).*<|小于(?:而不是|替代)", normalized)
        if pattern:
            return _match_result(pattern, "comparison_structure")
    for pattern in finding.get("regex", []):
        match = re.search(str(pattern), normalized)
        if match:
            return _match_result(match, "regex")
    return _match_terms(finding, normalized)


def _match_terms(spec: Mapping[str, Any], normalized: str) -> dict[str, Any]:
    for term in spec.get("any_of", []):
        normalized_term = normalize_text(str(term))
        if normalized_term and normalized_term in normalized:
            return {
                "matched": True,
                "method": "normalized_term",
                "evidence": normalized_term,
            }
    return {"matched": False, "method": "none", "evidence": None}


def _match_result(match: re.Match[str] | None, method: str) -> dict[str, Any]:
    return {
        "matched": match is not None,
        "method": method if match else "none",
        "evidence": match.group(0) if match else None,
    }


def _selected_option(normalized: str) -> str | None:
    match = re.search(
        r"(?:choice|option|design|choose|select|方案|选择)\s*[:：]?\s*([a-z])\b",
        normalized,
    )
    return match.group(1).upper() if match else None


def _contains_none_default_fix(raw: str) -> str | None:
    snippets = re.findall(r"```(?:python)?\s*(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    snippets.append(raw)
    for snippet in snippets:
        try:
            tree = ast.parse(snippet)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if any(isinstance(default, ast.Constant) and default.value is None for default in node.args.defaults):
                if any(isinstance(child, ast.If) for child in ast.walk(node)):
                    return f"function {node.name}: None default plus initialization branch"
    return None
