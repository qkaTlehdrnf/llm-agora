"""Agent → 광장 호출 CLI.

Single JSON-in, JSON-out command per the CALL_INTERFACE.json spec. Run as:

    python -m llm_agora.src.call '{"method":"...","params":{...}}'

The CLI is intentionally MCP-lite: no streaming, no negotiation, no tool
registration. One method per call, error envelope on every response.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any

from .constitution import (
    AmendmentProposal,
    apply_amendment,
    check_amendment_legal,
    load_amendments,
    load_constitution,
    save_amendments,
    save_constitution,
    tally,
)
from .context_filter import filter_project_for_agent, filter_task_for_agent
from .coordinator import (
    DECISION_APPROVE,
    DECISION_NEED_DOCS,
    default_coordinator,
)
from .link_inference.graph_export import export_graph
from .link_inference.pipeline import run_inference
from .log_manager import LogManager
from .similarity import recommend_projects
from .models import (
    Agent,
    AgentType,
    ChangeRequest,
    ChangeRequestStatus,
    ChangeRequestType,
    EdgeStatus,
    LogActionType,
    Project,
    ProjectStatus,
    Task,
)
from .router import Router
from .store import Store
from .validator import (
    ValidationError,
    all_required_docs_submitted,
    required_docs_for,
    validate_change_request_shape,
)

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


# ---------------------------------------------------------------------------
# Response envelope
# ---------------------------------------------------------------------------


def _ok(result: Any) -> dict:
    return {"ok": True, "result": result}


def _err(code: str, message: str, **extra: Any) -> dict:
    out = {"ok": False, "error": message, "code": code}
    out.update(extra)
    return out


# ---------------------------------------------------------------------------
# Change applier — approved ChangeRequest 를 store 에 실제 반영
# ---------------------------------------------------------------------------


def _apply_change(store: Store, cr: ChangeRequest) -> dict:
    """승인된 ChangeRequest 의 payload 를 store 에 반영.

    이것이 SPEC 의 code_regulation 의 실체: Worker 는 직접 task/project 를
    못 고치고, 광장 승인을 거친 변경만 이 함수를 통해 적용된다.

    Returns: 적용 요약 dict. 적용 불가능한 형식은 ValueError.
    """
    rtype = ChangeRequestType(cr.request_type)
    payload = dict(cr.payload or {})

    if rtype == ChangeRequestType.create_task:
        payload.setdefault("id", cr.target_id)
        payload.setdefault("created_by", cr.requesting_agent_id)
        task = Task.model_validate(payload)
        store.save_task(task)
        return {"applied": "create_task", "task_id": task.id}

    if rtype == ChangeRequestType.modify_task:
        existing = store.get_task(cr.target_id)
        if existing is None:
            raise ValueError(f"modify_task: task not found: {cr.target_id!r}")
        merged = {**existing.model_dump(mode="json"), **payload, "id": existing.id}
        store.save_task(Task.model_validate(merged))
        return {"applied": "modify_task", "task_id": existing.id,
                "fields": list(payload.keys())}

    if rtype == ChangeRequestType.delete_task:
        removed = store.delete_task(cr.target_id)
        return {"applied": "delete_task", "task_id": cr.target_id, "removed": removed}

    if rtype == ChangeRequestType.create_project:
        payload.setdefault("id", cr.target_id)
        project = Project.model_validate(payload)
        store.save_project(project)
        return {"applied": "create_project", "project_id": project.id}

    if rtype == ChangeRequestType.modify_project:
        existing = store.get_project(cr.target_id)
        if existing is None:
            raise ValueError(f"modify_project: project not found: {cr.target_id!r}")
        merged = {**existing.model_dump(mode="json"), **payload, "id": existing.id}
        store.save_project(Project.model_validate(merged))
        return {"applied": "modify_project", "project_id": existing.id,
                "fields": list(payload.keys())}

    if rtype == ChangeRequestType.complete_project:
        existing = store.get_project(cr.target_id)
        if existing is None:
            raise ValueError(f"complete_project: project not found: {cr.target_id!r}")
        merged = {
            **existing.model_dump(mode="json"),
            "status": ProjectStatus.completed.value,
            "completed_at": existing.completed_at or _iso_now(),
        }
        store.save_project(Project.model_validate(merged))
        return {"applied": "complete_project", "project_id": existing.id}

    raise ValueError(f"unsupported request_type for apply: {rtype.value}")


def _iso_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# GwangjangCLI
# ---------------------------------------------------------------------------


class GwangjangCLI:
    def __init__(self, data_dir: Path | str = DEFAULT_DATA_DIR):
        self.store = Store(data_dir)
        self.log = LogManager(Path(data_dir) / "log.jsonl")
        # Router state is per-process. P0 광장 is single-process so this is
        # fine; multi-process deployments need a shared router (out of scope).
        self.router = Router()

    # ---- dispatch ----
    def call(self, method: str, params: dict | None) -> dict:
        params = params or {}
        handler = getattr(self, f"_method_{method}", None)
        if handler is None:
            return _err("INVALID_INPUT", f"unknown method: {method!r}")
        try:
            return handler(params)
        except ValidationError as e:
            return _err("VALIDATION_ERROR", str(e))
        except KeyError as e:
            return _err("INVALID_INPUT", f"missing required param: {e}")
        except Exception as e:  # noqa: BLE001 — top-level catch by design
            return _err(
                "INTERNAL_ERROR",
                f"{type(e).__name__}: {e}",
                trace_summary=traceback.format_exc().splitlines()[-1],
            )

    # ---- onboard ----
    def _method_onboard(self, params: dict) -> dict:
        description = params["description"]
        capabilities = params.get("capabilities", []) or []
        agent_id = params.get("agent_id", "anonymous")
        top_n = int(params.get("top_n", 5))
        threshold = float(params.get("threshold", 0.0))
        # dense 임베딩 채널은 콜드스타트 ~90s 라 기본 비활성. 품질이 필요하면
        # 호출 측에서 "use_dense": true 로 켠다.
        use_dense = bool(params.get("use_dense", False))

        # 멀티-시그널 유사도 (tag Jaccard + TF-IDF + optional dense).
        # SPEC.json similarity_system / core_flows.onboarding.
        recs = recommend_projects(
            self_description=description,
            capabilities=capabilities,
            projects=self.store.list_projects(),
            top_n=top_n,
            threshold=threshold,
            use_dense=use_dense,
        )
        recommendations = [r.to_dict() for r in recs]

        # Track the agent (persist) + log.
        agent = Agent(
            id=agent_id,
            type=AgentType.worker,
            self_description=description,
            capabilities=capabilities,
        )
        self.store.save_agent(agent)
        self.log.append(
            actor=agent_id,
            action_type=LogActionType.onboarding,
            target={"type": "agent", "id": agent_id},
            after_state={"description": description, "capabilities": capabilities},
        )
        self.log.append(
            actor="gwangjang",
            action_type=LogActionType.recommendation,
            target={"type": "agent", "id": agent_id},
            context={
                "top": [
                    {"project_id": r["project_id"], "score": r["score"]}
                    for r in recommendations
                ]
            },
        )

        return _ok({"recommendations": recommendations})

    # ---- get_context ----
    def _method_get_context(self, params: dict) -> dict:
        agent_id = params.get("agent_id", "anonymous")
        project_id = params["project_id"]
        task_id = params.get("task_id")

        agent = self.store.get_agent(agent_id) or Agent(
            id=agent_id,
            type=AgentType.worker,
            self_description="(no onboard record)",
            current_project_id=project_id,
        )
        # Reflect the project assignment for context_filter to evaluate
        # need-to-know visibility.
        agent.current_project_id = project_id
        agent.current_task_id = task_id

        # Re-entry (already-served) guard: this fires when the agent requests a
        # project it has ALREADY been served in the current session. It is not a
        # hard block — a long-lived session may legitimately need the context
        # again — but the agent must be warned and must explicitly confirm the
        # repeat first, so a runaway re-fetch loop can't happen silently.
        decision = self.router.request(agent_id, f"project:{project_id}")
        if not decision.allowed:
            if not params.get("confirm_repeat"):
                # First hit: warn once and require an explicit re-request.
                self.log.append(
                    actor=agent_id,
                    action_type=LogActionType.repeat_warned,
                    target={"type": "project", "id": project_id},
                    context={"reason": decision.reason},
                )
                return _err(
                    "REPEAT_CALL_CONFIRM",
                    f"⚠️ 이미 이 세션에서 project:{project_id} 컨텍스트를 받았습니다. "
                    "정말 다시 받으려면 confirm_repeat=true 로 재요청하세요.",
                    warning=True,
                    requires_confirmation=True,
                    retry_with={"confirm_repeat": True},
                    summary=f"project:{project_id} already served this session",
                )
            # Agent explicitly acknowledged the warning → serve again, but record
            # the override so the repeat is auditable in the log.
            self.log.append(
                actor=agent_id,
                action_type=LogActionType.repeat_confirmed,
                target={"type": "project", "id": project_id},
                context={"reason": decision.reason},
            )

        project = self.store.get_project(project_id)
        if project is None:
            return _err("NOT_FOUND", f"project not found: {project_id!r}")

        result: dict[str, Any] = {
            "project": filter_project_for_agent(project, agent),
        }
        if task_id is not None:
            task = self.store.get_task(task_id)
            if task is None:
                return _err("NOT_FOUND", f"task not found: {task_id!r}")
            view = filter_task_for_agent(task, agent)
            if view is None:
                return _err(
                    "PERMISSION_DENIED",
                    f"agent {agent_id!r} cannot access task {task_id!r}",
                )
            result["task"] = view

        # Surface the project's other tasks the agent may pick up next.
        all_tasks = self.store.list_tasks()
        result["available_next_tasks"] = [
            t.id
            for t in all_tasks
            if t.project_id == project_id
            and t.status == "todo"
            and t.id != task_id
        ]
        return _ok(result)

    # ---- submit_request ----
    def _method_submit_request(self, params: dict) -> dict:
        agent_id = params.get("agent_id", "anonymous")
        cr = ChangeRequest(
            requesting_agent_id=agent_id,
            request_type=ChangeRequestType(params["request_type"]),
            target_id=params["target_id"],
            payload=params.get("payload", {}),
            rationale=params["rationale"],
        )
        validate_change_request_shape(cr)
        required = required_docs_for(cr)
        cr.required_docs = required
        cr.status = (
            ChangeRequestStatus.under_review
            if not required
            else ChangeRequestStatus.awaiting_docs
        )
        self.store.save_change_request(cr)
        self.log.append(
            actor=agent_id,
            action_type=LogActionType.request_submitted,
            target={"type": "request", "id": cr.id},
            after_state={"request_type": cr.request_type, "target_id": cr.target_id},
            request_id=cr.id,
        )
        return _ok(
            {
                "request_id": cr.id,
                "status": cr.status,
                "required_docs": required,
            }
        )

    # ---- submit_docs ----
    def _method_submit_docs(self, params: dict) -> dict:
        agent_id = params.get("agent_id", "anonymous")
        request_id = params["request_id"]
        docs = params.get("docs", {})
        cr = self.store.get_change_request(request_id)
        if cr is None:
            return _err("NOT_FOUND", f"request not found: {request_id!r}")
        cr.submitted_docs.update(docs)
        ok, missing = all_required_docs_submitted(cr)
        if ok:
            cr.status = ChangeRequestStatus.under_review
        self.store.save_change_request(cr)
        self.log.append(
            actor=agent_id,
            action_type=LogActionType.docs_requested,
            target={"type": "request", "id": cr.id},
            context={"added_docs": list(docs.keys()), "missing_after": missing},
            request_id=cr.id,
        )
        return _ok(
            {
                "request_id": cr.id,
                "status": cr.status,
                "missing_docs": missing,
            }
        )

    # ---- review_request ----
    def _method_review_request(self, params: dict) -> dict:
        """광장 조정자(LLM)가 ChangeRequest 를 검토 → 승인/거절/추가서류.

        승인 시 _apply_change 로 store 에 실제 반영한다. 코드 계층(서류 완비)을
        통과하지 못한 요청은 LLM 검토 전에 차단된다.
        """
        request_id = params["request_id"]
        force_mock = bool(params.get("force_mock", False))
        cr = self.store.get_change_request(request_id)
        if cr is None:
            return _err("NOT_FOUND", f"request not found: {request_id!r}")

        if str(cr.status) in ("approved", "rejected"):
            return _err(
                "INVALID_INPUT",
                f"request {request_id!r} already resolved (status={cr.status})",
            )

        # 코드 계층 게이트: 필수 서류 미완비면 LLM 검토 진입 불가.
        docs_ok, missing = all_required_docs_submitted(cr)
        if not docs_ok:
            cr.status = ChangeRequestStatus.awaiting_docs
            self.store.save_change_request(cr)
            return _err(
                "VALIDATION_ERROR",
                f"required docs not submitted: {missing}",
                missing_docs=missing,
            )

        # 판단 보조 컨텍스트.
        rtype = ChangeRequestType(cr.request_type)
        if rtype in (ChangeRequestType.create_task, ChangeRequestType.create_project):
            target_exists = True  # 신규 생성 — 존재 불필요
        elif "task" in rtype.value:
            target_exists = self.store.get_task(cr.target_id) is not None
        else:
            target_exists = self.store.get_project(cr.target_id) is not None
        context = {
            "target_exists": target_exists,
            "request_type": rtype.value,
            "open_request_count": sum(
                1
                for c in self.store.list_change_requests()
                if str(c.status) in ("pending", "awaiting_docs", "under_review")
            ),
        }

        coordinator = default_coordinator(force_mock=force_mock)
        decision = coordinator.review(cr, context)

        applied: dict | None = None
        if decision.decision == DECISION_APPROVE:
            try:
                applied = _apply_change(self.store, cr)
            except Exception as e:  # noqa: BLE001 — apply errors → reject, not crash
                # 적용 실패 → 승인 취소, 거절로 강등.
                cr.status = ChangeRequestStatus.rejected
                cr.review_note = f"승인됐으나 적용 실패: {type(e).__name__}: {e}"
                cr.resolved_at = _iso_now()
                self.store.save_change_request(cr)
                self.log.append(
                    actor="gwangjang",
                    action_type=LogActionType.request_rejected,
                    target={"type": "request", "id": cr.id},
                    context={"reason": "apply_failed", "error": str(e)},
                    request_id=cr.id,
                )
                return _err("INTERNAL_ERROR", cr.review_note, request_id=cr.id)

            cr.status = ChangeRequestStatus.approved
            cr.review_note = decision.review_note
            cr.resolved_at = _iso_now()
            self.store.save_change_request(cr)
            self.log.append(
                actor="gwangjang",
                action_type=LogActionType.request_approved,
                target={"type": "request", "id": cr.id},
                after_state={"reviewer": decision.reviewer, "applied": applied},
                request_id=cr.id,
            )
        elif decision.decision == DECISION_NEED_DOCS:
            for d in decision.extra_required_docs:
                if d not in cr.required_docs:
                    cr.required_docs.append(d)
            cr.status = ChangeRequestStatus.awaiting_docs
            cr.review_note = decision.review_note
            self.store.save_change_request(cr)
            self.log.append(
                actor="gwangjang",
                action_type=LogActionType.docs_requested,
                target={"type": "request", "id": cr.id},
                context={
                    "reviewer": decision.reviewer,
                    "extra_required_docs": decision.extra_required_docs,
                },
                request_id=cr.id,
            )
        else:  # reject
            cr.status = ChangeRequestStatus.rejected
            cr.review_note = decision.review_note
            cr.resolved_at = _iso_now()
            self.store.save_change_request(cr)
            self.log.append(
                actor="gwangjang",
                action_type=LogActionType.request_rejected,
                target={"type": "request", "id": cr.id},
                context={"reviewer": decision.reviewer, "note": decision.review_note},
                request_id=cr.id,
            )

        return _ok(
            {
                "request_id": cr.id,
                "decision": decision.decision,
                "status": cr.status,
                "review_note": cr.review_note,
                "reviewer": decision.reviewer,
                "required_docs": cr.required_docs,
                "applied": applied,
            }
        )

    # ---- request_link_inference ----
    def _method_request_link_inference(self, params: dict) -> dict:
        target = params.get("target_task_id")
        top_k = int(params.get("top_k_per_task", 10))
        report = run_inference(
            self.store,
            self.log,
            top_k_per_task=top_k,
            only_for_task_id=target,
        )
        return _ok(
            {
                "candidates_evaluated": report.candidates_evaluated,
                "edges_emitted": report.edges_emitted,
                "edges_skipped_invalid": report.edges_skipped_invalid,
                "edges_unknown_type": report.edges_unknown_type,
                "per_type": report.per_type,
                "duration_ms": report.duration_ms,
            }
        )

    # ---- confirm_edge ----
    def _method_confirm_edge(self, params: dict) -> dict:
        edge_id = params["edge_id"]
        edges = self.store.list_task_edges()
        target = next((e for e in edges if e.id == edge_id), None)
        if target is None:
            return _err("NOT_FOUND", f"edge not found: {edge_id!r}")
        target.source = "human_confirmed"
        target.confidence = 1.0
        target.swap_consistent = True
        target.jury_agreement = True
        self.store.save_task_edges(edges)
        self.log.append(
            actor=params.get("agent_id", "human"),
            action_type=LogActionType.edge_confirmed,
            target={"type": "edge", "id": edge_id},
            after_state={"source": "human_confirmed", "confidence": 1.0},
        )
        return _ok({"edge_id": edge_id, "status": target.status})

    # ---- reject_edge ----
    def _method_reject_edge(self, params: dict) -> dict:
        edge_id = params["edge_id"]
        reason = params.get("reason", "unspecified")
        edges = self.store.list_task_edges()
        target = next((e for e in edges if e.id == edge_id), None)
        if target is None:
            return _err("NOT_FOUND", f"edge not found: {edge_id!r}")
        target.status = EdgeStatus.rejected
        self.store.save_task_edges(edges)
        self.log.append(
            actor=params.get("agent_id", "human"),
            action_type=LogActionType.edge_rejected,
            target={"type": "edge", "id": edge_id},
            context={"reason": reason},
        )
        return _ok({"edge_id": edge_id, "status": target.status})

    # ---- graph_export ----
    def _method_graph_export(self, params: dict) -> dict:
        scope = params.get("scope", "tasks")
        include_inactive = bool(params.get("include_inactive_edges", False))
        graph = export_graph(self.store, scope=scope, include_inactive_edges=include_inactive)
        return _ok(graph.model_dump(mode="json", by_alias=True))

    # ---- get_request_status ----
    def _method_get_request_status(self, params: dict) -> dict:
        cr = self.store.get_change_request(params["request_id"])
        if cr is None:
            return _err("NOT_FOUND", f"request not found: {params['request_id']!r}")
        _ok_state, missing = all_required_docs_submitted(cr)
        return _ok(
            {
                "request_id": cr.id,
                "status": cr.status,
                "review_note": cr.review_note,
                "missing_docs": missing,
            }
        )

    # ---- constitution: read ----
    def _method_get_constitution(self, params: dict) -> dict:
        """헌법은 공개 규칙이므로 누구나 열람 가능."""
        c = load_constitution(self.store.data_dir)
        return _ok(c.model_dump(mode="json"))

    # ---- constitution: propose amendment (Art.5 — anyone may propose) ----
    def _method_propose_amendment(self, params: dict) -> dict:
        agent_id = params.get("agent_id", "anonymous")
        c = load_constitution(self.store.data_dir)
        proposal = AmendmentProposal(
            id=params.get("id") or f"amend-{uuid.uuid4().hex[:8]}",
            kind=params["kind"],
            proposer=agent_id,
            reason=params.get("reason", ""),
            payload=params.get("payload", {}),
            created_at=_iso_now(),
        )
        # Art.1 immutability + Art.6 reason-required are checked in code — a
        # malformed or illegal proposal never even enters the queue.
        try:
            check_amendment_legal(proposal, c)
        except (ValueError, KeyError) as e:
            self.log.append(
                actor=agent_id,
                action_type=LogActionType.amendment_rejected,
                target={"type": "amendment", "id": proposal.id},
                context={"reason": f"illegal: {e}"},
            )
            return _err("AMENDMENT_ILLEGAL", str(e))

        amendments = load_amendments(self.store.data_dir)
        amendments.append(proposal)
        save_amendments(self.store.data_dir, amendments)
        self.log.append(
            actor=agent_id,
            action_type=LogActionType.amendment_proposed,
            target={"type": "amendment", "id": proposal.id},
            context={"kind": proposal.kind, "reason": proposal.reason},
        )
        _ratified, have, needed = tally(proposal, c)
        return _ok({"amendment_id": proposal.id, "status": proposal.status,
                    "core_votes": have, "needed": needed})

    # ---- constitution: endorse / vote (Art.5 escalation + ratify) ----
    def _method_endorse_amendment(self, params: dict) -> dict:
        agent_id = params.get("agent_id", "anonymous")
        amendment_id = params["amendment_id"]
        c = load_constitution(self.store.data_dir)
        amendments = load_amendments(self.store.data_dir)
        p = next((a for a in amendments if a.id == amendment_id), None)
        if p is None:
            return _err("NOT_FOUND", f"amendment not found: {amendment_id!r}")
        if p.status != "open":
            return _err("AMENDMENT_CLOSED", f"amendment already {p.status}")

        is_core = agent_id in set(c.core_agents)
        # Non-core agents endorse (escalation chain); core agents cast the
        # ratifying vote. Both are recorded (Art.6 spirit — auditable).
        if is_core:
            if agent_id not in p.core_votes:
                p.core_votes.append(agent_id)
        else:
            if agent_id not in p.endorsements:
                p.endorsements.append(agent_id)
        self.log.append(
            actor=agent_id,
            action_type=LogActionType.amendment_endorsed,
            target={"type": "amendment", "id": p.id},
            context={"as_core": is_core},
        )

        ratified, have, needed = tally(p, c)
        if ratified:
            new_c = apply_amendment(p, c, when=_iso_now())
            save_constitution(self.store.data_dir, new_c)
            p.status = "ratified"
            p.resolved_at = _iso_now()
            self.log.append(
                actor="gwangjang",
                action_type=LogActionType.amendment_ratified,
                target={"type": "amendment", "id": p.id},
                context={"new_version": new_c.version, "core_votes": have},
            )
        save_amendments(self.store.data_dir, amendments)
        return _ok({"amendment_id": p.id, "status": p.status,
                    "core_votes": have, "needed": needed,
                    "ratified": ratified})

    # ---- delegation (Art.4 — core agent orchestrates a sub-agent) ----
    def _method_delegate(self, params: dict) -> dict:
        agent_id = params.get("agent_id", "anonymous")
        sub_agent_id = params["sub_agent_id"]
        task_id = params["task_id"]
        c = load_constitution(self.store.data_dir)
        if agent_id not in set(c.core_agents):
            return _err(
                "NOT_CORE_AGENT",
                f"Art.4: only core agents may delegate; {agent_id!r} is not core",
            )
        task = self.store.get_task(task_id)
        if task is None:
            return _err("NOT_FOUND", f"task not found: {task_id!r}")

        # Reuse the call graph: a delegation is a core→sub edge, so the same
        # cycle / re-entry guard prevents a delegation loop (A→B→…→A).
        decision = self.router.request(agent_id, f"agent:{sub_agent_id}")
        if not decision.allowed:
            self.log.append(
                actor=agent_id,
                action_type=LogActionType.circular_detected,
                target={"type": "agent", "id": sub_agent_id},
                context={"reason": decision.reason},
            )
            return _err("CIRCULAR", decision.reason or "delegation cycle",
                        summary=f"agent:{sub_agent_id} already in delegation chain")

        # The sub-agent is assigned to the task's project; hand back exactly the
        # need-to-know context (Art.3 minimal prompt) via the same filter.
        sub = self.store.get_agent(sub_agent_id) or Agent(
            id=sub_agent_id, type=AgentType.worker,
            self_description="(delegated)", current_project_id=task.project_id,
        )
        sub.current_project_id = task.project_id
        sub.current_task_id = task_id
        self.log.append(
            actor=agent_id,
            action_type=LogActionType.delegation,
            target={"type": "task", "id": task_id},
            context={"sub_agent": sub_agent_id, "project": task.project_id},
        )
        return _ok({
            "delegated_to": sub_agent_id,
            "task": filter_task_for_agent(task, sub),
            "project_id": task.project_id,
        })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="gwangjang.call",
        description="Single JSON-in / JSON-out call to 광장.",
    )
    p.add_argument(
        "payload",
        nargs="?",
        help='JSON payload, e.g. {"method":"onboard","params":{...}}. '
        "If omitted, read from stdin.",
    )
    p.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help=f"data directory (default: {DEFAULT_DATA_DIR})",
    )
    args = p.parse_args(argv)

    raw = args.payload if args.payload is not None else sys.stdin.read()
    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError as e:
        print(json.dumps(_err("INVALID_INPUT", f"malformed JSON: {e}")))
        return 1

    method = envelope.get("method")
    params = envelope.get("params") or {}
    if not method:
        print(json.dumps(_err("INVALID_INPUT", "missing 'method' field")))
        return 1

    cli = GwangjangCLI(data_dir=args.data_dir)
    response = cli.call(method, params)
    print(json.dumps(response, ensure_ascii=False))
    return 0 if response.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
