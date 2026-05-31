"""광장 중앙 조정자(Coordinator) LLM 인터페이스.

SPEC.json actors.gwangjang_agent 의 LLM 계층 구현. 중앙 조정자는 Worker 가 제출한
ChangeRequest 를 검토해 승인/거절/추가서류요구 중 하나를 결정한다.

  llm_layer.handles: "요청 내용의 논리적 타당성 검토 (광장 Agent)"

코드 계층(validator.py)이 형식·서류완비·DAG 같은 *우회 불가* 규칙을 강제한 뒤,
그 위에서 이 LLM 계층이 *내용 타당성* 판단을 내린다. 두 계층의 분리가 SPEC 의
핵심: code_regulation(우회 불가) + language/judgement(LLM).

judge.py 의 Judge 추상화와 동일한 패턴 — MockCoordinator 가 기본이고, env 키가
있으면 Gemini / Claude 백엔드로 자동 승격한다.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .models import ChangeRequest, ChangeRequestType

# 결정 종류.
DECISION_APPROVE = "approve"
DECISION_REJECT = "reject"
DECISION_NEED_DOCS = "need_docs"

_VALID_DECISIONS = {DECISION_APPROVE, DECISION_REJECT, DECISION_NEED_DOCS}

# 위험도 높은(되돌리기 어려운) 요청 유형.
_HIGH_RISK_TYPES = {
    ChangeRequestType.delete_task,
    ChangeRequestType.complete_project,
}


@dataclass(frozen=True)
class ReviewDecision:
    decision: str  # approve | reject | need_docs
    review_note: str
    extra_required_docs: list[str] = field(default_factory=list)
    reviewer: str = "coordinator"

    @property
    def approved(self) -> bool:
        return self.decision == DECISION_APPROVE

    def to_dict(self) -> dict:
        return {
            "decision": self.decision,
            "review_note": self.review_note,
            "extra_required_docs": self.extra_required_docs,
            "reviewer": self.reviewer,
        }


# ---------------------------------------------------------------------------
# Coordinator protocol
# ---------------------------------------------------------------------------


class Coordinator(ABC):
    name: str

    @abstractmethod
    def review(self, cr: ChangeRequest, context: dict) -> ReviewDecision:
        """ChangeRequest 검토 → ReviewDecision.

        context: 판단 보조용 메타 (예: {"target_exists": bool,
        "project_active": bool, "open_request_count": int}).
        """


# ---------------------------------------------------------------------------
# Mock coordinator — 결정론적 휴리스틱 (기본, API 불필요)
# ---------------------------------------------------------------------------


class MockCoordinator(Coordinator):
    """규칙 기반 검토자. 테스트/오프라인용 결정론적 판단.

    규칙 (순서대로):
      - rationale 가 너무 짧음(<15자) → reject (근거 불충분)
      - 고위험 유형(delete_task/complete_project)인데 impact 서류 없음 → need_docs
      - target_exists=False 이고 수정/삭제/완료 요청 → reject (대상 없음)
      - 그 외 → approve
    """

    def __init__(self, name: str = "mock-coordinator"):
        self.name = name

    def review(self, cr: ChangeRequest, context: dict) -> ReviewDecision:
        rationale = (cr.rationale or "").strip()
        rtype = ChangeRequestType(cr.request_type)

        if len(rationale) < 15:
            return ReviewDecision(
                DECISION_REJECT,
                f"근거(rationale)가 불충분합니다({len(rationale)}자). 변경의 "
                "필요성을 구체적으로 서술해 재제출하세요.",
                reviewer=self.name,
            )

        target_exists = context.get("target_exists", True)
        mutating = rtype in (
            ChangeRequestType.modify_task,
            ChangeRequestType.delete_task,
            ChangeRequestType.complete_project,
            ChangeRequestType.modify_project,
        )
        if mutating and not target_exists:
            return ReviewDecision(
                DECISION_REJECT,
                f"대상 {cr.target_id!r} 을(를) 찾을 수 없어 {rtype.value} 를 "
                "수행할 수 없습니다.",
                reviewer=self.name,
            )

        if rtype in _HIGH_RISK_TYPES and "impact_assessment" not in cr.submitted_docs:
            return ReviewDecision(
                DECISION_NEED_DOCS,
                f"{rtype.value} 는 되돌리기 어려운 고위험 작업입니다. "
                "영향 평가(impact_assessment) 서류를 제출하세요.",
                extra_required_docs=["impact_assessment"],
                reviewer=self.name,
            )

        return ReviewDecision(
            DECISION_APPROVE,
            f"{rtype.value} 승인. 근거 타당, 필수 서류 완비.",
            reviewer=self.name,
        )


# ---------------------------------------------------------------------------
# LLM coordinators (judge.py 와 동일한 env-키 패턴)
# ---------------------------------------------------------------------------


def _system_prompt() -> str:
    return (
        "You are the central coordinator (광장 Agent) of a multi-agent research "
        "codebase. A worker agent submitted a change request. The CODE LAYER has "
        "already enforced format, required-document, and DAG-cycle rules — your "
        "job is to judge the LOGICAL VALIDITY and RISK of the content.\n\n"
        "Return exactly one decision:\n"
        "- approve: the change is justified, scoped, and safe to apply.\n"
        "- reject: the rationale is weak, the target is wrong, or the change is "
        "harmful/duplicative.\n"
        "- need_docs: plausible but you need more evidence before approving "
        "(name the documents).\n\n"
        "Be stricter for delete_task / complete_project (hard to reverse).\n"
        "Answer in JSON: {\"decision\":\"approve|reject|need_docs\","
        "\"review_note\":\"one or two sentences (Korean ok)\","
        "\"required_docs\":[\"doc_type\", ...]}."
    )


def _user_prompt(cr: ChangeRequest, context: dict) -> str:
    return (
        f"Change request:\n"
        f"  type: {cr.request_type}\n"
        f"  target_id: {cr.target_id}\n"
        f"  rationale: {cr.rationale}\n"
        f"  payload: {json.dumps(cr.payload, ensure_ascii=False)[:800]}\n"
        f"  submitted_docs: {list(cr.submitted_docs.keys())}\n\n"
        f"Context: {json.dumps(context, ensure_ascii=False)}\n\n"
        "Return the JSON decision now."
    )


def _parse_decision(text: str, reviewer: str) -> ReviewDecision:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`").lstrip("json").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # 파싱 실패는 보수적으로 need_docs (자동 승인 금지).
        return ReviewDecision(
            DECISION_NEED_DOCS,
            "조정자 응답을 파싱하지 못했습니다. 명확한 근거 서류와 함께 재검토 요청하세요.",
            reviewer=reviewer,
        )
    decision = str(data.get("decision", "")).lower()
    if decision not in _VALID_DECISIONS:
        decision = DECISION_NEED_DOCS
    return ReviewDecision(
        decision=decision,
        review_note=str(data.get("review_note", ""))[:500],
        extra_required_docs=[str(d) for d in data.get("required_docs", []) or []],
        reviewer=reviewer,
    )


class GeminiCoordinator(Coordinator):
    name = "gemini-2.5-flash"

    def __init__(self, api_key: str | None = None):
        import google.generativeai as genai  # local import — optional dep

        key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GeminiCoordinator needs GOOGLE_API_KEY or GEMINI_API_KEY")
        genai.configure(api_key=key)
        self._model = genai.GenerativeModel(
            "gemini-2.5-flash", system_instruction=_system_prompt()
        )

    def review(self, cr: ChangeRequest, context: dict) -> ReviewDecision:
        resp = self._model.generate_content(_user_prompt(cr, context))
        return _parse_decision(resp.text or "", self.name)


class ClaudeCoordinator(Coordinator):
    name = "claude-haiku-4-5-20251001"

    def __init__(self, api_key: str | None = None):
        import anthropic  # local import — optional dep

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ClaudeCoordinator needs ANTHROPIC_API_KEY")
        self._client = anthropic.Anthropic(api_key=key)

    def review(self, cr: ChangeRequest, context: dict) -> ReviewDecision:
        msg = self._client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            system=_system_prompt(),
            messages=[{"role": "user", "content": _user_prompt(cr, context)}],
        )
        text = "".join(
            getattr(b, "text", "") for b in msg.content if getattr(b, "type", "") == "text"
        )
        return _parse_decision(text, self.name)


def default_coordinator(force_mock: bool = False) -> Coordinator:
    """기본 조정자. API 키 있으면 LLM, 없으면 MockCoordinator.

    jury 와 달리 조정자는 단일 결정권자이므로 1개 모델로 충분 — Gemini 우선,
    없으면 Claude, 둘 다 없으면 mock.
    """
    if not force_mock:
        if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
            try:
                return GeminiCoordinator()
            except Exception:
                pass
        if os.environ.get("ANTHROPIC_API_KEY"):
            try:
                return ClaudeCoordinator()
            except Exception:
                pass
    return MockCoordinator()


__all__ = [
    "ReviewDecision",
    "Coordinator",
    "MockCoordinator",
    "GeminiCoordinator",
    "ClaudeCoordinator",
    "default_coordinator",
    "DECISION_APPROVE",
    "DECISION_REJECT",
    "DECISION_NEED_DOCS",
]
