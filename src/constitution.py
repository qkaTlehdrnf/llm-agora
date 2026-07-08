"""광장 헌법 (Constitution) — code_regulation applied to the rules themselves.

The 광장's rules are not merely prose an agent can talk its way around: the
*constitution's own amendment process* is enforced in code here, exactly like
context isolation or the re-entry guard. A worker's prompt cannot bypass any of:

  * **Art.1 immutability** — every article may be amended EXCEPT Art.1 (the
    meta-rule "all articles are amendable"). Any amendment that would drop or
    weaken Art.1 is rejected in code.
  * **Art.6 reason-required** — a proposal, and every article it introduces or
    changes, must carry a non-empty ``reason``. Rules without a recorded reason
    cannot enter the constitution.
  * **Quorum thresholds** — REPLACE needs ⌈2/3⌉ of core agents (Art.2); ADD /
    amend needs ⌈1/2⌉ (Art.5). Tallied over the authoritative core-agent
    roster, which itself can only change through an amendment.

The seed constitution below is the user-authored core concept of 광장.
"""
from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Any, Literal
import json

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Amendment kinds.
AmendmentKind = Literal["add", "amend_article", "replace", "roster_change"]
_REPLACE = "replace"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class Article(BaseModel):
    """A single constitutional article. Art.6: ``reason`` is mandatory."""

    model_config = ConfigDict(extra="forbid")

    number: int = Field(ge=1)
    text: str = Field(min_length=1)
    reason: str = Field(min_length=1)  # Art.6 — enforced by the type itself.
    immutable: bool = False


class Constitution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    articles: list[Article]
    core_agents: list[str] = Field(default_factory=list)
    ratified_at: str | None = None

    @field_validator("articles")
    @classmethod
    def _art1_is_the_immutable_meta_rule(cls, arts: list[Article]) -> list[Article]:
        by_num = {a.number: a for a in arts}
        one = by_num.get(1)
        if one is None or not one.immutable:
            raise ValueError(
                "Art.1 must exist and be immutable (the meta-rule that all "
                "articles are amendable is itself the one thing that is not)."
            )
        return arts

    @model_validator(mode="after")
    def _unique_numbers(self) -> "Constitution":
        nums = [a.number for a in self.articles]
        if len(nums) != len(set(nums)):
            raise ValueError("duplicate article numbers")
        return self

    def article(self, number: int) -> Article | None:
        return next((a for a in self.articles if a.number == number), None)


class AmendmentProposal(BaseModel):
    """A proposed change. Art.5: anyone may propose; endorsements escalate the
    proposal upward; core-agent votes are the terminal ratification gate."""

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: AmendmentKind
    proposer: str
    reason: str = Field(min_length=1)  # Art.6.
    payload: dict[str, Any] = Field(default_factory=dict)
    endorsements: list[str] = Field(default_factory=list)  # escalation chain
    core_votes: list[str] = Field(default_factory=list)  # ratifying core agents
    status: Literal["open", "ratified", "rejected"] = "open"
    created_at: str | None = None
    resolved_at: str | None = None


# ---------------------------------------------------------------------------
# Quorum (Art.2 replace = 2/3, Art.5 add/amend = 1/2)
# ---------------------------------------------------------------------------
def required_core_votes(kind: AmendmentKind, n_core: int) -> int:
    """Minimum ratifying core votes for `kind` given `n_core` core agents."""
    if n_core <= 0:
        raise ValueError("no core agents: constitution cannot be amended")
    if kind == _REPLACE:
        return ceil(2 * n_core / 3)  # Art.2
    return ceil(n_core / 2)  # Art.5 (add / amend_article / roster_change)


# ---------------------------------------------------------------------------
# Legality — Art.1 immutability + Art.6 reason-required (uncircumventable)
# ---------------------------------------------------------------------------
def _articles_from_payload(kind: AmendmentKind, payload: dict) -> list[dict]:
    if kind in ("add", "amend_article"):
        art = payload.get("article")
        return [art] if art else []
    if kind == _REPLACE:
        return list((payload.get("constitution") or {}).get("articles", []))
    return []  # roster_change carries no articles


def check_amendment_legal(p: AmendmentProposal, current: Constitution) -> None:
    """Raise ValueError if the proposal violates an in-code invariant."""
    # Art.6 — the proposal itself and every article it introduces need a reason.
    if not (p.reason or "").strip():
        raise ValueError("Art.6: proposal has no recorded reason")
    for art in _articles_from_payload(p.kind, p.payload):
        if not str(art.get("reason", "")).strip():
            raise ValueError(
                f"Art.6: article {art.get('number','?')} has no recorded reason"
            )

    # Art.1 — the immutable meta-rule may never be amended away.
    if p.kind == "amend_article" and int(p.payload.get("article", {}).get("number", 0)) == 1:
        raise ValueError("Art.1 is immutable and cannot be amended")
    if p.kind == "roster_change":
        return
    if p.kind == _REPLACE:
        repl = Constitution.model_validate(p.payload["constitution"])  # runs Art.1 validator
        one = repl.article(1)
        cur_one = current.article(1)
        if one is None or not one.immutable or (cur_one and one.text != cur_one.text):
            raise ValueError(
                "Art.1 is immutable: a replacement must preserve Art.1 verbatim"
            )


# ---------------------------------------------------------------------------
# Tally + apply
# ---------------------------------------------------------------------------
def tally(p: AmendmentProposal, current: Constitution) -> tuple[bool, int, int]:
    """Return (ratified, have_votes, needed_votes) over the core roster."""
    core = set(current.core_agents)
    have = len({v for v in p.core_votes if v in core})
    needed = required_core_votes(p.kind, len(core))
    return have >= needed, have, needed


def apply_amendment(
    p: AmendmentProposal, current: Constitution, when: str | None = None
) -> Constitution:
    """Produce the amended Constitution. Caller must have checked legality+tally."""
    check_amendment_legal(p, current)  # defence in depth
    data = current.model_dump(mode="json")

    if p.kind == _REPLACE:
        nxt = Constitution.model_validate(p.payload["constitution"])
        data = nxt.model_dump(mode="json")
        data["core_agents"] = current.core_agents  # roster carries over
    elif p.kind == "add":
        data["articles"].append(p.payload["article"])
    elif p.kind == "amend_article":
        num = int(p.payload["article"]["number"])
        data["articles"] = [
            p.payload["article"] if a["number"] == num else a for a in data["articles"]
        ]
    elif p.kind == "roster_change":
        add = p.payload.get("add", [])
        remove = set(p.payload.get("remove", []))
        roster = [a for a in data["core_agents"] if a not in remove]
        roster += [a for a in add if a not in roster]
        data["core_agents"] = roster

    data["version"] = current.version + 1
    data["ratified_at"] = when
    return Constitution.model_validate(data)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def constitution_path(data_dir: Path | str) -> Path:
    return Path(data_dir) / "constitution.json"


def load_constitution(data_dir: Path | str) -> Constitution:
    path = constitution_path(data_dir)
    if path.exists():
        return Constitution.model_validate(json.loads(path.read_text(encoding="utf-8")))
    return seed_constitution()


def save_constitution(data_dir: Path | str, c: Constitution) -> None:
    constitution_path(data_dir).write_text(
        json.dumps(c.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def amendments_path(data_dir: Path | str) -> Path:
    return Path(data_dir) / "amendments.json"


def load_amendments(data_dir: Path | str) -> list[AmendmentProposal]:
    path = amendments_path(data_dir)
    if not path.exists():
        return []
    return [AmendmentProposal.model_validate(d) for d in json.loads(path.read_text("utf-8"))]


def save_amendments(data_dir: Path | str, items: list[AmendmentProposal]) -> None:
    amendments_path(data_dir).write_text(
        json.dumps([a.model_dump(mode="json") for a in items], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Seed — the user-authored 광장 constitution
# ---------------------------------------------------------------------------
def seed_constitution() -> Constitution:
    return Constitution(
        version=1,
        core_agents=["gwangjang"],  # bootstrap roster; grows via roster_change amendment
        articles=[
            Article(
                number=1,
                immutable=True,
                text="모든 헌법은 변경될 수 있다. (단, 이 조항 자체는 개정할 수 없다.)",
                reason="가변성을 유일하게 고정된 지점으로 두어, 헌법이 경직되지 않으면서도 "
                "'무엇이든 바꿀 수 있다'는 토대만은 흔들리지 않게 하기 위함.",
            ),
            Article(
                number=2,
                text="핵심 Agent의 2/3이 동의하는 순간, 헌법은 교체될 수 있다.",
                reason="전면 교체는 체계 전체를 바꾸므로, 단순 추가(1/2)보다 높은 합의 문턱을 둔다.",
            ),
            Article(
                number=3,
                text="Agent는 최대한 독립적인 TASK로 진행하며 최소한의 PROMPT를 유지한다. "
                "(독립적 = 추가 정보 없이 주어진 정보만으로 실행한 결과가 추가 정보가 있을 때와 "
                "동일하거나, 다른 TASK와 겹치는 작업 없이 자신의 작업이 온전히 반영되는 상태.)",
                reason="광장의 핵심 인사이트 — 최소 지식·최소 컨텍스트의 독립 TASK 다수가 "
                "코드 규제 아래 창발적으로 대형 시스템을 굴린다. context_filter·link_inference가 집행.",
            ),
            Article(
                number=4,
                text="핵심 Agent는 하위 Agent에게 핵심적인 업무를 지시하고, 긴 Prompt를 유지할 수 "
                "있으며, 주어진 목적을 위하여 하위 Agent를 Orchestration한다.",
                reason="독립 TASK(3조)를 조립해 목적을 달성할 조정자가 필요하다. "
                "조정 부담과 긴 컨텍스트는 핵심 Agent에 집중시켜 하위 Agent는 가볍게 유지한다.",
            ),
            Article(
                number=5,
                text="헌법은 누구나 제의할 수 있으며, 제의는 상위 Agent의 허락 하에 위로 결재가 "
                "올라가고, 최종적으로 핵심 Agent 절반의 동의 아래에 추가될 수 있다.",
                reason="개정 발의를 개방하되(누구나), 결재 상향과 핵심 Agent 정족수로 품질·책임을 건다.",
            ),
            Article(
                number=6,
                text="모든 규칙은 해당 규칙이 만들어진 이유와 함께 기록되어야 한다.",
                reason="이유 없는 규칙은 나중에 검증·폐기가 불가능하다. 실패 기반 규칙 누적(casebook) 방침과 동일 정신.",
            ),
        ],
    )


__all__ = [
    "Article",
    "Constitution",
    "AmendmentProposal",
    "required_core_votes",
    "check_amendment_legal",
    "tally",
    "apply_amendment",
    "load_constitution",
    "save_constitution",
    "load_amendments",
    "save_amendments",
    "seed_constitution",
]
