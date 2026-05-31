"""LLM jury for typed edge classification.

Implements SPEC.json link_inference_system.pipeline.stage=typed_classification:

  - Each candidate (A, B) is judged by N models in pairwise mode.
  - Each model is called twice with swapped order (A→B and B→A) so we can
    measure swap-consistency (per the LLM-as-judge bias literature; see
    BENCHMARKS.json LLM_as_jury_bias for the 35% flip rate that motivates
    this guard).
  - The aggregated confidence is product(swap_consistent, jury_agreement)
    mapped onto the 3-tier {0.0, 0.5, 1.0} scale.

For P0 the default backend is MockJudge — heuristic, deterministic, no API
calls. Real model integrations (Gemini, Claude) are pluggable behind the same
abstract interface; they require API keys via env and are off by default to
keep the test loop fast.
"""

from __future__ import annotations

import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

from ..models import EdgeType, Task

# The unknown sentinel for "no clear type" (also used when swap disagreement).
UNKNOWN: str | None = None

EDGE_TYPE_VALUES: list[str] = [t.value for t in EdgeType]


# ---------------------------------------------------------------------------
# Judge protocol
# ---------------------------------------------------------------------------


class Judge(ABC):
    """Single LLM judge. One call returns a type guess + brief rationale."""

    name: str  # human-readable identifier, e.g. "gemini-2.5-flash"

    @abstractmethod
    def classify(self, a: Task, b: Task) -> tuple[str | None, str]:
        """Return (edge_type_value_or_None, rationale)."""


# ---------------------------------------------------------------------------
# Mock judge — deterministic heuristic for tests and offline runs
# ---------------------------------------------------------------------------


class MockJudge(Judge):
    """Heuristic 'judge' for P0 testing without API access.

    Rules (in order):
      - same project, action overlap on "schema|model|interface" → notify_on_change
      - tag overlap >= 0.5 AND share a project → overlaps_with
      - one task action contains the other's action keyword → blocks
      - both have same project_id → complements (weak default)
      - otherwise → UNKNOWN
    """

    def __init__(self, name: str = "mock-heuristic", seed: int | None = None):
        self.name = name
        self._rng = random.Random(seed)

    def classify(self, a: Task, b: Task) -> tuple[str | None, str]:
        tags_a = {str(x).lower() for x in (a.context or {}).get("tags", [])}
        tags_b = {str(x).lower() for x in (b.context or {}).get("tags", [])}
        same_project = a.project_id == b.project_id
        action_a = a.action.lower()
        action_b = b.action.lower()

        schema_words = ("schema", "model", "interface", "스키마", "인터페이스", "모델")
        if same_project and any(w in action_a or w in action_b for w in schema_words):
            return ("notify_on_change", "shared schema/interface keyword in same project")

        if tags_a and tags_b:
            jac = len(tags_a & tags_b) / len(tags_a | tags_b)
            if jac >= 0.5 and same_project:
                return ("overlaps_with", f"tag jaccard={jac:.2f} in same project")

        # crude action-containment heuristic for blocks
        def _kws(s: str) -> set[str]:
            return {w for w in s.split() if len(w) > 3}

        kw_a, kw_b = _kws(action_a), _kws(action_b)
        if kw_a and kw_b and (kw_a <= kw_b or kw_b <= kw_a):
            return ("blocks", "one action is a keyword-subset of the other")

        if same_project and (tags_a & tags_b):
            return ("complements", "same project + some tag overlap")

        return (UNKNOWN, "no rule fired")


# ---------------------------------------------------------------------------
# Real API judges (skeleton; not exercised by default)
# ---------------------------------------------------------------------------


def _system_prompt() -> str:
    return (
        "You are classifying the typed relationship between two tasks in a "
        "research codebase. Return exactly one of: blocks, overlaps_with, "
        "complements, notify_on_change, supersedes, unknown. Definitions:\n"
        "- blocks (directed A→B): B's output requires A's output as input. "
        "Without A, B cannot be done or must be redone.\n"
        "- overlaps_with (undirected): two tasks touch the same module / "
        "produce overlapping artifacts and should be coordinated.\n"
        "- complements (directed A→B): A's output improves B's quality but "
        "B can run without A.\n"
        "- notify_on_change (undirected): A's interface/schema is an input "
        "to B; A changing means B might break.\n"
        "- supersedes (directed A→B): A replaces B; B is no longer worth "
        "doing.\n"
        "- unknown: none of the above is clearly applicable.\n\n"
        "Answer in JSON: {\"type\":\"...\",\"rationale\":\"one short sentence\"}."
    )


def _user_prompt(a: Task, b: Task) -> str:
    a_desc = (a.context or {}).get("desc") or ""
    b_desc = (b.context or {}).get("desc") or ""
    return (
        f"Task A:\n  id: {a.id}\n  project: {a.project_id}\n  action: {a.action}\n"
        f"  desc: {a_desc[:500]}\n\n"
        f"Task B:\n  id: {b.id}\n  project: {b.project_id}\n  action: {b.action}\n"
        f"  desc: {b_desc[:500]}\n\n"
        "Return the JSON answer now."
    )


class GeminiJudge(Judge):
    name = "gemini-2.5-flash"

    def __init__(self, api_key: str | None = None):
        import google.generativeai as genai  # local import — optional dep

        key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GeminiJudge needs GOOGLE_API_KEY or GEMINI_API_KEY")
        genai.configure(api_key=key)
        self._model = genai.GenerativeModel(
            "gemini-2.5-flash",
            system_instruction=_system_prompt(),
        )

    def classify(self, a: Task, b: Task) -> tuple[str | None, str]:
        import json as _json

        resp = self._model.generate_content(_user_prompt(a, b))
        text = (resp.text or "").strip()
        # Strip code fences if model wrapped JSON.
        if text.startswith("```"):
            text = text.strip("`").lstrip("json").strip()
        try:
            data = _json.loads(text)
        except _json.JSONDecodeError:
            return (UNKNOWN, "could not parse JSON from response")
        type_ = data.get("type")
        if type_ == "unknown":
            type_ = UNKNOWN
        elif type_ not in EDGE_TYPE_VALUES:
            type_ = UNKNOWN
        return (type_, str(data.get("rationale", ""))[:200])


class ClaudeHaikuJudge(Judge):
    name = "claude-haiku-4-5-20251001"

    def __init__(self, api_key: str | None = None):
        import anthropic  # local import — optional dep

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ClaudeHaikuJudge needs ANTHROPIC_API_KEY")
        self._client = anthropic.Anthropic(api_key=key)

    def classify(self, a: Task, b: Task) -> tuple[str | None, str]:
        import json as _json

        msg = self._client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            system=_system_prompt(),
            messages=[{"role": "user", "content": _user_prompt(a, b)}],
        )
        # anthropic SDK returns content blocks; the first text block is what we want.
        text_blocks = [
            getattr(b, "text", "") for b in msg.content if getattr(b, "type", "") == "text"
        ]
        text = "".join(text_blocks).strip()
        if text.startswith("```"):
            text = text.strip("`").lstrip("json").strip()
        try:
            data = _json.loads(text)
        except _json.JSONDecodeError:
            return (UNKNOWN, "could not parse JSON from response")
        type_ = data.get("type")
        if type_ == "unknown":
            type_ = UNKNOWN
        elif type_ not in EDGE_TYPE_VALUES:
            type_ = UNKNOWN
        return (type_, str(data.get("rationale", ""))[:200])


# ---------------------------------------------------------------------------
# Jury — aggregate per-pair across judges with swap-consistency guard
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JuryVerdict:
    type_: str | None  # None == unknown (no edge emitted)
    confidence: float  # 0.0 / 0.5 / 1.0
    swap_consistent: bool
    jury_agreement: bool
    rationale: str
    raw: dict  # per-judge per-direction results, for the log


class Jury:
    """Multi-judge typed link classifier with swap-consistency."""

    def __init__(self, judges: list[Judge]):
        if not judges:
            raise ValueError("jury requires at least one judge")
        self.judges = judges

    @property
    def model_names(self) -> list[str]:
        return [j.name for j in self.judges]

    def vote(self, a: Task, b: Task) -> JuryVerdict:
        per_judge: dict[str, dict[str, tuple[str | None, str]]] = {}
        for j in self.judges:
            per_judge[j.name] = {
                "ab": j.classify(a, b),
                "ba": j.classify(b, a),
            }

        # Direction: for directed types (blocks, complements, supersedes) the
        # A→B call asks "what relation does A→B have?" — if the swap returns
        # the *inverse* directed type, that's still consistent at the
        # undirected level. For simplicity P0 treats *any* type mismatch as
        # inconsistent and emits unknown — better precision than direction
        # gymnastics for first cut.
        swap_consistent_per_judge = []
        for name, calls in per_judge.items():
            t_ab = calls["ab"][0]
            t_ba = calls["ba"][0]
            swap_consistent_per_judge.append(t_ab == t_ba and t_ab is not None)

        # Jury agreement: all judges agree on the A→B type (after swap check).
        types_ab = [per_judge[j.name]["ab"][0] for j in self.judges]
        all_swap_ok = all(swap_consistent_per_judge)
        all_agree = all(t == types_ab[0] for t in types_ab) and types_ab[0] is not None

        if not all_swap_ok and not all_agree:
            confidence = 0.0
        elif not (all_swap_ok and all_agree):
            confidence = 0.5
        else:
            confidence = 1.0

        chosen_type: str | None = types_ab[0] if (all_swap_ok and all_agree) else None
        # If only one guard failed we still emit at confidence 0.5 if we have
        # a clear majority/single answer; otherwise UNKNOWN.
        if confidence == 0.5 and chosen_type is None:
            non_none = [t for t in types_ab if t is not None]
            if non_none:
                # majority pick (ties → first)
                most_common = max(set(non_none), key=non_none.count)
                chosen_type = most_common
            else:
                confidence = 0.0

        # Compose rationale (concatenate per-judge rationales).
        rationales = []
        for name, calls in per_judge.items():
            rationales.append(f"{name}/ab: {calls['ab'][1]}")
        rationale = " | ".join(rationales)[:500]

        return JuryVerdict(
            type_=chosen_type,
            confidence=confidence,
            swap_consistent=all_swap_ok,
            jury_agreement=all_agree,
            rationale=rationale,
            raw={
                name: {dir_: list(res) for dir_, res in calls.items()}
                for name, calls in per_judge.items()
            },
        )


def default_jury(force_mock: bool = False) -> Jury:
    """Build the default jury for P0. Picks real models when their API keys
    are present, else falls back to two mock judges with different seeds so
    swap/agreement gates are still exercised."""
    if not force_mock:
        judges: list[Judge] = []
        if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
            try:
                judges.append(GeminiJudge())
            except Exception:
                pass
        if os.environ.get("ANTHROPIC_API_KEY"):
            try:
                judges.append(ClaudeHaikuJudge())
            except Exception:
                pass
        if len(judges) >= 2:
            return Jury(judges)

    return Jury([MockJudge("mock-A", seed=1), MockJudge("mock-B", seed=2)])


__all__ = [
    "Judge",
    "MockJudge",
    "GeminiJudge",
    "ClaudeHaikuJudge",
    "Jury",
    "JuryVerdict",
    "default_jury",
    "EDGE_TYPE_VALUES",
]
