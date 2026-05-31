"""Pydantic models for the 광장 (LLM Agora) system.

Mirrors DATA_MODELS.json. Any schema change must update
both files in lockstep — DATA_MODELS.json is the LLM-readable spec, this file
is the runtime enforcement.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uuid() -> str:
    return str(uuid4())


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ProjectStatus(str, Enum):
    active = "active"
    paused = "paused"
    completed = "completed"


class TaskStatus(str, Enum):
    todo = "todo"
    in_progress = "in_progress"
    done = "done"
    blocked = "blocked"
    cancelled = "cancelled"
    superseded = "superseded"


class Priority(str, Enum):
    urgent = "urgent"
    high = "high"
    normal = "normal"
    low = "low"


class AgentType(str, Enum):
    gwangjang = "gwangjang"
    worker = "worker"


class ChangeRequestType(str, Enum):
    create_task = "create_task"
    modify_task = "modify_task"
    delete_task = "delete_task"
    create_project = "create_project"
    complete_project = "complete_project"
    modify_project = "modify_project"


class ChangeRequestStatus(str, Enum):
    pending = "pending"
    awaiting_docs = "awaiting_docs"
    under_review = "under_review"
    approved = "approved"
    rejected = "rejected"


class LogActionType(str, Enum):
    onboarding = "onboarding"
    recommendation = "recommendation"
    request_submitted = "request_submitted"
    docs_requested = "docs_requested"
    request_approved = "request_approved"
    request_rejected = "request_rejected"
    task_status_changed = "task_status_changed"
    circular_detected = "circular_detected"
    edge_inferred = "edge_inferred"
    edge_confirmed = "edge_confirmed"
    edge_rejected = "edge_rejected"


class EdgeType(str, Enum):
    blocks = "blocks"
    overlaps_with = "overlaps_with"
    complements = "complements"
    notify_on_change = "notify_on_change"
    supersedes = "supersedes"


class EdgeDirection(str, Enum):
    directed = "directed"
    undirected = "undirected"


class EdgeStatus(str, Enum):
    active = "active"
    drift_detected = "drift_detected"
    superseded_by_human_override = "superseded_by_human_override"
    rejected = "rejected"


class EdgeSource(str, Enum):
    llm_jury = "llm_jury"
    human_confirmed = "human_confirmed"
    inherited = "inherited"  # P2 GraphSAGE 등


# Mapping edge type → canonical direction (per SPEC.json link_inference_system.edge_types)
_EDGE_DIRECTION: dict[EdgeType, EdgeDirection] = {
    EdgeType.blocks: EdgeDirection.directed,
    EdgeType.overlaps_with: EdgeDirection.undirected,
    EdgeType.complements: EdgeDirection.directed,
    EdgeType.notify_on_change: EdgeDirection.undirected,
    EdgeType.supersedes: EdgeDirection.directed,
}


class NodeKind(str, Enum):
    task = "task"
    project = "project"
    lab = "lab"


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------


class Project(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    id: str
    title: str
    goal: str
    tags: list[str] = Field(default_factory=list)
    description_llm: str
    status: ProjectStatus = ProjectStatus.active
    domain: str
    tasks: list[str] = Field(default_factory=list)
    assigned_agents: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_now_iso)
    completed_at: str | None = None
    completion_criteria: str


class Task(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    id: str
    project_id: str
    action: str
    status: TaskStatus = TaskStatus.todo
    priority: Priority = Priority.normal
    prerequisite_task_ids: list[str] = Field(default_factory=list)
    done_when: str
    cmd: str | None = None
    assigned_agent: str | None = None
    created_at: str = Field(default_factory=_now_iso)
    created_by: str = "human"
    context: dict[str, Any] = Field(default_factory=dict)


class Agent(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    id: str
    type: AgentType
    self_description: str
    capabilities: list[str] = Field(default_factory=list)
    current_project_id: str | None = None
    current_task_id: str | None = None
    session_context: str | None = None


class ChangeRequest(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    id: str = Field(default_factory=_uuid)
    requesting_agent_id: str
    request_type: ChangeRequestType
    target_id: str
    payload: dict[str, Any]
    rationale: str
    required_docs: list[str] = Field(default_factory=list)
    submitted_docs: dict[str, str] = Field(default_factory=dict)
    status: ChangeRequestStatus = ChangeRequestStatus.pending
    review_note: str | None = None
    submitted_at: str = Field(default_factory=_now_iso)
    resolved_at: str | None = None


class LogEntry(BaseModel):
    """Append-only. Never mutated. Code layer enforces immutability of the
    on-disk file (no rewrite of existing lines)."""

    model_config = ConfigDict(use_enum_values=True, frozen=True)

    log_id: int
    timestamp: str = Field(default_factory=_now_iso)
    actor: str
    action_type: LogActionType
    target: dict[str, str]  # {"type": "project|task|request|edge", "id": "..."}
    before_state: dict[str, Any] | None = None
    after_state: dict[str, Any] | None = None
    request_id: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)


class CallGraphEdge(BaseModel):
    from_: str = Field(alias="from")
    to: str
    timestamp: str = Field(default_factory=_now_iso)

    model_config = ConfigDict(populate_by_name=True)


class CallGraph(BaseModel):
    nodes: list[str] = Field(default_factory=list)
    edges: list[CallGraphEdge] = Field(default_factory=list)


class SimilarityScore(BaseModel):
    project_id: str
    tag_score: float = Field(ge=0.0, le=1.0)
    semantic_score: float = Field(ge=0.0, le=1.0)
    final_score: float = Field(ge=0.0, le=1.0)
    available: bool = True


# ---------------------------------------------------------------------------
# Link inference (2026-05-28)
# ---------------------------------------------------------------------------


class TaskEdge(BaseModel):
    """A typed relationship between two tasks. See SPEC.json
    link_inference_system.edge_types for the semantic definitions."""

    model_config = ConfigDict(use_enum_values=True)

    id: str = Field(default_factory=lambda: f"edge-{uuid4().hex[:12]}")
    from_task_id: str
    to_task_id: str
    type: EdgeType
    direction: EdgeDirection
    confidence: Literal[0.0, 0.5, 1.0]
    source: EdgeSource = EdgeSource.llm_jury
    jury_models: list[str] = Field(default_factory=list)
    swap_consistent: bool
    jury_agreement: bool
    created_at: str = Field(default_factory=_now_iso)
    last_verified_at: str = Field(default_factory=_now_iso)
    status: EdgeStatus = EdgeStatus.active

    @model_validator(mode="before")
    @classmethod
    def _fill_direction(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if data.get("direction") is None and data.get("type") is not None:
            try:
                t = EdgeType(data["type"]) if isinstance(data["type"], str) else data["type"]
                data["direction"] = _EDGE_DIRECTION[t].value
            except (KeyError, ValueError):
                pass  # let downstream validators surface the real error
        return data

    @model_validator(mode="after")
    def _check_invariants(self) -> "TaskEdge":
        if self.from_task_id == self.to_task_id:
            raise ValueError(
                f"TaskEdge.{self.id}: self-loop is not allowed "
                f"(from_task_id == to_task_id == {self.from_task_id!r})"
            )

        # Direction must match the canonical mapping for the type.
        canonical = _EDGE_DIRECTION[EdgeType(self.type)].value
        if self.direction != canonical:
            raise ValueError(
                f"TaskEdge.{self.id}: type={self.type!r} requires "
                f"direction={canonical!r}, got {self.direction!r}"
            )

        # Confidence must be consistent with the two boolean gates.
        expected = (1.0 if self.swap_consistent else 0.5) * (
            1.0 if self.jury_agreement else 0.5
        )
        # expected ∈ {0.25, 0.5, 1.0} — but we round 0.25 down to 0.0
        # (treated as "boolean AND fails" → confidence 0).
        if not self.swap_consistent and not self.jury_agreement:
            expected = 0.0
        elif not (self.swap_consistent and self.jury_agreement):
            expected = 0.5
        else:
            expected = 1.0
        if abs(self.confidence - expected) > 1e-9:
            raise ValueError(
                f"TaskEdge.{self.id}: confidence={self.confidence} inconsistent "
                f"with swap_consistent={self.swap_consistent}, "
                f"jury_agreement={self.jury_agreement} (expected {expected})"
            )

        return self


# ---------------------------------------------------------------------------
# GraphJSON — export format for lab_relationship_visualizer
# ---------------------------------------------------------------------------


class GraphNode(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    id: str
    type: NodeKind
    label: str
    attrs: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    from_: str = Field(alias="from")
    to: str
    type: str  # EdgeType + future lab_similarity etc.
    direction: EdgeDirection
    confidence: float = Field(ge=0.0, le=1.0)
    weight: float | None = None

    model_config = ConfigDict(populate_by_name=True, use_enum_values=True)


class GraphJSON(BaseModel):
    version: Literal["1.0"] = "1.0"
    exported_at: str = Field(default_factory=_now_iso)
    exporter: str = "gwangjang graph export"
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)


__all__ = [
    "Project",
    "Task",
    "Agent",
    "ChangeRequest",
    "LogEntry",
    "CallGraph",
    "CallGraphEdge",
    "SimilarityScore",
    "TaskEdge",
    "GraphJSON",
    "GraphNode",
    "GraphEdge",
    "ProjectStatus",
    "TaskStatus",
    "Priority",
    "AgentType",
    "ChangeRequestType",
    "ChangeRequestStatus",
    "LogActionType",
    "EdgeType",
    "EdgeDirection",
    "EdgeStatus",
    "EdgeSource",
    "NodeKind",
]
