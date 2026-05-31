"""Rule enforcement for ChangeRequest and TaskEdge.

Mirrors SPEC.json code_layer.enforces:
  - 요청 형식 검증 (malformed request → 즉시 거절)
  - 서류 완비 여부 확인 (필요 서류 미제출 시 승인 불가)
  - blocks DAG invariant (link_inference_system 의 hard guarantee)
"""

from __future__ import annotations

from typing import Iterable

from .models import (
    ChangeRequest,
    ChangeRequestType,
    EdgeStatus,
    EdgeType,
    TaskEdge,
)


class ValidationError(Exception):
    """Raised when a request or edge violates a code-layer invariant."""


# Per PROTOCOL.json document_requirements_policy.minimum_always_required.
REQUIRED_DOCS_BY_REQUEST_TYPE: dict[ChangeRequestType, list[str]] = {
    ChangeRequestType.create_task: ["task_justification"],
    ChangeRequestType.delete_task: ["task_justification", "impact_assessment"],
    ChangeRequestType.complete_project: ["completion_evidence"],
}


# ---------------------------------------------------------------------------
# ChangeRequest
# ---------------------------------------------------------------------------


def validate_change_request_shape(cr: ChangeRequest) -> None:
    if not cr.rationale.strip():
        raise ValidationError(f"ChangeRequest.{cr.id}: rationale must be non-empty")
    if not cr.target_id.strip():
        raise ValidationError(f"ChangeRequest.{cr.id}: target_id must be non-empty")


def required_docs_for(cr: ChangeRequest) -> list[str]:
    return REQUIRED_DOCS_BY_REQUEST_TYPE.get(
        ChangeRequestType(cr.request_type), []
    )


def all_required_docs_submitted(cr: ChangeRequest) -> tuple[bool, list[str]]:
    """Return (ok, missing_doc_types). `ok=True` means the request can move
    from `awaiting_docs` → `under_review`."""
    required = required_docs_for(cr)
    missing = [d for d in required if d not in cr.submitted_docs]
    return (not missing, missing)


# ---------------------------------------------------------------------------
# TaskEdge — duplicate detection + blocks-DAG invariant
# ---------------------------------------------------------------------------


def _active_blocks_edges(edges: Iterable[TaskEdge]) -> list[TaskEdge]:
    out = []
    for e in edges:
        if EdgeType(e.type) != EdgeType.blocks:
            continue
        if EdgeStatus(e.status) != EdgeStatus.active:
            continue
        out.append(e)
    return out


def detect_blocks_cycle(edges: Iterable[TaskEdge]) -> list[str] | None:
    """DFS cycle detection on the subgraph of active `blocks` edges.

    Returns the cycle path (node ids in order) if one exists, else None.
    """
    blocks = _active_blocks_edges(edges)
    graph: dict[str, list[str]] = {}
    nodes: set[str] = set()
    for e in blocks:
        graph.setdefault(e.from_task_id, []).append(e.to_task_id)
        nodes.add(e.from_task_id)
        nodes.add(e.to_task_id)

    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {n: WHITE for n in nodes}
    parent: dict[str, str | None] = {n: None for n in nodes}

    def dfs(start: str) -> list[str] | None:
        stack: list[tuple[str, int]] = [(start, 0)]
        color[start] = GRAY
        while stack:
            node, idx = stack[-1]
            neighbors = graph.get(node, [])
            if idx < len(neighbors):
                stack[-1] = (node, idx + 1)
                nbr = neighbors[idx]
                nbr_color = color.get(nbr, WHITE)
                if nbr_color == GRAY:
                    # cycle: walk from node up via parent[] until we hit nbr
                    path = [nbr, node]
                    cur = parent.get(node)
                    while cur is not None and cur != nbr:
                        path.append(cur)
                        cur = parent.get(cur)
                    path.reverse()
                    path.append(nbr)  # close the cycle visually
                    return path
                if nbr_color == WHITE:
                    color[nbr] = GRAY
                    parent[nbr] = node
                    stack.append((nbr, 0))
            else:
                color[node] = BLACK
                stack.pop()
        return None

    for n in nodes:
        if color[n] == WHITE:
            cycle = dfs(n)
            if cycle:
                return cycle
    return None


def validate_task_edge_addition(
    new_edge: TaskEdge,
    existing_edges: Iterable[TaskEdge],
) -> None:
    """Reject if (from, to, type) duplicates an active edge, or if adding
    `new_edge` would create a cycle in the blocks subgraph."""
    existing = list(existing_edges)

    for e in existing:
        if EdgeStatus(e.status) != EdgeStatus.active:
            continue
        if (
            e.from_task_id == new_edge.from_task_id
            and e.to_task_id == new_edge.to_task_id
            and EdgeType(e.type) == EdgeType(new_edge.type)
        ):
            raise ValidationError(
                f"TaskEdge ({new_edge.from_task_id}, {new_edge.to_task_id}, "
                f"{new_edge.type}) already exists and is active"
            )

    if EdgeType(new_edge.type) == EdgeType.blocks:
        cycle = detect_blocks_cycle([*existing, new_edge])
        if cycle is not None:
            raise ValidationError(
                f"Adding blocks edge would create cycle: {' -> '.join(cycle)}"
            )
