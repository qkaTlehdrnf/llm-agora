"""GraphJSON export for lab_relationship_visualizer.

Produces DATA_MODELS.json GraphJSON: a node list (tasks/projects) + an edge
list (TaskEdges with type & confidence). The consumer
(lab_relationship_visualizer) decides on rendering.
"""

from __future__ import annotations

from typing import Literal

from ..models import (
    EdgeDirection,
    EdgeStatus,
    GraphEdge,
    GraphJSON,
    GraphNode,
    NodeKind,
)
from ..store import Store

ScopeT = Literal["tasks", "projects", "all"]


def export_graph(
    store: Store,
    scope: ScopeT = "tasks",
    include_inactive_edges: bool = False,
) -> GraphJSON:
    """Serialize the current store state into GraphJSON."""
    nodes: list[GraphNode] = []

    if scope in ("tasks", "all"):
        for t in store.list_tasks():
            ctx = t.context or {}
            attrs = {
                "status": t.status,
                "priority": t.priority,
                "project": t.project_id,
                "tags": ctx.get("tags", []),
            }
            if "desc" in ctx:
                attrs["desc"] = str(ctx["desc"])[:300]
            nodes.append(
                GraphNode(
                    id=t.id,
                    type=NodeKind.task,
                    label=t.action,
                    attrs=attrs,
                )
            )

    if scope in ("projects", "all"):
        for p in store.list_projects():
            nodes.append(
                GraphNode(
                    id=p.id,
                    type=NodeKind.project,
                    label=p.title,
                    attrs={
                        "domain": p.domain,
                        "status": p.status,
                        "tags": p.tags,
                        "task_count": len(p.tasks),
                    },
                )
            )

    edges: list[GraphEdge] = []
    for e in store.list_task_edges():
        if not include_inactive_edges and EdgeStatus(e.status) != EdgeStatus.active:
            continue
        edges.append(
            GraphEdge.model_validate(
                {
                    "from": e.from_task_id,
                    "to": e.to_task_id,
                    "type": e.type,
                    "direction": EdgeDirection(e.direction).value,
                    "confidence": e.confidence,
                    "weight": e.confidence,
                }
            )
        )

    return GraphJSON(nodes=nodes, edges=edges)


__all__ = ["export_graph"]
