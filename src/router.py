"""Call routing with cycle detection.

Implements SPEC.json core_flows.circular_detection + PROTOCOL.json
circular_detection_rules:
  - Definition: A→B→C→A or A→A pattern
  - Scope: single agent session
  - Detection: DFS on CallGraph, revisit a node already in the current path
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import CallGraph, CallGraphEdge


@dataclass(frozen=True)
class RouteDecision:
    allowed: bool
    reason: str | None = None
    summary_only: bool = False  # True when denied due to a cycle


class Router:
    """Per-agent CallGraph tracker.

    `request(agent_id, resource_id)` returns a RouteDecision. If a cycle would
    be created, the request is denied and the caller is expected to surface
    `summary_only=True` data (per SPEC.json on_cycle).

    The cycle definition: looking at the call edges already in the agent's
    graph, walking backwards from `resource_id` via predecessors reaches
    `agent_id`. That means agent → resource → ... → agent (= cycle).
    """

    def __init__(self) -> None:
        self._sessions: dict[str, CallGraph] = {}

    def _graph_for(self, agent_id: str) -> CallGraph:
        return self._sessions.setdefault(agent_id, CallGraph())

    def graph_for(self, agent_id: str) -> CallGraph:
        """Read-only accessor (returns the underlying CallGraph)."""
        return self._graph_for(agent_id)

    def request(self, agent_id: str, resource_id: str) -> RouteDecision:
        g = self._graph_for(agent_id)

        # Reverse adjacency: for each edge from→to, record to→from in `callers`.
        callers: dict[str, list[str]] = {}
        for e in g.edges:
            callers.setdefault(e.to, []).append(e.from_)

        # Walk up from resource_id; if we hit agent_id, that's a cycle.
        stack = [resource_id]
        seen: set[str] = set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            if cur == agent_id:
                return RouteDecision(
                    allowed=False,
                    reason=(
                        f"circular_dependency: requesting {resource_id!r} from "
                        f"{agent_id!r} would close a call cycle"
                    ),
                    summary_only=True,
                )
            stack.extend(callers.get(cur, []))

        # No cycle — record the edge.
        if resource_id not in g.nodes:
            g.nodes.append(resource_id)
        if agent_id not in g.nodes:
            g.nodes.append(agent_id)
        g.edges.append(
            CallGraphEdge.model_validate({"from": agent_id, "to": resource_id})
        )
        return RouteDecision(allowed=True)

    def reset_session(self, agent_id: str) -> None:
        """Clear the CallGraph for `agent_id`. Used at session end."""
        self._sessions.pop(agent_id, None)
