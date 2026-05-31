"""End-to-end orchestrator: candidates → jury → persist.

Used by both call.py (`request_link_inference` method) and CLI runs from
scripts/. Side effects: writes TaskEdge rows into the Store, appends
LogEntries via LogManager.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..log_manager import LogManager
from ..models import (
    EdgeSource,
    EdgeStatus,
    EdgeType,
    LogActionType,
    Task,
    TaskEdge,
)
from ..store import Store
from ..validator import ValidationError, validate_task_edge_addition
from .candidate_gen import CandidatePair, candidates_for_all
from .judge import Jury, JuryVerdict, default_jury


@dataclass(frozen=True)
class InferenceReport:
    candidates_evaluated: int
    edges_emitted: int
    edges_skipped_invalid: int
    edges_unknown_type: int
    per_type: dict[str, int]
    duration_ms: int


def _edge_from_verdict(
    pair: CandidatePair,
    verdict: JuryVerdict,
    jury_models: list[str],
) -> TaskEdge | None:
    """Build a TaskEdge from a verdict. Returns None if no edge should emit."""
    if verdict.type_ is None or verdict.confidence == 0.0:
        return None
    # Directed types use the candidate order as given (we sorted by id so this
    # is alphabetical, not semantic). For a real direction call we'd need the
    # judge to tell us — for P0 we keep direction-from-type and accept that
    # the source/target may need flipping later when judges are richer.
    return TaskEdge(
        from_task_id=pair.task_a_id,
        to_task_id=pair.task_b_id,
        type=EdgeType(verdict.type_),
        confidence=verdict.confidence,  # validator fills direction from type
        source=EdgeSource.llm_jury,
        jury_models=jury_models,
        swap_consistent=verdict.swap_consistent,
        jury_agreement=verdict.jury_agreement,
        status=EdgeStatus.active,
    )


def run_inference(
    store: Store,
    log: LogManager,
    jury: Jury | None = None,
    top_k_per_task: int = 10,
    use_dense: bool = True,
    only_for_task_id: str | None = None,
) -> InferenceReport:
    """Run the full pipeline.

    Args:
      only_for_task_id: if given, only emit edges touching that task. Used by
        `request_link_inference` for incremental updates.
    """
    import time

    jury = jury or default_jury()
    tasks: list[Task] = store.list_tasks()
    existing = store.list_task_edges()

    t0 = time.monotonic()
    candidates = candidates_for_all(
        tasks, top_k_per_task=top_k_per_task, use_dense=use_dense
    )

    if only_for_task_id is not None:
        candidates = [
            c
            for c in candidates
            if only_for_task_id in (c.task_a_id, c.task_b_id)
        ]

    by_id: dict[str, Task] = {t.id: t for t in tasks}
    per_type: dict[str, int] = {}
    emitted = 0
    skipped_invalid = 0
    unknown_type = 0

    for pair in candidates:
        a = by_id.get(pair.task_a_id)
        b = by_id.get(pair.task_b_id)
        if a is None or b is None:
            continue
        verdict = jury.vote(a, b)
        edge = _edge_from_verdict(pair, verdict, jury.model_names)
        if edge is None:
            unknown_type += 1
            continue
        try:
            validate_task_edge_addition(edge, existing)
        except ValidationError as e:
            skipped_invalid += 1
            log.append(
                actor="link_inference",
                action_type=LogActionType.edge_rejected,
                target={"type": "edge", "id": edge.id},
                context={"reason": str(e), "from": edge.from_task_id, "to": edge.to_task_id},
            )
            continue
        existing.append(edge)
        emitted += 1
        per_type[edge.type] = per_type.get(edge.type, 0) + 1
        log.append(
            actor="link_inference",
            action_type=LogActionType.edge_inferred,
            target={"type": "edge", "id": edge.id},
            after_state={
                "from": edge.from_task_id,
                "to": edge.to_task_id,
                "type": edge.type,
                "confidence": edge.confidence,
            },
            context={"channel_ranks": pair.channel_ranks, "rationale": verdict.rationale},
        )

    store.save_task_edges(existing)
    duration_ms = int((time.monotonic() - t0) * 1000)
    return InferenceReport(
        candidates_evaluated=len(candidates),
        edges_emitted=emitted,
        edges_skipped_invalid=skipped_invalid,
        edges_unknown_type=unknown_type,
        per_type=per_type,
        duration_ms=duration_ms,
    )


__all__ = ["InferenceReport", "run_inference"]
