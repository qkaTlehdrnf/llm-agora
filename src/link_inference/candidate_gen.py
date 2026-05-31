"""Candidate-pair generation for task linking.

Implements SPEC.json link_inference_system.pipeline.stage=candidate_generation:
three orthogonal signals fused via Reciprocal Rank Fusion (RRF):

  1. TF-IDF on task action+description
  2. tag Jaccard (lab_weighted_tagging / one_step_tagger output reused once
     task-level weighted_tags exist; falls back to plain tag overlap until
     then)
  3. dense embedding cosine (BGE-M3 if installed, else MiniLM, else skipped)

The output is a list of (task_a_id, task_b_id, fused_score) sorted descending.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..models import Task


@dataclass(frozen=True)
class CandidatePair:
    task_a_id: str
    task_b_id: str
    score: float
    channel_ranks: dict[str, int]  # per-channel rank for debugging


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------


def _task_text(t: Task) -> str:
    """Concatenated text view of a Task. action + desc + tags."""
    parts = [t.action]
    ctx = t.context or {}
    if desc := ctx.get("desc"):
        parts.append(str(desc))
    if tags := ctx.get("tags"):
        parts.append(" ".join(str(x) for x in tags))
    return "\n".join(parts)


def _task_tags(t: Task) -> set[str]:
    ctx = t.context or {}
    return {str(x).lower() for x in ctx.get("tags", [])}


def _tfidf_similarity(tasks: list[Task]) -> np.ndarray:
    """Return an n×n similarity matrix from TF-IDF cosine."""
    if not tasks:
        return np.zeros((0, 0))
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",  # robust to mixed Korean/English
        ngram_range=(2, 4),
        min_df=1,
    )
    texts = [_task_text(t) for t in tasks]
    mat = vectorizer.fit_transform(texts)
    return cosine_similarity(mat)


def _jaccard_similarity(tasks: list[Task]) -> np.ndarray:
    """Return an n×n similarity matrix from plain tag-set Jaccard."""
    n = len(tasks)
    sims = np.zeros((n, n))
    tag_sets = [_task_tags(t) for t in tasks]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = tag_sets[i], tag_sets[j]
            if not a and not b:
                sim = 0.0
            else:
                sim = len(a & b) / len(a | b)
            sims[i, j] = sim
            sims[j, i] = sim
    return sims


def _dense_similarity(tasks: list[Task]) -> np.ndarray | None:
    """Optional dense embedding channel. Returns None if no embedder available.

    Tries sentence-transformers with a small multilingual model. We don't
    download BGE-M3 by default — that's a 2GB+ artifact and would dominate
    cold-start latency in P0 testing.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None

    # Smallest credible multilingual model. Korean & English covered.
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    try:
        model = SentenceTransformer(model_name)
    except Exception:
        return None
    texts = [_task_text(t) for t in tasks]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    # L2-normalize then dot.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms
    return embeddings @ embeddings.T


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def _ranks_from_similarity(sim: np.ndarray) -> np.ndarray:
    """For each row i, rank the off-diagonal columns by similarity desc.

    Returns rank_matrix[i, j] = 1-indexed rank of j in i's sorted neighbors,
    or -1 if i == j. Lower rank = more similar.
    """
    n = sim.shape[0]
    ranks = np.full((n, n), -1, dtype=np.int32)
    for i in range(n):
        # mask self
        scores = sim[i].copy()
        scores[i] = -np.inf
        order = np.argsort(-scores)  # descending
        for rank, j in enumerate(order, start=1):
            if scores[j] == -np.inf:
                continue
            ranks[i, j] = rank
    return ranks


def reciprocal_rank_fusion(
    rank_matrices: dict[str, np.ndarray],
    k: int = 60,
) -> tuple[np.ndarray, dict[tuple[int, int], dict[str, int]]]:
    """RRF: score(i,j) = sum over channels of 1/(k + rank_channel(i,j)).

    Returns (fused_score_matrix, per_pair_channel_ranks).
    """
    if not rank_matrices:
        raise ValueError("at least one channel required")
    n = next(iter(rank_matrices.values())).shape[0]
    fused = np.zeros((n, n))
    per_pair_ranks: dict[tuple[int, int], dict[str, int]] = {}
    for ch_name, ranks in rank_matrices.items():
        for i in range(n):
            for j in range(n):
                if i == j or ranks[i, j] < 1:
                    continue
                fused[i, j] += 1.0 / (k + ranks[i, j])
                per_pair_ranks.setdefault((i, j), {})[ch_name] = int(ranks[i, j])
    return fused, per_pair_ranks


# ---------------------------------------------------------------------------
# Top-K candidate selection (per task + global)
# ---------------------------------------------------------------------------


def candidates_for_all(
    tasks: list[Task],
    top_k_per_task: int = 10,
    use_dense: bool = True,
) -> list[CandidatePair]:
    """Per-task top-K neighbors via RRF over TF-IDF + Jaccard (+ dense).

    Returns each (a, b) pair only once (a.id < b.id) to dedupe symmetric
    matches.
    """
    if len(tasks) < 2:
        return []

    channels: dict[str, np.ndarray] = {
        "tfidf": _ranks_from_similarity(_tfidf_similarity(tasks)),
        "jaccard": _ranks_from_similarity(_jaccard_similarity(tasks)),
    }
    if use_dense:
        dense = _dense_similarity(tasks)
        if dense is not None:
            channels["dense"] = _ranks_from_similarity(dense)

    fused, per_pair_ranks = reciprocal_rank_fusion(channels)

    seen: set[tuple[str, str]] = set()
    out: list[CandidatePair] = []
    for i, t_i in enumerate(tasks):
        # Pick top-K neighbors for task i.
        scores_i = fused[i].copy()
        scores_i[i] = -np.inf
        order = np.argsort(-scores_i)
        kept = 0
        for j in order:
            if kept >= top_k_per_task:
                break
            if scores_i[j] <= 0:
                break
            t_j = tasks[j]
            a, b = sorted([t_i.id, t_j.id])
            if (a, b) in seen:
                kept += 1
                continue
            seen.add((a, b))
            out.append(
                CandidatePair(
                    task_a_id=a,
                    task_b_id=b,
                    score=float(scores_i[j]),
                    channel_ranks=per_pair_ranks.get((i, int(j)), {}),
                )
            )
            kept += 1

    out.sort(key=lambda c: -c.score)
    return out


__all__ = [
    "CandidatePair",
    "candidates_for_all",
    "reciprocal_rank_fusion",
]
