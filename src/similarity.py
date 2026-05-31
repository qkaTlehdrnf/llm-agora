"""Worker 자기소개 → 프로젝트 추천 유사도 시스템.

SPEC.json similarity_system / core_flows.onboarding 의 구현:

  "Worker 자기소개 → 광장 similarity 계산 → 프로젝트 추천 → Worker 선택"

세 개의 직교 시그널을 결합한다 (link_inference 의 candidate_generation 과 동일한
설계 철학 — 단, 여기서는 task-task 가 아니라 agent-project 매칭):

  1. tag_score      — Jaccard(capabilities, project.tags)
  2. lexical_score  — TF-IDF char_wb cosine (혼합 한/영 강건)
  3. dense_score    — multilingual MiniLM cosine (설치 시에만; 없으면 skip)

lexical + dense 를 의미(semantic) 채널로 묶어 평균하고, tag 채널과 가중합한다.
출력은 models.SimilarityScore — DATA_MODELS.json 의 기존 스키마를 그대로 채운다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import Project, SimilarityScore

# 의사결정 가중치. tag 매칭은 강한 신호지만 sparse 하므로 의미 채널과 균형.
_TAG_WEIGHT = 0.4
_SEMANTIC_WEIGHT = 0.6


def _agent_text(self_description: str, capabilities: list[str]) -> str:
    parts = [self_description or ""]
    if capabilities:
        parts.append(" ".join(str(c) for c in capabilities))
    return "\n".join(p for p in parts if p.strip())


def _project_text(p: Project) -> str:
    parts = [p.title, p.goal, p.description_llm, p.domain]
    if p.tags:
        parts.append(" ".join(str(t) for t in p.tags))
    return "\n".join(str(x) for x in parts if x)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _lexical_scores(agent_text: str, project_texts: list[str]) -> np.ndarray:
    """agent_text vs 각 project_text 의 TF-IDF cosine. 반환 shape=(n_projects,)."""
    if not project_texts:
        return np.zeros(0)
    corpus = [agent_text, *project_texts]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    mat = vectorizer.fit_transform(corpus)
    # row 0 = agent, rows 1.. = projects
    sims = cosine_similarity(mat[0:1], mat[1:]).ravel()
    return sims


def _encode_texts(texts: list[str]) -> np.ndarray | None:
    """multilingual MiniLM 으로 인코딩. 미설치/실패 시 None.

    candidate_gen._dense_similarity 와 동일한 모델 선택 — BGE-M3 는 콜드스타트
    비용 때문에 기본 비활성. 정규화된 임베딩 반환.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    try:
        model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    except Exception:
        return None
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return emb / norms


def _dense_scores(
    agent_text: str, project_texts: list[str], use_dense: bool
) -> np.ndarray | None:
    if not use_dense or not project_texts:
        return None
    emb = _encode_texts([agent_text, *project_texts])
    if emb is None:
        return None
    agent_vec = emb[0:1]
    proj_vecs = emb[1:]
    return (agent_vec @ proj_vecs.T).ravel()


@dataclass(frozen=True)
class Recommendation:
    project_id: str
    title: str
    goal: str
    score: float  # final_score, 0..1
    tag_score: float
    semantic_score: float
    available: bool

    def to_dict(self) -> dict:
        return {
            "project_id": self.project_id,
            "title": self.title,
            "goal": self.goal,
            "score": round(self.score, 3),
            "tag_score": round(self.tag_score, 3),
            "semantic_score": round(self.semantic_score, 3),
            "available": self.available,
        }


def recommend_projects(
    self_description: str,
    capabilities: list[str],
    projects: list[Project],
    top_n: int = 5,
    threshold: float = 0.0,
    use_dense: bool = True,
) -> list[Recommendation]:
    """Worker 프로필에 가장 잘 맞는 프로젝트 top-N 추천.

    Args:
        self_description: Worker 자기소개 자유 텍스트.
        capabilities: Worker 역량 태그 리스트.
        projects: 후보 프로젝트 (보통 store.list_projects()).
        top_n: 반환 개수 상한.
        threshold: final_score 하한 (이하 제외). config similarity_threshold 연동.
        use_dense: dense 임베딩 채널 사용 여부.

    Returns: final_score 내림차순 Recommendation 리스트.
    """
    if not projects:
        return []

    agent_text = _agent_text(self_description, capabilities)
    cap_set = {str(c).lower() for c in capabilities}
    project_texts = [_project_text(p) for p in projects]

    lexical = _lexical_scores(agent_text, project_texts)
    dense = _dense_scores(agent_text, project_texts, use_dense)

    recs: list[Recommendation] = []
    for i, p in enumerate(projects):
        tag_set = {str(t).lower() for t in p.tags}
        tag_score = _jaccard(cap_set, tag_set)

        lex = float(lexical[i]) if lexical.size else 0.0
        sem_channels = [lex]
        if dense is not None:
            sem_channels.append(float(dense[i]))
        semantic_score = float(np.clip(np.mean(sem_channels), 0.0, 1.0))

        final = _TAG_WEIGHT * tag_score + _SEMANTIC_WEIGHT * semantic_score
        final = float(np.clip(final, 0.0, 1.0))

        # status 는 디스크 로드 시 문자열("active"), 기본값 생성 시 enum 둘 다
        # 가능하므로 .value 로 정규화해 비교한다.
        available = getattr(p.status, "value", p.status) == "active"
        recs.append(
            Recommendation(
                project_id=p.id,
                title=p.title,
                goal=p.goal,
                score=final,
                tag_score=tag_score,
                semantic_score=semantic_score,
                available=available,
            )
        )

    recs.sort(key=lambda r: -r.score)
    recs = [r for r in recs if r.score >= threshold]
    return recs[:top_n]


def to_similarity_scores(recs: list[Recommendation]) -> list[SimilarityScore]:
    """Recommendation → models.SimilarityScore (DATA_MODELS.json 스키마)."""
    return [
        SimilarityScore(
            project_id=r.project_id,
            tag_score=r.tag_score,
            semantic_score=r.semantic_score,
            final_score=r.score,
            available=r.available,
        )
        for r in recs
    ]


__all__ = [
    "Recommendation",
    "recommend_projects",
    "to_similarity_scores",
]
