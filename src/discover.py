"""폴더 구조 자동 디스커버리 — 프로젝트/태스크 휴리스틱 스캔.

DEPLOYMENT.json discovery_logic 의 휴리스틱 단계 구현. "최소 LLM 호출 — 휴리스틱
으로 90% 처리, 애매한 부분만 LLM" 철학을 따른다. LLM 정제(--llm)는 후속 단계의
훅으로 남겨둔다.

출력은 후보(candidate) 리스트일 뿐 — 실제 등록은 CLI 의 사용자 확인(add/skip/edit)
또는 --yes 자동 승인 뒤에 이뤄진다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .utils import DEFAULT_EXCLUDED_PATHS

# 프로젝트 후보를 식별하는 매니페스트/마커 파일.
_PROJECT_MANIFESTS = (
    "pyproject.toml",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "setup.py",
)
_PROJECT_DOC_MARKERS = ("README.md", "CLAUDE.md")

# 태스크를 담고 있을 법한 파일.
_TASK_FILES = ("tasks.json", "TODO.md", "ROADMAP.md")


@dataclass
class ProjectCandidate:
    id: str
    path: str
    indicators: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"id": self.id, "path": self.path, "indicators": self.indicators}


@dataclass
class TaskCandidate:
    source_file: str
    kind: str  # "tasks_json" | "todo_md" | "roadmap_md"
    count_hint: int | None = None

    def to_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "kind": self.kind,
            "count_hint": self.count_hint,
        }


@dataclass
class DiscoveryResult:
    projects: list[ProjectCandidate] = field(default_factory=list)
    tasks: list[TaskCandidate] = field(default_factory=list)
    existing_gwangjang_data: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "projects": [p.to_dict() for p in self.projects],
            "tasks": [t.to_dict() for t in self.tasks],
            "existing_gwangjang_data": self.existing_gwangjang_data,
        }


def _is_excluded(name: str, excluded: set[str]) -> bool:
    return name in excluded or name.startswith(".")


def _scan_for_projects(
    root: Path, excluded: set[str], max_depth: int = 3
) -> list[ProjectCandidate]:
    """projects/{name}/ · src/{name}/ · 매니페스트 보유 폴더를 후보로."""
    out: list[ProjectCandidate] = []
    seen: set[str] = set()

    def consider(d: Path) -> None:
        if not d.is_dir():
            return
        indicators: list[str] = []
        for mani in _PROJECT_MANIFESTS:
            if (d / mani).exists():
                indicators.append(mani)
        for doc in _PROJECT_DOC_MARKERS:
            if (d / doc).exists():
                indicators.append(doc)
        if (d / ".git").exists():
            indicators.append(".git submodule")
        if not indicators:
            return
        rel = str(d.relative_to(root))
        if rel in seen:
            return
        seen.add(rel)
        out.append(ProjectCandidate(id=d.name, path=rel, indicators=indicators))

    # 1) projects/ · src/ 직속 하위 (현재 레포 패턴)
    for container in ("projects", "src"):
        cdir = root / container
        if cdir.is_dir():
            for child in sorted(cdir.iterdir()):
                if child.is_dir() and not _is_excluded(child.name, excluded):
                    consider(child)

    # 2) 루트 기준 얕은 BFS — 매니페스트 보유 폴더.
    def walk(d: Path, depth: int) -> None:
        if depth > max_depth:
            return
        for child in sorted(d.iterdir()):
            if not child.is_dir():
                continue
            if _is_excluded(child.name, excluded):
                continue
            consider(child)
            walk(child, depth + 1)

    walk(root, 1)
    return out


def _count_tasks_json(fp: Path) -> int | None:
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if isinstance(data, dict) and "tasks" in data:
        data = data["tasks"]
    if isinstance(data, list):
        return len(data)
    return None


def _scan_for_tasks(
    root: Path, excluded: set[str], max_depth: int = 3
) -> list[TaskCandidate]:
    out: list[TaskCandidate] = []

    def walk(d: Path, depth: int) -> None:
        if depth > max_depth:
            return
        for child in sorted(d.iterdir()):
            if child.is_dir():
                if _is_excluded(child.name, excluded):
                    continue
                walk(child, depth + 1)
                continue
            if child.name == "tasks.json":
                out.append(
                    TaskCandidate(
                        source_file=str(child.relative_to(root)),
                        kind="tasks_json",
                        count_hint=_count_tasks_json(child),
                    )
                )
            elif child.name == "TODO.md":
                out.append(
                    TaskCandidate(
                        source_file=str(child.relative_to(root)), kind="todo_md"
                    )
                )
            elif child.name == "ROADMAP.md":
                out.append(
                    TaskCandidate(
                        source_file=str(child.relative_to(root)), kind="roadmap_md"
                    )
                )

    walk(root, 1)
    return out


def discover(root: Path | str, excluded: list[str] | None = None) -> DiscoveryResult:
    """루트 폴더 구조를 휴리스틱 스캔 → 프로젝트/태스크 후보 산출."""
    root = Path(root).resolve()
    ex = set(excluded if excluded is not None else DEFAULT_EXCLUDED_PATHS)

    result = DiscoveryResult()
    result.projects = _scan_for_projects(root, ex)
    result.tasks = _scan_for_tasks(root, ex)

    # 기존 광장-호환 데이터 발견 (DEPLOYMENT.json interactions_with_existing_system).
    for legacy in (
        root / "docs" / "tasks.json",
        root / "docs" / "agents" / "MANIFEST.json",
    ):
        if legacy.exists():
            result.existing_gwangjang_data.append(str(legacy.relative_to(root)))

    return result


__all__ = [
    "ProjectCandidate",
    "TaskCandidate",
    "DiscoveryResult",
    "discover",
]
