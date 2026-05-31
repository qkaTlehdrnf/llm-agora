"""Thin JSON-file data store for 광장.

Wraps data/ JSON files behind typed methods. Pure I/O — no business logic.
Validation happens via Pydantic on load (the source of truth is on disk).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .models import (
    Agent,
    ChangeRequest,
    Project,
    Task,
    TaskEdge,
)


class Store:
    def __init__(self, data_dir: Path | str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ---- generic ----
    def _load(self, name: str) -> list[dict]:
        path = self.data_dir / name
        if not path.exists():
            return []
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return []
        return json.loads(text)

    def _save(self, name: str, items: Iterable[dict]) -> None:
        path = self.data_dir / name
        path.write_text(
            json.dumps(list(items), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ---- Project ----
    def list_projects(self) -> list[Project]:
        return [Project.model_validate(d) for d in self._load("projects.json")]

    def get_project(self, project_id: str) -> Project | None:
        for p in self.list_projects():
            if p.id == project_id:
                return p
        return None

    def save_project(self, project: Project) -> None:
        projects = {p.id: p for p in self.list_projects()}
        projects[project.id] = project
        self._save(
            "projects.json",
            [p.model_dump(mode="json") for p in projects.values()],
        )

    # ---- Task ----
    def list_tasks(self) -> list[Task]:
        return [Task.model_validate(d) for d in self._load("tasks.json")]

    def get_task(self, task_id: str) -> Task | None:
        for t in self.list_tasks():
            if t.id == task_id:
                return t
        return None

    def save_task(self, task: Task) -> None:
        tasks = {t.id: t for t in self.list_tasks()}
        tasks[task.id] = task
        self._save(
            "tasks.json",
            [t.model_dump(mode="json") for t in tasks.values()],
        )

    def delete_task(self, task_id: str) -> bool:
        """tasks.json 에서 영구 삭제. Returns: 실제로 제거됐는지.

        주의: SPEC 의 supersedes 처럼 '상태 보존' 이 필요한 경우 delete 대신
        save_task 로 status 만 바꾸는 편이 권장된다. delete 는 명시적 삭제용.
        """
        tasks = self.list_tasks()
        remaining = [t for t in tasks if t.id != task_id]
        if len(remaining) == len(tasks):
            return False
        self._save("tasks.json", [t.model_dump(mode="json") for t in remaining])
        return True

    # ---- Agent ----
    def list_agents(self) -> list[Agent]:
        return [Agent.model_validate(d) for d in self._load("agents.json")]

    def get_agent(self, agent_id: str) -> Agent | None:
        for a in self.list_agents():
            if a.id == agent_id:
                return a
        return None

    def save_agent(self, agent: Agent) -> None:
        agents = {a.id: a for a in self.list_agents()}
        agents[agent.id] = agent
        self._save(
            "agents.json",
            [a.model_dump(mode="json") for a in agents.values()],
        )

    # ---- ChangeRequest ----
    def list_change_requests(self) -> list[ChangeRequest]:
        return [
            ChangeRequest.model_validate(d)
            for d in self._load("change_requests.json")
        ]

    def get_change_request(self, request_id: str) -> ChangeRequest | None:
        for cr in self.list_change_requests():
            if cr.id == request_id:
                return cr
        return None

    def save_change_request(self, cr: ChangeRequest) -> None:
        all_crs = {c.id: c for c in self.list_change_requests()}
        all_crs[cr.id] = cr
        self._save(
            "change_requests.json",
            [c.model_dump(mode="json") for c in all_crs.values()],
        )

    # ---- TaskEdge ----
    def list_task_edges(self) -> list[TaskEdge]:
        return [TaskEdge.model_validate(d) for d in self._load("task_edges.json")]

    def save_task_edges(self, edges: list[TaskEdge]) -> None:
        self._save(
            "task_edges.json",
            [e.model_dump(mode="json") for e in edges],
        )

    def append_task_edge(self, edge: TaskEdge) -> None:
        edges = self.list_task_edges()
        edges.append(edge)
        self.save_task_edges(edges)
