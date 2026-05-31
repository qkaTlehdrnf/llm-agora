"""Need-to-know information partitioning for Worker Agents.

Implements PROTOCOL.json information_partitioning_rules:
  - Worker는 배정된 프로젝트의 컨텍스트만 받음
  - Worker는 현재 태스크의 직접 dependency만 볼 수 있음
  - 다른 Agent의 세션 컨텍스트는 절대 공유 안 됨
  - 광장만이 전체 프로젝트/태스크 목록을 가짐
"""

from __future__ import annotations

from typing import Any

from .models import Agent, AgentType, Project, Task


# Fields a Worker may see on a Project they are NOT assigned to.
_PROJECT_PUBLIC_FIELDS = {"id", "title", "goal", "domain"}

# Fields nobody but 광장 sees, even on the assigned project.
_PROJECT_HIDDEN_FROM_WORKER = {"assigned_agents"}

# Fields hidden from non-self Workers on Tasks (kept minimal).
_TASK_HIDDEN_FROM_WORKER = {"assigned_agent"}


def _is_gwangjang(agent: Agent) -> bool:
    return AgentType(agent.type) == AgentType.gwangjang


def filter_project_for_agent(project: Project, agent: Agent) -> dict[str, Any]:
    """Return a dict view of `project` filtered to what `agent` may see."""
    full = project.model_dump(mode="json")
    if _is_gwangjang(agent):
        return full
    if agent.current_project_id == project.id:
        return {k: v for k, v in full.items() if k not in _PROJECT_HIDDEN_FROM_WORKER}
    return {k: v for k, v in full.items() if k in _PROJECT_PUBLIC_FIELDS}


def filter_task_for_agent(task: Task, agent: Agent) -> dict[str, Any] | None:
    """Return a dict view of `task` if visible to `agent`, else None.

    Visibility rule:
      - 광장: sees everything.
      - Worker: sees tasks in their currently assigned project only.
        Within that project, hides cross-agent fields.
    """
    full = task.model_dump(mode="json")
    if _is_gwangjang(agent):
        return full
    if agent.current_project_id != task.project_id:
        return None
    return {k: v for k, v in full.items() if k not in _TASK_HIDDEN_FROM_WORKER}


def filter_task_list_for_agent(
    tasks: list[Task], agent: Agent
) -> list[dict[str, Any]]:
    """Apply `filter_task_for_agent` and drop hidden tasks."""
    out: list[dict[str, Any]] = []
    for t in tasks:
        view = filter_task_for_agent(t, agent)
        if view is not None:
            out.append(view)
    return out
