"""Append-only log persistence.

Implements SPEC.json log_system.append_only + DATA_MODELS.json
LogEntry.immutability_rule: the on-disk JSONL file is only ever opened in
append mode. log_id is assigned by the manager (monotonically increasing),
never accepted from callers — this prevents an actor from injecting a
spoofed-low id to slip an entry past a "latest-only" reader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .models import LogActionType, LogEntry


class LogManager:
    """JSONL-backed append-only log.

    Concurrency note: this is a single-process implementation. For multi-
    writer scenarios, wrap `append()` with a file lock (e.g. fcntl.flock).
    P0 광장 runs single-process so we keep it minimal.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()
        self._next_id: int | None = None  # lazy

    def _compute_next_id(self) -> int:
        last = 0
        for entry in self.iter_entries():
            if entry.log_id > last:
                last = entry.log_id
        return last + 1

    def append(
        self,
        actor: str,
        action_type: LogActionType | str,
        target: dict[str, str],
        before_state: dict | None = None,
        after_state: dict | None = None,
        request_id: str | None = None,
        context: dict | None = None,
    ) -> LogEntry:
        if self._next_id is None:
            self._next_id = self._compute_next_id()
        entry = LogEntry(
            log_id=self._next_id,
            actor=actor,
            action_type=(
                LogActionType(action_type)
                if isinstance(action_type, str)
                else action_type
            ),
            target=target,
            before_state=before_state,
            after_state=after_state,
            request_id=request_id,
            context=context or {},
        )
        self._next_id += 1
        with self.path.open("a", encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")
        return entry

    def iter_entries(self) -> Iterator[LogEntry]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield LogEntry.model_validate_json(line)

    def all(self) -> list[LogEntry]:
        return list(self.iter_entries())

    def filter(
        self,
        actor: str | None = None,
        action_type: LogActionType | str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
    ) -> list[LogEntry]:
        action_value = (
            action_type.value
            if isinstance(action_type, LogActionType)
            else action_type
        )
        out = []
        for e in self.iter_entries():
            if actor is not None and e.actor != actor:
                continue
            if action_value is not None and e.action_type != action_value:
                continue
            if target_type is not None and e.target.get("type") != target_type:
                continue
            if target_id is not None and e.target.get("id") != target_id:
                continue
            out.append(e)
        return out
