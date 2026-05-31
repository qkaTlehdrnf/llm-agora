"""광장 루트 탐지 · 설정 · 데이터 경로 유틸리티.

DEPLOYMENT.json root_detection / config_file_schema 의 런타임 구현.

광장 루트는 git 의 `.git` 과 동일한 역할을 하는 `.gwangjang/` 디렉터리로 표시된다.
현재 디렉터리에서 상위로 거슬러 올라가며 탐색한다 (git 방식).

데이터 경로는 두 가지 모드를 지원한다:
  1. 설치형(deployment): `.gwangjang/data/` — `gwangjang init` 로 생성.
  2. 레거시(monorepo): `data/` — 기존 `python -m ...call`.

Store 는 두 경우 모두 동일한 flat JSON 레이아웃(projects.json / tasks.json /
agents.json / change_requests.json / task_edges.json / log.jsonl)을 사용하므로,
CLI 는 단지 올바른 data_dir 만 골라주면 된다.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# 디렉터리 마커. .git 과 동일한 역할.
MARKER_DIR = ".gwangjang"
CONFIG_NAME = "config.json"
DATA_SUBDIR = "data"

# Store 가 기대하는 데이터 파일들 (flat 레이아웃).
DATA_FILES_EMPTY_LIST = (
    "projects.json",
    "tasks.json",
    "agents.json",
    "change_requests.json",
    "task_edges.json",
)
LOG_FILE = "log.jsonl"

# 현재 광장 버전. DEPLOYMENT.json installation.package_name 과 동기.
GWANGJANG_VERSION = "0.1.0"

DEFAULT_EXCLUDED_PATHS = [
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "archived",
    ".git",
    MARKER_DIR,
]


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Root detection
# ---------------------------------------------------------------------------


def find_gwangjang_root(start: Path | str | None = None) -> Path | None:
    """현재(또는 지정) 디렉터리에서 상위로 올라가며 `.gwangjang/` 탐색.

    Returns: 마커를 포함하는 디렉터리(루트)의 절대 경로, 없으면 None.
    git 이 `.git` 을 찾는 방식과 동일 — 파일시스템 루트까지 거슬러 올라간다.
    """
    cur = Path(start).resolve() if start is not None else Path.cwd().resolve()
    # start 가 파일이면 부모부터.
    if cur.is_file():
        cur = cur.parent
    for candidate in [cur, *cur.parents]:
        if (candidate / MARKER_DIR).is_dir():
            return candidate
    return None


def marker_dir(root: Path | str) -> Path:
    return Path(root) / MARKER_DIR


def data_dir_for_root(root: Path | str) -> Path:
    """설치형 루트의 데이터 디렉터리 (`<root>/.gwangjang/data/`)."""
    return marker_dir(root) / DATA_SUBDIR


def config_path_for_root(root: Path | str) -> Path:
    return marker_dir(root) / CONFIG_NAME


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def default_config(root: Path | str) -> dict[str, Any]:
    return {
        "version": GWANGJANG_VERSION,
        "created_at": iso_now(),
        "root_path": str(Path(root).resolve()),
        "llm_backend": "mock",  # mock | vllm | gemini | claude
        "similarity_threshold": 0.3,
        "excluded_paths": list(DEFAULT_EXCLUDED_PATHS),
        "data_path": f"{MARKER_DIR}/{DATA_SUBDIR}",
    }


def load_config(root: Path | str) -> dict[str, Any]:
    p = config_path_for_root(root)
    if not p.exists():
        return default_config(root)
    return json.loads(p.read_text(encoding="utf-8"))


def save_config(root: Path | str, config: dict[str, Any]) -> None:
    p = config_path_for_root(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Init scaffolding
# ---------------------------------------------------------------------------


def init_root(path: Path | str) -> tuple[Path, bool]:
    """`<path>/.gwangjang/` 디렉터리 구조를 생성.

    Returns: (root_path, created). `created=False` 면 이미 존재해 손대지 않음.
    DEPLOYMENT.json cli_subcommands.init.behavior_if_exists 준수.
    """
    root = Path(path).resolve()
    mk = marker_dir(root)
    if mk.exists():
        return (root, False)

    data = data_dir_for_root(root)
    data.mkdir(parents=True, exist_ok=True)

    # 빈 데이터 파일들 시드 (Store 와 호환되는 flat 레이아웃).
    for fname in DATA_FILES_EMPTY_LIST:
        fp = data / fname
        if not fp.exists():
            fp.write_text("[]", encoding="utf-8")
    log_fp = data / LOG_FILE
    if not log_fp.exists():
        log_fp.write_text("", encoding="utf-8")

    save_config(root, default_config(root))
    return (root, True)


__all__ = [
    "MARKER_DIR",
    "GWANGJANG_VERSION",
    "DEFAULT_EXCLUDED_PATHS",
    "iso_now",
    "find_gwangjang_root",
    "marker_dir",
    "data_dir_for_root",
    "config_path_for_root",
    "default_config",
    "load_config",
    "save_config",
    "init_root",
]
