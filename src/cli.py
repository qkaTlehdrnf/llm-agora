"""`gwangjang` 콘솔 진입점 — 설치형 CLI.

DEPLOYMENT.json cli_subcommands 의 MVP 구현:
  gwangjang init [path]      — 현재/지정 폴더에 .gwangjang/ 초기화
  gwangjang discover [--llm] — 폴더 구조 스캔 → 프로젝트/태스크 후보
  gwangjang status [--json]  — 광장 루트 · 등록 카운트 · 최근 로그
  gwangjang root             — 현재 광장 루트 출력 (없으면 exit 1)
  gwangjang call '<JSON>'    — Agent 호출 (src/call.py 위임)
  gwangjang start [--daemon] — 서비스 시작 (현재는 health check)

두 가지 실행 경로를 모두 지원:
  - 설치형:   `gwangjang ...`              (pyproject console_scripts)
  - monorepo: `python -m llm_agora.src.cli ...`
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import discover as discover_mod
from .call import GwangjangCLI
from .store import Store
from .utils import (
    GWANGJANG_VERSION,
    data_dir_for_root,
    find_gwangjang_root,
    init_root,
    load_config,
)


# ---------------------------------------------------------------------------
# data_dir 해석
# ---------------------------------------------------------------------------

# 레거시 monorepo 기본 데이터 경로.
_LEGACY_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _resolve_data_dir(explicit: str | None = None) -> Path:
    """사용할 data 디렉터리 결정.

    우선순위: --data-dir 명시 > .gwangjang/ 루트 탐지 > 레거시 monorepo data/.
    """
    if explicit:
        return Path(explicit)
    root = find_gwangjang_root()
    if root is not None:
        return data_dir_for_root(root)
    return _LEGACY_DATA_DIR


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def _cmd_init(args: argparse.Namespace) -> int:
    target = Path(args.path).resolve() if args.path else Path.cwd()
    root, created = init_root(target)
    if not created:
        print(
            f"⚠️  이미 광장이 초기화되어 있습니다: {root}/.gwangjang/ "
            "(중단)",
            file=sys.stderr,
        )
        return 1
    print(f"✅ 광장 초기화 완료: {root}/.gwangjang/")
    print(f"   데이터: {data_dir_for_root(root)}")
    print("   다음: `gwangjang discover` 로 프로젝트/태스크 자동 식별")
    return 0


def _cmd_discover(args: argparse.Namespace) -> int:
    root = find_gwangjang_root()
    if root is None:
        print(
            "❌ 광장 루트(.gwangjang/)를 찾을 수 없습니다. 먼저 `gwangjang init`.",
            file=sys.stderr,
        )
        return 1
    config = load_config(root)
    result = discover_mod.discover(root, excluded=config.get("excluded_paths"))

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0

    print(f"🔍 디스커버리 결과 (루트: {root})\n")
    print(f"📦 프로젝트 후보 {len(result.projects)}개:")
    for p in result.projects:
        print(f"   - {p.id}  ({p.path})  [{', '.join(p.indicators)}]")
    print(f"\n📋 태스크 소스 후보 {len(result.tasks)}개:")
    for t in result.tasks:
        hint = f" ~{t.count_hint}개" if t.count_hint is not None else ""
        print(f"   - {t.source_file}  [{t.kind}{hint}]")
    if result.existing_gwangjang_data:
        print("\n🔗 기존 광장-호환 데이터:")
        for f in result.existing_gwangjang_data:
            print(f"   - {f}")

    if args.llm:
        print(
            "\n(--llm 정제는 후속 단계 — 현재는 휴리스틱 결과만 출력합니다.)",
            file=sys.stderr,
        )
    print(
        "\n등록은 아직 자동화되지 않았습니다. 후보를 검토 후 "
        "`gwangjang call` 의 create_project/create_task 로 등록하세요."
    )
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    root = find_gwangjang_root()
    data_dir = _resolve_data_dir()
    store = Store(data_dir)
    projects = store.list_projects()
    tasks = store.list_tasks()
    agents = store.list_agents()
    crs = store.list_change_requests()
    edges = store.list_task_edges()

    open_requests = [
        c for c in crs if str(c.status) in ("pending", "awaiting_docs", "under_review")
    ]

    # 최근 로그 요약 (마지막 5줄).
    log_fp = Path(data_dir) / "log.jsonl"
    recent: list[dict] = []
    if log_fp.exists():
        lines = [ln for ln in log_fp.read_text(encoding="utf-8").splitlines() if ln.strip()]
        for ln in lines[-5:]:
            try:
                recent.append(json.loads(ln))
            except json.JSONDecodeError:
                continue

    summary = {
        "root": str(root) if root else None,
        "mode": "deployment" if root else "legacy-monorepo",
        "data_dir": str(data_dir),
        "version": GWANGJANG_VERSION,
        "counts": {
            "projects": len(projects),
            "tasks": len(tasks),
            "agents": len(agents),
            "open_requests": len(open_requests),
            "task_edges": len(edges),
        },
        "recent_log": recent,
    }

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    print(f"🏛️  광장 status  (v{GWANGJANG_VERSION})")
    print(f"   루트: {summary['root'] or '(없음 — 레거시 monorepo 모드)'}")
    print(f"   데이터: {data_dir}")
    c = summary["counts"]
    print(
        f"   프로젝트 {c['projects']} · 태스크 {c['tasks']} · 에이전트 "
        f"{c['agents']} · 미해결요청 {c['open_requests']} · 엣지 {c['task_edges']}"
    )
    if recent:
        print("   최근 로그:")
        for e in recent:
            print(
                f"     [{e.get('log_id')}] {e.get('action_type')} "
                f"by {e.get('actor')}"
            )
    return 0


def _cmd_root(args: argparse.Namespace) -> int:
    root = find_gwangjang_root()
    if root is None:
        print("", file=sys.stderr)  # silent — 스크립트 친화
        return 1
    print(str(root))
    return 0


def _cmd_call(args: argparse.Namespace) -> int:
    raw = args.payload if args.payload is not None else sys.stdin.read()
    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError as e:
        print(json.dumps({"ok": False, "error": f"malformed JSON: {e}", "code": "INVALID_INPUT"}))
        return 1
    method = envelope.get("method")
    params = envelope.get("params") or {}
    if not method:
        print(json.dumps({"ok": False, "error": "missing 'method' field", "code": "INVALID_INPUT"}))
        return 1
    data_dir = _resolve_data_dir(args.data_dir)
    cli = GwangjangCLI(data_dir=data_dir)
    response = cli.call(method, params)
    print(json.dumps(response, ensure_ascii=False))
    return 0 if response.get("ok") else 1


def _cmd_start(args: argparse.Namespace) -> int:
    root = find_gwangjang_root()
    data_dir = _resolve_data_dir()
    if args.daemon:
        print(
            "데몬 모드(Unix socket)는 후속 단계(phase 7+)입니다. "
            "현재는 stateless CLI 모드만 지원합니다.",
            file=sys.stderr,
        )
        return 1
    print("🏛️  광장 서비스 (stateless CLI 모드)")
    print(f"   루트: {root or '(레거시 monorepo)'}")
    print(f"   데이터: {data_dir}")
    print(f"   버전: {GWANGJANG_VERSION}")
    print("   각 `gwangjang call` 은 새 프로세스로 처리됩니다. 준비 완료.")
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gwangjang",
        description="LLM 광장 — 폴더 자동 인식형 멀티-에이전트 조정 시스템.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("init", help="현재/지정 폴더에 광장 초기화")
    sp.add_argument("path", nargs="?", help="초기화할 경로 (기본: 현재 폴더)")
    sp.set_defaults(func=_cmd_init)

    sp = sub.add_parser("discover", help="폴더 구조 스캔 → 프로젝트/태스크 후보")
    sp.add_argument("--llm", action="store_true", help="LLM 정제 (후속 단계)")
    sp.add_argument("--json", action="store_true", help="JSON 출력")
    sp.set_defaults(func=_cmd_discover)

    sp = sub.add_parser("status", help="광장 루트 · 등록 카운트 · 최근 로그")
    sp.add_argument("--json", action="store_true", help="JSON 출력")
    sp.set_defaults(func=_cmd_status)

    sp = sub.add_parser("root", help="현재 광장 루트 경로 출력 (없으면 exit 1)")
    sp.set_defaults(func=_cmd_root)

    sp = sub.add_parser("call", help="Agent → 광장 호출 (JSON in/out)")
    sp.add_argument("payload", nargs="?", help="JSON payload (생략 시 stdin)")
    sp.add_argument("--data-dir", default=None, help="데이터 디렉터리 강제 지정")
    sp.set_defaults(func=_cmd_call)

    sp = sub.add_parser("start", help="광장 서비스 시작 (health check)")
    sp.add_argument("--daemon", action="store_true", help="데몬 모드 (후속 단계)")
    sp.set_defaults(func=_cmd_start)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
