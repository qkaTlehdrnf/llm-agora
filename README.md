# LLM 광장 (LLM Agora)

> A coordination substrate for many specialized LLM agents — each agent succeeds
> without knowing the whole system, because **code (not prompts) enforces the
> collaboration boundaries**.

`version 0.1-design`

---

## 🎯 Objective (목표 / main objective)

**회사 조직에서 영감을 받았습니다.** 한 회사의 구성원은 회사 전체를 다 알지
못해도, 각자의 전문성만으로 맡은 일을 해내고 — 그 합이 거대한 시스템을 굴립니다.

광장의 목표는 이 구조를 LLM 에이전트 협업에 그대로 가져오는 것입니다:

> **단순 전문 Agent 다수 + Agent당 최소 지식 + 코드 기반 행동 규제
> → 창발적으로(emergent) 거대 시스템을 구동한다.**

핵심 통찰(key insight): **전체 지식 없이도 개발 성공이 가능하다.** 그 조건은
"Agent당 최소 컨텍스트"와 "코드 기반 행동 규제"의 조합입니다.

### 두 가지 메커니즘 (two mechanisms)

1. **language_exchange — LLM 간 구조화 문서 교환**
   에이전트끼리 구조화된 문서를 생성·공유해 정보를 전달합니다(문서 쓰기/읽기).

2. **code_regulation — 코드가 행동을 강제**
   무엇을 할 수 있고 없는지를 *프롬프트가 아니라 코드*가 결정합니다.
   프롬프트로 규칙을 설명하면 LLM이 우회할 수 있지만, **코드는 우회 불가**입니다.
   이것이 신뢰할 수 있는 협업 경계선입니다.

### 두 종류의 행위자 (actors)

| | 역할 | 아는 것 |
|---|---|---|
| **광장 Agent** (Coordinator) | 중앙 조정자. 프로젝트 추천, 요청 승인/거절, 정보 배분 통제, 로그 보존, 이미 받은 컨텍스트 재요청 경고·재확인 | 전체 프로젝트/태스크 메타데이터 + 로그 (세부 구현은 모름) |
| **Worker Agent** (Specialist) | 배정된 컨텍스트 안에서 태스크 수행, 변경 요청 제출 | 배정된 프로젝트 + 현재 태스크 컨텍스트만 (다른 프로젝트는 광장이 차단) |

Worker는 직접 프로젝트/태스크를 수정할 수 없고(반드시 광장 경유), 한 세션에서
**이미 받은 컨텍스트를 다시 요청하면 코드가 경고하고 재확인을 요구하며**(무한
재요청 루프 방지), 다른 Agent의 컨텍스트에 직접 접근할 수 없습니다 —
모두 **코드 레이어가 강제**합니다.

> 참고: task edge 의 `blocks` 관계가 만드는 **DAG 사이클**은 별개로
> `validator.py` 가 검출합니다(아래 § link_inference). `router.py` 는 그와 다른,
> "한 세션 내 동일 컨텍스트 재요청" 감지기입니다.

---

## 📜 헌법 (Constitution)

광장은 **헌법** 아래에서 동작합니다. 규칙 그 자체의 개정 절차마저 프롬프트가
아니라 코드가 강제합니다(1조 불변, 6조 이유필수, 개정 정족수). 정본은
[`CONSTITUTION.md`](./CONSTITUTION.md), 기계형은 [`data/constitution.json`](./data/constitution.json),
집행 코드는 [`src/constitution.py`](./src/constitution.py) 입니다.

| 조 | 요지 | 집행 |
|---|---|---|
| 1조 (불변) | 모든 헌법은 변경 가능 — 단 1조 자신은 개정 불가 | 개정안이 1조를 건드리면 코드가 거부 |
| 2조 | 핵심 Agent **2/3** 동의 → 헌법 **교체** | `required_core_votes("replace")` |
| 3조 | 최대한 **독립 TASK · 최소 PROMPT** | `context_filter` + `link_inference` |
| 4조 | 핵심 Agent가 하위 Agent **Orchestration** | `delegate` (+ 위임 순환 가드) |
| 5조 | **누구나 발의** → 결재 상향 → 핵심 **1/2** 로 **추가** | `propose_amendment`/`endorse_amendment` |
| 6조 | 모든 규칙은 **이유와 함께 기록** | `Article.reason` 필수 |

호출: `get_constitution` · `propose_amendment` · `endorse_amendment` · `delegate`
([`CALL_INTERFACE.json`](./CALL_INTERFACE.json)).

---

## 🔗 핵심 서브시스템: link_inference_system

태스크 간 **typed edge(유형이 있는 관계)** 를 자동 추론합니다. "작업 겹침 방지,
상호 보완 인지, 변경점 추적"을 코드 수준으로 구현한 것입니다.

| edge type | 의미 | 동작 |
|---|---|---|
| `blocks` | A 산출물이 B의 필수 입력 (DAG) | A 완료 전 B 시작 차단 |
| `overlaps_with` | 산출물 경로/모듈이 겹침 | 시작 전 조정 알림 (머지 충돌 예방) |
| `complements` | A 산출물이 B 품질에 기여 (B는 없이도 동작은 함) | A 완료 시 "산출물 사용 가능" 알림 |
| `notify_on_change` | A의 인터페이스/스키마가 B의 입력 | **A 변경 commit 시 B에 자동 알림** ← 광장의 USP |
| `supersedes` | A 완료가 B를 대체 | B를 `superseded` 상태로 보존 |

추론 파이프라인: `BM25 + weighted-tag Jaccard + dense embedding → RRF 융합 →
LLM-as-jury(Gemini + Claude) pairwise 분류`. 사람 라벨 없이, **swap-consistency +
2-model 합의** 두 가드로 신뢰성을 확보합니다.

---

## 🗂️ 저장소 구조 — 프레임워크 vs 인스턴스 데이터 분리

이 저장소는 **재사용 가능한 프레임워크(공개)** 와 **당신의 실제 데이터(비공개)** 를
명확히 분리합니다.

```
llm_agora/
├── README.md            ← (당신이 보는 이 문서) 목표 / main objective
├── src/                 ← 프레임워크 코드 (code layer + LLM layer)
│   ├── cli.py              `gwangjang` 콘솔 명령 (init/discover/status/root/call/start)
│   ├── call.py             단방향 호출 인터페이스 (JSON in/out)
│   ├── utils.py            .gwangjang/ 루트 탐지 + 설정 + init
│   ├── discover.py         폴더 구조 휴리스틱 디스커버리 (프로젝트/태스크)
│   ├── similarity.py       Worker 자기소개 → 프로젝트 추천 (tag+TF-IDF+dense)
│   ├── coordinator.py      광장 Agent — 변경요청 LLM 검토(승인/거절/추가서류)
│   ├── constitution.py     헌법 — 개정 정족수·1조 불변·6조 이유필수를 코드로 강제
│   ├── store.py            JSON 파일 데이터 스토어 (data_dir 파라미터)
│   ├── validator.py        형식 검증 + DAG 사이클 검출
│   ├── router.py           세션 내 동일 컨텍스트 재요청 감지(경고+재확인)
│   ├── context_filter.py   need-to-know 가시성 필터
│   ├── log_manager.py      append-only 로그
│   └── link_inference/     후보 생성 → jury → 파이프라인 → graph export
│
├── *.json               ← 설계 명세 (LLM-최적화). SPEC.json 부터 읽으세요.
│   SPEC → DATA_MODELS → PROTOCOL → BENCHMARKS → ...
│
├── data.example/        ← 합성 예시 데이터 (공개). 스키마/동작 참고용.
│
└── data/                ← 당신의 실제 "관리중인 프로젝트" 데이터 (비공개, .gitignore)
                            여기에 진짜 projects/tasks/edges/log 가 쌓입니다.
```

- **`data.example/`** (공개): 가상의 소프트웨어 프로젝트 3개로 스키마와 edge 추론
  결과를 보여주는 합성 샘플입니다. 누구나 이걸로 바로 돌려볼 수 있습니다.
- **`data/`** (비공개, `.gitignore`): 당신의 인스턴스가 실제로 조정하는
  프로젝트/태스크가 저장되는 곳. 커밋되지 않습니다.

> **사람용 README ↔ LLM용 SPEC**: 이 문서는 사람을 위한 목표 요약이고,
> 기계가 읽는 완전한 명세는 [`SPEC.json`](./SPEC.json) 에 있습니다
> (그 안 `philosophy` 섹션이 위 Objective의 출처입니다).

---

## 🚀 Quickstart

```bash
# 0) 설치 (개발 모드) — `gwangjang` 콘솔 명령 + llm_agora 패키지 등록
pip install -e .

# 1) 예시 데이터로 새 인스턴스 시작
cp -r data.example data

# 2) 호출 인터페이스 사용 (예: 그래프 export)
python -m llm_agora.src.call \
  '{"method":"graph_export","params":{"scope":"all"}}'

# 3) 태스크 간 typed edge 추론
python -m llm_agora.src.call \
  '{"method":"request_link_inference","params":{"top_k_per_task":10}}'
```

데이터 경로는 `--data-dir` 로 바꿀 수 있습니다 (기본값 `data/`). 실제 LLM jury는
`GOOGLE_API_KEY` + `ANTHROPIC_API_KEY` 환경변수가 있으면 자동 활성화되고,
없으면 휴리스틱 MockJudge로 동작합니다.

---

## 📦 설치형 CLI — 임의 폴더에서 `gwangjang`

광장을 패키지로 설치하면 git 처럼 **아무 폴더에서나** 시작할 수 있습니다.
`.gwangjang/` 디렉터리가 루트 마커(=git 의 `.git`) 역할을 합니다.

```bash
# 설치 (개발 모드)
pip install -e .

# 임의 프로젝트 폴더에서
cd ~/my-workspace
gwangjang init                # .gwangjang/ 생성
gwangjang discover            # 폴더 스캔 → 프로젝트/태스크 후보 식별
gwangjang status              # 등록 카운트 + 최근 로그
gwangjang call '{"method":"onboard","params":{...}}'   # Agent 호출
```

설치 후 `gwangjang ...` 콘솔 명령과 `python -m llm_agora.src.cli ...` 는
동등합니다. 루트 탐지는 현재 폴더에서 상위로 거슬러 올라가며 `.gwangjang/` 를
찾고, 없으면 패키지에 내장된 `data/` 로 폴백합니다.

### Worker onboarding (similarity 추천)

```bash
gwangjang call '{"method":"onboard","params":{
  "agent_id":"w1",
  "description":"연구실 웹페이지 크롤링·논문 태깅 전문가",
  "capabilities":["crawling","tagging","llm"],
  "top_n":4
}}'
# → tag Jaccard + TF-IDF(+옵션 dense) 결합으로 프로젝트 top-N 추천
#   "use_dense": true 로 dense 임베딩 채널 활성화(콜드스타트 ~90s)
```

### 변경요청 검토 (광장 Agent)

Worker 는 직접 task/project 를 못 고치고, **광장 승인을 거친 변경만** 반영됩니다.

```bash
# 1) Worker 변경요청 제출 → 광장이 필수 서류 목록 반환
gwangjang call '{"method":"submit_request","params":{"agent_id":"w1",
  "request_type":"create_task","target_id":"task-x",
  "payload":{"project_id":"demo","action":"...","done_when":"..."},
  "rationale":"왜 이 변경이 필요한지"}}'

# 2) 서류 제출 → 3) 광장 Agent 검토(승인 시 store 에 실제 반영)
gwangjang call '{"method":"submit_docs","params":{"request_id":"...","docs":{...}}}'
gwangjang call '{"method":"review_request","params":{"request_id":"..."}}'
# 결정: approve(적용) / reject / need_docs. GOOGLE/ANTHROPIC 키 있으면 LLM,
# 없으면 MockCoordinator 휴리스틱. "force_mock": true 로 강제 가능.
```

---

## 📄 더 읽기

- [`CONSTITUTION.md`](./CONSTITUTION.md) — 헌법 정본 (조문 + 개정 절차 + 코드 대응)
- [`SPEC.json`](./SPEC.json) — 전체 시스템 명세 (시작점)
- [`DATA_MODELS.json`](./DATA_MODELS.json) — 데이터 모델
- [`PROTOCOL.json`](./PROTOCOL.json) — 호출/협업 프로토콜
- [`CALL_INTERFACE.json`](./CALL_INTERFACE.json) — 호출 인터페이스 명세
- [`BENCHMARKS.json`](./BENCHMARKS.json) — 평가 지표/목표치
