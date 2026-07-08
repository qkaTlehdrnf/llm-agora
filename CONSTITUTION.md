# 광장 헌법 (Constitution)

> 광장의 규칙은 프롬프트로 설명되는 권고가 아니라 **코드가 강제하는 경계**다.
> 이 문서는 헌법의 정본(正本)이며, 기계가 읽는 형태는
> [`data/constitution.json`](./data/constitution.json), 집행 코드는
> [`src/constitution.py`](./src/constitution.py) 이다. 조문·개정 절차·정족수는
> 모두 코드에서 검증되며 하위 Agent의 프롬프트로 우회할 수 없다.

`version 1` · 핵심 Agent 로스터(seed): `["gwangjang"]`

---

## 조문 (Articles)

각 조는 **만들어진 이유(reason)** 와 함께 기록된다 (6조의 자기적용).

### 1조 — 가변성 (불변 · IMMUTABLE)
> 모든 헌법은 변경될 수 있다. **단, 이 조항 자체는 개정할 수 없다.**

**이유**: 가변성을 유일하게 고정된 지점으로 두어, 헌법이 경직되지 않으면서도
"무엇이든 바꿀 수 있다"는 토대만은 흔들리지 않게 하기 위함. — 코드는 1조를
개정·삭제·약화하는 어떤 개정안도 거부한다(전면 교체 시에도 1조는 원문 보존 필수).

### 2조 — 전면 교체 정족수
> 핵심 Agent의 **2/3** 이 동의하는 순간, 헌법은 교체될 수 있다.

**이유**: 전면 교체는 체계 전체를 바꾸므로 단순 추가(1/2)보다 높은 합의 문턱을 둔다.

### 3조 — 독립 TASK · 최소 PROMPT
> Agent는 최대한 **독립적인 TASK** 로 진행하며 **최소한의 PROMPT** 를 유지한다.
> (독립적 = 추가 정보 없이 주어진 정보만으로 실행한 결과가 추가 정보가 있을 때와
> 동일하거나, 다른 TASK와 겹치는 작업 없이 자신의 작업이 온전히 반영되는 상태.)

**이유**: 광장의 핵심 인사이트 — 최소 지식·최소 컨텍스트의 독립 TASK 다수가
코드 규제 아래 창발적으로 대형 시스템을 굴린다. `context_filter`(need-to-know)와
`link_inference`(overlaps_with·blocks-DAG)가 이 조를 기계적으로 집행한다.

### 4조 — 핵심 Agent의 Orchestration
> 핵심 Agent는 하위 Agent에게 핵심적인 업무를 지시하고, 긴 Prompt를 유지할 수
> 있으며, 주어진 목적을 위하여 하위 Agent를 **Orchestration** 한다.

**이유**: 독립 TASK(3조)를 조립해 목적을 달성할 조정자가 필요하다. 조정 부담과
긴 컨텍스트는 핵심 Agent에 집중시켜 하위 Agent는 가볍게 유지한다.

### 5조 — 발의와 비준
> 헌법은 **누구나 제의** 할 수 있으며, 제의는 상위 Agent의 허락 하에 위로 결재가
> 올라가고, 최종적으로 핵심 Agent **절반(1/2)** 의 동의 아래에 추가될 수 있다.

**이유**: 개정 발의를 개방하되(누구나), 결재 상향과 핵심 Agent 정족수로 품질·책임을 건다.

### 6조 — 이유의 기록
> 모든 규칙은 해당 규칙이 만들어진 **이유와 함께 기록** 되어야 한다.

**이유**: 이유 없는 규칙은 나중에 검증·폐기가 불가능하다. 실패 기반 규칙 누적
(casebook) 방침과 동일 정신. — 코드는 이유 없는 조항/발의를 거부한다.

---

## 개정 절차 (Amendment process)

1. **발의** — 누구나 `propose_amendment` (5조). 발의 단계에서 코드가 즉시 검증:
   - 1조를 건드리면 → `AMENDMENT_ILLEGAL` (불변).
   - 이유(reason)가 비면 → `AMENDMENT_ILLEGAL` (6조).
2. **결재 상향 / 표결** — `endorse_amendment`.
   - 비핵심 Agent = **결재 상향**(endorsement, 정족수엔 미포함).
   - 핵심 Agent = **비준표**(core vote).
3. **비준** — 핵심 Agent 정족수 충족 순간 자동 적용(`version`+1, `constitution.json` 교체).

| 개정 종류 | 정족수 | 근거 |
|---|---|---|
| `replace` (전면 교체) | 핵심 Agent **⌈2/3⌉** | 2조 |
| `add` (조항 추가) | 핵심 Agent **⌈1/2⌉** | 5조 |
| `amend_article` (조항 수정) | 핵심 Agent **⌈1/2⌉** | 5조 (1조 대상은 불가) |
| `roster_change` (핵심 Agent 로스터 변경) | 핵심 Agent **⌈1/2⌉** | 5조 |

> "핵심 Agent"는 `constitution.json`의 `core_agents` 로스터로 정의되며, 로스터
> 자체의 변경도 개정(`roster_change`)을 거쳐야 한다 — 무한 자기부여 방지.

---

## 조문 ↔ 집행 코드 대응 (Constitution ⇄ code)

헌법은 완전한 신규 규범이 아니라, 상당 부분 **기존 코드 장치에 헌법적 이름을 부여**한 것이다.

| 조 | 집행 코드 | 호출 메서드 | 검증 |
|---|---|---|---|
| 1조 (불변) | `constitution.py` `Constitution` 검증자 + `check_amendment_legal` | `propose_amendment` | 1조 개정 시도 차단 ✅ |
| 2조 (교체 2/3) | `required_core_votes("replace", n)` = ⌈2n/3⌉ | `endorse_amendment` | n=3→2표에서 비준 ✅ |
| 3조 (독립·최소) | `context_filter`(need-to-know) + `link_inference`(`overlaps_with`, blocks-DAG) | `get_context` | 격리·겹침·의존 사이클 집행 ✅ |
| 4조 (orchestration) | `_method_delegate` + `Router`(위임 순환 가드) | `delegate` | 비핵심 위임 차단·순환 차단 ✅ |
| 5조 (발의·1/2) | `required_core_votes(add, n)` = ⌈n/2⌉ + endorsement/vote 분리 | `propose_amendment` / `endorse_amendment` | 누구나 발의·1/2 비준 ✅ |
| 6조 (이유 기록) | `Article.reason` 필수 필드 + `check_amendment_legal` + 로그 `rationale` | 전 메서드 | 이유 없는 조항 차단 ✅ |

> 참고 — 두 종류의 "cycle": 3조의 blocks-DAG 위상 사이클은 `validator.py`가,
> 4조 위임 체인의 호출 순환은 `router.py`가 각각 검출한다.
