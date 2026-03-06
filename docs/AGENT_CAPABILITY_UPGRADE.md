# AI 에이전트 역량 강화 방법론

**작성일**: 2026-03-06  
**목적**: 문서 정리 이상의 수단으로, 다음 주부터 에이전트 코딩 역량이 체감 가능하게 향상되도록 함.

---

## 1. 핵심 원리

| 구분 | 문서 중심 | 역량 강화 중심 |
|------|-----------|----------------|
| **저장** | HANDOFF.md, TROUBLESHOOTING.md | `.cursor/rules/*.mdc` (자동 적용) |
| **적용 시점** | 사람이 읽을 때 | 에이전트가 매 턴 컨텍스트에 포함 |
| **형태** | "참고하세요" | "BEFORE X, do Y" / "AVOID X, INSTEAD Y" |
| **검증** | 없음 | Preflight, Self-Review 절차 |

**결론**: 규칙(Rules)은 에이전트가 항상 읽고 따르므로, 경험을 규칙으로 변환해야 역량이 올라감.

---

## 2. 구성 요소

| 요소 | 경로 | 역할 |
|------|------|------|
| **Preflight** | `.cursor/rules/agent-preflight.mdc` | 작업 전 필수 검증 (날짜, CI, 문서, 커밋) |
| **Anti-patterns** | `.cursor/rules/agent-anti-patterns.mdc` | 피해야 할 구체적 패턴 (정/오 예시) |
| **Experience Replay** | `docs/EXPERIENCE_REPLAY.md` | 실패→원인→해결→규칙 매핑. 새 실패 시 추가 |
| **Self-Review** | `.cursor/commands/self-review.md` | 커밋 전 점검 절차 (`/self-review`) |
| **Skill** | `.cursor/skills/connector-vision-auto-teaching/SKILL.md` | Reflection Workflow 절차 |

---

## 3. 사용 절차

### 3.1 매 턴 (자동)

- `agent-preflight.mdc`, `agent-anti-patterns.mdc`는 `alwaysApply: true` → 에이전트가 매 대화에 자동 적용.

### 3.2 커밋 전 (수동/권장)

- `/self-review` 실행 또는 `.cursor/commands/self-review.md` 절차 따름.

### 3.3 새 실패 발생 시

1. `docs/EXPERIENCE_REPLAY.md`에 항목 추가 (실패, 원인, 해결, 규칙화)
2. `agent-preflight` 또는 `agent-anti-patterns`에 대응 규칙 없으면 추가
3. 필요 시 스킬에 절차 반영

---

## 4. 참고 연구

- **Self-improving agents**: 사용자 수정·되돌리기 패턴을 규칙으로 반영 시, 수정률 0.45 → 0.07로 감소 (Pradeep, 2026).
- **Cursor Rules**: "Add rules only when the agent makes the same mistake repeatedly" (Cursor Blog).
- **Verification**: 명시적 검증 단계, 정/오 예시가 규칙 효과를 높임 (Lambda Curry, Cursor Docs).
