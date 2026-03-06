# Experience Replay — 경험 → 규칙 매핑

**목적**: AI 에이전트가 과거 실패를 읽고, 동일 실수 재발을 막는 규칙으로 변환할 수 있도록 구조화.

**사용법**: 새 실패 발생 시 이 문서에 항목 추가. agent-preflight.mdc, agent-anti-patterns.mdc에 대응 규칙이 없으면 규칙 추가.

---

## 형식

각 항목은 다음 구조를 따름:

```
## N. [짧은 제목] (YYYY-MM-DD)

**실패**: [무엇이 잘못되었는지 한 줄]
**원인**: [근본 원인]
**해결**: [구체적 조치]
**규칙화**: [agent-preflight 또는 agent-anti-patterns에 반영된 규칙 요약]
```

---

## 1. 날짜 오기입 (2026-03-06)

**실패**: WEEKLY_HANDOFF_2026_03_07.md 생성. 실제 날짜는 2026-03-06(금).

**원인**: "금요일"에서 3/7을 추측. user_info의 `Today's date` 미확인.

**해결**: user_info에서 날짜 확인 후 문서 작성. 03_07 → 03_06 수정.

**규칙화**: agent-preflight §1 (날짜 작성 전), agent-anti-patterns §1 (날짜 추측)

---

## 2. CI test_exe_stdout_fix 실패 (2026-03-06)

**실패**: GitHub Actions에서 Test EXE stdout fix 단계 exit code 1. 로컬 통과.

**원인**: CI cwd/sys.path 차이로 `run_pin_gui` import 실패. assert 실패 시 메시지 없음.

**해결**: `sys.path.insert(0, root)`, `os.chdir(root)`, `(code, msg)` 반환, working-directory 명시.

**규칙화**: agent-preflight §2 (CI/테스트 스크립트 수정 전), agent-anti-patterns §2 (CI 전용 실패 무시)

---

## 3. 문서 중심 인수인계 (2026-03-06)

**실패**: HANDOFF, TROUBLESHOOTING 등 문서만 추가. 에이전트가 "다음에 참고"할 실행 규칙 부재.

**원인**: 문서는 사람이 읽어야 적용됨. 에이전트는 Rules/Skills를 매 턴 자동 적용.

**해결**: agent-preflight.mdc, agent-anti-patterns.mdc 생성. "BEFORE X, do Y" 형태로 실행 가능 규칙화.

**규칙화**: agent-anti-patterns §3 (문서만 정리)

---

## 4. 검증 생략 (2026-03-06)

**실패**: 날짜 수정 후 HANDOFF 링크·파일명 일치 여부 미확인. 잘못된 링크 가능성.

**원인**: "문서 수정 완료"로 간주하고 실제 동작 검증 생략.

**해결**: 수정 후 링크 클릭 가능 여부, 파일 존재 여부, 테스트 실행으로 회귀 확인.

**규칙화**: agent-preflight §4 (커밋·푸시 전), agent-anti-patterns §4 (검증 생략)

---

## 규칙 업데이트 절차

1. 새 실패 발생 → 이 문서에 항목 추가
2. agent-preflight 또는 agent-anti-patterns에 대응 규칙 있는지 확인
3. 없으면 해당 규칙 파일에 "BEFORE X, do Y" 또는 "AVOID X, INSTEAD Y" 추가
4. 필요 시 스킬(SKILL.md)에 절차 반영
