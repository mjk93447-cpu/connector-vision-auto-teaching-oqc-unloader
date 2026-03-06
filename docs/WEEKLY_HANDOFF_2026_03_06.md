# 주간 인수인계 — 2026-03-06 (금)

**작성일**: 2026년 3월 6일 금요일  
**목적**: 주말 이후 월요일 개발 재개 시 컨텍스트 보존. 메모리·경험 손실 대비.

---

## 1. 금주(3/4~3/6) 완료 작업 요약

### 1.1 EXE NoneType write 오류 해결 (ROADMAP 10.11a)

| 항목 | 내용 |
|------|------|
| **증상** | 오프라인 Windows EXE에서 학습 ~25분 후 `'NoneType' object has no attribute 'write'` 에러로 중단 |
| **원인** | PyInstaller `console=False` 시 `sys.stdout`/`sys.stderr`가 `None`. Ultralytics YOLO가 progress bar 출력 시 크래시 |
| **해결** | `run_pin_gui.py` 최상단에서 None이면 `open(os.devnull,'w')`로 대체 (모든 import 전) |
| **추가** | GUI Training log 영역 (epoch별 Loss/P/R, ETA) |
| **검증** | `build_pin_exe.bat` → `test_exe_train.bat` → `exe_test_ok.txt` 생성 확인 |

### 1.2 GitHub Actions CI 수정

| 항목 | 내용 |
|------|------|
| **증상** | `Test EXE stdout fix (NoneType write)` 단계에서 CI 실패 |
| **원인** | CI 환경 cwd/path 차이로 `run_pin_gui` import 실패 가능성 |
| **해결** | `test_exe_stdout_fix.py`에 프로젝트 루트 sys.path 추가, chdir, 에러 메시지 개선 |
| **추가** | workflow paths에 `tools_scripts/**` 추가, `working-directory` 명시 |

### 1.3 ROADMAP 10.16~10.18 (3/6)

- **10.16**: 학습 속도·탐지 품질 균형 실측 (pin_large 10ep Recall 0%, pin_synthetic 50ep Recall 24.5%)
- **10.17**: geometry_refinement 평가 시 비활성화 필요 (`--no-geometry-refinement`로 Recall/Precision 정확 측정)
- **10.18**: ROI 비활성화·Geometry Refinement 개선 완료 — masked prior 활용 시 Recall/Precision 100%

---

## 2. 주요 파일·경로

| 경로 | 역할 |
|------|------|
| `run_pin_gui.py` | EXE 진입점, stdout/stderr None 처리, `--test-train` |
| `pin_detection/gui.py` | GUI, Training log (results.csv 폴링) |
| `build_pin_exe.bat` | 로컬 EXE 빌드 |
| `test_exe_train.bat` | EXE 학습 테스트 (`--test-train`) |
| `pin_detection_gui.spec` | PyInstaller spec |
| `docs/EXE_NONETYPE_FIX_PLAN.md` | NoneType 해결 계획·검증 |
| `docs/TROUBLESHOOTING.md` | 에러 사례·진단·해결 |
| `tools_scripts/test_exe_stdout_fix.py` | None stdout/stderr 처리 검증 |
| `tools_scripts/repro_exe_training_with_none_stdout.py` | None stdout 환경 학습 시뮬레이션 |

---

## 3. 다음 주 권장 작업 (ROADMAP 기준)

| 우선순위 | 작업 | 출처 |
|----------|------|------|
| P2 | Excel vs bbox 개수 불일치 경고 | 10.12 |
| P2 | 대형 이미지 ROI (Inference 시 masked 페어 옵션) | 10.12 |
| P3 | Help 검색 (Ctrl+F) | 10.12 |
| P3 | 최근 경로 기억 (설정 저장/복원) | 10.12 |
| - | 데이터셋 자동 최적화 (imgsz/epochs 추천) | 10.14 |

---

## 4. 로컬 검증 명령

```batch
REM EXE 빌드
build_pin_exe.bat

REM EXE 학습 테스트
test_exe_train.bat

REM None stdout 처리 단위 테스트
python tools_scripts/test_exe_stdout_fix.py
```

---

## 5. 참고 문서

- **ROADMAP.md** 10.11a: EXE NoneType 해결 완료
- **docs/EXE_NONETYPE_FIX_PLAN.md**: 상세 계획
- **docs/TROUBLESHOOTING.md**: 에러 사례·진단 절차
