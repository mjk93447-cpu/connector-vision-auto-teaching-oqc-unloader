# EXE 오프라인 학습 시 NoneType write 오류 해결 계획

**작성일**: 2026-03-06  
**증상**: 오프라인 Windows EXE에서 학습 25분 진행 후 `'NoneType' object has no attribute 'write'` 오류로 중단

---

## 1. 원인 진단

| 항목 | 설명 |
|------|------|
| **발생 환경** | PyInstaller EXE, `console=False` (GUI 모드) |
| **근본 원인** | Windows GUI 앱에서 `sys.stdout`, `sys.stderr`가 `None`으로 설정됨 (pythonw.exe 동작) |
| **트리거** | Ultralytics YOLO 학습 중 progress bar, 로그 등이 stdout/stderr에 write 시도 |
| **참조** | [PyInstaller #7782](https://github.com/pyinstaller/pyinstaller/issues/7782), StackOverflow |

---

## 2. 해결 방안

### 2.1 Phase A: stdout/stderr 안전 리다이렉트 (필수)

| 작업 | 설명 | 목표 |
|------|------|------|
| **run_pin_gui.py 최상단** | `sys.stdout`/`sys.stderr`가 None이면 `open(os.devnull,'w')`로 대체 | 크래시 방지 |
| **적용 시점** | `import` 전, `os.environ` 설정 직후 | ultralytics 등 모든 라이브러리 적용 |

### 2.2 Phase B: GUI 학습 로그 표시 (사용성)

| 작업 | 설명 | 목표 |
|------|------|------|
| **Training log 영역** | Text 위젯으로 epoch별 성능·진행도 표시 | 사용자 가시성 |
| **results.csv 폴링** | 기존 graph poll 확장, 최신 epoch 요약 텍스트 출력 | ETA·진행률 표시 |
| **실시간 로그 (선택)** | stdout 캡처 스트림으로 YOLO 출력 수집 → GUI 표시 | 상세 로그 |

---

## 3. 구현 순서

1. **Phase A**: run_pin_gui.py에 stdout/stderr 안전 처리
2. **Phase B**: GUI에 Training log Text 위젯, results.csv 기반 요약
3. **검증**: EXE 빌드 후 오프라인 환경에서 장시간 학습 테스트

---

## 4. 검증 기준

| 항목 | 목표 | 결과 |
|------|------|------|
| NoneType write | 25분+ 학습 시 오류 미발생 | ✓ 2026-03-06 EXE `--test-train` 3 epoch 완료, 크래시 없음 |
| GUI 로그 | Epoch별 Loss/P/R, 예상 남은 시간 표시 | ✓ Training log 영역 구현 |

## 5. 검증 실행 방법

```batch
REM 빌드 (프로젝트 루트)
build_pin_exe.bat

REM EXE 학습 테스트 (test_data/pin_synthetic 사용)
test_exe_train.bat
```

성공 시 `pin_models_exe_test/exe_test_ok.txt` 생성.
