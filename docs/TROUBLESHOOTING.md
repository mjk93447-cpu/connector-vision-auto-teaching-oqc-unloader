# 문제 해결 가이드 — Pin Detection / EXE

이 문서는 시행착오를 거친 에러 사례와 진단·해결 절차를 체계적으로 정리합니다.  
**최종 수정**: 2026-03-06

---

## 1. EXE NoneType write 오류

### 1.1 증상

```
'NoneType' object has no attribute 'write'
```

- **발생 시점**: 오프라인 Windows EXE에서 YOLO 학습 진행 중 (~25분)
- **환경**: PyInstaller 빌드 EXE, `console=False` (GUI 모드)

### 1.2 진단 절차

| 단계 | 확인 항목 | 방법 |
|------|-----------|------|
| 1 | PyInstaller console 설정 | `pin_detection_gui.spec` → `console=False` 여부 |
| 2 | stdout/stderr 상태 | EXE 실행 시 `sys.stdout`, `sys.stderr`가 `None`인지 |
| 3 | 트리거 위치 | Ultralytics YOLO progress bar, tqdm, 로그 출력 |

### 1.3 근본 원인

- Windows GUI 앱(pythonw.exe)에서는 `sys.stdout`, `sys.stderr`가 `None`
- Ultralytics YOLO 학습 시 progress bar·로그가 stdout에 write 시도 → `None.write()` 호출 → AttributeError

### 1.4 해결 방법

**run_pin_gui.py 최상단** (모든 import 전):

```python
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")
```

### 1.5 검증

- `tools_scripts/test_exe_stdout_fix.py`: None 환경에서 import 후 크래시 없음 확인
- `test_exe_train.bat`: EXE `--test-train` 3 epoch 완료, `exe_test_ok.txt` 생성

---

## 2. GitHub Actions — Test EXE stdout fix 실패

### 2.1 증상

- CI 단계 `Test EXE stdout fix (NoneType write)`에서 exit code 1
- 로컬에서는 통과, GitHub Actions에서만 실패

### 2.2 진단 절차

| 단계 | 확인 항목 | 방법 |
|------|-----------|------|
| 1 | 작업 디렉터리 | CI 기본 cwd가 workspace root인지 |
| 2 | sys.path | `run_pin_gui` import 시 프로젝트 루트 포함 여부 |
| 3 | 예외 유형 | AssertionError vs AttributeError vs ImportError |

### 2.3 근본 원인

- CI 환경에서 cwd 또는 sys.path 차이로 `run_pin_gui` import 실패
- `assert` 실패 시 AssertionError 미처리로 exit 1

### 2.4 해결 방법

**tools_scripts/test_exe_stdout_fix.py**:

```python
# 프로젝트 루트를 path에 추가
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)
os.chdir(_root)
```

- `assert` 대신 `(code, msg)` 반환, 실패 시 메시지 출력
- `working-directory: ${{ github.workspace }}` 명시 (workflow)

### 2.5 검증

- 로컬: `python tools_scripts/test_exe_stdout_fix.py`
- CI: push 후 Actions 로그 확인

---

## 3. PyInstaller 빌드 관련

### 3.1 PowerShell에서 빌드 시 stderr 이슈

- **증상**: `python -m PyInstaller` 실행 시 INFO 출력이 stderr로 처리되어 PowerShell RemoteException
- **해결**: `build_pin_exe.bat`에서 `cmd /c`로 실행, 또는 `--log-level WARN` 사용
- **CI**: `shell: cmd` 사용 (build-pin-exe.yml)

### 3.2 numpy.f2py hidden import 경고

- **증상**: PyInstaller 분석 시 `expecttest`, `torch.testing._internal.opinfo` 등 ModuleNotFoundError
- **영향**: 대부분 비치명적, EXE 빌드 완료됨
- **대응**: 필요 시 spec에 `excludes` 또는 `hiddenimports` 추가

---

## 4. 진단 체크리스트 (EXE 학습 실패 시)

1. [ ] EXE가 최신 빌드인가? (GitHub Actions 아티팩트 또는 `build_pin_exe.bat`)
2. [ ] `run_pin_gui.py`에 stdout/stderr None 처리 있는가?
3. [ ] `--test-train`으로 단축 테스트 시 동일 오류 재현되는가?
4. [ ] `test_data/pin_synthetic` 데이터 존재하는가?
5. [ ] 오프라인 환경에서 `YOLO_OFFLINE=true` 설정되어 있는가?

---

## 5. 참고 링크

- [PyInstaller #7782](https://github.com/pyinstaller/pyinstaller/issues/7782) — GUI app stdout
- `docs/EXE_NONETYPE_FIX_PLAN.md` — 상세 해결 계획
