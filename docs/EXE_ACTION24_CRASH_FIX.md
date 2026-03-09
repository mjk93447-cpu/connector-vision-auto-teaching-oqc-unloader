# Action #24 EXE 크래시 수정 — Building dataset 21/21 후 silent exit

**날짜**: 2026-03-09  
**대상**: pin_detection_gui Windows EXE (Actions #24)

---

## 1. 증상

| 항목 | 내용 |
|------|------|
| **발생 시점** | Train 탭, Start training 클릭 후 |
| **로그** | "Building dataset (1/21)" … "(21/21)" 한 줄씩 출력 |
| **크래시** | 21/21 도달 직후 EXE 즉시 종료 |
| **에러** | 경고/에러 팝업 없음 (silent exit) |

---

## 2. 원인

| 항목 | 설명 |
|------|------|
| **발생 환경** | PyInstaller EXE, GUI Train tab |
| **트리거** | Dataset prep 완료 → `model.train()` 호출 → DataLoader workers spawn |
| **근본 원인** | Windows multiprocessing spawn: PyInstaller EXE에서 worker 프로세스 생성 시 충돌 |
| **참조** | [PyInstaller multiprocessing](https://github.com/pyinstaller/pyinstaller/wiki/Recipe-Multiprocessing) |

---

## 3. 조치

| 파일 | 변경 |
|------|------|
| `pin_detection/train.py` | `_default_workers()`: `sys.frozen` 시 0 반환 |
| `pin_detection/train.py` | `model.train()` 호출 전: frozen 시 `n_workers=0` 강제 |

---

## 4. 재현

```bash
python tools_scripts/repro_exe_train_crash.py
```

- 21쌍 합성 데이터 생성
- `sys.frozen=True` 시뮬레이션 → workers=0 적용
- Building dataset (21/21) → model.train() → 완료 확인

---

## 5. 검증

- EXE 빌드 후 Train 탭에서 21+ 이미지로 학습 시작
- Building dataset (21/21) 이후 Epoch 1/… 진행 확인
- 학습 완료까지 정상 동작 확인
