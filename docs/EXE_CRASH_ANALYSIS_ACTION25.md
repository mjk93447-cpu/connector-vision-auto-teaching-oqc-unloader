# EXE 크래시 정밀 분석 결과 (Action #25)

**분석일**: 2026-03-09

---

## 1. 재현 테스트 결과

| 환경 | 결과 |
|------|------|
| **Python (repro_exe_train_crash.py)** | ✓ 정상 완료 (21 이미지, 2 epochs) |
| **EXE (Actions #25)** | ✗ Building dataset (21/21) 후 ~2분 렉 → 크래시 |

**결론**: 크래시는 **EXE 전용** 이슈. Python에서는 재현되지 않음.

---

## 2. 디버그 로그 (Python 재현 시)

```
[2026-03-09 13:45:49] 1. Dataset prep done
[2026-03-09 13:45:49] 2. Loading YOLO model
[2026-03-09 13:45:49] 2a. Model path
[2026-03-09 13:45:49] 2b. Model loaded
[2026-03-09 13:45:49] 3. Starting model.train() - workers=0 epochs=2
[2026-03-09 13:46:17] 4. model.train() completed
```

Python에서는 3→4 구간이 ~28초. EXE에서는 이 구간에서 2분 후 크래시.

---

## 3. 원인 후보 (우선순위)

### 후보 A: YOLO cache='ram' (가능성 높음)

| 근거 | 내용 |
|------|------|
| Ultralytics/YOLOv5 #10276 | cache 시 RAM 할당 실패로 크래시 보고 |
| YOLO 기본값 | cache=True → cache='ram' 사용 |
| EXE 환경 | 메모리 레이아웃·할당이 일반 Python과 다름 |

**조치**: EXE에서 `cache='disk'` 사용.

### 후보 B: matplotlib graph poll (가능성 있음)

| 근거 | 내용 |
|------|------|
| 2분 시점 | 첫 epoch 완료 후 results.csv 생성 |
| graph poll | 1초마다 results.csv 확인, 있으면 _draw_graph() 호출 |
| PyInstaller + matplotlib | TkAgg 백엔드, 폰트 등에서 크래시 보고 다수 |

**조치**: graph poll의 matplotlib draw를 EXE에서 지연/비활성화 또는 예외 처리 강화.

### 후보 C: PyTorch/NumPy (가능성 있음)

| 근거 | 내용 |
|------|------|
| PyInstaller + PyTorch | DLL 로딩, 메모리 매핑 이슈 |
| 첫 배치 | model.train() 첫 forward 시 C 레벨 크래시 가능 |

---

## 4. 적용 권장 수정

1. **cache='disk' (EXE)**  
   - `train.py`: `sys.frozen`일 때 `cache='disk'`로 변경

2. **graph poll 방어**  
   - `_draw_graph` 호출을 try/except로 감싸고, 실패 시 해당 주기만 스킵

3. **진단용 EXE**  
   - `pin_detection_gui_debug.exe` (console=True)로 동일 시나리오 재현 후 콘솔 traceback 확인

---

## 5. 대용량 공장 시나리오 검증 (2026-03-09)

| 시나리오 | 결과 |
|----------|------|
| **5000×4000 픽셀, 20장, 40핀/장, 복잡 배경, roi_map** | ✓ 정상 완료 (frozen 모드) |

- `tools_scripts/generate_pin_test_data.py --large-factory`: 5000×4000 합성 데이터 + roi_map.json 생성
- `tools_scripts/test_exe_large_train.py`: sys.frozen=True, cache=disk, workers=0, 2 epochs
- 디버그 로그: Dataset prep → model.train() 완료 (~25초)
