# Pin Detection EXE GUI — 정밀 분석

## 1. 구성 요소

| 구성 | 파일 | 역할 |
|------|------|------|
| 진입점 | run_pin_gui.py | YOLO_OFFLINE 설정, gui.main() 호출 |
| GUI | pin_detection/gui.py | PinDetectionGUI, Train/Inference/Help 탭 |
| 빌드 | pin_detection_gui.spec | PyInstaller, yolo26n.pt 번들, excludes |

---

## 2. Train 탭 — 요소별 분석

| 요소 | 현재 동작 | 이슈 |
|------|------|------|
| **Unmasked folder** | Browse → 경로 선택 | ✓ |
| **Masked folder** | Browse → 경로 선택 | ✓ |
| **Excel folder** | Browse → 경로 선택 | **미사용**: train_pin_model에 전달 안 됨 |
| **Output folder** | 기본 pin_models | ✓ |
| **Epochs** | Spinbox 10–500 | ✓ |
| **Image size** | Spinbox 320–1280 | ✓, imgsz 초과 시 1280 클램프 |
| **Workers** | Spinbox 0–16 | ✓ |
| **CPU** | os.cpu_count() 표시 | ✓ |
| **Dataset** | N images, W×H | unmasked 폴더 스캔 후 표시 |
| **Est. time** | 공식 기반 | ✓ |
| **Progress** | indeterminate | 학습 중 스피너 |
| **Training metrics** | matplotlib, results.csv 폴링 1초 | ✓ |
| **Start training** | 버튼 | **Stop 버튼 없음** |

### 학습 흐름
1. prepare_yolo_dataset_from_dirs → data.yaml
2. train_pin_model(epochs, imgsz, workers)
3. 완료 시 model_path에 best.pt 설정

---

## 3. Inference 탭 — 요소별 분석

| 요소 | 현재 동작 | 이슈 |
|------|------|------|
| **Input image** | Browse → 단일 파일 | ✓ |
| **Model (.pt)** | Browse 또는 Train 완료 시 자동 설정 | ✓ |
| **Excel format ref** | Browse → 출력 형식 참조 | ✓ |
| **Run inference** | 버튼 | ✓ |

### 추론 파라미터 (GUI 미노출)
- conf_threshold: **0.25 고정**
- use_geometry_refinement: **True 고정**
- cap_precision: **True 고정**
- masked_path: **대형 이미지 ROI용 — GUI에서 미지원**

### 출력
- `{stem}_masked.png`: 입력 이미지와 동일 폴더
- `result.xlsx`: 입력 이미지와 동일 폴더
- **출력 폴더 선택 불가**

---

## 4. Help 탭

- 정적 텍스트, 스크롤
- 검색 없음

---

## 5. 공통/기타

| 항목 | 현재 | 이슈 |
|------|------|------|
| **창 크기** | 720×620, min 500×500 | ✓ |
| **DPI** | tk scaling 1.2 | 고해상도에서 다를 수 있음 |
| **스레드** | 학습·추론 daemon Thread | ✓ |
| **예외** | messagebox.showerror | ✓ |
| **상태바** | status_var | ✓ |

---

## 6. Excel I/O

| 기능 | 현재 | 이슈 |
|------|------|------|
| **헤더 추론** | 위핀, 아래핀, OK, NG, 간격 등 | 영어 전용 헤더 미지원 |
| **출력** | format_ref 있으면 동일 구조 | ✓ |

---

## 7. 개선 전략 요약

| 우선순위 | 개선 | 목표 | 상태 |
|----------|------|------|------|
| P1 | Conf threshold UI | Recall 100% 달성용 조정 | ✓ 완료 |
| P1 | 출력 폴더 선택 | Inference 결과 저장 위치 지정 | ✓ 완료 |
| P1 | 학습 중단 버튼 | Stop training (epoch end) | ✓ 완료 |
| P2 | Val split UI | Train 탭 val_split 0.01–0.5 | ✓ 완료 |
| P2 | Excel 폴더 활용 | Train 시 Excel 검증 또는 경고 | 예정 |
| P2 | 대형 이미지 ROI | Inference 시 masked 페어 선택 옵션 | 예정 |
| P3 | 영어 Excel 헤더 | upper_count, lower_count, judgment | ✓ 완료 |
| P3 | Help 검색 | Ctrl+F | 예정 |
| P3 | 최근 경로 기억 | 설정 저장/복원 | 예정 |
