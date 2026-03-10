# EXE 아티팩트 #33 테스트 이슈 및 개발 계획

**분석일**: 2026-03-10

---

## 1. 발견된 문제

| # | 증상 | 우선순위 |
|---|------|----------|
| 1 | Train: 2쌍, imgsz 5504, epochs 100, workers 4 → Building dataset OK, epoch 로그 타이밍에서 렉 후 크래시 | P1 |
| 2 | Edit ROI: 브러쉬로 핀 지정이 어려움. 작은 스퀘어를 핀마다 수십 개 지정 가능해야 함 | P1 |
| 3 | 초록색 마커가 원본 이미지와 겹쳐 YOLO 학습 혼동. 빨간색 등 구분 색으로 변경 필요 | P1 |
| 4 | 잘못 지정한 마스킹 지우기(erase) 기능 없음 | P1 |
| 5 | ROI·타겟 마스킹이 YOLO에서 최우선 활용되도록 코드 전반 수정 | P1 |
| 6 | "Select folder first" 에러 반복 발생 | P2 |

---

## 2. 합성 테스트 데이터 경로 (고정)

| 경로 | 용도 | 생성 명령 |
|------|------|----------|
| `test_data/pin_synthetic/train/unmasked`, `.../masked` | 기본 학습용 (640×480) | `python tools_scripts/generate_pin_test_data.py` |
| `test_data/pin_large_factory/unmasked`, `.../masked` | 대용량·고화질 (5000×4000) | `python tools_scripts/generate_pin_test_data.py --output-dir test_data/pin_large_factory --large-factory --large-factory-n 10` |

**EXE 테스트 시 "Select folder first" 방지**:
1. EXE를 **프로젝트 루트**에서 실행 (test_data와 같은 디렉터리). 또는 `run_exe_test.bat` 사용
2. Train 탭 **"Load test data"** 클릭 → unmasked/masked/output 자동 채움
3. 없으면: `python tools_scripts/generate_pin_test_data.py --output-dir test_data/pin_large_factory --large-factory --large-factory-n 10` 실행 후 1–2 반복

---

## 3. 개발 계획 (단계별)

### Phase A: EXE 크래시 수정 (문제 1)

| 작업 | 내용 |
|------|------|
| A1 | train.py: EXE에서 workers 강제 0 (기존), cache='disk' (기존) 확인 |
| A2 | gui.py: _draw_graph를 try/except로 감싸고, frozen 시 matplotlib draw 지연/방어 |
| A3 | graph poll: results.csv 첫 생성 시점(epoch 1 완료)에서 크래시 가능 → draw 실패 시 해당 주기만 스킵 |

### Phase B: ROI Editor 강화 (문제 2, 3, 4)

| 작업 | 내용 |
|------|------|
| B1 | **스퀘어 도구**: 작은 사각형(예: 8×8~16×16) 클릭으로 핀마다 개별 지정. 수십 개 가능 |
| B2 | **색상 변경**: 타겟 마스킹을 초록→빨간색 (255,0,0). YOLO 학습 시 원본 초록과 구분 |
| B3 | **지우기**: Erase 모드로 잘못 지정한 마스킹 제거 (우클릭 또는 별도 버튼) |
| B4 | 브러쉬 유지(선택), 스퀘어가 기본·주력 도구 |

### Phase C: 어노테이션·파이프라인 (문제 5)

| 작업 | 내용 |
|------|------|
| C1 | annotation.py: RED 마스크 추출 추가. GREEN(기존)과 RED 모두 지원 (하위 호환) |
| C2 | dataset.py: masked 이미지에서 RED 우선 추출, 없으면 GREEN fallback |
| C3 | inference.py: masked prior에서 RED 우선 |
| C4 | generate_pin_test_data.py: --red-markers 옵션 (새 합성 데이터용) |

### Phase D: Select folder first (문제 6)

| 작업 | 내용 |
|------|------|
| D1 | output_dir 기본값 "pin_models" 유지, 비어 있으면 자동 채움 |
| D2 | Apply suggested / Edit ROI / Train 클릭 시: unmasked, masked, output 모두 검증, 누락 시 구체적 메시지 |
| D3 | "Select folder first" → "Select unmasked folder, masked folder, and output folder first." 등으로 명확화 |

---

## 4. 구현 순서

1. **A** (크래시) → 2. **D** (폴더 에러) → 3. **B** (ROI Editor) → 4. **C** (어노테이션)
5. 합성 데이터 생성 → 6. EXE 빌드 → 7. 실제 EXE 테스트 → 8. 결과 표 정리 → 9. 커밋/빌드 판단

---

## 5. 테스트 결과 표

| 항목 | 결과 | 비고 |
|------|------|------|
| EXE graph poll matplotlib | ✓ | EXE에서 graph 비활성화 (frozen 시 skip) |
| Edit ROI 스퀘어·빨간 마스킹 | ✓ | Square, Brush, Erase 모드, TARGET_MARKER_RGB |
| Edit ROI 지우기 | ✓ | Erase 모드, unmasked 복원 |
| Select folder 에러 | ✓ | **GUI 시작 시 test_data 자동 채움** (에이전트 테스트 시 폴더 미지정 방지) |
| annotation RED 우선 | ✓ | extract_red_mask, masked_array_to_annotations |
| generate_pin_test_data --red-markers | ✓ | bbox_to_red_region |
| pytest (45개) | ✓ | 통과 |
| repro_exe_train_crash | ✓ | frozen, workers=0, cache=disk |
| test_exe_large_train | ✓ | pin_large_factory, roi_map |
| test_gui_auto_fill | ✓ | 시작 시 경로 자동 채움 검증 |
