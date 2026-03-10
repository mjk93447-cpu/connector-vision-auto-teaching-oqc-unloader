# EXE Action #33 테스트 결과 보고서

**테스트일**: 2026-03-10

---

## 1. 수정 사항 요약

| # | 문제 | 수정 내용 |
|---|------|----------|
| 1 | Train epoch 로그 타이밍에서 크래시 | train.py: EXE에서 imgsz > 1920 시 1920으로 cap (OOM 방지) |
| 2 | 브러쉬로 핀 지정 어려움 | roi_editor: Square 기본 모드, SQUARE_SIZE 8×8 |
| 3 | 초록 마커 YOLO 혼동 | TARGET_MARKER_RGB=(255,0,0), annotation RED 우선 |
| 4 | 마스킹 지우기 없음 | Erase 모드 (기존 구현 확인) |
| 5 | ROI·타겟 YOLO 최우선 | roi.py extract_pin_roi RED 우선 |
| 6 | Select folder first | _resolve_synthetic_paths 경로 검색 강화 |

---

## 2. 테스트 결과 표

| 항목 | 결과 | 비고 |
|------|------|------|
| pytest (45개) | ✓ | 통과 |
| test_exe_stdout_fix | ✓ | None stdout/stderr |
| repro_exe_train_crash | ✓ | frozen, workers=0, cache=disk |
| test_exe_large_train | ✓ | pin_large_factory, roi_map |
| test_gui_auto_fill | ✓ | 시작 시 경로 자동 채움 |
| test_load_test_data_paths | ✓ | 합성 데이터 경로 해석 |

---

## 3. 합성 테스트 데이터 경로 (고정)

| 경로 | 용도 |
|------|------|
| `test_data/pin_synthetic/train/unmasked`, `.../masked` | 기본 (640×480) |
| `test_data/pin_large_factory/unmasked`, `.../masked` | 대용량 (5000×4000) |

생성: `python tools_scripts/generate_pin_test_data.py --output-dir test_data/pin_large_factory --large-factory --large-factory-n 10`

---

## 4. EXE 빌드 판단

| 조건 | 상태 |
|------|------|
| 단위 테스트 통과 | ✓ |
| EXE 시뮬레이션 통과 | ✓ |
| GUI 자동 채움 검증 | ✓ |
| ROADMAP 대조 | ✓ |

**결론**: GitHub 커밋 및 EXE 아티팩트 빌드 진행 가능.
