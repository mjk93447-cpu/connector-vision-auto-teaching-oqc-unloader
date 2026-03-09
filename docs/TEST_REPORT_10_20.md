# 10.20 기능 테스트 리포트

**날짜**: 2026-03-06  
**버전**: 10.20 (수동 ROI 지정·imgsz 제한 해제)

---

## Cycle 1: 기능 테스트 및 버그 탐색

### 테스트 환경
- Python 3.12.6, Windows
- test_data: pin_synthetic, pin_large_5496x3672

### 자동 테스트 결과 (pytest)
| 테스트 | 결과 |
|--------|------|
| test_roi_editor_load_save | ✓ PASS |
| test_dataset_roi_map_integration | ✓ PASS |
| test_dataset_roi_map_from_file | ✓ PASS |
| test_imgsz_no_cap | ✓ PASS |
| test_add_one_pair_with_roi | ✓ PASS |

### 수동 검증
- 학습(1 epoch, imgsz=320): ✓
- roi_map 연동 학습: ✓

### 발견·수정 이슈
| # | 이슈 | 조치 |
|---|------|------|
| 1 | Epochs Spinbox from_=10 → 3 epochs 기본값과 불일치 | from_=3으로 수정 |
| 2 | ROI Editor 첫 로드 시 canvas 크기 0 가능 | root.after(80, _load_image)로 지연 로드 |
| 3 | ROI Editor 키보드 네비게이션 없음 | Left/Right 키 바인딩 추가 |
| 4 | **Graph poll save_dir 오류**: GUI가 output_dir/pin_run 사용, YOLO는 runs/detect/<project>/pin_run 사용 | save_dir = Path("runs")/ "detect" / Path(out_dir).name / "pin_run" |

### GUI 요소 점검
- [x] Train 탭: Unmasked/Masked/Output 폴더 선택
- [x] Epochs, imgsz, Workers Spinbox
- [x] Apply suggested
- [x] Edit ROI 버튼
- [x] Start training / Stop training
- [x] Inference 탭
- [x] Help 탭

---

## Cycle 2: 수정 후 재테스트 — 완료

- 위 이슈 1–4 수정 반영
- pytest 5/5 통과
- 학습·roi_map 연동 smoke test 통과

---

## Cycle 3: 최종 검증 — 완료

### 전문가 관점 평가
| 항목 | 평가 | 비고 |
|------|------|------|
| **기능 완성도** | 양호 | Train/Inference/ROI Editor 정상 동작 |
| **사용성** | 양호 | imgsz 자유 지정, ROI 수동 지정, ETA 표시 |
| **안정성** | 양호 | imgsz cap 제거, graph poll 경로 수정 |
| **문서화** | 양호 | ROADMAP, ROI_MANUAL_STRATEGY, TEST_REPORT |

### 기준점
- [x] 핵심 기능(학습, 추론, ROI 지정) 동작
- [x] imgsz 1280/640 제한 제거
- [x] roi_map 연동
- [x] 자동 테스트 통과

---

## EXE 빌드 및 배포

- **로컬 빌드**: `build_pin_exe_onedir.bat` 또는 `python -m PyInstaller pin_detection_gui_onedir.spec --noconfirm`
- **GitHub Actions**: build-pin-exe 워크플로
- **배포**: Releases에 `pin-v10.20` 태그로 업로드
