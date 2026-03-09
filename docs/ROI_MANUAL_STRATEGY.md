# 수동 ROI 지정·imgsz 제한 해제 전략

**작성일**: 2026-03-06  
**근거**: 10.19 최신 아티팩트 테스트 결과

---

## 1. 이슈 요약

| # | 증상 | 요구 |
|---|------|------|
| 1 | ROI 자동 추출·크기 제한 계속 문제 | 사용자가 이미지별로 드래그로 ROI 사각형 지정, 연속 지정, 저장 후 YOLO 학습 시 정확히 사용 |
| 2 | image size capped 에러 (EXE) | 1280/640 제한·강제 제거, 사용자 지정 ROI·박스 영역만 사용 |

---

## 2. 기존 코드·문서 호환성 검토

### 2.1 충돌·변경 대상

| 항목 | 현재 | 변경 후 |
|------|------|----------|
| **ROADMAP 10.18** | ROI 비활성화, 전체 이미지 | 사용자 수동 ROI (이미지별) |
| **10.19 imgsz** | 640 고정, Spinbox 제거 | imgsz 사용자 지정, cap 제거 |
| **dataset.py** | use_roi, extract_pin_roi (자동) | roi_map (사용자 지정) |
| **gui.py** | imgsz 640 고정, capped 1280 | imgsz Spinbox 복원, cap 제거 |
| **train.py** | imgsz 640 기본 | imgsz 사용자 지정, cap 없음 |

### 2.2 유지·호환

| 항목 | 비고 |
|------|------|
| masked/unmasked 쌍 | 어노테이션은 masked에서 추출 (유지) |
| prepare_yolo_dataset_from_dirs | roi_map 파라미터 추가, 기존 use_roi와 병행 |
| geometry_refinement | 추론 시 masked prior 유지 |
| CLI | --roi-map 경로 추가 (선택) |

### 2.3 데이터 형식

**roi_map.json** (예시):
```json
{
  "01": [100, 100, 900, 500],
  "02": [150, 80, 1200, 600],
  "02_masked": [150, 80, 1200, 600]
}
```
- 키: `unmasked` 파일 stem (또는 `masked` stem, 둘 다 매핑)
- 값: `[x1, y1, x2, y2]` 픽셀 좌표

---

## 3. 구현 전략

### Phase A: imgsz 제한 제거

| 작업 | 파일 | 내용 |
|------|------|------|
| A1 | gui.py | imgsz Spinbox 복원, cap 제거, _imgsz_max 삭제 |
| A2 | gui.py | "image size capped" messagebox 제거 |
| A3 | dataset.py | analyze_dataset_for_training imgsz 1280/640 제한 제거 |
| A4 | train.py | imgsz cap 없음 (유지) |

### Phase B: ROI Editor GUI

| 작업 | 파일 | 내용 |
|------|------|------|
| B1 | roi_editor.py (신규) | 이미지 표시, 마우스 드래그로 사각형, 좌표 저장 |
| B2 | roi_editor | 이미지 순차 탐색 (Prev/Next), 저장/취소 |
| B3 | gui.py | "Edit ROI" 버튼 → ROI Editor 열기 |
| B4 | roi_map 저장 | dataset_dir/roi_map.json 또는 output_dir/roi_map.json |

### Phase C: Dataset·Train에 roi_map 연동

| 작업 | 파일 | 내용 |
|------|------|------|
| C1 | dataset.py | _add_one_pair(..., roi: tuple | None) |
| C2 | dataset.py | prepare_yolo_dataset_from_dirs(..., roi_map: dict | None) |
| C3 | roi_map 사용 | stem에 해당하는 roi가 있으면 crop, 없으면 전체 이미지 |
| C4 | train.py | roi_map 전달 (CLI/GUI) |

### Phase D: 문서 정리

| 작업 | 내용 |
|------|------|
| D1 | ROADMAP 10.20 | 수동 ROI·imgsz 제한 해제 반영 |
| D2 | ROI_OFF_GEOMETRY_REFINEMENT_PLAN | 10.18 ROI off → user ROI로 변경 |

---

## 4. 구현 순서

1. **Phase A**: imgsz cap 제거 (즉시 적용)
2. **Phase B**: ROI Editor GUI
3. **Phase C**: dataset·train roi_map 연동
4. **Phase D**: 문서 업데이트

---

## 5. ROI Editor 상세 UX

1. **진입**: Train 탭에서 unmasked·masked 폴더 선택 후 "Edit ROI" 클릭
2. **화면**: 현재 이미지 표시, 마우스 드래그로 사각형 그리기
3. **네비게이션**: Prev / Next / 이미지 번호 (1/20)
4. **저장**: "Save ROI" → roi_map.json 저장
5. **학습**: "Start training" 시 roi_map 사용 (있으면)
