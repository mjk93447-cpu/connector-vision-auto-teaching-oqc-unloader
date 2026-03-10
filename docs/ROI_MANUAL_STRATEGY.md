# 수동 ROI 지정 전략 (ROADMAP 10.20)

**작성일**: 2026-03-06  
**갱신일**: 2026-03-05 (10.0 원칙 반영)

---

## 1. 핵심 원칙 (ROADMAP 10.0)

| 원칙 | 설명 |
|------|------|
| **ROI = 분석 영역** | Edit ROI로 지정한 사각형이 학습·추론의 분석 대상 |
| **imgsz = ROI 박스 크기** | ROI 박스 max(w,h)에서만 파생. 수동 입력·min/max 제한 없음 |
| **단일 소스** | roi_map.json → crop·imgsz 자동 |

---

## 2. 이슈 요약

| # | 증상 | 요구 |
|---|------|------|
| 1 | ROI 자동 추출·크기 제한 계속 문제 | 사용자가 이미지별로 드래그로 ROI 사각형 지정, 저장 후 YOLO 학습에 정확히 사용 |
| 2 | image size capped 에러 (EXE) | imgsz = ROI 박스 크기만. 수동 입력·제한 없음 |

---

## 3. 구현 현황 (완료)

### 3.1 imgsz

| 항목 | 구현 |
|------|------|
| **gui.py** | imgsz Spinbox 제거, ROI 기반 읽기전용 라벨 |
| **dataset.py** | imgsz_from_roi_map, min/max 제한 없음 |
| **train.py** | load_roi_map_and_imgsz → imgsz 사용 |

### 3.2 유지·호환

| 항목 | 비고 |
|------|------|
| masked/unmasked 쌍 | 어노테이션은 masked에서 추출 (유지) |
| prepare_yolo_dataset_from_dirs | roi_map 파라미터, output_dir.parent/roi_map.json 자동 로드 |
| geometry_refinement | 추론 시 masked prior 유지 |

### 3.3 데이터 형식

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

## 4. 구현 전략 (완료)

### Phase A: imgsz ROI 전용 ✓

| 작업 | 파일 | 내용 |
|------|------|------|
| A1 | gui.py | imgsz Spinbox 제거, ROI 기반 읽기전용 라벨 |
| A2 | dataset.py | imgsz_from_roi_map, min/max 제한 없음 |
| A3 | train.py | load_roi_map_and_imgsz → imgsz |

### Phase B: ROI Editor GUI ✓

| 작업 | 파일 | 내용 |
|------|------|------|
| B1 | roi_editor.py (신규) | 이미지 표시, 마우스 드래그로 사각형, 좌표 저장 |
| B2 | roi_editor | 이미지 순차 탐색 (Prev/Next), 저장/취소 |
| B3 | gui.py | "Edit ROI" 버튼 → ROI Editor 열기 |
| B4 | roi_map 저장 | dataset_dir/roi_map.json 또는 output_dir/roi_map.json |

### Phase C: Dataset·Train에 roi_map 연동 ✓

| 작업 | 파일 | 내용 |
|------|------|------|
| C1 | dataset.py | _add_one_pair(..., roi: tuple | None) |
| C2 | dataset.py | prepare_yolo_dataset_from_dirs(..., roi_map: dict | None) |
| C3 | roi_map 사용 | stem에 해당하는 roi가 있으면 crop, 없으면 use_roi 시 자동 ROI |
| C4 | train.py | roi_map 자동 로드, imgsz 파생 |

---

## 5. ROI Editor 상세 UX

1. **진입**: Train 탭에서 unmasked·masked 폴더 선택 후 "Edit ROI" 클릭
2. **화면**: 현재 이미지 표시, 마우스 드래그로 사각형 그리기
3. **네비게이션**: Prev / Next / 이미지 번호 (1/20)
4. **저장**: "Save ROI" → roi_map.json 저장
5. **학습**: "Start training" 시 roi_map 사용 (있으면)
