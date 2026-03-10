# ROI 전달 경로 및 roi_map 검증

Last updated: 2025-03-05

## 목적

실제 공장 테스트 전 ROI 전달 경로와 roi_map 사용 흐름을 검증한다.

## ROI 전달 경로 요약

| 단계 | 경로 | 설명 |
|------|------|------|
| **생성** | `output_dir/roi_map.json` | ROI Editor 저장, generate_pin_test_data --large-factory |
| **로드 (학습)** | `output_dir.parent/roi_map.json` → `output_dir/roi_map.json` | dataset.py prepare_yolo_dataset_from_dirs |
| **추론** | 미사용 | inference.py는 roi_map 미사용 (전체 이미지 또는 masked prior) |

## 상세 흐름

### 1. roi_map.json 생성

- **ROI Editor** (`pin_detection/roi_editor.py`): 사용자가 Train 탭에서 "Edit ROI" 클릭 → 드래그로 ROI 지정 → "Save ROI map" → `output_dir/roi_map.json` 저장
- **합성 데이터** (`tools_scripts/generate_pin_test_data.py --large-factory`): 핀 bbox union + margin으로 자동 생성 → `out_dir/roi_map.json`

### 2. 학습 시 roi_map 로드 (dataset.py)

```python
# prepare_yolo_dataset_from_dirs(output_dir = dataset_dir)
# dataset_dir = output_dir/dataset (train.py에서 output_dir/dataset 전달)

roi_map_path = output_dir.parent / "roi_map.json"  # 1순위: pin_models/roi_map.json
if not roi_map_path.exists():
    roi_map_path = output_dir / "roi_map.json"    # 2순위: pin_models/dataset/roi_map.json
```

- **GUI 학습**: output_dir = 사용자 지정 (예: `pin_models/my_run`) → roi_map은 `pin_models/my_run/roi_map.json` (Edit ROI 저장 위치와 동일)
- **test_exe_large_train**: out_dir = `pin_models_exe_large_test`, roi_map 복사 → `pin_models_exe_large_test/roi_map.json` → dataset_dir = `pin_models_exe_large_test/dataset` → output_dir.parent = `pin_models_exe_large_test` ✓

### 3. _add_one_pair ROI 적용

- `roi_map[stem]` 존재 시: `roi` 파라미터로 전달 → unmasked/masked 모두 crop 후 annotation
- `roi_map` 없음 + `use_roi=True` + max(w,h)>2000: `extract_pin_roi(masked_path)` 자동 추출

### 4. 추론 (inference.py)

- **roi_map 미사용**: ROI crop 비활성화 (과거 ROI가 작아 핀 손실 발생)
- masked_path 제공 시: masked 이미지에서 annotation 직접 추출 (prior)
- 그 외: 전체 이미지 YOLO 추론

## 검증 결과

- `tests/test_gui_10_20.py`: test_dataset_roi_map_integration, test_dataset_roi_map_from_file, test_add_one_pair_with_roi
- `tools_scripts/test_exe_large_train.py`: roi_map 복사 후 학습 정상 완료
- `tools_scripts/verify_roi_path.py`: 경로 일관성 검증 스크립트

## Changelog

- 2025-03-05: 초안 작성, ROI 전달 경로 문서화
