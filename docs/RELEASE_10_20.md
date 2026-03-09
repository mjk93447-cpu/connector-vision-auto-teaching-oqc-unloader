# Release 10.20 — 수동 ROI 지정·imgsz 제한 해제

**날짜**: 2026-03-06  
**상태**: 완료

---

## 변경 사항 요약

### 1. imgsz 제한 해제
- **image size capped** 에러 제거
- imgsz Spinbox 복원 (320–4096, 사용자 자유 지정)
- 1280/640 강제 제한 제거

### 2. 수동 ROI 지정 (ROI Editor)
- **Edit ROI** 버튼: Train 탭에서 unmasked/masked/output 폴더 선택 후 사용
- 이미지별 드래그로 ROI 사각형 지정
- Prev/Next, Left/Right 키로 이미지 탐색
- roi_map.json 저장 (output_dir)
- YOLO 학습 시 roi_map 자동 적용

### 3. 버그 수정
- Epochs Spinbox: from_=3 (3 epochs 기본 지원)
- Graph poll: YOLO 실제 저장 경로 `runs/detect/<project>/pin_run` 사용
- ROI Editor: 지연 로드, 키보드 네비게이션

---

## 사용 방법

### ROI 지정
1. Train 탭에서 Unmasked/Masked/Output 폴더 선택
2. **Edit ROI** 클릭
3. 각 이미지에서 드래그로 ROI 사각형 지정
4. **Save ROI map** 클릭

### 학습
- roi_map.json이 output 폴더에 있으면 자동 적용
- imgsz는 320–4096 범위에서 자유 지정

---

## EXE 빌드

```bash
python -m PyInstaller pin_detection_gui_onedir.spec --noconfirm
```

출력: `dist/pin_detection_gui/` (one-dir)

---

## 테스트

```bash
python -m pytest tests/test_gui_10_20.py -v
```
