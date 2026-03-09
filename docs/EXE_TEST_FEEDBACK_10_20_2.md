# EXE 테스트 피드백 — Action #22 / 10.20.2

**날짜**: 2026-03-09  
**대상**: pin_detection_gui_windows 아티팩트  
**반영**: 2026-03-09 (ROADMAP 10.20.2 EXE 테스트 피드백)

---

## 1. 이슈 요약

| # | 증상 | 우선순위 |
|---|------|----------|
| 1 | Train 탭 불필요/구식 UI 요소 | P1 |
| 2 | Edit ROI: masked만 표시, unmasked/masked 쌍·YOLO 지도데이터 미적용 | P1 |
| 3 | Preparing dataset 10분, P=0 R=0, 학습 최적화 필요 | P1 |

---

## 2. 이슈별 솔루션

### Issue 1: Train 탭 UI 정리

| 항목 | 현재 | 조치 |
|------|------|------|
| Excel file (optional) | Train 탭에 있음, 학습에 미사용 | Train 탭에서 제거 (Inference에 excel_format 있음) |
| 라벨 정리 | "(YOLO input size, no cap)" 등 | 최신 상황에 맞게 간결화 |
| Suggested/Apply suggested | analyze_dataset 백그라운드 | 유지 (필요) |
| CPU/Dataset/Est. time | 유지 | 유지 |

### Issue 2: Edit ROI — unmasked/masked 쌍, YOLO 지도데이터

**현재**: masked 이미지만 표시, ROI 드래그

**목표**:
- 파일명 기준 unmasked/masked 쌍 탐색
- **Unmasked**: YOLO 입력 (어떤 영역을 찾을지)
- **Masked**: 지도데이터 (어떻게 마스킹할지 = annotation)
- ROI: unmasked·masked 동일 좌표 적용
- YOLO: images = unmasked(crop), labels = masked에서 추출(crop 후)

**구현**:
1. ROI Editor: unmasked/masked 탭 또는 좌우 분할로 둘 다 표시
2. ROI는 unmasked 기준으로 드래그 (masked와 동일 좌표)
3. dataset: unmasked crop → images, masked crop → annotations (기존 로직 유지)
4. roi_map stem = unmasked stem

### Issue 3: Preparing 10분, P=0 R=0

**원인 추정**:
- 대형 이미지(5496×3672) 전처리 지연
- annotation 품질 (녹색 마스크 추출)
- YOLO 기본 설정 (epochs 3, lr 등)

**솔루션**:
1. **데이터 전처리**: 대형 이미지 리사이즈 또는 ROI 필수, 병렬 처리
2. **로그**: "Preparing dataset..." 직후 "Building dataset (1/20)..." 등 진행 표시
3. **YOLO 최적화**: epochs↑, lr 조정, mosaic=0.5(소량 데이터), imgsz=640 고수
4. **Annotation 검증**: 녹색 마스크 임계값, min_area 조정

---

## 3. 구현 완료 (2026-03-09)

| 이슈 | 구현 내용 |
|------|-----------|
| 1 | Train 탭 Excel 제거, imgsz 라벨 "(320–4096)" |
| 2 | ROI Editor: unmasked/masked 좌우 분할, ROI unmasked 기준, stem=unmasked |
| 3 | use_roi=True 기본, epochs 100 기본, on_progress "Building dataset (n/N)", analyze epochs 100 |
