# 중장기 개발 계획 — Connector Vision Auto Teaching

## 1. 개요

| 항목 | 내용 |
|------|------|
| **1차 목표** | 커넥터 핀 자동 마스킹 — 오프라인 Windows EXE |
| **모델** | YOLO26 (Ultralytics) |
| **범위** | 독립 실행형, OQC 모듈 연동 제외 |

---

## 2. 마일스톤

| 단계 | 목표 | 산출물 |
|------|------|--------|
| **Phase 1** | 프로젝트 셋업, YOLO26 연동 | Python 패키지, 학습/추론 CLI |
| **Phase 2** | 학습 파이프라인 | 마스킹 전/후 → YOLO 어노테이션 생성 → 학습 |
| **Phase 3** | 추론 파이프라인 | 이미지 → 핀 탐지 → 마스킹 이미지, 엑셀 출력 |
| **Phase 4** | GUI·EXE 패키징 | Windows EXE, 사용자 워크플로 |

---

## 3. Phase 1 — 프로젝트 셋업

### 3.1 작업

- [x] `pin_detection/` 모듈 생성
- [x] GUI (tkinter): 학습/추론 탭, 폴더 업로드
- [x] YOLO26(ultralytics) 의존성 추가 (`requirements-pin.txt`)
- [x] 학습 CLI: 마스킹 전/후 이미지 + 엑셀 → 모델 학습
- [x] 추론 CLI: 이미지 → 마스킹 이미지 + 엑셀
- [x] openpyxl로 엑셀 입출력

### 3.3 사용법

```bash
pip install -r requirements-pin.txt

# GUI 실행
python tools.py pin gui

# 학습 (10쌍)
python tools.py pin train --unmasked-dir ./unmasked --masked-dir ./masked

# 추론
python tools.py pin inference --model pin_models/.../best.pt --image new.jpg
```

### 3.2 의존성

```
ultralytics>=8.3  # YOLO26
openpyxl          # Excel
```

---

## 4. Phase 2 — 학습 파이프라인

### 4.1 핵심 로직

1. **마스킹 전/후 비교**: 파일명 매칭, 쌍 개수 검증
2. **마스킹 후 이미지**에서 초록색 영역 추출 → 클러스터 → bbox
3. **마스킹 전 이미지**에 bbox 적용 → YOLO 형식 어노테이션
4. YOLO26 학습 (class: pin, mosaic=0.5, copy_paste=0.1)

### 4.2 위/아래 구분

- Y 좌표 기준: 상단 50% → 위핀, 하단 50% → 아래핀
- 또는 여백(검은 영역) 감지로 구분선 추정

---

## 5. Phase 3 — 추론 파이프라인

### 5.1 출력

- 마스킹 이미지: 탐지된 핀 위치에 초록색 점
- 엑셀: 학습 시 인식한 형식으로 핀 개수, OK/NG, 좌우간격

### 5.2 좌우간격(mm)

- `mm_per_pixel = 0.5 / (핀 bbox 너비 픽셀)`
- 좌우 인접 핀 간 픽셀 거리 × mm_per_pixel

---

## 6. Phase 4 — GUI·EXE

### 6.1 GUI 기능

| 기능 | 설명 |
|------|------|
| **모드 선택** | 학습 모드 / 추론 모드 |
| **학습 모드** | 마스킹 전 이미지 폴더, 마스킹 후 이미지 폴더, 엑셀 폴더(또는 파일) 업로드 |
| **학습 실행** | YOLO26 로컬 학습, 진행률 표시 |
| **추론 모드** | 학습된 모델 선택, 이미지 선택 → 마스킹 이미지·엑셀 출력 |

### 6.2 로컬 학습 과정 (구체화)

1. 사용자가 마스킹 전/후 폴더, 엑셀 폴더 선택
2. 마스킹 전/후 이미지 쌍 비교·검증 (파일명 매칭, 개수 확인)
3. 마스킹 후 이미지에서 초록색 영역 추출 → bbox → YOLO 어노테이션
4. YOLO26 학습 (epochs, imgsz 설정 가능)
5. 학습 완료 시 모델 저장 경로 표시

### 6.3 성능 지표 (우선순위)

| 지표 | 목표 | 비고 |
|------|------|------|
| **Recall** | 100% | 위 20개·아래 20개 중 **하나라도 놓치면 안 됨** |
| **Precision** | 100% | 20개보다 **더 감지해서도 안 됨** (과탐지 금지) |
| **영역 마스킹** | 정확 | 위치는 recall보다 덜 중요 |

### 6.4 EXE 패키징

- PyInstaller, tkinter 기반. `run_pin_gui.py` 진입점.
- GitHub Actions: `.github/workflows/build-pin-exe.yml`
- 아티팩트: push 시 `pin_detection_gui_windows`, 태그 `pin-v1` 시 Releases

---

## 7. 확인 사항 (확정)

| 항목 | 답변 |
|------|------|
| **Q1** | 마스킹 전 10개, 후 10개 → **총 10쌍**. 엑셀도 세트로 10개. |
| **Q2** | 제품 기종별 별도 학습 **불필요**. 핀 개수·크기 거의 동일, 간격·외곽만 약간 차이. 추가 티칭 없음에 중점. |
| **Q3** | 헤더 **고정**. 핀 개수도 동일, 양식 항상 동일. |
| **Q4** | 마스킹 전/후 **동일 촬영**. 마스킹만 적용(핀 영역 초록색). 해상도·크기·위치 동일. |

---

## 8. 개발 진행 원칙

- 확인 필요 사항은 문서에 기록하고, 합리적 가정으로 우선 진행.
- 모순·결함 발견 시 즉시 설명하고 질문하여 해소.
- ROADMAP은 검토 결과에 따라 반영·수정.

---

## 9. 개발 현황 분석 (2026-03-05)

### 9.1 완료 항목

- 데이터셋: 마스킹 전/후 쌍 매칭, unmasked/masked 동일 해상도 검증
- 학습/추론 파이프라인, GUI, CLI, 엑셀 입출력
- 자체 테스트: `tests/test_pin_dataset.py` (11개)

### 9.2 성능 지표 미적용

| 지표 | 목표 | 현황 |
|------|------|------|
| Recall | 100% | 측정·평가 없음 |
| Precision | 100% | cap_at_20_per_row만 적용, 실제 Precision 미측정 |
| 영역 마스킹 | 정확 | IoU/위치 정확도 미측정 |

### 9.3 상세 분석

- `docs/PIN_DETECTION_ANALYSIS.md` 참고

---

## 10. 중장기 필수 추가 개발

### 10.0 핵심 설계 원칙 (2026-03-05 정리)

**분석 영역·이미지 크기 일원화**:

| 원칙 | 설명 |
|------|------|
| **ROI = 분석 영역** | 사용자가 Edit ROI로 지정한 사각형이 학습·추론의 분석 대상. crop 영역 = 분석 영역. |
| **imgsz = ROI 박스 크기** | imgsz는 ROI 박스 max(w,h)에서만 파생. 수동 입력·min/max 제한 없음. |
| **단일 소스** | roi_map.json에 ROI가 있으면 그 크기 사용. 없으면 전체 이미지가 분석 영역(analyze_dataset fallback). |

**전략 이력**:
- 10.18 ROI OFF: 과거 실험용. 10.20 수동 ROI로 **대체됨**.
- 10.20 수동 ROI: 현재 **유일 기본 전략**. Edit ROI → roi_map.json → crop·imgsz 자동.

---

### 10.1 Phase 5 — 성능 평가 (필수)

| 작업 | 설명 | 산출물 |
|------|------|--------|
| **Recall/Precision 평가** | masked(GT) vs 추론 bbox IoU 매칭 | `pin eval` 또는 `tools.py pin eval` |
| **Validation split** | train 80% / val 20% 분리, data.yaml 수정 | 과적합 방지, 조기 종료 |
| **평가 스크립트** | `--model`, `--unmasked-dir`, `--masked-dir` → Recall, Precision, F1 출력 | CLI `pin eval` |

### 10.2 Phase 6 — 운영·품질 강화

| 작업 | 설명 | 산출물 |
|------|------|--------|
| **배치 추론** | `--image-dir` 지원, 폴더 단위 추론 | CLI `pin inference --image-dir` |
| **Confidence 튜닝** | Recall 100% 달성용 conf threshold 탐색 | 평가 스크립트 옵션 |
| **추론 속도 벤치마크** | 이미지당 ms, FPS 측정 | `pin benchmark` 또는 tools.py 연동 |

### 10.3 Phase 7 — 품질 검증 (권장)

| 작업 | 설명 |
|------|------|
| **Excel ↔ 어노테이션 검증** | 학습 시 Excel 핀 개수 vs bbox 개수 불일치 경고 |
| **위/아래 클래스 분리** | class 0: upper, class 1: lower (선택) |
| **모델 비교** | YOLO26n vs s/m Recall·Precision·속도 트레이드오프 |

### 10.4 우선순위 요약

1. **필수**: Recall/Precision 평가 스크립트, Validation split
2. **중요**: 배치 추론, conf 튜닝, 속도 벤치마크
3. **권장**: Excel 검증, 클래스 분리, 모델 비교

---

### 10.5 정확한 핀 감지 및 오탐지 방지 (중장기 전략)

FPC/FFC 커넥터(20p×2, 총 40핀) 흑백 공장 이미지 기반 테스트 결과를 반영한 추가 개발 계획.

| 작업 | 설명 | 목표 |
|------|------|------|
| **Recall 100% 달성** | conf threshold 스캔, max-dist 매칭 검증, FN 원인 분석 | 위·아래 각 20개 누락 없음 |
| **Precision 100% 달성** | dust/scratch 등 fake pin(노이즈)에 대한 오탐 방지 | 20개 초과 감지 금지 |
| **합성 데이터 강화** | FPC 스타일 직사각형 패드, 흑백, fake dot 다양화 | 실제 공장 이미지와 유사도 향상 |
| **후처리 규칙** | 상·하단 각 20개 고정 시 초과 감지 제거(cap_at_20_per_row) | 과탐지 억제 |
| **노이즈 학습** | fake pin 위치를 negative 샘플로 활용(선택) | 오탐지 억제 |
| **실제 공장 데이터 검증** | 합성 학습 모델 → 실제 흑백 이미지 평가 | domain gap 측정·보완 |

---

### 10.6 기하학적 제약 활용 — Precision/Recall 100% 달성 전략

**도메인 특성**: 핀 위치가 거의 일정함. 커넥터 모델이 달라져도 간격·핀 크기는 10~20% 정도만 변하고, **항상 위·아래 20개씩 수평 정렬**된 고정 패턴.

| 전략 | 설명 | 목표 |
|------|------|------|
| **고정 개수 제약** | 항상 20+20개 → 부족 시 보간, 초과 시 제거 | Recall↑ Precision↑ |
| **균일 간격 보간** | 탐지된 핀들의 간격으로 누락 슬롯 위치 추정 | FN 보정 |
| **그리드 스냅** | 20개 슬롯 그리드에 탐지 매칭, 빈 슬롯은 보간 | 위치 정규화 |
| **행 Y 밴드** | 상·하단 예상 y 범위 밖 탐지는 제거 | FP 억제 |
| **Confidence + 기하학** | 19개 탐지 시 1개 보간, 21개 시 최저 conf 제거 | 정확도 극대화 |

---

### 10.7 BMP/JPG 교차 페어링 (이미지 포맷 호환)

**배경**: 공장 데이터에서 unmasked는 BMP, masked는 JPG 등 서로 다른 포맷으로 저장되는 경우가 있음. 기존에는 동일 파일명·동일 확장자만 매칭하여 "Masked pair not found for 2.bmp" 오류 발생.

| 작업 | 설명 | 목표 |
|------|------|------|
| **확장자 무시 매칭** | unmasked `2.bmp` ↔ masked `2.jpg` 등 stem(기본명) 기준 매칭 | BMP/JPG/PNG 교차 페어링 |
| **매칭 우선순위** | 1) 동일명 2) stem_masked.suffix 3) stem + 다른 확장자 | 명시적 규칙 |
| **지원 포맷** | .bmp, .jpg, .jpeg, .png | IMG_EXTS와 일치 |
| **단위 테스트** | bmp↔jpg, jpg↔png 등 교차 케이스 검증 | 회귀 방지 |

---

### 10.8 완전 오프라인·학습시간 정확도 (필수)

**배경**: EXE는 공장 오프라인 Windows 환경에서 동작해야 하나, curl/GitHub 접속 시도가 발생. 학습 예상 2분인데 실제 10분+ 미완료.

| 문제 | 원인 | 해결책 |
|------|------|--------|
| **curl/GitHub 접속** | ultralytics: 모델(yolo26n.pt) 다운로드, pip 업데이트 체크, 폰트(Arial.ttf) 다운로드 | `YOLO_OFFLINE=true` 선설정, yolo26n.pt 번들, 폰트 로컬화 |
| **예상시간 부정확** | imgsz 대형 시 (imgsz/640)² 비례로 급증, CPU 전용·실측 미반영 | CPU 보정계수, 보수적 ETA 표시 |
| **학습 미완료** | 대형 imgsz 시 메모리·시간 폭증, workers>0이 Windows multiprocessing 이슈 | EXE에서 workers=0, batch 자동 스케일 |

| 작업 | 설명 | 목표 |
|------|------|------|
| **YOLO_OFFLINE** | `run_pin_gui.py` 최상단에서 `os.environ["YOLO_OFFLINE"]="true"` 설정 (ultralytics import 전) | 네트워크 호출 차단 |
| **yolo26n.pt 번들** | PyInstaller datas에 포함, `_MEIPASS` 경로로 로드 | 오프라인 학습 가능 |
| **imgsz** | ROI 박스 크기에서만 파생. GUI 수동 입력 없음 (10.0 원칙) | ROI=분석영역 일원화 |
| **예상시간 공식** | CPU 보정, imgsz² 스케일, 보수적 표시 | 사용자 기대에 부합 |
| **workers 기본값** | Windows에서 4 이하 권장. EXE: workers=0 강제 | 안정적 학습 |

---

### 10.9 대형 이미지 ROI 기반 학습 (5496×3672 대응)

**배경**: 5496×3672 등 대형 이미지를 전체 학습하면 연산·메모리 폭증. 핀은 이미지 일부에만 존재.

| 전략 | 설명 | 목표 |
|------|------|------|
| **1단계: 핀 영역 탐지** | masked 이미지의 초록 픽셀(핀 위치)에서 전체 bbox 계산 | 대략적 핀 클러스터 위치 |
| **2단계: ROI 설정** | bbox 주변에 margin(15~20%) 추가, 이미지 경계 클램프 | 넉넉한 ROI |
| **3단계: ROI crop 학습** | unmasked/masked를 ROI로 crop 후 YOLO 학습 | 연산량 대폭 감소 |
| **4단계: 추론 시 동일** | 입력 이미지에서 ROI 추정 또는 학습 시 ROI 정보 저장 후 crop 추론 | 일관된 파이프라인 |

| 작업 | 설명 | 목표 |
|------|------|------|
| **extract_pin_roi** | masked 이미지 → (x1,y1,x2,y2) ROI, margin_ratio=0.15 | `pin_detection/roi.py` |
| **prepare_* with ROI** | max(w,h)>2000 시 자동 ROI crop 후 dataset 생성 | 대형 이미지 지원 |
| **추론 ROI** | 학습 시 사용한 ROI 전략을 추론에 적용 (또는 전체 이미지 저해상도 스캔) | 5496×3672 추론 가능 |
| **5496×3672 벤치마크** | 20장 학습 시간 측정 | **실측: ~6.7분** (25 epochs, imgsz 640, ROI 적용, CPU) |

---

### 10.10 처리속도·예외처리·엣지케이스 강화

| 항목 | 설명 | 목표 |
|------|------|------|
| **처리속도** | 모델 캐싱, 배치 추론, conf 최적화 | 이미지당 목표 <2초 (CPU) |
| **핀 영역 밴드 필터** | y 범위 밖 탐지 제거 (상단 0.05~0.45, 하단 0.45~0.95) | 오탐지·이상치 제거 |
| **수평 범위 검증** | 보간 위치가 이미지 x [0.02, 0.98] 내인지 검증 | 잘못된 보간 방지 |
| **빈 행 처리** | 한 행 0개 탐지 시 반대 행 간격으로 템플릿 생성 | 극단적 FN 대응 |
| **이상치 제거** | 행 내 y 중앙에서 2σ 이상 이탈 탐지 제거 | 잘못된 탐지 필터 |
| **출력 범위 클램프** | (xc,yc,w,h) ∈ [0,1] 보장 | 안전한 후처리 |

---

### 10.11 ROI 검증 및 추론 ROI (완료)

**배경**: ROI crop 시 핀 누락 방지 검증, 대형 이미지 추론 시 ROI 적용으로 속도·일관성 확보.

| 작업 | 설명 | 목표 |
|------|------|------|
| **ROI 핀 보존 테스트** | test_roi_preserves_all_pins: 40핀 crop 전후 개수 일치 검증 | 회귀 방지 |
| **추론 ROI** | run_inference(masked_path=) 제공 시 max(w,h)>2000이면 ROI crop 후 추론, 좌표 역변환 | 5496×3672 추론 가속 |
| **평가 스크립트 ROI** | run_pin_experiment: masked_path 전달하여 대형 이미지 평가 시 ROI 자동 적용 | 일관된 파이프라인 |
| **validate_roi_pins** | tools_scripts/validate_roi_pins.py: 대형 데이터셋 ROI 검증 | 수동 검증용 |

**엣지 케이스 대응**:
- margin_ratio 0.15: 핀이 ROI 경계 근처일 경우 0.20 권장
- extract_green_mask: g>150, r<g-60 조건 — 흐린 초록 누락 시 threshold 완화 검토

---

### 10.11a EXE NoneType write 오류 해결 (2026-03-06) — 완료

**배경**: 오프라인 Windows EXE에서 학습 25분 진행 후 `'NoneType' object has no attribute 'write'` 오류로 중단.

**원인**: PyInstaller `console=False` 시 `sys.stdout`/`sys.stderr`가 `None`. Ultralytics YOLO가 progress 출력 시 크래시.

**해결** (상세: `docs/EXE_NONETYPE_FIX_PLAN.md`):
- `run_pin_gui.py` 최상단에서 `stdout`/`stderr`가 None이면 `open(os.devnull,'w')`로 대체
- GUI에 Training log 영역 추가: epoch별 Loss/P/R, ETA 표시

**검증** (2026-03-06):
- `build_pin_exe.bat` → `dist\pin_detection_gui.exe` 빌드 성공
- `test_exe_train.bat` → `--test-train` 3 epoch 학습 완료, `exe_test_ok.txt` 생성 확인
- NoneType write 오류 미발생, EXE 학습 정상 동작

---

### 10.12 EXE GUI 정밀 개선 (사용성·기능 강화)

**배경**: `docs/EXE_GUI_ANALYSIS.md` 기반. Train/Inference 탭 요소별 분석 후 개선 전략 수립.

| 우선순위 | 작업 | 설명 | 목표 |
|----------|------|------|------|
| **P1** | Conf threshold UI | Inference 탭 conf Spinbox (0.01–0.5) | ✓ 완료 |
| **P1** | 출력 폴더 선택 | Inference 결과 저장 경로 Browse | ✓ 완료 |
| **P1** | 학습 중단 버튼 | Stop training, stop_event 콜백 | ✓ 완료 |
| **P2** | Val split UI | Train 탭 val_split 0.01–0.5 Spinbox | ✓ 완료 |
| **P2** | Excel 폴더 활용 | Train 시 Excel vs bbox 개수 불일치 경고 | 예정 |
| **P2** | 대형 이미지 ROI | Inference 시 masked 페어 선택(옵션) | 예정 |
| **P3** | 영어 Excel 헤더 | upper_count, lower_count, judgment 등 infer | ✓ 완료 |
| **P3** | Help 검색 | Ctrl+F 또는 검색 필드 | 예정 |
| **P3** | 최근 경로 기억 | 설정 저장/복원 (JSON) | 예정 |

---

### 10.13 셀 ID 기반 페어링 및 단일 엑셀 다중 행 (완료)

**배경**: 공장 데이터 파일명 형식 `YYYYMMDD_HHMMSS_A2HD{cell_no}`. 20개 셀 데이터가 한 엑셀에 행별로 저장.

| 작업 | 설명 | 목표 |
|------|------|------|
| **extract_cell_id** | 파일명에서 A2HD+alphanumeric 추출 | `pin_detection/dataset.py` |
| **셀 ID 페어링** | _find_masked_pair: cell_id 우선 매칭, stem 폴백 | 마스킹 전후 셀별 연속 학습 |
| **단일 엑셀 다중 행** | load_excel_multi_row, find_row_index_by_cell_id | 20행(셀별) 형식 지원 |
| **Excel 행 업데이트** | write_result_excel(update_existing, cell_id) | 추론 결과를 해당 셀 행에 반영 |
| **GUI** | Excel folder → Excel file, Help 텍스트 갱신 | 사용자 가이드 반영 |

---

### 10.14 데이터셋 자동 최적화 (이미지 특성 스캔 → 세팅 추천)

**배경**: 데이터 선택 시 epochs·val_split·mosaic 등 추천. imgsz는 ROI에서만 파생(10.0 원칙).

| 전략 | 설명 | 목표 |
|------|------|------|
| **빠른 스캔** | 샘플 3~5장만 분석, 백그라운드 실행 | UI 반응성 |
| **imgsz** | roi_map 있으면 ROI 박스 크기. 없으면 전체 이미지 max(w,h) (분석영역=전체) | 10.0 원칙 준수 |
| **데이터량** | n < 15 → epochs 150 권장, n ≥ 20 → 100 | 과적합/과소적합 방지 |
| **자동 적용** | "Apply suggested" 버튼으로 epochs, val_split, mosaic 적용 | 사용자 편의 |

| 작업 | 설명 | 목표 |
|------|------|------|
| **analyze_dataset_for_training** | unmasked_dir, masked_dir, output_dir → imgsz(ROI/전체), epochs, val_split, note | `pin_detection/dataset.py` |
| **GUI "Apply suggested"** | 추천값 적용 버튼. imgsz는 ROI 기반 읽기전용 표시 | 원클릭 적용 |
| **스캔 결과 표시** | "imgsz:640 (from ROI), epochs:100" 등 요약 | 투명성 |

---

### 10.15 학습 속도 최적화 (20장 20분+ → 목표 5분 이내) — 완료

**배경**: 20장 학습, epochs 10, workers 8에도 20분 이상 소요. CPU 환경에서 속도 극대화 필요.

| 전략 | 설명 | 목표 | 상태 |
|------|------|------|------|
| **cache=True** | RAM 캐시, 디스크 I/O 제거 | ✓ |
| **batch=16** | step 수 최소화 | ✓ |
| **mosaic=0** | mosaic 비활성화 (기본) | epoch당 시간 감소 | ✓ |
| **rect=True** | rectangular training | ✓ |
| **plots=False** | 플롯 생성 생략 | ✓ |
| **ROI min** | 가로 30%, 세로 10% (커넥터 가로형) | ✓ |

**실측 (20장 5496×3672, ROI crop, CPU)**:
- 10 epochs, mosaic=0: ~3.7 min (219 sec)
- 25 epochs, mosaic=0: ~5.4 min
- mosaic=0.5: 유사 또는 약간 느림 (augmentation 오버헤드)

**ROI 제약**: `extract_pin_roi` — min_width_ratio=0.30, min_height_ratio=0.10 (커넥터 가로형)

**Mosaic 자동 제안** (`analyze_dataset_for_training`):
- ROI crop 또는 n_images ≥ 20 → mosaic=0.0 (속도 우선)
- 그 외 → mosaic=0.5 (품질 우선, 소량 데이터 증강)
- GUI "Apply suggested" 시 mosaic 값 자동 적용

---

### 10.16 최적화 루프 4 — 학습 속도·탐지 품질 균형 (2026-03-06)

**실측 결과**:

| 데이터셋 | 이미지 | Epochs | Mosaic | 학습 시간 | Recall (max_dist=30) |
|----------|--------|--------|--------|-----------|----------------------|
| pin_large_5496x3672 | 20 | 10 | 0 | ~3.7 min | 0% |
| pin_synthetic | 10 | 50 | 0.5 | ~3.7 min | 24.5% |

**분석**:
- pin_large: 10 epochs로는 수렴 부족, 50+ epochs 또는 mosaic=0.5 권장
- pin_synthetic: 50 epochs + mosaic=0.5 → Recall 24.5% (파이프라인 동작 확인)
- pin_large 0% Recall: domain 차이 또는 epoch 부족 가능성

**다음 루프 전략**:
1. pin_large 50 epochs, mosaic=0.5로 재학습 후 Recall 측정
2. imgsz=512 시도 (640 대비 속도 향상, 품질 trade-off)
3. val_split=0으로 소량 데이터 시 전체 학습

---

### 10.17 최적화 루프 5 — pin_large 50ep, geometry_refinement 영향 (2026-03-06)

**실측**:
- pin_large 50ep mosaic=0.5: 학습 ~4.9 min, val mAP50 0.099, R 0.59
- **평가 시 geometry_refinement**: `--no-geometry-refinement` 없으면 0% Recall (refinement가 20+20 그리드로 강제 배치해 GT와 매칭 실패)
- **평가 시 geometry_refinement 비활성화**: Recall 30%, Precision 36% (--iou 0 --max-dist 50)

**결론**:
- refine_to_fixed_grid는 20+20 고정 시 사용자 출력용으로 적합하나, 평가(IoU/center-distance 매칭) 시에는 raw 모델 출력으로 측정하는 것이 정확함
- `run_pin_experiment --no-geometry-refinement --max-dist 50` 으로 Recall/Precision 정확 측정

**imgsz=512 속도**:
- pin_large 10ep, imgsz=512: ~3.2 min (imgsz=640 대비 ~14% 단축)

---

### 10.18 ROI 비활성화·Geometry Refinement 개선 (2026-03-06) — 10.20으로 대체됨

**배경**: ROI가 핀 영역을 충분히 커버하지 못해 geometry refinement가 제대로 동작하지 않음. Geometry refinement는 모든 핀 정확 탐지에 필수.

**전략** (상세: `docs/ROI_OFF_GEOMETRY_REFINEMENT_PLAN.md`):

| Phase | 작업 | 목표 | 상태 |
|-------|------|------|------|
| **A** | ROI 비활성화 | 학습·추론 모두 전체 이미지 사용 | 10.20으로 대체 |
| **B** | epochs=3 | 학습 속도 극대화 (~2분 이내) | ✓ ~1.8 min |
| **C** | Geometry refinement 개선 | masked prior 활용 → 100% P/R | ✓ 유지 |
| **D** | 영역별 스캔 효율화 | 핀 클러스터 탐색 (필요 시) | 보류 |

**핵심**: masked_path 제공 시 masked에서 핀 위치 추출 → refinement 입력 → **100% Recall/Precision**

**대체**: 10.20 수동 ROI 지정이 현재 기본 전략. 사용자가 Edit ROI로 영역을 지정하면 그 영역이 분석 대상. 10.18의 "ROI OFF"는 더 이상 기본 옵션이 아님.

---

### 10.19 EXE 아티팩트 테스트 이슈 (2026-03-06) — 완료

**배경**: 최신 아티팩트 테스트 결과. 상세: `docs/EXE_ARTIFACT_ISSUES_STRATEGY.md`

| # | 증상 | 우선순위 | 대응 |
|---|------|----------|------|
| 1 | EXE 첫 실행 DLL 에러, 관리자 권한으로만 실행 | P1 | manifest asInvoker, one-dir 옵션, VC++ Redist 안내 |
| 2 | Masked folder 지정 시 ~2분 렉, 조작 불가 | P1 | analyze_dataset를 백그라운드 스레드로, _update_eta에서 제거 |
| 3 | imgsz 입력창 노출, 입력 시 극심한 렉 | P2 | imgsz Spinbox 제거, ROI 기반 읽기전용 표시 (10.0 원칙) |

**구현 순서** (2026-03-06 완료):
1. GUI: _update_eta에서 analyze 제거, ETA만 계산. Apply suggested 시에만 백그라운드 analyze ✓
2. GUI: imgsz Spinbox 제거, ROI 박스 크기 기반 읽기전용 라벨 ✓
3. EXE: spec uac_admin=False, one-dir 빌드 옵션 ✓
4. 문서: DOWNLOAD_EXE.md VC++ Redist 안내 ✓

---

### 10.20 수동 ROI 지정·imgsz 제한 해제 (2026-03-06) — 완료

**배경**: 10.19 최신 아티팩트 테스트. ROI 자동 추출·크기 제한 계속 문제. image size capped 에러 발생.

**요구**:
1. 학습 시 이미지별로 사용자가 드래그로 ROI 사각형 지정, 연속 지정, 저장 후 YOLO 학습에 정확히 사용
2. imgsz = ROI 박스 크기만 사용. 수동 입력·min/max 제한 없음 (10.0 원칙)

**전략** (상세: `docs/ROI_MANUAL_STRATEGY.md`):

| Phase | 작업 | 목표 | 상태 |
|-------|------|------|------|
| **A** | imgsz ROI 전용 | ROI 박스 크기에서만 파생, 수동 입력 제거 | ✓ |
| **B** | ROI Editor GUI | 이미지별 드래그로 ROI 지정, Prev/Next, roi_map.json 저장 | ✓ |
| **C** | roi_map 연동 | prepare_yolo_dataset_from_dirs(roi_map), _add_one_pair(roi) | ✓ |
| **D** | 문서 정리 | ROADMAP, ROI_OFF 계획 업데이트 | ✓ |

**구현 내용**:
1. **Phase A**: gui.py imgsz Spinbox 제거, ROI 기반 읽기전용 라벨. dataset.py imgsz_from_roi_map, min/max 제한 없음.
2. **Phase B**: `pin_detection/roi_editor.py` 신규. Train 탭 "Edit ROI" 버튼. 드래그로 ROI 지정, Prev/Next, roi_map.json 저장(output_dir).
3. **Phase C**: `_add_one_pair(..., roi=...)`, `prepare_yolo_dataset_from_dirs`에서 output_dir.parent/roi_map.json 자동 로드, stem별 roi 적용.
4. **Phase D**: ROADMAP 10.20 완료 반영.

**10.20.1 버그 수정** (테스트 사이클):
- Epochs Spinbox from_=3 (3 epochs 기본 지원)
- Graph poll save_dir: output_dir/pin_run 및 runs/detect/<name>/pin_run 후보 지원 (절대/상대 경로)
- ROI Editor: 지연 로드, Left/Right 키 바인딩
- run_pin_gui --test-train: pin_synthetic/train fallback

**10.20.2 강화 평가·개선** (V2 평가모델, 평균 95점 목표):
- Train/Edit ROI: output_dir·경로 존재 검증, 명확한 에러 메시지
- ROI Editor: 제목·Left/Right 키 안내
- Help 탭: ROI Editor 단계, imgsz=ROI 박스 크기(수동입력 없음) 안내
- 테스트: test_train_validation, test_edit_roi_validation, test_exe_stdout_fix

**10.20.2 EXE 테스트 피드백** (Action #22, 10.20.2, 상세: `docs/EXE_TEST_FEEDBACK_10_20_2.md`):

| 이슈 | 조치 | 상태 |
|------|------|------|
| **1. Train 탭 불필요 UI** | Excel 입력 제거 (학습 미사용), imgsz 라벨 ROI 기반 읽기전용 | ✓ |
| **2. Edit ROI masked만 표시** | unmasked/masked 좌우 분할 표시, ROI는 unmasked 기준 드래그, stem=unmasked | ✓ |
| **3. Preparing 10분, P=0 R=0** | use_roi=True(대형 이미지 자동 ROI), epochs 100 기본, on_progress 로그, Apply suggested epochs 100 | ✓ |

**YOLO EXE 검증** (2026-03-09):
- `tools_scripts/run_yolo_exe_validation.py`: 20 학습쌍 + 10 노이즈 테스트
- 10 epochs 학습 → 10/10 OK (40핀 정확 마스킹)
- 상세: `docs/YOLO_EXE_VALIDATION_REPORT.md`

**Action #24 EXE 크래시 수정** (Building dataset 21/21 → silent exit):

| 증상 | Building dataset (1/21)…(21/21) 출력 후 EXE 즉시 종료, 에러 팝업 없음 |
|------|----------------------------------------------------------------------|
| **원인** | PyInstaller EXE + YOLO workers>0 → Windows multiprocessing spawn 시 크래시 |
| **조치** | `sys.frozen` 시 workers=0 강제 (train.py) |
| **재현** | `python tools_scripts/repro_exe_train_crash.py` (21 이미지, frozen 시뮬레이션) |
| **상세** | `docs/EXE_ACTION24_CRASH_FIX.md` |

**Action #25 EXE 크래시 + 대용량 공장 시나리오 검증** (2026-03-09):

| 항목 | 내용 |
|------|------|
| **원인 후보** | cache='ram' (Ultralytics #10276), matplotlib graph poll |
| **조치** | EXE에서 cache='disk', workers=0, matplotlib 예외 처리 |
| **대용량 검증** | 5000×4000 픽셀, 20장, 40핀/장, roi_map, 복잡 배경 → frozen 모드 정상 완료 |
| **스크립트** | `generate_pin_test_data.py --large-factory`, `test_exe_large_train.py` |
| **상세** | `docs/EXE_CRASH_ANALYSIS_ACTION25.md` |

---

### 10.22 #31 테스트 결과 반영 및 추가 개발 (2026-03-09)

**#31 결과**:
1. ✓ EXE 아티팩트 빌드 성공, 크래시 해결
2. ✗ P/R 매우 낮음 (epochs 100에도 ~0.1 수준). 목표: Recall 99.9%, Precision 99%
3. ✗ conf=0.01로 해도 Upper:0, Lower:0 — 40핀 입력에 0개 탐지
4. 추정 원인: ROI 오수신 또는 **masked 십자형(+) 핀 마커** 미인식

**추가 개발 (완료)**:

| 작업 | 설명 | 산출물 |
|------|------|--------|
| **Masked 십자형 마커 인식** | extract_green_mask 완화, morphology dilation으로 thin cross 병합 | `annotation.py` |
| **결과물 날짜시간별 저장** | pin_results/YYYYMMDD_HHMMSS/ 구조 | `results_path.py`, gui, cli |

**annotation.py 개선**:
- GREEN_G_MIN 150→120, R/B diff 60→50 (십자형·색편차 허용)
- cluster_to_bbox: dilate_thin=True, min_area=2 (얇은 십자 병합)
- _dilate_for_thin_markers: scipy binary_dilation으로 cross arm 연결

**결과 경로**:
- 기본: `pin_results/YYYYMMDD_HHMMSS/` (프로젝트 루트)
- 사용자 지정 폴더 시: `지정폴더/YYYYMMDD_HHMMSS/`

**완료** (10.22 사전조치):
- ROI 전달 경로·roi_map 검증: docs/ROI_PATH_VERIFICATION.md, tools_scripts/verify_roi_path.py
- 합성 데이터 십자형 마커: generate_pin_test_data.py --cross-markers
- Debug exe 빌드 제외 (크래시 해결됨)

**향후 과제** (Recall 99.9% 달성):
- 실제 공장 masked 데이터(십자형)로 학습/추론 검증

---

### 10.23 Geometry Refinement 설계 정리 (2026-03-05)

**설계 원칙** (상세: `docs/GEOMETRY_REFINEMENT_DESIGN.md`):

| 단계 | 역할 |
|------|------|
| 1차 | YOLO — 최신 모델 성능으로 탐색 (primary) |
| 2차 | Masked prior — 십자 핀 마스킹으로 FN 보완 |
| 3차 | Geometry refinement — FN/FP 보조 보정 |

**Geometry refinement (보조)**:
- FN (< 20/행): 균일 간격 보간
- FP (> 20/행): confidence 상위 20개
- FP (엉뚱한 위치): spacing anomaly 검증 → 제거

**구현**: `inference.py` YOLO 우선 → masked 병합 → `geometry_refinement.refine_to_fixed_grid`

---

### 10.24 ROI 정밀화 (2026-03-05)

**배경**: 사이즈가 매우 크고 탐지 객체가 매우 작은 경우, 정확한 학습을 위해 ROI 설정 정밀화 필요. 위/아래 핀 조명 차이로 밝기가 다른 경우 분할 ROI·순차 마스킹 필요.

**1. 정밀 ROI 설정**

| 작업 | 설명 | 목표 |
|------|------|------|
| **이미지 확대** | 마우스 스크롤로 줌 인/아웃 | 대형 이미지에서 소형 핀 정밀 지정 |
| **사각형 도구** | 기존 ROI 드래그 유지 | 영역 지정 |
| **브러쉬 도구** | 작은 핀(connector pin) 객체를 마스킹 | YOLO에 정확히 어떤 객체를 찾을지 알림 |

**2. 분할 ROI·순차 마스킹**

| 작업 | 설명 | 목표 |
|------|------|------|
| **2분할 ROI** | 위핀 20개 / 아래핀 20개 영역 각각 지정 | 조명 차이 대응 |
| **순차 마스킹** | 위핀 20개 마스킹 → 아래핀 20개 마스킹 | 영역별 정밀 마스킹 |

**데이터 형식** (`roi_map.json` 확장):
- 단일: `"stem": [x1,y1,x2,y2]` (기존 호환)
- 분할: `"stem": {"upper":[x1,y1,x2,y2], "lower":[x1,y1,x2,y2]}`

**구현 순서**:
1. ROI Editor: 스크롤 줌 (MouseWheel)
2. ROI Editor: 브러쉬 도구 (마스킹 모드, 초록색 그리기)
3. ROI Editor: 분할 ROI 모드 (Upper ROI / Lower ROI)
4. ROI Editor: 순차 마스킹 (Upper 마스킹 → Lower 마스킹)
5. dataset.py: 분할 roi_map 지원 (union bbox로 crop 또는 각각 crop)
