# Pin Detection 실험 테스트 리포트

Last updated: 2026-03-06

## 1. 개요

로드맵에 따라 pin_detection 모듈의 학습·추론·GUI 기능을 합성 테스트 데이터로 검증하고, 성능 지표를 측정하였다.

---

## 2. 테스트 환경

| 항목 | 값 |
|------|-----|
| OS | Windows 10 |
| Python | 3.12.6 |
| CPU | 11th Gen Intel Core i7-1165G7 @ 2.80GHz |
| 모델 | YOLO26n (Ultralytics 8.4.19) |
| GPU | 미사용 (CPU 추론) |

---

## 3. 테스트 데이터

### 3.1 합성 데이터 (FPC/FFC 커넥터 모방)

- **참조**: [20p FPC/FFC adapter board](https://hubtronics.in/image/cache/catalog/sagar/20p-fpc-ffc-adapter-board-550x550.jpg)
- **스크립트**: `tools_scripts/generate_pin_test_data.py`
- **특징**:
  | 항목 | 설정 |
  |------|------|
  | 배경 | 검은색 (흑백 공장 이미지 시뮬레이션) |
  | 핀 형태 | 직사각형 패드 (12×4 px, 가로>세로) |
  | 핀 색상 | 밝은 회색 (210-255, 흑백) |
  | 레이아웃 | 위 20핀 + 아래 20핀, 수평 정렬, 마주보는 형태 |
  | 블러 | 25% 확률 GaussianBlur |
  | 노이즈 | 이미지당 6개 가짜 점 (dust/scratch 시뮬레이션) |
  | 마스킹 | 초록색은 **실제 핀 위치에만** (가짜 점에는 없음) |

- **형식**: 640×480, 위 20핀 + 아래 20핀 (총 40핀/이미지)
- **쌍 수**: 10쌍 (unmasked + masked)
- **meta.json**: fake_centers 저장 (FP-on-fake 평가용)

```bash
python tools_scripts/generate_pin_test_data.py --n-pairs 10 --blur-prob 0.25 --n-fake-pins 6
```

### 3.2 데이터 검증

- unmasked/masked 동일 해상도 검증 통과
- `validate_pair_dimensions` 정상 동작
- 초록색 영역 → YOLO 어노테이션 추출 정상

---

## 4. 기능별 테스트 결과

### 4.1 학습 (Train)

| 항목 | 결과 |
|------|------|
| CLI 학습 | ✅ 정상 (`tools.py pin train --unmasked-dir ... --masked-dir ...`) |
| 데이터셋 준비 | ✅ 10쌍 매칭, 어노테이션 생성 |
| YOLO26 학습 | ✅ 30 epochs 완료 (~4.5분) |
| 모델 저장 | ✅ `runs/detect/pin_models_test/pin_run/weights/best.pt` |

**학습 중 YOLO 메트릭**: Precision=0, Recall=0, mAP50=0 (소형 객체·적은 데이터로 인한 한계)

### 4.2 추론 (Inference)

| 항목 | 결과 |
|------|------|
| CLI 추론 | ✅ 정상 |
| conf=0.25 (기본) | 위 0, 아래 0 → NG (소형 핀 미탐지) |
| conf=0.01 | 위 20, 아래 20 → OK |
| 마스킹 이미지 | ✅ 초록색 점 출력 |
| 엑셀 출력 | ✅ result.xlsx (핀 개수, OK/NG, 좌우간격) |

**결론**: 기본 conf=0.25는 소형 핀에 과도하게 높음. conf=0.01에서 40개 탐지.

### 4.3 GUI

| 항목 | 결과 |
|------|------|
| GUI 실행 | ✅ `python run_pin_gui.py` 또는 `python tools.py pin gui` |
| 학습 탭 | ✅ 폴더 선택, Epochs/imgsz/Workers 설정, 학습 실행 |
| 추론 탭 | ✅ 이미지·모델 선택, 추론 실행 |
| 학습 그래프 | ✅ Loss, Precision, Recall 실시간 표시 |
| EXE 빌드 | ⏳ PyInstaller 빌드 (ultralytics/torch 의존성으로 10분 이상 소요) |

**참고**: EXE는 GitHub Actions (`build-pin-exe.yml`)에서 push 시 자동 빌드. GUI 기능은 `python run_pin_gui.py`로 동일 검증 가능.

---

## 5. 성능 지표 (Recall/Precision)

### 5.1 평가 방법

- **GT**: masked 이미지에서 추출한 bbox (40개/이미지)
- **Pred**: 추론 bbox (cap_at_20_per_row 적용 → 최대 40개)
- **매칭**: 중심 거리 ≤ N px → TP

### 5.2 실험 결과 (FPC 스타일, train 40 / test 20, val_split 0.2, 80 epochs)

| 데이터 | TP | FP | FN | Recall | Precision | F1 | mAP50 |
|--------|-----|-----|-----|--------|------------|-----|-------|
| 테스트 20장 | 757 | 43 | 43 | 94.62% | 94.62% | 94.62% | 0.811 |

- **FP on fake (노이즈)**: 0건
- **추론 속도**: ~3100 ms/img (CPU)

### 5.3 IoU 매칭

- IoU 0.25/0.5: 소형 bbox에서 YOLO 예측과 GT bbox 크기·위치 차이로 IoU 낮음 → 중심 거리(max_dist) 매칭 사용

### 5.4 분석

- **목표**: Recall 100%, Precision 100% (SPEC)
- **현황**: train 40 / test 20, val_split 0.2, 80 epochs → Recall/Precision ~95%, mAP50 0.81
- **원인 추정**:
  1. 핀 크기 작음 (12×4 px) → YOLO 소형 객체 한계
  2. 학습 데이터 10장만 사용, Validation split 미적용
  3. 합성 데이터와 실제 공장 흑백 이미지 도메인 차이
  4. **ROADMAP 10.5**: Recall/Precision 100% 달성 및 오탐지 방지 전략 추가됨

---

## 6. 추론 속도

| 항목 | 값 |
|------|-----|
| 이미지당 평균 | ~850–1050 ms (CPU) |
| 10장 배치 | ~15–17초 |
| 구성 | preprocess ~1 ms, inference ~70–95 ms, postprocess ~0.1 ms |

**참고**: GPU 사용 시 추론 속도 대폭 개선 예상.

---

## 7. 결론 및 권장 사항

### 7.1 정상 동작 확인

- ✅ 데이터셋 준비 (쌍 매칭, 검증)
- ✅ 학습 파이프라인
- ✅ 추론 파이프라인 (마스킹 이미지, 엑셀)
- ✅ GUI (학습/추론 탭)

### 7.2 개선 필요

1. **conf threshold**: 기본 0.25 → 소형 핀용으로 0.01–0.05 권장 (또는 데이터별 튜닝)
2. **Recall/Precision**: 실제 데이터로 100% 목표 검증 필요
3. **핀 크기**: 실제 커넥터 이미지에서 핀 크기 확인 후 imgsz·augmentation 조정
4. **학습 데이터**: 10쌍 → 50–100쌍 이상 확대 권장

### 7.3 ROADMAP Phase 5 연계

- `pin eval` 스크립트: `tools_scripts/run_pin_experiment.py`로 프로토타입 구현
- Validation split, 배치 추론 등 Phase 5·6 작업 진행 권장

---

## 8. 재현 방법

### 8.1 전체 파이프라인 (권장)

```bash
# 1. 데이터 생성 (train 40 + test 20, 별도 시드)
python -m tools_scripts.generate_pin_test_data --train-pairs 40 --test-pairs 20

# 2. 학습 (val_split 0.2, 80 epochs)
python tools.py pin train --unmasked-dir test_data/pin_synthetic/train/unmasked --masked-dir test_data/pin_synthetic/train/masked --output-dir pin_models --epochs 80 --val-split 0.2

# 3. 평가 (테스트 세트, mAP50, Confusion Matrix)
python tools.py pin eval --model runs/detect/pin_models/pin_run/weights/best.pt --unmasked-dir test_data/pin_synthetic/test/unmasked --masked-dir test_data/pin_synthetic/test/masked --conf 0.01 --max-dist 40 --run-map50 --save-dir pin_eval_results

# 4. 추론 (단일)
python tools.py pin inference --model runs/detect/pin_models/pin_run/weights/best.pt --image test_data/pin_synthetic/test/unmasked/001.jpg --conf 0.01

# 5. GUI
python run_pin_gui.py
```

### 8.2 원클릭 파이프라인

```bash
python -m tools_scripts.run_full_pin_pipeline --epochs 80
```

---

## 9. Changelog

- 2026-03-06: 초안 작성, 합성 데이터 실험 결과 반영
