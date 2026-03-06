# Pin Detection 개발 현황 분석 및 성능 평가

Last updated: 2026-03-05

## 1. 목적

pin_detection 모듈의 개발 완성도, 성능 지표 적용 현황, 개선 필요 사항을 정리한다.

---

## 2. 개발 완료 항목

| 영역 | 구현 내용 | 상태 |
|------|-----------|------|
| **데이터셋** | 마스킹 전/후 쌍 매칭, 초록색 영역 → YOLO bbox | 완료 |
| **검증** | unmasked/masked 동일 해상도 검증, 빈 폴더 검증 | 완료 |
| **학습** | YOLO26n, mosaic/copy_paste, CLI/GUI | 완료 |
| **추론** | 단일 이미지, cap_at_20_per_row(Precision 보장) | 완료 |
| **엑셀** | 형식 추론, 결과 출력 | 완료 |
| **테스트** | test_pin_dataset.py (11개) | 완료 |

---

## 3. 성능 지표 적용 현황

### 3.1 SPEC 목표 (SPEC_1ST_GOAL.md, ROADMAP §6.3)

| 지표 | 목표 | 현재 적용 |
|------|------|-----------|
| **Recall** | 100% | ❌ 측정·평가 없음 |
| **Precision** | 100% | ⚠️ cap_at_20_per_row로 상한만 적용, 실제 Precision 미측정 |
| **영역 마스킹** | 정확 | ⚠️ IoU/위치 정확도 미측정 |

### 3.2 평가 부재

- **학습 중**: YOLO 내부 metrics(precision/recall)만 표시, GT 대비 Recall/Precision 검증 없음
- **학습 후**: hold-out 테스트 없음, 모델 품질 정량 평가 불가
- **추론**: 단일 이미지 결과만 출력, 배치 평가 스크립트 없음

---

## 4. 개선 필요 사항 (우선순위)

### 4.1 필수 (성능 검증)

| 항목 | 설명 | 영향 |
|------|------|------|
| **Recall/Precision 평가** | masked 이미지(GT) vs 추론 결과 비교 | 목표 달성 여부 확인 불가 |
| **Validation split** | train/val 동일 폴더 사용 → 과적합 위험 | 조기 종료·모델 선택 불가 |
| **평가 스크립트** | `pin eval` 또는 `tools.py pin eval` | 배치 평가, conf threshold 튜닝 |

### 4.2 중요 (운영·품질)

| 항목 | 설명 | 영향 |
|------|------|------|
| **배치 추론** | 폴더 단위 추론, 결과 요약 | 대량 이미지 처리 불편 |
| **Confidence threshold 튜닝** | Recall 100% 달성용 conf 최적화 | 과탐지/미탐지 균형 |
| **추론 속도 벤치마크** | 이미지당 ms 측정 | EXE 배포 시 성능 보장 |

### 4.3 권장 (장기)

| 항목 | 설명 |
|------|------|
| **Excel ↔ 어노테이션 검증** | 학습 시 Excel 핀 개수 vs 추출 bbox 개수 불일치 경고 |
| **위/아래 클래스 분리** | class 0: upper, class 1: lower (선택) |
| **모델 비교** | YOLO26n vs s/m 성능·속도 트레이드오프 |

---

## 5. 평가 메트릭 정의

### 5.1 Recall (핀 미탐지 방지)

```
Recall = TP / (TP + FN)
- GT: masked 이미지에서 추출한 bbox
- Pred: 추론 bbox (IoU > 0.5 매칭)
- 목표: 100% (20+20 중 하나라도 놓치면 NG)
```

### 5.2 Precision (과탐지 방지)

```
Precision = TP / (TP + FP)
- 위/아래 각 20개 초과 시 FP로 간주
- cap_at_20_per_row로 상한은 적용 중
- 목표: 100% (20개 초과 감지 금지)
```

### 5.3 IoU 매칭

- GT bbox와 Pred bbox 간 IoU > 0.5 → TP
- 매칭되지 않은 GT → FN
- 매칭되지 않은 Pred → FP

---

## 6. 참고: Edge Detection 대비

| 항목 | Edge (Sobel) | Pin (YOLO) |
|------|--------------|------------|
| 평가 스크립트 | `tools.py eval` | ❌ 없음 |
| 벤치마크 | `edge_performance_eval` | ❌ 없음 |
| 튜닝 | `tools.py tune` | ❌ 없음 |
| 테스트 | test_smoke, test_auto_* | test_pin_dataset |

---

## 7. Changelog

- 2026-03-05: 초안 작성, ROADMAP §9·§10 연계
