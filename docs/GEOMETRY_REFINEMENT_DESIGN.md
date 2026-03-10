# Geometry Refinement 설계 (보조 수단)

**갱신일**: 2026-03-05

---

## 관련 문서

- **ROI 정밀화**: ROADMAP 10.24 — 줌, 브러쉬, 분할 ROI

---

## 1. 설계 원칙

| 단계 | 역할 | 설명 |
|------|------|------|
| **1차** | YOLO | 최신 모델 성능으로 탐색 (primary) |
| **2차** | Masked prior | 십자 핀 마스킹 위치로 FN 보완 |
| **3차** | Geometry refinement | 미검출·오검출 보정 (보조) |

**Geometry refinement는 보조 수단**: YOLO + masked prior 이후, 그래도 발생하는 FN/FP를 기하학적 특징으로 보정.

---

## 2. 기하학적 특징

- 위·아래 각 20개 고정
- 거의 완전히 평행, 일정한 간격
- confidence가 떨어지는 객체에 대해 이 특징 활용

---

## 3. 케이스별 처리

| 케이스 | 처리 |
|--------|------|
| **FN (< 20/행)** | 추가 탐색: 균일 간격 보간으로 누락 슬롯 채움 |
| **FP (> 20/행)** | confidence 상위 20개 유지 |
| **FP (엉뚱한 위치)** | 정밀 검증: spacing anomaly (간격 이상) → 해당 구간 최저 confidence 제거 |

### 3.1 FN (미검출)

- 행 내 탐지 < 20 → 큰 gap 구간에 보간
- 빈 행 → 반대 행 간격으로 템플릿 생성

### 3.2 FP (오검출)

- **초과 (> 20)**: confidence 상위 20개만 유지
- **엉뚱한 위치**: 인접 gap이 median의 2.5배 이상 → spacing anomaly → 해당 구간 최저 confidence 제거 (반복)

---

## 4. 구현

| 파일 | 역할 |
|------|------|
| `inference.py` | YOLO 우선 → masked prior 병합 → geometry refinement |
| `geometry_refinement.py` | `refine_to_fixed_grid`, `_verify_and_remove_fp_wrong_location` |

---

## 5. 파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `GAP_ANOMALY_RATIO` | 2.5 | gap이 median의 이 배수 이상이면 FP 후보 |
| `min_per_row_for_interp` | 4 | 이 개수 미만이면 보간 생략 |
| `merge_dist_px` | 15 | YOLO-masked 중복 판정 거리 (px) |
