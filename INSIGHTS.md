# Insights — 분석·평가·튜닝 결과 통합

이 문서는 프로젝트 분석, 테스트 결과, 튜닝 전략, 성능 리뷰를 통합합니다.

---

## 1. 프로젝트 개요

- **목적**: OQC 언로더 장비용 커넥터 핀·몰드 위치 자동 티칭. 제품 기종별 수동 teaching 자동화.
- **포크**: Edge (Sobel 에지 배치 처리). 비전 파이프라인, 자동 최적화, GUI 재사용.

### OQC 비전 모듈 (1번 질문)

- **기존 모듈**: 있음. 사내 보안 이슈로 소스 공개는 어려움.
- **연동·임플리멘테이션**: 다음 과제로 보류. 현재는 개발 가능성 검증 단계.

### 1차 목표

- **산출물**: 주어진 커넥터 핀/몰드 이미지에서 **핀 개수**와 **핀 간격(mm)** 측정 — **오프라인 Windows EXE**.
- **범위**: 독립 실행형, 기존 모듈 연동 미포함.

#### 캘리브레이션

- EXE를 오프라인 PC로 복사·설치 후, 해당 PC의 로컬 이미지로 캘리브레이션 가능.

#### 입력 (로컬 환경)

| 항목 | 설명 |
|------|------|
| 마스킹 전 커넥터 탑뷰 사진 | 원본 이미지 × 10 |
| 마스킹 후 커넥터 탑뷰 사진 | 핀 위치(초록색) × 10, 동일 촬영에 마스킹만 적용 |
| 엑셀 파일 | 위/아래 핀 개수, OK/NG, 좌우간격 × 10 (세트, 양식 고정) |

#### 출력

| 항목 | 중요도 | 비고 |
|------|--------|------|
| 핀 개수 | **필수** | 항상 정확히 출력 |
| 핀 간 좌우간격 | 대략적 | 엑셀 출력 |
| 마스킹 이미지 | **필수** | 인풋 이미지에 핀 위치 정확 표시 후 출력 |

#### 가정

- 핀 좌우 길이: **0.5mm 고정** (정확도는 크게 중요하지 않음).

#### 아키텍처 (YOLO26 기반)

- **모델**: YOLO26 (소형 객체 탐지). 1회 로컬 학습 후 추가 티칭 없음.
- **학습 입력**: 마스킹 전/후 사진, 엑셀(위/아래 핀 개수, OK/NG, 좌우간격).
- **추론**: 마스킹 없는 사진 → 위핀 20개·아래핀 20개 탐지 → 마스킹 이미지 + 엑셀(핀 개수, OK/NG, 좌우간격) 출력.
- **상세**: `SPEC_1ST_GOAL.md` 참고.

---

## 2. 서브시스템 성숙도 (1–100)

| 서브시스템 | 점수 | 상태 |
|------------|-----|------|
| Core Edge Detector | 86 | Mature |
| Boundary Band Filter | 83 | Mature |
| Auto Thresholding | 80 | Stable |
| GUI Batch Processing | 82 | Mature |
| ROI Editor + Cache | 80 | Stable |
| Auto Optimization Search | 74 | Improving |
| Auto Optimization Scoring | 76 | Improving |
| Synthetic Evaluation | 78 | Stable |

**개선 백로그**: Bayesian/surrogate 모델링, GPU 가속, 데이터셋별 캘리브레이션, 다중 ROI 워크플로.

---

## 3. 목표 점수별 전략 (Target Score Tuning)

| 목표 | 전략 | 조기종료 | CLI 예산 | 라운드 |
|------|------|----------|----------|--------|
| ≤0.35 | aggressive_low | 0.20 | 250 | 30 |
| ≤0.45 | medium_04 | 0.22 | 400 | 35 |
| ≤0.55 | medium_05 | 0.25 | 600 | 40 |
| >0.55 | careful_06 | 0.28 | 900 | 45 |

- **CLI**: `python tools.py tune` (단일), `python tools.py tune --batch` (0.4/0.5/0.6 배치 + 3회 개선 루프).
- **개선 루프**: 테스트 → 평가 → 전략 도출 → 설정 반영 → 반복.

---

## 4. Boundary Score 공식

GT(ground truth) 있을 때 `compute_boundary_optimized_score`:

- **alignment**: F1 (tolerance=1px)
- **thinness**: min(1, gt_pixels / pred_pixels)
- **connectivity**: 1 - 0.25×(n_components - 1)
- **single_line**: 1 - 0.4×endpoint_ratio - 0.25×branch_ratio

`score = 0.45×alignment + 0.25×thinness + 0.20×connectivity + 0.10×single_line`

---

## 5. Auto Optimization 성능

- **병렬화**: Batch10 **1.99x** 단축. 전체 **36%** 단축 (workers=4).
- **권장**: auto_candidate_workers=4.
- **누적**: 초기 ~19.5s → 최종 8.27s (58% 단축).

---

## 6. GPU 가속

- **CuPy**: NVIDIA GPU 시 **3~10배** 예상. 5~7배 현실적.
- **내장 GPU**: 1.2~2.5배.
- **설치**: `pip install cupy-cuda12x`. 벤치마크: `python tools.py gpu`.

---

## 7. Automated Test 결과 (인사이트)

- **workers=0**: prep ~0.6s, first_eval ~1.6–2.5s, batch10 ~15–21s, total ~17–24s.
- **workers=4**: prep ~0.56s, first_eval ~2.1–2.3s (프로세스 오버헤드), batch10 ~16–17s.
- **first_score**: 0.02~0.05 (합성 이미지 기준).

## 8. Boundary Score 4회 루프 최종 가중치

`boundary_score_eval` 4회 루프 후: w_align 0.38, w_thin 0.19, w_conn 0.31, w_line 0.11. connectivity_penalty 0.41, endpoint/branch_penalty 0.4.

## 9. Branch/Endpoint 메트릭 권장

`branch_endpoint_impact_test`: **reduce_or_disable** — endpoint/branch 가중치 없이도 good > bad 랭킹 유지. 영향 제한적.

## 10. Target Score Strategy Config (legacy)

`tools.py tune --batch` 실행 시 자동 생성. target 0.4/0.5/0.6별 eval_budget, round_size. 필요 시 `legacy/target_score_strategy_config.json` 참고 후 재실행으로 복구.

## 11. 배포·테스트

- **테스트**: `run_tests.bat` 또는 `python -m unittest discover`.
- **푸시**: `push_all.bat` (테스트 → 커밋 → 푸시).
- **EXE**: GitHub Actions → Artifacts. `release_ver20.bat`으로 태그 푸시.
