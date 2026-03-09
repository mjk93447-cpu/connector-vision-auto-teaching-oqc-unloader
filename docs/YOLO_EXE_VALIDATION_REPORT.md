# YOLO EXE Validation Report

**날짜**: 2026-03-09  
**대상**: 업데이트된 pin_detection (10.20.2 EXE 피드백 반영 후)

---

## 1. 테스트 개요

| 항목 | 내용 |
|------|------|
| 학습 데이터 | 합성 20세트 (unmasked 20 + masked 20) |
| 학습 epochs | 10 |
| 테스트 데이터 | 노이즈 큰 합성 10세트 (blur_prob=0.6, n_fake_pins=12) |
| 목표 | 위 20 + 아래 20 = 총 40핀 정확 위치 마스킹 |

---

## 2. 실행 방법

```bash
python tools_scripts/run_yolo_exe_validation.py
```

- `test_data/yolo_exe_validation/train`: 20 학습 쌍 생성
- `test_data/yolo_exe_validation/test_noisy`: 10 노이즈 테스트 쌍 생성
- `pin_models/yolo_exe_validation/pin_run/weights/best.pt`: 학습 모델
- `yolo_result/`: 추론 결과 이미지 및 analysis.json

---

## 3. 결과 요약

| 이미지 | Upper | Lower | Total | OK |
|--------|-------|-------|-------|-----|
| noisy_01 ~ noisy_10 | 20 | 20 | 40 | ✓ |
| **합계** | — | — | — | **10/10 OK** |

- **추론 conf_threshold**: 0.01 (노이즈 데이터에서 저신뢰도 탐지 허용)
- **geometry_refinement**: 20+20 고정 그리드 보간 적용
- 모든 노이즈 테스트 이미지에서 40핀 정확 탐지·마스킹 확인

---

## 4. 결론

- 10 epochs 학습으로 합성 데이터 기준 40핀 탐지 성공
- 노이즈가 큰 테스트(blur 60%, fake 12개)에서도 100% OK
- `yolo_result/*_masked.jpg`: 녹색 마스킹이 핀 위치에 정확히 적용됨
