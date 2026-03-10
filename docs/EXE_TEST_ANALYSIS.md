# EXE 테스트 결과 정밀 분석

## 1. 기존 테스트 결과 요약

| 실험 | 데이터 | Epochs | P | R | mAP50 | 비고 |
|------|--------|--------|---|---|-------|------|
| pin_models_exe_test | unmasked+labels, 5쌍 | 5 | 0 | 0 | 0 | Pin Masking flow |
| pin_models_pin_masking_test | unmasked+labels, 5쌍 | 20 | 0 | 0 | 0 | 동일 |
| pin_models_action36_test | masked, 30쌍 | 15 | 0.13 | 1.0 | 0.97 | Epoch 8-9 |
| **pin_models_pr50_test** | **unmasked+labels, 20쌍** | **50** | **0.894** | **0.787** | **0.95** | **Epoch 18-21** |

## 2. 핵심 발견

**pin_models_pr50_test** (unmasked+labels, 20쌍, 50 epochs):
- Epoch 18-21: **P=89.4%, R=78.7%**, mAP50=0.95
- **50% 이상 P/R 달성** — 공장 데이터 50%+ 가능성 확보

## 3. 결론

- 합성 공장 유사 데이터(작은 핀, mold, dust)에서 **P≥50%, R≥50%** 달성
- 실제 공장 데이터 + Pin Masking으로 50%+ 기대 가능
- **EXE artifact 빌드 진행**
