# Connector Vision Auto Teaching for OQC Unloader Machine

OQC 언로더 장비용 **커넥터 핀·몰드 위치 자동 티칭** 프로그램입니다.  
제품 기종마다 커넥터 크기와 생김세가 달라서 매번 수동으로 하던 사전 teaching 과정을 자동화합니다.

---

## 목적

- **핵심 기능**: connector의 pin과 mold의 정확한 위치를 자동으로 찾아주는 프로그램에서, 제품 기종별로 다른 커넥터에 대해 **사전 teaching을 자동화**하는 Auto Teaching 프로그램.
- **배경**: 기존에는 제품 기종이 바뀔 때마다 커넥터 크기/형상이 달라져 teaching을 새로 해야 했음 → 이를 자동화하여 OQC unloader machine에 적용.

## 1차 목표

- **산출물**: 커넥터 탑뷰 이미지 → **핀 개수**·**핀 간 좌우간격**·**마스킹 이미지** — **오프라인 Windows EXE**.
- **범위**: 독립 실행형. 기존 OQC 비전 모듈 연동은 다음 과제.

### 아키텍처

- **YOLO26** 기반 (소형 객체 탐지). 1회 로컬 학습 후 추가 티칭 없음.

### 입출력

| 구분 | 내용 |
|------|------|
| **학습 (1회)** | 마스킹 전/후 사진, 엑셀(위/아래 핀 개수, OK/NG, 좌우간격) |
| **추론** | 마스킹 없는 사진 → 마스킹 이미지, 엑셀(핀 개수, OK/NG, 좌우간격) |
| **판정** | 위 20개·아래 20개 → OK, 그 외 → NG |
| **가정** | 핀 좌우 길이 0.5mm 고정 |

### 상세 사양

- **SPEC_1ST_GOAL.md** 참고.

---

## 프로젝트 구조 (Fork from Edge)

이 저장소는 에지 디텍터(Edge) 프로젝트를 포크하여, 비전/위치 검출·파라미터 자동 탐색 등 유사 기능을 활용할 수 있도록 구성되어 있습니다.

- **개발 환경**: Python 3.8+, Cursor IDE 권장
- **문서**: `FORMAL_DOCUMENTATION.md`, `DEVELOPMENT_NOTES.md` 스타일 유지
- **실행/테스트**: 기존 Edge 스크립트 참고 후, Auto Teaching 전용 진입점으로 단계적 전환 예정

---

## 빠른 시작

```bash
python -m venv .venv
.venv\Scripts\activate
pip install numpy pillow
python sobel_edge_detection.py   # GUI 실행
python tools.py --help           # eval, tune, benchmark 등 CLI 도구
```

### 1차 목표 (핀 자동 마스킹)

```bash
pip install -r requirements-pin.txt
# GUI 실행 (학습/추론)
python tools.py pin gui
# 학습 (10쌍)
python tools.py pin train --unmasked-dir ./unmasked --masked-dir ./masked
# 추론
python tools.py pin inference --model pin_models/.../best.pt --image new.jpg
```

---

## 배포

- **테스트**: `run_tests.bat` 또는 `python -m unittest discover`
- **푸시**: `push_all.bat` (테스트 → 커밋 → 푸시)
- **Edge EXE**: `release_ver20.bat` → 태그 v20 → Releases
- **Pin EXE**: `release_pin.bat` → 태그 pin-v1 → Releases. 또는 push 시 Actions → Artifacts

## 문서

- **ROADMAP.md**: 중장기 개발 계획
- **SPEC_1ST_GOAL.md**: 1차 목표 상세 사양 (YOLO26 핀 자동 마스킹)
- **AGENTS.md**: 에이전트/개발 가이드
- **FORMAL_DOCUMENTATION.md**, **DEVELOPMENT_NOTES.md**: 요구사항·아키텍처·이력 (Edge 포크)
- **INSIGHTS.md**: 분석·평가·튜닝 결과 통합
