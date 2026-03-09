# EXE 아티팩트 테스트 이슈 — 심층 분석 및 대응 전략

**작성일**: 2026-03-06  
**근거**: 최신 아티팩트 테스트 결과

---

## 1. 이슈 요약

| # | 증상 | 심각도 | 우선순위 |
|---|------|--------|----------|
| 1 | EXE 첫 실행 시 DLL 에러, 관리자 권한으로만 실행 가능 | 높음 | P1 |
| 2 | Masked images folder 지정 시 ~2분 렉, 조작 불가 | 높음 | P1 |
| 3 | Image size 입력창이 ROI 비활성화 시에도 노출, 입력 시 극심한 렉 | 중간 | P2 |

---

## 2. 이슈 1: DLL 에러 및 관리자 권한

### 2.1 심층 분석

| 항목 | 내용 |
|------|------|
| **가능 원인** | (1) VC++ Redistributable 미설치 (2) PyInstaller one-file 추출 시 temp 권한 (3) Windows Defender/AV 차단 (4) manifest에 admin 요청 |
| **PyInstaller one-file** | 실행 시 `_MEI*` temp 폴더에 추출. 일부 환경에서 쓰기 권한 부족 → DLL 로드 실패 |
| **관리자 권한** | UAC 승인 시 temp 경로·권한이 달라져 동작할 수 있음 |
| **spec 현황** | `runtime_tmpdir=None`, `console=False`, manifest 미명시 |

### 2.2 대응 전략

| Phase | 작업 | 목표 |
|-------|------|------|
| **A** | manifest 추가 | `requestedExecutionLevel asInvoker` — 관리자 요청 제거 |
| **B** | one-dir 빌드 옵션 | one-file 실패 시 one-dir로 fallback. 문서화 |
| **C** | VC++ Redist 문서화 | README/다운로드 안내에 VC++ 2015–2022 x64 설치 링크 |
| **D** | runtime_tmpdir 검토 | `os.environ['TEMP']` 또는 사용자 writable 경로 지정 (필요 시) |

### 2.3 구현 순서

1. spec에 `uac_admin=False` 또는 manifest 파일로 `asInvoker` 설정
2. `--onedir` 빌드 스크립트 추가, DOWNLOAD_EXE.md에 one-dir 사용법 안내
3. DLL 에러 시 VC++ Redist 설치 안내 메시지 (선택)

---

## 3. 이슈 2: Masked folder 지정 시 ~2분 렉

### 3.1 심층 분석

| 항목 | 내용 |
|------|------|
| **호출 경로** | Browse → `_on_masked_browse` → `_update_eta` → `analyze_dataset_for_training` |
| **실행 스레드** | **메인(GUI) 스레드** — UI 블로킹 |
| **analyze_dataset_for_training** | max_samples=5장에 대해: (1) unmasked 열기 (2) masked 찾기·열기 (3) `masked_array_to_annotations` — 초록 영역 추출·클러스터링 |
| **대형 이미지** | 5496×3672 등에서 `masked_array_to_annotations`가 무거움. 5장 × 대형 = 수십 초~2분 |
| **결론** | 무거운 연산이 메인 스레드에서 동기 실행됨 |

### 3.2 대응 전략

| Phase | 작업 | 목표 |
|-------|------|------|
| **A** | 비동기 스캔 | `_update_eta` 내 `analyze_dataset_for_training`을 `threading.Thread`로 실행 |
| **B** | UI 피드백 | 스캔 중 "Scanning dataset..." 표시, Apply suggested 비활성화 |
| **C** | 경량 1차 표시 | 스캔 전에 파일 개수·첫 이미지 크기만 빠르게 표시 (이미 구현됨), suggested는 백그라운드 완료 후 |
| **D** | max_samples 축소 | 5 → 3 또는 2 (대형 이미지 시) — 선택 |

### 3.3 구현 순서

1. `_update_eta` 분리: (1) 즉시: n_images, w×h만 표시 (2) 백그라운드: analyze → suggested 표시
2. `root.after(0, ...)`로 스레드 완료 시 GUI 업데이트
3. 스캔 중 Apply suggested 비활성화, "Scanning..." 라벨

---

## 4. 이슈 3: Image size 입력 시 렉 및 ROI 비활성화와 불일치

### 4.1 심층 분석

| 항목 | 내용 |
|------|------|
| **ROI 비활성화** | ROADMAP 10.18: 학습·추론 모두 ROI off, 전체 이미지 사용. imgsz=640 고정 권장 |
| **GUI 현황** | imgsz Spinbox가 항상 노출. `sb_imgsz.bind("<KeyRelease>", _update_eta)` |
| **렉 원인** | **키 입력마다** `_update_eta` 호출 → `analyze_dataset_for_training` 실행 (메인 스레드) |
| **결론** | KeyRelease 시 전체 데이터셋 스캔이 매 키마다 실행됨 |

### 4.2 대응 전략

| Phase | 작업 | 목표 |
|-------|------|------|
| **A** | KeyRelease 디바운스 | 300–500ms 디바운스. 입력 멈춘 뒤 1회만 _update_eta |
| **B** | ROI off 시 imgsz 단순화 | ROI 비활성화 시: imgsz 고정 640, Spinbox 숨기거나 읽기 전용 |
| **C** | analyze 호출 제거 (ETA만) | ETA 계산은 n, imgsz, epochs, workers만 사용. analyze는 "Apply suggested" 시에만 |

### 4.3 구현 순서

1. `_update_eta`에서 `analyze_dataset_for_training` 호출 제거. ETA는 `_estimate_training_time(n, imgsz, epochs, workers)`만 사용
2. "Apply suggested" 클릭 시에만 별도 스레드로 analyze 실행
3. imgsz KeyRelease에 디바운스 (또는 analyze 제거로 해소)
4. ROI off 정책에 맞춰 imgsz UI: (옵션 A) 640 고정·라벨만 표시 (옵션 B) Spinbox 유지하되 analyze 연동 제거

---

## 5. 통합 구현 순서

| 단계 | 작업 | 이슈 |
|------|------|------|
| 1 | _update_eta에서 analyze 제거, ETA만 계산 | #2, #3 |
| 2 | "Apply suggested"를 별도 버튼 동작으로, 백그라운드 스레드에서 analyze | #2 |
| 3 | imgsz KeyRelease 디바운스 (또는 1–2로 해소 시 생략) | #3 |
| 4 | ROI off 시 imgsz 640 고정, Spinbox 숨김 또는 읽기 전용 | #3 |
| 5 | spec manifest asInvoker, one-dir 빌드 옵션 | #1 |
| 6 | DOWNLOAD_EXE.md VC++ Redist 안내 | #1 |

---

## 6. 검증 기준

| 이슈 | 검증 |
|------|------|
| #1 | 일반 사용자 권한으로 EXE 실행, DLL 에러 없음 |
| #2 | Masked folder 선택 후 5초 이내 UI 응답, 2분 렉 없음 |
| #3 | imgsz 입력 시 렉 없음. ROI off 시 imgsz 640 고정 또는 UI 정리 |

---

## 7. 구현 완료 (2026-03-06)

| 이슈 | 구현 |
|------|------|
| #2 | _update_eta에서 analyze 제거. 백그라운드 스레드로 "Apply suggested" 스캔. 즉시 n/w×h/ETA 표시 |
| #3 | imgsz Spinbox 제거, "640 (fixed, ROI off)" 라벨로 대체. KeyRelease 바인딩 제거 |
| #1 | uac_admin=False, manifest_asinvoker.xml, build_pin_exe_onedir.bat, DOWNLOAD_EXE.md VC++ 안내 |
