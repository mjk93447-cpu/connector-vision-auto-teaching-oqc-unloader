# EXE 로컬 테스트 시나리오

## 1. 사전 조건

| 항목 | 경로/값 | 확인 |
|------|---------|------|
| 프로젝트 루트 | `c:\connector-vision-auto-teaching-oqc-unloader` | EXE 실행 cwd |
| EXE 경로 | `dist\pin_detection_gui.exe` | PyInstaller 빌드 산출물 |
| 합성 데이터 (우선) | `test_data\pin_unmasked_labels\` | unmasked + labels + roi_map.json |
| 합성 데이터 (폴백) | `test_data\pin_synthetic\unmasked`, `...\masked` | 대체 |
| 출력 디렉터리 | `pin_models_exe_test\` | 성공 시 `exe_test_ok.txt` 생성 |

## 2. 합성 데이터 경로 (정확한 지정)

```
test_data/
└── pin_unmasked_labels/
    ├── unmasked/    ← 01.jpg ~ 20.jpg (필수)
    ├── labels/      ← 01.txt ~ 20.txt (YOLO v26 format, 필수)
    └── roi_map.json ← 필수
```

생성 명령:
```bash
python tools_scripts\generate_unmasked_with_labels.py --fast -n 20
```

## 3. 테스트 시나리오 (순서)

### 3.1 사전 검증
```powershell
cd c:\connector-vision-auto-teaching-oqc-unloader

# 1) EXE 존재 확인
Test-Path dist\pin_detection_gui.exe

# 2) 합성 데이터 확인
Test-Path test_data\pin_unmasked_labels\unmasked
Test-Path test_data\pin_unmasked_labels\labels
Test-Path test_data\pin_unmasked_labels\roi_map.json
(Get-ChildItem test_data\pin_unmasked_labels\labels\*.txt).Count -ge 1

# 3) 기존 출력 정리 (선택)
Remove-Item -Recurse -Force pin_models_exe_test -ErrorAction SilentlyContinue
```

### 3.2 실행 테스트 (--test-train)
```powershell
cd c:\connector-vision-auto-teaching-oqc-unloader
# 빠른 검증: PIN_EXE_TEST_EPOCHS=2 (기본 5)
$env:PIN_EXE_TEST_EPOCHS = "2"
dist\pin_detection_gui.exe --test-train
```

- **성공**: `pin_models_exe_test\exe_test_ok.txt` 생성, exit 0
- **실패**: `exe_test_ok.txt` 생성 안 됨, exit 1

### 3.3 배치 스크립트 (test_exe_train.bat)
```powershell
cd c:\connector-vision-auto-teaching-oqc-unloader
.\test_exe_train.bat
```

- EXE 없으면 에러 → build_pin_exe.bat 먼저 실행
- 데이터 없으면 에러 → `generate_unmasked_with_labels.py --fast -n 5` 실행

### 3.4 GUI 모드 (선택)
```powershell
cd c:\connector-vision-auto-teaching-oqc-unloader
dist\pin_detection_gui.exe
```
- GUI 시작 시 test_data 자동 채움
- Train 탭에서 unmasked+labels 경로 확인 후 학습 실행

## 4. 에러 대응

| 증상 | 원인 | 조치 |
|------|------|------|
| `dist\pin_detection_gui.exe` 없음 | 빌드 미완료 | `python -m PyInstaller --noconfirm pin_detection_gui.spec` |
| `exe_test_ok.txt` 미생성 | 학습 실패 | PIN_DEBUG=1 설정 후 `%TEMP%\pin_train_debug.log` 확인 |
| `unmasked` 없음 | 데이터 미생성 | `generate_unmasked_with_labels.py --fast -n 5` |
| `labels` 비어 있음 | 라벨 없음 | 동일 스크립트로 재생성 |
| cwd 오류 | EXE를 dist에서 실행 | **반드시 프로젝트 루트에서 실행** |

## 5. 검증 체크리스트

- [ ] `dist\pin_detection_gui.exe` 존재
- [ ] `test_data\pin_unmasked_labels\unmasked` 에 이미지 ≥1
- [ ] `test_data\pin_unmasked_labels\labels` 에 *.txt ≥1
- [ ] `test_data\pin_unmasked_labels\roi_map.json` 존재
- [ ] 프로젝트 루트에서 `dist\pin_detection_gui.exe --test-train` 실행
- [ ] `pin_models_exe_test\exe_test_ok.txt` 생성
- [ ] `test_exe_train.bat` exit 0
