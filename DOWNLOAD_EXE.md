# EXE 다운로드

## Pin Detection GUI (프로토타입)

1. **Releases**: https://github.com/mjk93447-cpu/connector-vision-auto-teaching-oqc-unloader/releases  
   - `pin-v1` 또는 `pin-v2` 태그에서 `pin_detection_gui.exe` 다운로드

2. **Artifacts**: https://github.com/mjk93447-cpu/connector-vision-auto-teaching-oqc-unloader/actions  
   - "Build Pin Detection EXE" 워크플로 실행 후 Artifacts에서 `pin_detection_gui_windows` 다운로드

### DLL 에러 발생 시 (관리자 권한 없이 실행 불가)

- **VC++ Redistributable** 설치: [Microsoft Visual C++ 2015–2022 x64](https://aka.ms/vs/17/release/vc_redist.x64.exe)  
- 오프라인 PC: 위 설치 파일을 USB 등으로 복사 후 설치
- 여전히 실패 시: one-dir 빌드 사용 (`build_pin_exe_onedir.bat`)

## Edge Batch GUI

- Releases의 `v20` 등 태그에서 `edge_batch_gui.exe` 다운로드
