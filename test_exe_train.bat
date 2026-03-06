@echo off
REM Test EXE training - run from project root
REM Requires: test_data\pin_synthetic\unmasked, test_data\pin_synthetic\masked

if not exist "dist\pin_detection_gui.exe" (
    echo ERROR: dist\pin_detection_gui.exe not found. Run build_pin_exe.bat first.
    exit /b 1
)
if not exist "test_data\pin_synthetic\unmasked" (
    echo ERROR: test_data\pin_synthetic\unmasked not found
    exit /b 1
)

echo Running EXE with --test-train...
dist\pin_detection_gui.exe --test-train

if exist "pin_models_exe_test\exe_test_ok.txt" (
    echo SUCCESS: Training completed. exe_test_ok.txt found.
    type pin_models_exe_test\exe_test_ok.txt
    exit /b 0
) else (
    echo FAIL: exe_test_ok.txt not found. Training may have failed.
    exit /b 1
)
