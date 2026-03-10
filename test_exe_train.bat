@echo off
REM Test EXE training - run from project root
REM Requires: test_data\pin_unmasked_labels (or pin_synthetic)
REM PIN_EXE_TEST_EPOCHS=2 for faster local test (default 5)

set PIN_EXE_TEST_EPOCHS=2

if not exist "dist\pin_detection_gui.exe" (
    echo ERROR: dist\pin_detection_gui.exe not found. Run build_pin_exe.bat first.
    exit /b 1
)
if not exist "test_data\pin_unmasked_labels\unmasked" (
    if not exist "test_data\pin_synthetic\unmasked" (
        if not exist "test_data\pin_synthetic\train\unmasked" (
            echo ERROR: test_data\pin_unmasked_labels or pin_synthetic not found. Run: python tools_scripts\generate_unmasked_with_labels.py --fast -n 5
            exit /b 1
        )
    )
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
