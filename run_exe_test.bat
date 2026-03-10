@echo off
REM EXE test: Run from project root so "Load test data" finds test_data.
REM 1. Build EXE (or download artifact)
REM 2. Copy pin_detection_gui.exe to project root
REM 3. Run this batch from project root
REM 4. In GUI: Train tab -> "Load test data" -> Start training

if not exist "test_data\pin_large_factory\unmasked" (
    echo Generating test_data...
    python tools_scripts/generate_pin_test_data.py --output-dir test_data/pin_large_factory --large-factory --large-factory-n 10
)
if exist "pin_detection_gui.exe" (
    pin_detection_gui.exe
) else if exist "dist\pin_detection_gui.exe" (
    dist\pin_detection_gui.exe
) else (
    echo Run: python run_pin_gui.py
    python run_pin_gui.py
)
