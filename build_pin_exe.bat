@echo off
REM Build Pin Detection EXE - use cmd to avoid PowerShell stderr issues
REM Run from project root. Output: dist\pin_detection_gui.exe

echo Building pin_detection_gui.exe...
python -m PyInstaller --noconfirm --log-level WARN pin_detection_gui.spec
if %ERRORLEVEL% neq 0 (
    echo BUILD FAILED
    exit /b 1
)
echo Build complete: dist\pin_detection_gui.exe
exit /b 0
