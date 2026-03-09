@echo off
REM One-dir build — DLL in folder. Use when one-file has DLL/admin issues.
REM Output: dist\pin_detection_gui\pin_detection_gui.exe

echo Building pin_detection_gui (one-dir)...
python -m PyInstaller --noconfirm --log-level WARN pin_detection_gui_onedir.spec
if %ERRORLEVEL% neq 0 (
    echo BUILD FAILED
    exit /b 1
)
echo Build complete: dist\pin_detection_gui\pin_detection_gui.exe
exit /b 0
