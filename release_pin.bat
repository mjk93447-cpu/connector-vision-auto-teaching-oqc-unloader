@echo off
chcp 65001 >nul
cd /d "%~dp0"
:: Pin Detection 프로토타입 릴리스 — pin-v1 태그로 EXE 빌드 트리거
echo [1/3] origin main 푸시 중...
git push origin main
if %ERRORLEVEL% neq 0 (
  echo 푸시 실패. 먼저 push_all.bat 또는 git add/commit 후 다시 실행하세요.
  pause
  exit /b 1
)
echo [2/3] 태그 pin-v1 생성...
git tag -a pin-v1 -m "Pin Detection GUI prototype (YOLO26)" 2>nul || echo 태그 pin-v1 이미 존재할 수 있음
echo [3/3] 태그 pin-v1 푸시 (Actions에서 EXE 빌드 후 Releases에 올라갑니다)...
git push origin pin-v1
if %ERRORLEVEL% neq 0 (
  echo 태그 푸시 실패. 이미 pin-v1이 있으면: git push origin pin-v1 --force
  pause
  exit /b 1
)
echo.
echo 완료. GitHub Actions에서 빌드가 끝나면 Releases에서 exe를 받을 수 있습니다.
pause
