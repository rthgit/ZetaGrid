@echo off
set IP=69.30.85.214
set PORT=22185
set USER=root

echo ===================================================
echo 📥 ZETAGRID DOWNLOADER (FROM A40)
echo ===================================================
echo Source: %USER%@%IP%:%PORT%
echo.

echo Where do you want to save the models?
echo 1. Desktop (C:\Users\PC\Desktop\cpu-da\repaired_models)
echo 2. Drive E: (E:\ZETAGRID)
set /p TARGET="Choose (1/2): "

if "%TARGET%"=="1" set DEST="C:\Users\PC\Desktop\cpu-da\repaired_models"
if "%TARGET%"=="2" set DEST="E:\ZETAGRID"

if not exist %DEST% mkdir %DEST%

echo.
echo 🚀 DOWNLOADING 25B v2 (REPAIRED)...
scp -P %PORT% %USER%@%IP%:/workspace/zetagrid_50b/repaired_checkpoints/zeta_25B_v2.pt %DEST%
if %ERRORLEVEL% NEQ 0 (
    echo ⚠️  zeta_25B_v2.pt not found or download failed.
) else (
    echo ✅ zeta_25B_v2.pt DOWNLOADED!
)

echo.
echo 🚀 DOWNLOADING 25B CODE v3 (If exists)...
scp -P %PORT% %USER%@%IP%:/workspace/zetagrid_50b/code_checkpoints/zeta_25B_code_v3.pt %DEST%
if %ERRORLEVEL% NEQ 0 (
    echo ℹ️  Code model not found yet (Run Phase 7 first).
) else (
    echo ✅ zeta_25B_code_v3.pt DOWNLOADED!
)

echo.
echo DONE. Models are in %DEST%
pause
