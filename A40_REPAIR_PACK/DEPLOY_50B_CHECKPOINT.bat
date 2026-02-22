@echo off
set IP=69.30.85.214
set PORT=22185
set USER=root

echo ===================================================
echo 🚀 ZETAGRID DEPLOYER (50B CHECKPOINT)
echo ===================================================
echo Target: %USER%@%IP%:%PORT%
echo File: zeta50b_sft_step2000.pt (50GB+)
echo.

if not exist "C:\Users\PC\Desktop\cpu-da\zeta50b_sft_step2000.pt" (
    echo ❌ 50B CHECKPOINT NOT FOUND!
    echo Looked at: C:\Users\PC\Desktop\cpu-da\zeta50b_sft_step2000.pt
    pause
    exit /b
)

echo 🚀 UPLOADING 50B CHECKPOINT...
echo This will take a while. Do not close this window.
scp -P %PORT% "C:\Users\PC\Desktop\cpu-da\zeta50b_sft_step2000.pt" %USER%@%IP%:/workspace/zetagrid_50b/
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Upload Failed!
    pause
    exit /b
)

echo.
echo ✅ 50B CHECKPOINT UPLOADED!
echo You can now run the 50B Repair: python RUN_REPAIR_A40.py 50B
pause
