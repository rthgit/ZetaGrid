@echo off
set IP=69.30.85.214
set PORT=22185
set USER=root

echo ===================================================
echo 🛠️  ZETAGRID EMERGENCY FIX UPLOADER
echo ===================================================
echo This will upload the CORRECTED scripts to A40.
echo (Restoring D_FF=16384 for TRUE 25B Architecture)
echo.

echo 1. Stopping any running python processes (Optional, manual)...
echo.

echo 2. Uploading RUN_REPAIR_A40.py (Fixed)...
scp -P %PORT% ..\RUN_REPAIR_A40.py %USER%@%IP%:/workspace/zetagrid_50b/

echo 3. Uploading RUN_CODE_A40.py (Fixed)...
scp -P %PORT% ..\RUN_CODE_A40.py %USER%@%IP%:/workspace/zetagrid_50b/

echo.
echo ===================================================
echo ✅ UPLOAD COMPLETE.
echo NOW:
echo 1. SSH into A40: ssh -p %PORT% %USER%@%IP%
echo 2. RUN: python RUN_REPAIR_A40.py 25B
echo    (This will restart the repair with correct params)
echo ===================================================
pause
