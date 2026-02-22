@echo off
set IP=69.30.85.214
set PORT=22185
set USER=root

echo ===================================================
echo 📥 ZETAGRID 25B (CODE & REPAIR) DOWNLOADER
echo ===================================================
echo Source: %USER%@%IP%:%PORT%
echo.

echo 1. Downloading Base Repaired Model (v2)...
rem This is the 6.3GB model we just made.
scp -P %PORT% %USER%@%IP%:/workspace/zetagrid_50b/repaired_checkpoints/zeta_25B_v2.pt "E:\ZETAGRID\"

echo 2. Downloading Code Checkpoint (Step 100)...
rem This is the newly trained coder.
scp -P %PORT% %USER%@%IP%:/workspace/zetagrid_50b/code_checkpoints/zeta_code_step100.pt "E:\ZETAGRID\"

echo.
echo ===================================================
echo ✅ DOWNLOAD COMPLETE.
echo Now you have:
echo - E:\ZETAGRID\zeta_25B_v2.pt
echo - E:\ZETAGRID\zeta_code_step100.pt
echo.
echo Use convert_rth_to_gguf.py to make GGUF.
echo ===================================================
pause
