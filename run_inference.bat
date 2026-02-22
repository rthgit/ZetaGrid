@echo off
title RTH-LM 50B FAST INFERENCE
echo ======================================================
echo RTH-LM 50B - LOCAL FAST INFERENCE (SFT)
echo ======================================================
echo.
echo Make sure you have downloaded:
echo 1. zetagrid_25b_production.npy
echo 2. zeta50b_sft_step2000.pt
echo.
echo Launching...
python FAST_INFERENCE_50B.py
pause
