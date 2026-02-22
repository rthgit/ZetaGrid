#!/bin/bash
# scripts/diag_parts.sh
set -e

# Find Vitis
FOUND=$(ls /mnt/d/Xilinx/Vitis_HLS/*/bin/vitis_hls.bat 2>/dev/null | tail -n 1 || true)
if [ -z "$FOUND" ]; then
    echo "Vitis not found on D:"
    exit 1
fi

echo "Found: $FOUND"
WIN_CMD=$(wslpath -w "$FOUND")
TCL_PATH=$(realpath zed_hls/scripts/probe_parts.tcl)
TCL_WIN=$(wslpath -w "$TCL_PATH")

echo "Running: $WIN_CMD -f $TCL_WIN"
cmd.exe /C "$WIN_CMD" -f "$TCL_WIN"
