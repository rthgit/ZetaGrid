#!/bin/bash
set -euo pipefail

# ==============================================================================
# ZED-HLS RUNNER (v1.0)
# Automates the execution of a single HLS kernel
# ==============================================================================

KERNEL="$1"
DATE_TAG="$(date +%F)"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KDIR="$ROOT/kernels/$KERNEL"
OUT="$ROOT/reports/$DATE_TAG/$KERNEL"

# --- 1. DETECTION OF VITIS HLS ---
# We verify if vitis_hls is in PATH. If typically installed on Windows, we might need to source a bat.
# For now, we assume 'vitis_hls' works or the user has aliased it.
VITIS_CMD="vitis_hls"

if ! command -v $VITIS_CMD &> /dev/null; then
    # Try finding it in common Windows paths executable from WSL (C: and D:)
    # We use ls and head to pick the first one roughly. Added || true to prevent set -e exit if not found.
    FOUND_C=$(ls /mnt/c/Xilinx/Vitis_HLS/*/bin/vitis_hls.bat 2>/dev/null | tail -n 1 || true)
    FOUND_D=$(ls /mnt/d/Xilinx/Vitis_HLS/*/bin/vitis_hls.bat 2>/dev/null | tail -n 1 || true)
    
    if [ -n "$FOUND_D" ]; then
        FOUND="$FOUND_D"
    else
        FOUND="$FOUND_C"
    fi
    
    if [ -n "$FOUND" ]; then
        echo "[INFO] Found Vitis HLS on Windows: $FOUND"
        VITIS_CMD="$FOUND"
    else
        echo "[ERROR] 'vitis_hls' not found in PATH or standard locs."
        exit 1
    fi
fi

# Check if we are running a Windows Batch file from WSL
IS_WIN_BAT=0
if [[ "$VITIS_CMD" == *".bat" ]]; then
    IS_WIN_BAT=1
fi

# --- 2. SETUP ---
echo "=== ZED-HLS RUN: $KERNEL ==="
echo "Target: $OUT"
mkdir -p "$OUT"
cp "$KDIR/config.json" "$OUT/config.json" 2>/dev/null || echo "{}" > "$OUT/config.json"

# Log Environment
echo "TS: $(date -Is)" | tee "$OUT/run.log"
echo "Host: $(hostname)" | tee -a "$OUT/run.log"

# --- 3. EXECUTION ---
pushd "$KDIR" >/dev/null

echo "--> Launching Vitis HLS..."
# Run Vitis HLS using the build.tcl script
if [ $IS_WIN_BAT -eq 1 ]; then
    # Convert path to Windows format for cmd.exe
    WIN_CMD=$(wslpath -w "$VITIS_CMD")
    # Execute via cmd.exe. 
    # Note: build.tcl is in current dir, so "build.tcl" is fine relative path.
    cmd.exe /C "$WIN_CMD" -f build.tcl | tee "$OUT/hls_console.log"
else
    "$VITIS_CMD" -f build.tcl | tee "$OUT/hls_console.log"
fi
EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -ne 0 ]; then
    echo "[ERROR] Vitis HLS failed! Check $OUT/hls_console.log"
    popd >/dev/null
    exit $EXIT_CODE
fi

# Collect Report Files (XML/Rpt)
# Vitis typically puts them in <project>/<solution>/syn/report or simila
# We blindly copy all .rpt and .xml files we find in the project di
echo "--> Collecting Artifacts..."
find . -name "*.rpt" -exec cp {} "$OUT/" \;
find . -name "*.xml" -exec cp {} "$OUT/" \;
find . -name "*.json" -exec cp {} "$OUT/" \; 2>/dev/null || true

popd >/dev/null

# --- 4. PARSING ---
if [ -f "$ROOT/scripts/parse_reports.py" ]; then
    echo "--> Parsing Metrics..."
    python3 "$ROOT/scripts/parse_reports.py" "$KDIR" "$OUT" | tee -a "$OUT/parse.log"
else
    echo "[WARN] Parser script not found. Skipping metrics extraction."
fi

echo "=== RUN COMPLETE ==="
echo "Report: $OUT"
