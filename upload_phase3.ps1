
# UPLOAD PHASE 3 FILES TO RUNPOD
# Save as: upload_phase3.ps1

# ============================================================
# CONFIGURATION
# ============================================================

$POD_HOST = "root@69.30.85.39"
$POD_PORT = "22065"
$SSH_KEY = "$env:USERPROFILE\.ssh\id_ed25519" 

$REMOTE_BASE = "/workspace/zetagrid_50b"

# Local Files to Upload
$FILES = @(
    "zetagrid_50b_seed.pt",
    "A40_TRAIN_50B_PHASE3.py"
)

# ============================================================
# SCRIPT
# ============================================================

Write-Host "`n╔══════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║       ZETAGRID PHASE 3 UPLOADER (RUNPOD)                ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════╝`n" -ForegroundColor Cyan

if (-not (Test-Path $SSH_KEY)) {
    Write-Host "❌ SSH Key not found at: $SSH_KEY" -ForegroundColor Red
    exit 1
}

# 1. Upload files
foreach ($file in $FILES) {
    if (Test-Path $file) {
        $size = (Get-Item $file).Length / 1MB
        Write-Host "📤 Uploading $file ($($size.ToString('0.00')) MB)..." -ForegroundColor Yellow
        
        $remote_path = "$REMOTE_BASE/$file"
        $scpCmd = "scp -P $POD_PORT -i `"$SSH_KEY`" `"$file`" ${POD_HOST}:${remote_path}"
        
        try {
            Invoke-Expression $scpCmd
            Write-Host "   ✅ Uploaded." -ForegroundColor Green
        }
        catch {
            Write-Host "   ❌ Failed: $_" -ForegroundColor Red
        }
    }
    else {
        Write-Host "⚠️ File not found locally: $file" -ForegroundColor Red
    }
}

Write-Host "`nReady to launch training!" -ForegroundColor Green
Write-Host "Command to run on RunPod:" -ForegroundColor White
Write-Host "ssh -p $POD_PORT -i `"$SSH_KEY`" $POD_HOST `"cd $REMOTE_BASE && python A40_TRAIN_50B_PHASE3.py`"" -ForegroundColor White
Write-Host "`nPress Enter to exit..."
Read-Host
