
# UPLOAD PHASE 3 FIX TO RUNPOD
# Save as: upload_phase3_fix.ps1

$POD_HOST = "root@69.30.85.39"
$POD_PORT = "22065"
$SSH_KEY = "$env:USERPROFILE\.ssh\id_ed25519" 

$REMOTE_BASE = "/workspace/zetagrid_50b"
$FILE = "A40_TRAIN_50B_PHASE3_FIX.py"

Write-Host "📤 Uploading FIX ($FILE)..." -ForegroundColor Yellow
$scpCmd = "scp -P $POD_PORT -i `"$SSH_KEY`" `"$FILE`" ${POD_HOST}:${REMOTE_BASE}/$FILE"

try {
    Invoke-Expression $scpCmd
    Write-Host "✅ Uploaded." -ForegroundColor Green
    Write-Host "`nRun on Pod:"
    Write-Host "python A40_TRAIN_50B_PHASE3_FIX.py" -ForegroundColor White
}
catch {
    Write-Host "❌ Failed: $_" -ForegroundColor Red
}

Write-Host "Press Enter to exit..."
Read-Host
