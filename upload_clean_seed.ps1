
# UPLOAD CLEAN SEED TO RUNPOD
# Save as: upload_clean_seed.ps1

$POD_HOST = "root@69.30.85.39"
$POD_PORT = "22065"
$SSH_KEY = "$env:USERPROFILE\.ssh\id_ed25519" 

$REMOTE_BASE = "/workspace/zetagrid_50b"
$FILE = "zetagrid_50b_seed_clean.pt"

if (-not (Test-Path $FILE)) {
    Write-Host "❌ File not found: $FILE" -ForegroundColor Red
    Write-Host "⚠️  Did you run EXPAND_25B_TO_50B_CLEAN.py first?" -ForegroundColor Yellow
    exit 1
}

$size = (Get-Item $FILE).Length / 1MB
Write-Host "📤 Uploading NO-NOISE SEED ($($size.ToString('0.00')) MB)..." -ForegroundColor Yellow

$scpCmd = "scp -P $POD_PORT -i `"$SSH_KEY`" `"$FILE`" ${POD_HOST}:${REMOTE_BASE}/zetagrid_50b_seed.pt"

# Note: We rename it to 'zetagrid_50b_seed.pt' on the remote so the script picks it up automatically!

try {
    Invoke-Expression $scpCmd
    Write-Host "✅ Uploaded." -ForegroundColor Green
    Write-Host "`nNow run on Pod:"
    Write-Host "python A40_TRAIN_50B_PHASE3_ULTIMATE_FIX.py" -ForegroundColor White
}
catch {
    Write-Host "❌ Failed: $_" -ForegroundColor Red
}

Write-Host "Press Enter to exit..."
Read-Host
