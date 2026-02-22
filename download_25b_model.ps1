
# DOWNLOAD ZETAGRID 25B FROM RUNPOD A40
# Save as: download_25b_model.ps1

# ============================================================
# CONFIGURATION
# ============================================================

$POD_HOST = "root@69.30.85.39"
$POD_PORT = "22065"
# Assuming standard Windows SSH path
$SSH_KEY = "$env:USERPROFILE\.ssh\id_ed25519" 

$REMOTE_FILE = "/workspace/zetagrid_50b/phase2_checkpoints/zeta25b_step15000.pt"
$LOCAL_DIR = "$env:USERPROFILE\Desktop\cpu-da\models"

# Create local directory if it doesn't exist
if (-not (Test-Path $LOCAL_DIR)) {
    New-Item -ItemType Directory -Force -Path $LOCAL_DIR | Out-Null
}

$LOCAL_FILE = Join-Path $LOCAL_DIR "zeta25b_step15000.pt"

# ============================================================
# SCRIPT
# ============================================================

Write-Host "`n╔══════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║       ZETAGRID 25B DOWNLOADER (RUNPOD A40)              ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════╝`n" -ForegroundColor Cyan

Write-Host "  Remote File: $REMOTE_FILE" -ForegroundColor Yellow
Write-Host "  Local Dest : $LOCAL_FILE" -ForegroundColor Yellow
Write-Host "  Size       : ~500 MB (Trainable Params)" -ForegroundColor Yellow

# Verify local key exists
if (-not (Test-Path $SSH_KEY)) {
    Write-Host "`n❌ SSH Key not found at: $SSH_KEY" -ForegroundColor Red
    Write-Host "Please update the `$SSH_KEY variable in this script." -ForegroundColor Red
    exit 1
}

# Construct SCP command
# Using scp from standard OpenSSH in Windows
$scpCmd = "scp -P $POD_PORT -i `"$SSH_KEY`" ${POD_HOST}:${REMOTE_FILE} `"$LOCAL_FILE`""

Write-Host "`n🚀 Starting Download..." -ForegroundColor Green

try {
    # Execute SCP
    Invoke-Expression $scpCmd
    
    if (Test-Path $LOCAL_FILE) {
        $size = (Get-Item $LOCAL_FILE).Length / 1MB
        Write-Host "`n✅ DOWNLOAD COMPLETE!" -ForegroundColor Green
        Write-Host "   File: $LOCAL_FILE" -ForegroundColor White
        Write-Host "   Size: $($size.ToString('0.00')) MB" -ForegroundColor White
    }
    else {
        Write-Host "`n❌ Download command ran, but file not found locally." -ForegroundColor Red
    }
}
catch {
    Write-Host "`n❌ ERROR: $_" -ForegroundColor Red
}

Write-Host "`nPress Enter to exit..." 
Read-Host
