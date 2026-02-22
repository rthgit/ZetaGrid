
# DEPLOY ZETAGRID RELEASE TO GIT
# Save as: deploy_to_git.ps1

Write-Host "🚀 ZETAGRID GIT DEPLOYMENT MANAGER" -ForegroundColor Cyan

# 1. Initialize Git if needed
if (-not (Test-Path .git)) {
    Write-Host "   Initializing Git in $(Get-Location)..."
    git init
}

# 2. Configure .gitignore (Crucial to avoid uploading 7GB files by accident)
$gitignore = @(
    "*.npy",        # Ignore large Genome
    "*.pt",         # Ignore large Checkpoints
    "*.bin",        # Ignore Datasets
    "__pycache__/",
    "wandb/"
)
Set-Content -Path .gitignore -Value $gitignore
Write-Host "   ✅ Configured .gitignore (Ignoring .npy, .pt)"

# 3. Add Important Files
# We WANT to add the Model Card, Instructions, Python Scripts
# We might want to add the quantized model if it's small (0.06GB is OK for git!)
# But wait, .qulp is binary. 60MB is fine for Git.

git add ZETAGRID_25B_CARD.md
git add RELEASE_INSTRUCTIONS.md
git add ZETAGRID_INFERENCE.py
# --- 3. Gather Files ---
$FilesToRelease = @(
    "README.md",
    "rth_logo.png",
    "ZENODO_PAPER.md",
    "SCALING_REPORT_120B.md",
    "ZETAGRID_25B_CARD.md",
    "ZETAGRID_INFERENCE.py",
    "RELEASE_INSTRUCTIONS.md",
    "QULP_2BIT_QUANTIZER.py",
    "EXPAND_25B_TO_50B_CLEAN.py",
    "README_RELEASE.md"
)

foreach ($file in $FilesToRelease) {
    if (Test-Path $file) {
        git add $file
        Write-Host "   ✅ Added $file" -ForegroundColor Green
    }
}


# Force add the quantized model (it's small enough, <100MB)
if (Test-Path zeta25b_2bit.qulp) {
    if ((Get-Item zeta25b_2bit.qulp).Length -lt 100MB) {
        git add -f zeta25b_2bit.qulp
        Write-Host "   ✅ Added zeta25b_2bit.qulp (Lite Model)"
    }
    else {
        Write-Host "   ⚠️ Skipping zeta25b_2bit.qulp (Too large >100MB)" -ForegroundColor Yellow
    }
}


git commit -m "ZetaGrid 25B Phase 2 Release"

# 4. Push to Remote
$REMOTE_URL = "https://github.com/rthgit/ZetaGrid.git"

if (-not (git remote | Select-String "origin")) {
    git remote add origin $REMOTE_URL
    Write-Host "   ✅ Remote 'origin' added: $REMOTE_URL"
}
else {
    git remote set-url origin $REMOTE_URL
    Write-Host "   ✅ Remote 'origin' updated."
}

Write-Host "`n🚀 PUSHING TO GITHUB..." -ForegroundColor Cyan
git push -u origin master
# If master fails, try main
if ($LASTEXITCODE -ne 0) {
    git push -u origin main
}

Write-Host "`n✅ DEPLOYMENT COMPLETE!" -ForegroundColor Green
