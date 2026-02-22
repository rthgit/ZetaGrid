
# FORCE DEPLOY (Use with Caution - Overwrites Remote)
# Save as: force_deploy.ps1

Write-Host "🚀 ZETAGRID FORCE DEPLOY" -ForegroundColor Red

# 1. Initialize & Add
if (-not (Test-Path .git)) { git init }

# Configure safe directory just in case
git config --global --add safe.directory E:/ZETAGRID

# Add all release files (excluding heavy ones via .gitignore logic or manual check)
git add ZETAGRID_25B_CARD.md
git add RELEASE_INSTRUCTIONS.md
git add ZETAGRID_INFERENCE.py
git add QULP_2BIT_QUANTIZER.py
git add EXPAND_25B_TO_50B_CLEAN.py
git add README.md 2>$null

git commit -m "ZetaGrid 25B Release (Force Push)"

# 2. Remote & Force Push
$REMOTE_URL = "https://github.com/rthgit/ZetaGrid.git"

if (-not (git remote | Select-String "origin")) {
    git remote add origin $REMOTE_URL
}
else {
    git remote set-url origin $REMOTE_URL
}

Write-Host "⚠️  OVERWRITING REMOTE REPOSITORY..." -ForegroundColor Yellow
git push -u origin master --force

if ($LASTEXITCODE -ne 0) {
    Write-Host "Trying main branch..."
    git push -u origin main --force
}

Write-Host "✅ DONE!" -ForegroundColor Green
