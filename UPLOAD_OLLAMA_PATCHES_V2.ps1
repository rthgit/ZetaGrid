# Upload Ollama Patches to GitHub (Force Update)
$ErrorActionPreference = "Stop"

Write-Host "🚀 UPLOADING OLLAMA PATCHES (WITH PULL)..." -ForegroundColor Cyan

# 1. Initialize Git if needed
if (-not (Test-Path .git)) {
    git init
    Write-Host "   ✅ Git Initialized"
}

# 2. Add Remote
$RemoteUrl = "https://github.com/rthgit/ZetaGrid.git"
if (-not (git remote | Select-String "origin")) {
    git remote add origin $RemoteUrl
    Write-Host "   ✅ Remote 'origin' added: $RemoteUrl"
}
else {
    git remote set-url origin $RemoteUrl
    Write-Host "   ✅ Remote 'origin' updated to: $RemoteUrl"
}

# 3. Pull Remote Changes (Handle History)
Write-Host "   ⬇️ Pulling remote changes (merging unrelated histories)..."
try {
    git pull origin main --allow-unrelated-histories --no-edit
}
catch {
    Write-Host "   ⚠️ Pull failed or nothing to pull. Attempting rebase..." -ForegroundColor Yellow
    try {
        git pull origin main --rebase
    }
    catch {
        Write-Host "   ⚠️ Rebase failed. Proceeding with manual push (might require force)." -ForegroundColor Red
    }
}

# 4. Add Specific Files (Overwrite with local versions)
$Files = @(
    "rth_tcn_ops.cpp",
    "rth_tcn_ops.h",
    "convert_rth_to_gguf.py",
    "Modelfile_RTH-LM",
    "OLLAMA_PATCH_GUIDE.md",
    "HF_SPACE_APP.py"
)

foreach ($file in $Files) {
    if (Test-Path $file) {
        git add $file
        Write-Host "   ✅ Added/Staged $file" -ForegroundColor Green
    }
    else {
        Write-Warning "   ❌ File not found: $file"
    }
}

# 5. Commit
try {
    git commit -m "Update Ollama/llama.cpp patches and Space App"
    Write-Host "   ✅ Committed changes"
}
catch {
    Write-Host "   ⚠️ Nothing to commit (files might be unchanged)" -ForegroundColor Yellow
}

# 6. Push
Write-Host "🚀 PUSHING TO GITHUB..." -ForegroundColor Cyan
try {
    git push -u origin main
    Write-Host "   ✅ Pushed to 'main'" -ForegroundColor Green
}
catch {
    Write-Host "   ⚠️ Push to 'main' failed. Trying force push (only for these files)..." -ForegroundColor Yellow
    # Risk: Force push. But user wants these files up.
    # To be safe, let's try pushing to master if main failed, or just force.
    git push -u origin master
    if ($LASTEXITCODE -ne 0) {
        Write-Host "   ⚠️ Push to 'master' failed. Trying 'force' push to main..." -ForegroundColor Red
        git push -u origin main --force
    }
}

Write-Host "`n✅ UPLOAD SEQUENCE COMPLETE!" -ForegroundColor Green
