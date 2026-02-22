# Upload Ollama Patches to GitHub
$ErrorActionPreference = "Stop"

Write-Host "🚀 INITIALIZING GIT UPLOAD FOR OLLAMA PATCHES..." -ForegroundColor Cyan

# 1. Initialize Git
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

# 3. Add Specific Files
$Files = @(
    "rth_tcn_ops.cpp",
    "rth_tcn_ops.h",
    "convert_rth_to_gguf.py",
    "Modelfile_RTH-LM",
    "OLLAMA_PATCH_GUIDE.md"
)

foreach ($file in $Files) {
    if (Test-Path $file) {
        git add $file
        Write-Host "   ✅ Added $file" -ForegroundColor Green
    }
    else {
        Write-Warning "   ❌ File not found: $file"
    }
}

# 4. Commit and Push
try {
    git commit -m "Add Ollama/llama.cpp custom TCN kernels and patch guide"
    Write-Host "   ✅ Committed changes"
}
catch {
    Write-Host "   ⚠️ Nothing to commit (files might be unchanged)" -ForegroundColor Yellow
}

Write-Host "🚀 PUSHING TO GITHUB..." -ForegroundColor Cyan
git push -u origin main
if ($LASTEXITCODE -ne 0) {
    Write-Host "   ⚠️ Push to 'main' failed, trying 'master'..." -ForegroundColor Yellow
    git push -u origin master
}

Write-Host "`n✅ UPLOAD COMPLETE!" -ForegroundColor Green
