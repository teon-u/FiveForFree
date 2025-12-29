<#
.SYNOPSIS
    Build FiveForFree Launcher exe
.DESCRIPTION
    Uses PyInstaller to create a standalone exe file for the launcher.
.EXAMPLE
    .\build-launcher.ps1
#>

param(
    [switch]$Clean,
    [switch]$Help
)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not $ProjectRoot) { $ProjectRoot = "F:\Git\real_multi_agetns\projects\FiveForFree" }

function Show-Help {
    Write-Host ""
    Write-Host "FiveForFree Launcher Builder" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: .\build-launcher.ps1 [options]" -ForegroundColor White
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  -Clean    Remove build artifacts before building"
    Write-Host "  -Help     Show this help message"
    Write-Host ""
    exit 0
}

if ($Help) { Show-Help }

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " FiveForFree Launcher Builder" -ForegroundColor Cyan
Write-Host " $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check for PyInstaller
Write-Host "[1/4] Checking PyInstaller..." -ForegroundColor Yellow
$VenvPip = "$ProjectRoot\venv\Scripts\pip.exe"
$VenvPython = "$ProjectRoot\venv\Scripts\python.exe"

if (Test-Path $VenvPython) {
    $Python = $VenvPython
    $Pip = $VenvPip
} else {
    $Python = "python"
    $Pip = "pip"
}

& $Pip show pyinstaller 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Installing PyInstaller..." -ForegroundColor Gray
    & $Pip install pyinstaller
}
Write-Host "  [OK] PyInstaller ready" -ForegroundColor Green

# Clean if requested
if ($Clean) {
    Write-Host ""
    Write-Host "[2/4] Cleaning build artifacts..." -ForegroundColor Yellow
    Remove-Item -Path "$ProjectRoot\build" -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -Path "$ProjectRoot\dist" -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "  [OK] Cleaned" -ForegroundColor Green
} else {
    Write-Host "[2/4] Skip cleaning (use -Clean to remove old builds)" -ForegroundColor Gray
}

# Build
Write-Host ""
Write-Host "[3/4] Building exe..." -ForegroundColor Yellow
Push-Location $ProjectRoot

& $Python -m PyInstaller `
    --onefile `
    --windowed `
    --name "FiveForFree Launcher" `
    --clean `
    launcher.py

$ExitCode = $LASTEXITCODE
Pop-Location

if ($ExitCode -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] Build failed with exit code: $ExitCode" -ForegroundColor Red
    exit $ExitCode
}

Write-Host "  [OK] Build complete" -ForegroundColor Green

# Show result
Write-Host ""
Write-Host "[4/5] Result:" -ForegroundColor Yellow
$ExePath = "$ProjectRoot\dist\FiveForFree Launcher.exe"
if (Test-Path $ExePath) {
    $FileInfo = Get-Item $ExePath
    Write-Host "  Location: $ExePath" -ForegroundColor White
    Write-Host "  Size: $([math]::Round($FileInfo.Length / 1MB, 2)) MB" -ForegroundColor White
    Write-Host "  [OK] Build complete" -ForegroundColor Green
} else {
    Write-Host "  [WARN] exe not found at expected location" -ForegroundColor Yellow
}

# Copy exe to project root for easier access
Write-Host ""
Write-Host "[5/5] Copying exe to project root..." -ForegroundColor Yellow
$DestPath = "$ProjectRoot\FiveForFree Launcher.exe"
if (Test-Path $ExePath) {
    Copy-Item -Path $ExePath -Destination $DestPath -Force
    if (Test-Path $DestPath) {
        Write-Host "  Copied to: $DestPath" -ForegroundColor White
        Write-Host "  [OK] exe ready for use!" -ForegroundColor Green
    } else {
        Write-Host "  [WARN] Copy failed" -ForegroundColor Yellow
    }
} else {
    Write-Host "  [SKIP] No exe to copy" -ForegroundColor Gray
}

Write-Host ""
Write-Host "[SUCCESS] Launcher exe created!" -ForegroundColor Green
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
