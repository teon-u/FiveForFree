<#
.SYNOPSIS
    FiveForFree - System Starter with Health Check
.DESCRIPTION
    Starts the FiveForFree backend and frontend systems.
    Performs health checks before starting.
.PARAMETER SkipHealthCheck
    Skip the pre-start health check
.PARAMETER BackendOnly
    Start only the backend system
.PARAMETER FrontendOnly
    Start only the frontend development server
.EXAMPLE
    .\start-fiveforfree.ps1
    .\start-fiveforfree.ps1 -BackendOnly
    .\start-fiveforfree.ps1 -FrontendOnly
#>

param(
    [switch]$SkipHealthCheck,
    [switch]$BackendOnly,
    [switch]$FrontendOnly,
    [switch]$Help
)

if ($Help) {
    Get-Help $MyInvocation.MyCommand.Path -Detailed
    exit 0
}

# Get project root (parent of scripts folder)
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not $ProjectRoot) { $ProjectRoot = "F:\Git\real_multi_agetns\projects\FiveForFree" }

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " FiveForFree System Starter" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Health Check
if (-not $SkipHealthCheck) {
    Write-Host "[1/3] Running Health Check..." -ForegroundColor Yellow
    Write-Host ""

    $allPassed = $true

    # Check Python
    try {
        $pythonVersion = python --version 2>&1
        Write-Host "  [OK] Python: $pythonVersion" -ForegroundColor Green
    } catch {
        Write-Host "  [FAIL] Python not found" -ForegroundColor Red
        $allPassed = $false
    }

    # Check Node.js
    try {
        $nodeVersion = node --version 2>&1
        Write-Host "  [OK] Node.js: $nodeVersion" -ForegroundColor Green
    } catch {
        Write-Host "  [FAIL] Node.js not found" -ForegroundColor Red
        $allPassed = $false
    }

    # Check required folders/files
    $healthChecks = @(
        @{Name = "scripts/run_system.py"; Path = "$ProjectRoot\scripts\run_system.py" },
        @{Name = "frontend folder"; Path = "$ProjectRoot\frontend" },
        @{Name = "src folder"; Path = "$ProjectRoot\src" },
        @{Name = "config folder"; Path = "$ProjectRoot\config" },
        @{Name = ".env file"; Path = "$ProjectRoot\.env" }
    )

    foreach ($check in $healthChecks) {
        if (Test-Path $check.Path) {
            Write-Host "  [OK] $($check.Name)" -ForegroundColor Green
        } else {
            Write-Host "  [FAIL] $($check.Name)" -ForegroundColor Red
            $allPassed = $false
        }
    }

    Write-Host ""

    if (-not $allPassed) {
        Write-Host "[ERROR] Health check failed! Please fix the issues above." -ForegroundColor Red
        exit 1
    }

    Write-Host "[OK] Health check passed!" -ForegroundColor Green
    Write-Host ""
}

# Start Systems
$stepNum = if ($SkipHealthCheck) { 1 } else { 2 }
$totalSteps = if ($SkipHealthCheck) { 2 } else { 3 }

if (-not $FrontendOnly) {
    Write-Host "[$stepNum/$totalSteps] Starting Backend System..." -ForegroundColor Yellow

    Start-Process powershell -ArgumentList @(
        "-NoExit",
        "-Command",
        "Set-Location '$ProjectRoot'; Write-Host 'Starting FiveForFree Backend...' -ForegroundColor Cyan; python scripts/run_system.py"
    )

    Write-Host "  [STARTED] Backend (python scripts/run_system.py)" -ForegroundColor Yellow
    Start-Sleep -Seconds 2
    $stepNum++
}

if (-not $BackendOnly) {
    Write-Host "[$stepNum/$totalSteps] Starting Frontend Dev Server..." -ForegroundColor Yellow

    Start-Process powershell -ArgumentList @(
        "-NoExit",
        "-Command",
        "Set-Location '$ProjectRoot\frontend'; Write-Host 'Starting FiveForFree Frontend...' -ForegroundColor Cyan; npm run dev"
    )

    Write-Host "  [STARTED] Frontend (npm run dev)" -ForegroundColor Yellow
    Start-Sleep -Seconds 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " FiveForFree started successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Access the application:" -ForegroundColor Gray
Write-Host "  Frontend: http://localhost:5173" -ForegroundColor White
Write-Host "  Backend API: http://localhost:8000" -ForegroundColor White
Write-Host ""
