<#
.SYNOPSIS
    FiveForFree - December 10 Backtest Simulation
.DESCRIPTION
    Runs backtest simulation:
    - Train models on data before December 10, 2025
    - Simulate trading on December 10-20, 2025
.PARAMETER TestDays
    Number of days for test period (default: 10)
.PARAMETER ModelType
    Model type to use: lightgbm or xgboost (default: lightgbm)
.PARAMETER MaxTickers
    Maximum number of tickers to process
.EXAMPLE
    .\run-dec10-backtest.ps1
    .\run-dec10-backtest.ps1 -TestDays 10 -ModelType lightgbm
#>

param(
    [string]$CutoffDate = "2025-12-10",
    [int]$TestDays = 10,
    [string]$ModelType = "lightgbm",
    [int]$MaxTickers = 0,
    [switch]$Verbose,
    [switch]$Help
)

# Get project root
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not $ProjectRoot) { $ProjectRoot = "F:\Git\real_multi_agetns\projects\FiveForFree" }

function Show-Banner {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host " FiveForFree December 10 Backtest" -ForegroundColor Cyan
    Write-Host " $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
}

function Show-Help {
    Show-Banner
    Write-Host "Usage: .\run-dec10-backtest.ps1 [options]" -ForegroundColor White
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  -CutoffDate   Training cutoff date (default: 2025-12-10)"
    Write-Host "  -TestDays     Number of test days (default: 10)"
    Write-Host "  -ModelType    Model type: lightgbm or xgboost (default: lightgbm)"
    Write-Host "  -MaxTickers   Limit number of tickers (default: all)"
    Write-Host "  -Verbose      Show detailed output"
    Write-Host "  -Help         Show this help message"
    Write-Host ""
    Write-Host "This script:" -ForegroundColor Yellow
    Write-Host "  1. Trains models on data before cutoff date"
    Write-Host "  2. Simulates trades on test period"
    Write-Host "  3. Reports performance metrics"
    Write-Host ""
    exit 0
}

if ($Help) { Show-Help }

Show-Banner

# Check Python venv
$VenvPython = "$ProjectRoot\venv\Scripts\python.exe"
$SystemPython = "python"

if (Test-Path $VenvPython) {
    Write-Host "[OK] Using virtual environment Python" -ForegroundColor Green
    $Python = $VenvPython
} else {
    Write-Host "[WARN] Using system Python (venv not found)" -ForegroundColor Yellow
    $Python = $SystemPython
}

# Build arguments
$Args = @(
    "$ProjectRoot\scripts\run_dec10_backtest.py",
    "--cutoff-date", $CutoffDate,
    "--test-days", $TestDays,
    "--model-type", $ModelType
)

if ($MaxTickers -gt 0) {
    $Args += "--max-tickers"
    $Args += $MaxTickers
}

if ($Verbose) {
    $Args += "--verbose"
}

Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Cutoff Date:  $CutoffDate"
Write-Host "  Test Days:    $TestDays"
Write-Host "  Model Type:   $ModelType"
if ($MaxTickers -gt 0) {
    Write-Host "  Max Tickers:  $MaxTickers"
}
Write-Host ""

Write-Host "Starting backtest simulation..." -ForegroundColor Cyan
Write-Host ""

# Run the script
Push-Location $ProjectRoot
& $Python $Args
$ExitCode = $LASTEXITCODE
Pop-Location

Write-Host ""
if ($ExitCode -eq 0) {
    Write-Host "[OK] Backtest completed successfully!" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Backtest failed with exit code: $ExitCode" -ForegroundColor Red
}

exit $ExitCode
