<#
.SYNOPSIS
    FiveForFree - Enhanced Backtest with All Improvements
.DESCRIPTION
    Runs enhanced backtest with all 4 CEO-requested improvements:
    1. S+ Grade (win_rate >= 80% AND PF >= 4.0)
    2. Rebalancing every 5 days
    3. Transaction costs (0.25% commission, 0.1% slippage, -5% stop loss)
    4. Concentration strategy (S+ -> top 2, else A-grade 4)
.PARAMETER CutoffDate
    Training cutoff date (default: 2025-12-10)
.PARAMETER TestDays
    Number of test days (default: 10)
.PARAMETER RebalanceDays
    Rebalancing period in days (default: 5)
.EXAMPLE
    .\run-enhanced-backtest.ps1
    .\run-enhanced-backtest.ps1 -CutoffDate "2025-12-10" -TestDays 10 -RebalanceDays 5
#>

param(
    [string]$CutoffDate = "2025-12-10",
    [int]$TestDays = 10,
    [int]$RebalanceDays = 5,
    [double]$Commission = 0.25,
    [double]$Slippage = 0.10,
    [double]$StopLoss = 5.0,
    [switch]$Verbose,
    [switch]$Help
)

# Get project root
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not $ProjectRoot) { $ProjectRoot = "F:\Git\real_multi_agetns\projects\FiveForFree" }

function Show-Banner {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host " FiveForFree Enhanced Backtest" -ForegroundColor Cyan
    Write-Host " All 4 CEO Improvements" -ForegroundColor Yellow
    Write-Host " $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
}

function Show-Help {
    Show-Banner
    Write-Host "Usage: .\run-enhanced-backtest.ps1 [options]" -ForegroundColor White
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  -CutoffDate     Training cutoff date (default: 2025-12-10)"
    Write-Host "  -TestDays       Number of test days (default: 10)"
    Write-Host "  -RebalanceDays  Rebalancing period in days (default: 5)"
    Write-Host "  -Commission     Commission % (default: 0.25)"
    Write-Host "  -Slippage       Slippage % (default: 0.10)"
    Write-Host "  -StopLoss       Stop loss % (default: 5.0)"
    Write-Host "  -Verbose        Show detailed output"
    Write-Host "  -Help           Show this help message"
    Write-Host ""
    Write-Host "Improvements implemented:" -ForegroundColor Yellow
    Write-Host "  1. S+ Grade: win_rate >= 80% AND PF >= 4.0"
    Write-Host "  2. Rebalancing: Every 5 days recalculation"
    Write-Host "  3. Transaction Costs: commission, slippage, stop loss"
    Write-Host "  4. Concentration: S+ -> top 2 (50%), else A-grade 4 (25%)"
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
    "$ProjectRoot\scripts\run_enhanced_backtest.py",
    "--cutoff-date", $CutoffDate,
    "--test-days", $TestDays,
    "--rebalance-days", $RebalanceDays,
    "--commission", $Commission,
    "--slippage", $Slippage,
    "--stop-loss", $StopLoss
)

if ($Verbose) {
    $Args += "--verbose"
}

Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Cutoff Date:     $CutoffDate"
Write-Host "  Test Days:       $TestDays"
Write-Host "  Rebalance Days:  $RebalanceDays"
Write-Host "  Commission:      $Commission%"
Write-Host "  Slippage:        $Slippage%"
Write-Host "  Stop Loss:       $StopLoss%"
Write-Host ""

Write-Host "Improvements:" -ForegroundColor Cyan
Write-Host "  [1] S+ Grade (WR>=80%, PF>=4.0)"
Write-Host "  [2] 5-Day Rebalancing"
Write-Host "  [3] Transaction Costs Applied"
Write-Host "  [4] Concentration Strategy"
Write-Host ""

Write-Host "Starting enhanced backtest..." -ForegroundColor Cyan
Write-Host ""

# Run the script
Push-Location $ProjectRoot
& $Python $Args
$ExitCode = $LASTEXITCODE
Pop-Location

Write-Host ""
if ($ExitCode -eq 0) {
    Write-Host "[OK] Enhanced backtest completed successfully!" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Enhanced backtest failed with exit code: $ExitCode" -ForegroundColor Red
}

exit $ExitCode
