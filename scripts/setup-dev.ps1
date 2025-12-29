<#
.SYNOPSIS
    FiveForFree - Development Environment Setup
.DESCRIPTION
    Automatically sets up the development environment:
    - Creates Python virtual environment
    - Installs Python dependencies
    - Installs Node.js dependencies
    - Copies .env.example to .env if needed
.PARAMETER SkipPython
    Skip Python environment setup
.PARAMETER SkipNode
    Skip Node.js setup
.PARAMETER Force
    Force reinstall even if already configured
.EXAMPLE
    .\setup-dev.ps1           # Full setup
    .\setup-dev.ps1 -SkipNode # Python only
    .\setup-dev.ps1 -Force    # Force reinstall
#>

param(
    [switch]$SkipPython,
    [switch]$SkipNode,
    [switch]$Force,
    [switch]$Help
)

# Get project root
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not $ProjectRoot) { $ProjectRoot = "F:\Git\real_multi_agetns\projects\FiveForFree" }

function Show-Banner {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host " FiveForFree Development Setup" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
}

function Show-Help {
    Show-Banner
    Write-Host "Usage: .\setup-dev.ps1 [options]" -ForegroundColor White
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  -SkipPython  Skip Python virtual environment setup"
    Write-Host "  -SkipNode    Skip Node.js dependencies installation"
    Write-Host "  -Force       Force reinstall even if already configured"
    Write-Host "  -Help        Show this help message"
    Write-Host ""
    Write-Host "What this script does:" -ForegroundColor Yellow
    Write-Host "  1. Creates Python virtual environment (venv)"
    Write-Host "  2. Installs Python dependencies from requirements.txt"
    Write-Host "  3. Installs Node.js dependencies (npm install)"
    Write-Host "  4. Copies .env.example to .env if needed"
    Write-Host ""
    exit 0
}

if ($Help) { Show-Help }

Show-Banner

$totalSteps = 4
$currentStep = 0

# Check prerequisites
Write-Host "[CHECK] Verifying prerequisites..." -ForegroundColor Yellow
Write-Host ""

$hasErrors = $false

# Check Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  [OK] Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  [FAIL] Python not found" -ForegroundColor Red
    Write-Host "         Please install Python 3.8+ from https://python.org" -ForegroundColor Gray
    $hasErrors = $true
}

# Check pip
try {
    $pipVersion = pip --version 2>&1
    Write-Host "  [OK] pip: Available" -ForegroundColor Green
} catch {
    Write-Host "  [FAIL] pip not found" -ForegroundColor Red
    $hasErrors = $true
}

# Check Node.js
try {
    $nodeVersion = node --version 2>&1
    Write-Host "  [OK] Node.js: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "  [FAIL] Node.js not found" -ForegroundColor Red
    Write-Host "         Please install Node.js 18+ from https://nodejs.org" -ForegroundColor Gray
    $hasErrors = $true
}

# Check npm
try {
    $npmVersion = npm --version 2>&1
    Write-Host "  [OK] npm: $npmVersion" -ForegroundColor Green
} catch {
    Write-Host "  [FAIL] npm not found" -ForegroundColor Red
    $hasErrors = $true
}

Write-Host ""

if ($hasErrors) {
    Write-Host "[ERROR] Prerequisites not met. Please install missing tools." -ForegroundColor Red
    exit 1
}

Write-Host "[OK] All prerequisites met!" -ForegroundColor Green
Write-Host ""

# Step 1: Python Virtual Environment
$currentStep++
if (-not $SkipPython) {
    Write-Host "[$currentStep/$totalSteps] Setting up Python virtual environment..." -ForegroundColor Yellow

    $venvPath = "$ProjectRoot\venv"

    if ((Test-Path $venvPath) -and (-not $Force)) {
        Write-Host "  [SKIP] Virtual environment already exists" -ForegroundColor Gray
        Write-Host "         Use -Force to recreate" -ForegroundColor Gray
    } else {
        if (Test-Path $venvPath) {
            Write-Host "  Removing existing venv..." -ForegroundColor Gray
            Remove-Item -Recurse -Force $venvPath
        }

        Write-Host "  Creating virtual environment..." -ForegroundColor Cyan
        python -m venv $venvPath

        if (Test-Path $venvPath) {
            Write-Host "  [OK] Virtual environment created at: $venvPath" -ForegroundColor Green
        } else {
            Write-Host "  [FAIL] Failed to create virtual environment" -ForegroundColor Red
            exit 1
        }
    }
    Write-Host ""
} else {
    Write-Host "[$currentStep/$totalSteps] Skipping Python venv (--SkipPython)" -ForegroundColor Gray
    Write-Host ""
}

# Step 2: Install Python dependencies
$currentStep++
if (-not $SkipPython) {
    Write-Host "[$currentStep/$totalSteps] Installing Python dependencies..." -ForegroundColor Yellow

    $requirementsPath = "$ProjectRoot\requirements.txt"
    $venvPip = "$ProjectRoot\venv\Scripts\pip.exe"

    if (Test-Path $requirementsPath) {
        if (Test-Path $venvPip) {
            Write-Host "  Installing from requirements.txt..." -ForegroundColor Cyan
            & $venvPip install -r $requirementsPath --quiet
            Write-Host "  [OK] Python dependencies installed" -ForegroundColor Green
        } else {
            Write-Host "  [WARN] venv pip not found, using global pip" -ForegroundColor Yellow
            pip install -r $requirementsPath --quiet
            Write-Host "  [OK] Python dependencies installed (global)" -ForegroundColor Green
        }
    } else {
        Write-Host "  [SKIP] requirements.txt not found" -ForegroundColor Gray
    }
    Write-Host ""
} else {
    Write-Host "[$currentStep/$totalSteps] Skipping Python deps (--SkipPython)" -ForegroundColor Gray
    Write-Host ""
}

# Step 3: Install Node.js dependencies
$currentStep++
if (-not $SkipNode) {
    Write-Host "[$currentStep/$totalSteps] Installing Node.js dependencies..." -ForegroundColor Yellow

    $frontendPath = "$ProjectRoot\frontend"
    $nodeModulesPath = "$frontendPath\node_modules"

    if ((Test-Path $nodeModulesPath) -and (-not $Force)) {
        Write-Host "  [SKIP] node_modules already exists" -ForegroundColor Gray
        Write-Host "         Use -Force to reinstall" -ForegroundColor Gray
    } else {
        if (Test-Path "$frontendPath\package.json") {
            Write-Host "  Running npm install..." -ForegroundColor Cyan
            Push-Location $frontendPath
            npm install --silent
            Pop-Location
            Write-Host "  [OK] Node.js dependencies installed" -ForegroundColor Green
        } else {
            Write-Host "  [SKIP] package.json not found in frontend/" -ForegroundColor Gray
        }
    }
    Write-Host ""
} else {
    Write-Host "[$currentStep/$totalSteps] Skipping Node.js deps (--SkipNode)" -ForegroundColor Gray
    Write-Host ""
}

# Step 4: Environment file
$currentStep++
Write-Host "[$currentStep/$totalSteps] Setting up environment file..." -ForegroundColor Yellow

$envPath = "$ProjectRoot\.env"
$envExamplePath = "$ProjectRoot\.env.example"

if ((Test-Path $envPath) -and (-not $Force)) {
    Write-Host "  [SKIP] .env already exists" -ForegroundColor Gray
} elseif (Test-Path $envExamplePath) {
    Copy-Item $envExamplePath $envPath -Force
    Write-Host "  [OK] Copied .env.example to .env" -ForegroundColor Green
    Write-Host "  [INFO] Please update .env with your API keys" -ForegroundColor Yellow
} else {
    Write-Host "  [SKIP] .env.example not found" -ForegroundColor Gray
}

Write-Host ""

# Summary
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Update .env with your API keys (if needed)"
Write-Host "  2. Run: .\fff.ps1 start" -ForegroundColor White
Write-Host ""
Write-Host "Quick commands:" -ForegroundColor Yellow
Write-Host "  .\fff.ps1 start    # Start the system"
Write-Host "  .\fff.ps1 status   # Check status"
Write-Host "  .\fff.ps1 stop     # Stop the system"
Write-Host ""
