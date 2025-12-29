<#
.SYNOPSIS
    FiveForFree - Unified CLI Tool
.DESCRIPTION
    Integrated command-line interface for FiveForFree system management.
    Supports start, stop, status, and logs commands.
.PARAMETER Command
    The command to execute: start, stop, status, logs
.PARAMETER Target
    Target for logs command: backend, frontend, all
.EXAMPLE
    .\fff.ps1 start          # Start entire system
    .\fff.ps1 stop           # Stop entire system
    .\fff.ps1 status         # Check system status
    .\fff.ps1 logs backend   # View backend logs
    .\fff.ps1 logs frontend  # View frontend logs
#>

param(
    [Parameter(Position=0)]
    [ValidateSet("start", "stop", "status", "logs", "help")]
    [string]$Command = "help",

    [Parameter(Position=1)]
    [string]$Target = "all",

    [switch]$Help
)

# Get project root
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not $ProjectRoot) { $ProjectRoot = "F:\Git\real_multi_agetns\projects\FiveForFree" }

# Port configuration
$BackendPort = 8000
$FrontendPort = 5173

function Show-Banner {
    Write-Host ""
    Write-Host "  _____ _          _____          _____              " -ForegroundColor Cyan
    Write-Host " |  ___(_)_   ____| ____|__  _ __|  ___|_ __ ___  ___" -ForegroundColor Cyan
    Write-Host " | |_  | \ \ / / _ \  _|/ _ \| '__| |_ | '__/ _ \/ _ \" -ForegroundColor Cyan
    Write-Host " |  _| | |\ V /  __/ | | (_) | |  |  _|| | |  __/  __/" -ForegroundColor Cyan
    Write-Host " |_|   |_| \_/ \___|_|  \___/|_|  |_|  |_|  \___|\___|" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  FiveForFree CLI v1.0" -ForegroundColor Gray
    Write-Host ""
}

function Show-Help {
    Show-Banner
    Write-Host "Usage: .\fff.ps1 <command> [target]" -ForegroundColor White
    Write-Host ""
    Write-Host "Commands:" -ForegroundColor Yellow
    Write-Host "  start          Start the entire system (backend + frontend)"
    Write-Host "  stop           Stop all running processes"
    Write-Host "  status         Check system status"
    Write-Host "  logs <target>  View logs (backend, frontend, all)"
    Write-Host "  help           Show this help message"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  .\fff.ps1 start"
    Write-Host "  .\fff.ps1 status"
    Write-Host "  .\fff.ps1 logs backend"
    Write-Host ""
}

function Test-PortInUse {
    param([int]$Port)
    $connection = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
    return $null -ne $connection
}

function Get-ProcessByPort {
    param([int]$Port)
    $connection = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
    if ($connection) {
        return Get-Process -Id $connection.OwningProcess -ErrorAction SilentlyContinue
    }
    return $null
}

function Start-System {
    Show-Banner
    Write-Host "[START] Starting FiveForFree System..." -ForegroundColor Yellow
    Write-Host ""

    # Check if already running
    if (Test-PortInUse -Port $BackendPort) {
        Write-Host "[WARN] Backend already running on port $BackendPort" -ForegroundColor Yellow
    } else {
        Write-Host "[1/2] Starting Backend..." -ForegroundColor Cyan
        Start-Process powershell -ArgumentList @(
            "-NoExit",
            "-Command",
            "Set-Location '$ProjectRoot'; Write-Host 'FiveForFree Backend' -ForegroundColor Cyan; python scripts/run_system.py"
        )
        Write-Host "      Backend started (port $BackendPort)" -ForegroundColor Green
        Start-Sleep -Seconds 2
    }

    if (Test-PortInUse -Port $FrontendPort) {
        Write-Host "[WARN] Frontend already running on port $FrontendPort" -ForegroundColor Yellow
    } else {
        Write-Host "[2/2] Starting Frontend..." -ForegroundColor Cyan
        Start-Process powershell -ArgumentList @(
            "-NoExit",
            "-Command",
            "Set-Location '$ProjectRoot\frontend'; Write-Host 'FiveForFree Frontend' -ForegroundColor Cyan; npm run dev"
        )
        Write-Host "      Frontend started (port $FrontendPort)" -ForegroundColor Green
    }

    Write-Host ""
    Write-Host "[OK] System started!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Access:" -ForegroundColor White
    Write-Host "  Frontend: http://localhost:$FrontendPort" -ForegroundColor Gray
    Write-Host "  API:      http://localhost:$BackendPort" -ForegroundColor Gray
    Write-Host ""
}

function Stop-System {
    Show-Banner
    Write-Host "[STOP] Stopping FiveForFree System..." -ForegroundColor Yellow
    Write-Host ""

    $stopped = 0

    # Stop backend
    $backendProc = Get-ProcessByPort -Port $BackendPort
    if ($backendProc) {
        Write-Host "[1/2] Stopping Backend (PID: $($backendProc.Id))..." -ForegroundColor Cyan
        Stop-Process -Id $backendProc.Id -Force -ErrorAction SilentlyContinue
        Write-Host "      Backend stopped" -ForegroundColor Green
        $stopped++
    } else {
        Write-Host "[1/2] Backend not running" -ForegroundColor Gray
    }

    # Stop frontend
    $frontendProc = Get-ProcessByPort -Port $FrontendPort
    if ($frontendProc) {
        Write-Host "[2/2] Stopping Frontend (PID: $($frontendProc.Id))..." -ForegroundColor Cyan
        Stop-Process -Id $frontendProc.Id -Force -ErrorAction SilentlyContinue
        Write-Host "      Frontend stopped" -ForegroundColor Green
        $stopped++
    } else {
        Write-Host "[2/2] Frontend not running" -ForegroundColor Gray
    }

    Write-Host ""
    if ($stopped -gt 0) {
        Write-Host "[OK] Stopped $stopped process(es)" -ForegroundColor Green
    } else {
        Write-Host "[INFO] No processes were running" -ForegroundColor Gray
    }
    Write-Host ""
}

function Show-Status {
    Show-Banner
    Write-Host "[STATUS] FiveForFree System Status" -ForegroundColor Yellow
    Write-Host ""

    # Backend status
    $backendRunning = Test-PortInUse -Port $BackendPort
    Write-Host -NoNewline "  Backend  (port $BackendPort): "
    if ($backendRunning) {
        $proc = Get-ProcessByPort -Port $BackendPort
        Write-Host "RUNNING (PID: $($proc.Id))" -ForegroundColor Green
    } else {
        Write-Host "STOPPED" -ForegroundColor Red
    }

    # Frontend status
    $frontendRunning = Test-PortInUse -Port $FrontendPort
    Write-Host -NoNewline "  Frontend (port $FrontendPort): "
    if ($frontendRunning) {
        $proc = Get-ProcessByPort -Port $FrontendPort
        Write-Host "RUNNING (PID: $($proc.Id))" -ForegroundColor Green
    } else {
        Write-Host "STOPPED" -ForegroundColor Red
    }

    Write-Host ""

    # Health check
    Write-Host "  Health Check:" -ForegroundColor White

    if ($backendRunning) {
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:$BackendPort/health" -TimeoutSec 3 -ErrorAction SilentlyContinue
            Write-Host "    API Health: " -NoNewline
            Write-Host "OK" -ForegroundColor Green
        } catch {
            Write-Host "    API Health: " -NoNewline
            Write-Host "UNHEALTHY" -ForegroundColor Yellow
        }
    }

    if ($frontendRunning) {
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:$FrontendPort" -TimeoutSec 3 -ErrorAction SilentlyContinue
            Write-Host "    Frontend:   " -NoNewline
            Write-Host "OK" -ForegroundColor Green
        } catch {
            Write-Host "    Frontend:   " -NoNewline
            Write-Host "UNHEALTHY" -ForegroundColor Yellow
        }
    }

    Write-Host ""
}

function Show-Logs {
    param([string]$LogTarget)

    Show-Banner

    $logsDir = "$ProjectRoot\logs"

    switch ($LogTarget.ToLower()) {
        "backend" {
            Write-Host "[LOGS] Backend Logs" -ForegroundColor Yellow
            $logFile = "$logsDir\backend.log"
            if (Test-Path $logFile) {
                Get-Content $logFile -Tail 50 -Wait
            } else {
                Write-Host "No backend log file found at: $logFile" -ForegroundColor Gray
                Write-Host "Tip: Backend logs may be in console output" -ForegroundColor Gray
            }
        }
        "frontend" {
            Write-Host "[LOGS] Frontend Logs" -ForegroundColor Yellow
            $logFile = "$logsDir\frontend.log"
            if (Test-Path $logFile) {
                Get-Content $logFile -Tail 50 -Wait
            } else {
                Write-Host "No frontend log file found at: $logFile" -ForegroundColor Gray
                Write-Host "Tip: Frontend logs may be in console output" -ForegroundColor Gray
            }
        }
        default {
            Write-Host "[LOGS] Available log targets:" -ForegroundColor Yellow
            Write-Host "  .\fff.ps1 logs backend" -ForegroundColor Gray
            Write-Host "  .\fff.ps1 logs frontend" -ForegroundColor Gray
        }
    }
}

# Main execution
if ($Help -or $Command -eq "help") {
    Show-Help
    exit 0
}

switch ($Command) {
    "start" { Start-System }
    "stop" { Stop-System }
    "status" { Show-Status }
    "logs" { Show-Logs -LogTarget $Target }
    default { Show-Help }
}
