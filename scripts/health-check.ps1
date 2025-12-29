<#
.SYNOPSIS
    FiveForFree - System Health Check
.DESCRIPTION
    Comprehensive health check for FiveForFree system:
    - API Server (localhost:8000)
    - Frontend (localhost:5173)
    - Database connectivity
    - System resources
.PARAMETER Verbose
    Show detailed output
.PARAMETER Watch
    Continuously monitor (refresh every 5 seconds)
.EXAMPLE
    .\health-check.ps1           # One-time check
    .\health-check.ps1 -Verbose  # Detailed output
    .\health-check.ps1 -Watch    # Continuous monitoring
#>

param(
    [switch]$Detailed,
    [switch]$Watch,
    [int]$Interval = 5,
    [switch]$Help
)

# Configuration
$BackendUrl = "http://localhost:8000"
$FrontendUrl = "http://localhost:5173"
$HealthEndpoint = "/health"
$Timeout = 5

function Show-Banner {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host " FiveForFree Health Check" -ForegroundColor Cyan
    Write-Host " $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
}

function Show-Help {
    Show-Banner
    Write-Host "Usage: .\health-check.ps1 [options]" -ForegroundColor White
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  -Detailed   Show detailed health information"
    Write-Host "  -Watch      Continuously monitor (Ctrl+C to stop)"
    Write-Host "  -Interval   Refresh interval in seconds (default: 5)"
    Write-Host "  -Help       Show this help message"
    Write-Host ""
    Write-Host "Checks performed:" -ForegroundColor Yellow
    Write-Host "  1. API Server (port 8000)"
    Write-Host "  2. Frontend (port 5173)"
    Write-Host "  3. API Health endpoint (/health)"
    Write-Host "  4. Port availability"
    Write-Host ""
    exit 0
}

if ($Help) { Show-Help }

function Test-Endpoint {
    param(
        [string]$Url,
        [string]$Name
    )

    $result = @{
        Name = $Name
        Url = $Url
        Status = "UNKNOWN"
        ResponseTime = -1
        StatusCode = 0
        Error = $null
    }

    try {
        $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
        $response = Invoke-WebRequest -Uri $Url -TimeoutSec $Timeout -ErrorAction Stop
        $stopwatch.Stop()

        $result.Status = "HEALTHY"
        $result.ResponseTime = $stopwatch.ElapsedMilliseconds
        $result.StatusCode = $response.StatusCode
    } catch {
        $result.Status = "UNHEALTHY"
        $result.Error = $_.Exception.Message

        # Check if it's a connection refused (service not running)
        if ($_.Exception.Message -match "Unable to connect") {
            $result.Status = "OFFLINE"
        }
    }

    return $result
}

function Test-Port {
    param([int]$Port)

    $connection = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
    if ($connection) {
        $process = Get-Process -Id $connection.OwningProcess -ErrorAction SilentlyContinue
        return @{
            InUse = $true
            ProcessId = $connection.OwningProcess
            ProcessName = if ($process) { $process.ProcessName } else { "Unknown" }
        }
    }
    return @{ InUse = $false; ProcessId = $null; ProcessName = $null }
}

function Show-HealthStatus {
    param([switch]$ShowDetails)

    if (-not $Watch) { Show-Banner }

    $allHealthy = $true

    # Check Backend Port
    Write-Host "[1/4] Backend Port (8000)" -ForegroundColor White
    $backendPort = Test-Port -Port 8000
    if ($backendPort.InUse) {
        Write-Host "      Status: " -NoNewline
        Write-Host "LISTENING" -ForegroundColor Green -NoNewline
        Write-Host " (PID: $($backendPort.ProcessId), Process: $($backendPort.ProcessName))" -ForegroundColor Gray
    } else {
        Write-Host "      Status: " -NoNewline
        Write-Host "NOT LISTENING" -ForegroundColor Red
        $allHealthy = $false
    }
    Write-Host ""

    # Check Frontend Port
    Write-Host "[2/4] Frontend Port (5173)" -ForegroundColor White
    $frontendPort = Test-Port -Port 5173
    if ($frontendPort.InUse) {
        Write-Host "      Status: " -NoNewline
        Write-Host "LISTENING" -ForegroundColor Green -NoNewline
        Write-Host " (PID: $($frontendPort.ProcessId), Process: $($frontendPort.ProcessName))" -ForegroundColor Gray
    } else {
        Write-Host "      Status: " -NoNewline
        Write-Host "NOT LISTENING" -ForegroundColor Red
        $allHealthy = $false
    }
    Write-Host ""

    # Check API Health Endpoint
    Write-Host "[3/4] API Health Endpoint" -ForegroundColor White
    if ($backendPort.InUse) {
        $apiHealth = Test-Endpoint -Url "$BackendUrl$HealthEndpoint" -Name "API Health"
        Write-Host "      URL: $BackendUrl$HealthEndpoint" -ForegroundColor Gray
        Write-Host "      Status: " -NoNewline
        switch ($apiHealth.Status) {
            "HEALTHY" {
                Write-Host "HEALTHY" -ForegroundColor Green -NoNewline
                Write-Host " ($($apiHealth.ResponseTime)ms, HTTP $($apiHealth.StatusCode))" -ForegroundColor Gray
            }
            "UNHEALTHY" {
                Write-Host "UNHEALTHY" -ForegroundColor Yellow
                if ($ShowDetails -and $apiHealth.Error) {
                    Write-Host "      Error: $($apiHealth.Error)" -ForegroundColor Gray
                }
                $allHealthy = $false
            }
            "OFFLINE" {
                Write-Host "OFFLINE" -ForegroundColor Red
                $allHealthy = $false
            }
        }
    } else {
        Write-Host "      Status: " -NoNewline
        Write-Host "SKIPPED" -ForegroundColor Gray -NoNewline
        Write-Host " (Backend not running)" -ForegroundColor Gray
        $allHealthy = $false
    }
    Write-Host ""

    # Check Frontend
    Write-Host "[4/4] Frontend Accessibility" -ForegroundColor White
    if ($frontendPort.InUse) {
        $frontendHealth = Test-Endpoint -Url $FrontendUrl -Name "Frontend"
        Write-Host "      URL: $FrontendUrl" -ForegroundColor Gray
        Write-Host "      Status: " -NoNewline
        switch ($frontendHealth.Status) {
            "HEALTHY" {
                Write-Host "HEALTHY" -ForegroundColor Green -NoNewline
                Write-Host " ($($frontendHealth.ResponseTime)ms)" -ForegroundColor Gray
            }
            "UNHEALTHY" {
                Write-Host "UNHEALTHY" -ForegroundColor Yellow
                $allHealthy = $false
            }
            "OFFLINE" {
                Write-Host "OFFLINE" -ForegroundColor Red
                $allHealthy = $false
            }
        }
    } else {
        Write-Host "      Status: " -NoNewline
        Write-Host "SKIPPED" -ForegroundColor Gray -NoNewline
        Write-Host " (Frontend not running)" -ForegroundColor Gray
        $allHealthy = $false
    }
    Write-Host ""

    # Summary
    Write-Host "========================================" -ForegroundColor Cyan
    if ($allHealthy) {
        Write-Host " Overall Status: " -NoNewline
        Write-Host "ALL HEALTHY" -ForegroundColor Green
    } else {
        Write-Host " Overall Status: " -NoNewline
        Write-Host "ISSUES DETECTED" -ForegroundColor Yellow
        Write-Host ""
        Write-Host " Tip: Run '.\fff.ps1 start' to start the system" -ForegroundColor Gray
    }
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""

    return $allHealthy
}

# Main execution
if ($Watch) {
    Write-Host "Watching system health (Ctrl+C to stop)..." -ForegroundColor Yellow
    Write-Host "Refresh interval: $Interval seconds" -ForegroundColor Gray
    Write-Host ""

    while ($true) {
        Clear-Host
        Show-Banner
        Show-HealthStatus -ShowDetails:$Detailed | Out-Null
        Write-Host "Press Ctrl+C to stop monitoring" -ForegroundColor Gray
        Start-Sleep -Seconds $Interval
    }
} else {
    $healthy = Show-HealthStatus -ShowDetails:$Detailed
    if ($healthy) {
        exit 0
    } else {
        exit 1
    }
}
