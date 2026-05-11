# =============================================================================
# Katraswing -- Start signal server + GitHub auto-update watcher
# Run this once in the interactive session (same session as MT5).
# - Launches the signal server in a minimized window
# - Starts a watcher that polls GitHub every 5 min and auto-updates on new commits
# =============================================================================

$ErrorActionPreference = "Continue"   # never let a single error kill the loop

$INSTALL_DIR = "C:\katraswing"
$FINNHUB_KEY = "d7j16r1r01qn2qavovt0d7j16r1r01qn2qavovtg"

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-OK($msg)   { Write-Host "    OK: $msg" -ForegroundColor Green }

# Start-Service fails silently on a Paused service — resume it instead.
function Start-StreamlitService {
    $svc = Get-Service katraswing-streamlit -ErrorAction SilentlyContinue
    if ($svc.Status -eq 'Paused') {
        Resume-Service katraswing-streamlit -ErrorAction SilentlyContinue
    } elseif ($svc.Status -ne 'Running') {
        Start-Service katraswing-streamlit -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 2
}

# Find the python process running mt5_signal_server.py, then kill both it
# and its parent PowerShell window — no PID file needed.
function Stop-SignalServer {
    $pythonProcs = Get-WmiObject Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like "*mt5_signal_server*" }
    foreach ($p in $pythonProcs) {
        Stop-Process -Id ([int]$p.ParentProcessId) -Force -ErrorAction SilentlyContinue
        Stop-Process -Id ([int]$p.ProcessId)       -Force -ErrorAction SilentlyContinue
    }
    # Fallback: kill any leftover python processes
    Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

function Start-SignalServer {
    Start-Process powershell -ArgumentList @(
        "-NoExit", "-Command",
        "cd $INSTALL_DIR; python mt5_signal_server.py --interval 30 --risk-pct 1.0 --finnhub-key $FINNHUB_KEY"
    ) -WindowStyle Minimized
}

Set-Location $INSTALL_DIR

# Kill any existing signal server
Write-Step "Stopping any existing signal server"
Stop-SignalServer

# Start signal server
Write-Step "Starting signal server"
Start-SignalServer
Start-Sleep -Seconds 3
Write-OK "Signal server launched"

# Ensure Streamlit service is running
Write-Step "Starting Streamlit service"
Start-StreamlitService
$stl = Get-Service katraswing-streamlit -ErrorAction SilentlyContinue
Write-Host "  katraswing-streamlit : $($stl.Status)" -ForegroundColor $(if ($stl.Status -eq 'Running') {'Green'} else {'Red'})

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Services started. Starting GitHub auto-update watcher..." -ForegroundColor Green
Write-Host "  This window will check GitHub every 5 minutes." -ForegroundColor Green
Write-Host "  Minimize it -- do NOT close it." -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""

# ── Auto-update watcher loop ─────────────────────────────────────────────────
Set-Location $INSTALL_DIR

while ($true) {
    $checkTime = Get-Date -Format "HH:mm:ss"

    git fetch origin main --quiet 2>$null

    $local  = git rev-parse HEAD 2>$null
    $remote = git rev-parse origin/main 2>$null

    if ($local -ne $remote) {
        Write-Host ""
        Write-Host "[$checkTime] New commits detected -- updating..." -ForegroundColor Yellow
        try {
            Write-Host "  [1/6] Stopping signal server..." -ForegroundColor DarkGray
            Stop-SignalServer

            Write-Host "  [2/6] Stopping Streamlit..." -ForegroundColor DarkGray
            Stop-Service katraswing-streamlit -ErrorAction SilentlyContinue
            Start-Sleep -Seconds 2

            Write-Host "  [3/6] git pull..." -ForegroundColor DarkGray
            git pull

            Write-Host "  [4/6] pip install..." -ForegroundColor DarkGray
            python -m pip install -r requirements.txt -q

            Write-Host "  [5/6] Starting Streamlit..." -ForegroundColor DarkGray
            Start-StreamlitService

            Write-Host "  [6/6] Starting signal server..." -ForegroundColor DarkGray
            Start-SignalServer

            $newHash = git rev-parse --short HEAD
            Write-Host "[$checkTime] Update complete -- now at commit $newHash" -ForegroundColor Green
        } catch {
            Write-Host "[$checkTime] Update ERROR: $_" -ForegroundColor Red
        }
    } else {
        Write-Host "[$checkTime] Up to date ($($local.Substring(0,7)))" -ForegroundColor DarkGray
    }

    Start-Sleep -Seconds 300
}
