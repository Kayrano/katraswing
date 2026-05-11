# =============================================================================
# Katraswing -- Start signal server + GitHub auto-update watcher
# Run this once in the interactive session (same session as MT5).
# - Launches the signal server in a minimized window
# - Starts a watcher that polls GitHub every 5 min and auto-updates on new commits
# =============================================================================

$INSTALL_DIR = "C:\katraswing"
$FINNHUB_KEY = "d7j16r1r01qn2qavovt0d7j16r1r01qn2qavovtg"

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-OK($msg)   { Write-Host "    OK: $msg" -ForegroundColor Green }

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
Start-Service katraswing-streamlit -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
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

        # Stop signal server window + python process
        Stop-SignalServer

        # Stop Streamlit
        Stop-Service katraswing-streamlit -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2

        # Pull + install
        git pull
        python -m pip install -r requirements.txt -q

        # Restart Streamlit
        Start-Service katraswing-streamlit -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2

        # Relaunch signal server (new window, PID saved)
        Start-SignalServer

        $newHash = git rev-parse --short HEAD
        Write-Host "[$checkTime] Update complete -- now at commit $newHash" -ForegroundColor Green
    } else {
        Write-Host "[$checkTime] Up to date ($($local.Substring(0,7)))" -ForegroundColor DarkGray
    }

    Start-Sleep -Seconds 300
}
