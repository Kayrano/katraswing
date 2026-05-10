# =============================================================================
# Katraswing -- Pull latest code and restart services
# Run as Administrator whenever you push new changes from your local machine.
# Signal server runs as a foreground window (not NSSM service).
# =============================================================================
#Requires -RunAsAdministrator

$INSTALL_DIR  = "C:\katraswing"
$FINNHUB_KEY  = "d7j16r1r01qn2qavovt0d7j16r1r01qn2qavovtg"

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-OK($msg)   { Write-Host "    OK: $msg" -ForegroundColor Green }

Set-Location $INSTALL_DIR

# --- Stop signal server (foreground process) ---------------------------------
Write-Step "Stopping signal server"
Get-Process python -ErrorAction SilentlyContinue |
    Where-Object { $_.MainWindowTitle -like "*katraswing*" -or
                   (Get-WmiObject Win32_Process -Filter "ProcessId=$($_.Id)" -ErrorAction SilentlyContinue).CommandLine -like "*mt5_signal_server*" } |
    Stop-Process -Force -ErrorAction SilentlyContinue
# Fallback: if above finds nothing, stop ALL python (streamlit restarts below)
$remaining = Get-Process python -ErrorAction SilentlyContinue
if ($remaining) {
    Write-Host "    Stopping all python processes..." -ForegroundColor Yellow
    $remaining | Stop-Process -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 2
Write-OK "Signal server stopped"

# --- Stop Streamlit service --------------------------------------------------
Write-Step "Stopping Streamlit service"
Stop-Service katraswing-streamlit -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
Write-OK "Streamlit stopped"

# --- Pull latest code --------------------------------------------------------
Write-Step "Pulling latest code"
git pull
Write-OK "Code updated"

# --- Update Python packages --------------------------------------------------
Write-Step "Updating Python packages"
python -m pip install -r requirements.txt -q
Write-OK "Packages updated"

# --- Restart Streamlit service -----------------------------------------------
Write-Step "Starting Streamlit service"
Start-Service katraswing-streamlit -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3
$stl = Get-Service katraswing-streamlit -ErrorAction SilentlyContinue
Write-Host "  katraswing-streamlit : $($stl.Status)" -ForegroundColor $(if ($stl.Status -eq 'Running') {'Green'} else {'Red'})

# --- Relaunch signal server as minimized window ------------------------------
Write-Step "Starting signal server"
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "cd C:\katraswing; python mt5_signal_server.py --interval 60 --risk-pct 1.0 --finnhub-key $FINNHUB_KEY"
) -WindowStyle Minimized
Start-Sleep -Seconds 3

$sig = Get-Process python -ErrorAction SilentlyContinue
Write-Host "  katraswing-signals   : $(if ($sig) {'Running'} else {'NOT RUNNING'})" -ForegroundColor $(if ($sig) {'Green'} else {'Red'})

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Update complete!" -ForegroundColor Green
Write-Host "  Signal logs: Get-Content C:\katraswing\logs\signals_stdout.log -Wait -Tail 30" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
