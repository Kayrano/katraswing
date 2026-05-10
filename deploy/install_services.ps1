# =============================================================================
# Katraswing - Install Windows Services via NSSM
# Run as Administrator AFTER setup_windows.ps1 has completed.
# Both services auto-start on boot and auto-restart on crash.
# =============================================================================
#Requires -RunAsAdministrator

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$INSTALL_DIR = "C:\katraswing"
$PYTHON      = (Get-Command python).Source
$LOG_DIR     = "$INSTALL_DIR\logs"
$FINNHUB_KEY = "d7j16r1r01qn2qavovt0d7j16r1r01qn2qavovtg"

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-OK($msg)   { Write-Host "    OK: $msg" -ForegroundColor Green }

# Verify NSSM is available
if (-not (Get-Command nssm -ErrorAction SilentlyContinue)) {
    Write-Error "nssm.exe not found. Run setup_windows.ps1 first."
    exit 1
}

New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null

# Helper: install or reinstall a service
function Install-NSSMService {
    param(
        [string]$Name,
        [string]$Exe,
        [string]$AppArgs,
        [string]$WorkDir,
        [string]$StdoutLog,
        [string]$StderrLog
    )
    $existing = Get-Service -Name $Name -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Host "    Removing existing service: $Name"
        $ErrorActionPreference = "SilentlyContinue"
        nssm stop $Name 2>$null
        $ErrorActionPreference = "Stop"
        nssm remove $Name confirm 2>$null
        Start-Sleep -Seconds 2
    }

    nssm install $Name $Exe "$AppArgs"
    nssm set $Name AppDirectory $WorkDir
    nssm set $Name AppStdout    $StdoutLog
    nssm set $Name AppStderr    $StderrLog
    nssm set $Name AppRotateFiles   1
    nssm set $Name AppRotateOnline  1
    nssm set $Name AppRotateBytes   10485760
    nssm set $Name Start            SERVICE_AUTO_START
    nssm set $Name AppRestartDelay  3000
    nssm set $Name AppThrottle      30000
}

# Service 1: Signal Server
Write-Step "Installing katraswing-signals service"
Install-NSSMService `
    -Name      "katraswing-signals" `
    -Exe       $PYTHON `
    -AppArgs   "mt5_signal_server.py --interval 60 --risk-pct 1.0 --finnhub-key $FINNHUB_KEY" `
    -WorkDir   $INSTALL_DIR `
    -StdoutLog "$LOG_DIR\signals_stdout.log" `
    -StderrLog "$LOG_DIR\signals_stderr.log"
Write-OK "katraswing-signals installed"

# Service 2: Streamlit Dashboard
Write-Step "Installing katraswing-streamlit service"
Install-NSSMService `
    -Name      "katraswing-streamlit" `
    -Exe       $PYTHON `
    -AppArgs   "-m streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --browser.gatherUsageStats false --server.headless true" `
    -WorkDir   $INSTALL_DIR `
    -StdoutLog "$LOG_DIR\streamlit_stdout.log" `
    -StderrLog "$LOG_DIR\streamlit_stderr.log"
Write-OK "katraswing-streamlit installed"

# Start both services
Write-Step "Starting services"
Start-Service katraswing-signals
Start-Sleep -Seconds 3
Start-Service katraswing-streamlit
Start-Sleep -Seconds 3

$sig = Get-Service katraswing-signals
$stl = Get-Service katraswing-streamlit
Write-Host ""
Write-Host "  katraswing-signals   : $($sig.Status)" -ForegroundColor $(if ($sig.Status -eq 'Running') {'Green'} else {'Red'})
Write-Host "  katraswing-streamlit : $($stl.Status)" -ForegroundColor $(if ($stl.Status -eq 'Running') {'Green'} else {'Red'})
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Services installed and started." -ForegroundColor Green
Write-Host "" -ForegroundColor Green
Write-Host "  Check status : Get-Service katraswing-*" -ForegroundColor Green
Write-Host "  Signal logs  : $LOG_DIR\signals_stdout.log" -ForegroundColor Green
Write-Host "  Dashboard    : http://<your-azure-ip>:8501" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
