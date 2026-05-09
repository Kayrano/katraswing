# =============================================================================
# Katraswing — Install Windows Services via NSSM
# Run as Administrator AFTER setup_windows.ps1 has completed.
# Both services auto-start on boot and auto-restart on crash.
# =============================================================================
#Requires -RunAsAdministrator

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$INSTALL_DIR  = "C:\katraswing"
$PYTHON       = (Get-Command python).Source
$LOG_DIR      = "$INSTALL_DIR\logs"

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-OK($msg)   { Write-Host "    OK: $msg" -ForegroundColor Green }

# Verify NSSM is available
if (-not (Get-Command nssm -ErrorAction SilentlyContinue)) {
    Write-Error "nssm.exe not found. Run setup_windows.ps1 first."
    exit 1
}

New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null

# ── Helper: install or reinstall a service ────────────────────────────────────
function Install-NSSMService {
    param(
        [string]$Name,
        [string]$Exe,
        [string]$Args,
        [string]$WorkDir,
        [string]$StdoutLog,
        [string]$StderrLog
    )
    # Remove existing service cleanly
    $existing = Get-Service -Name $Name -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Host "    Removing existing service: $Name"
        nssm stop  $Name 2>$null
        nssm remove $Name confirm 2>$null
        Start-Sleep -Seconds 2
    }

    nssm install $Name $Exe $Args
    nssm set $Name AppDirectory   $WorkDir
    nssm set $Name AppStdout      $StdoutLog
    nssm set $Name AppStderr      $StderrLog
    nssm set $Name AppRotateFiles 1
    nssm set $Name AppRotateOnline 1
    nssm set $Name AppRotateBytes 10485760     # rotate at 10 MB
    nssm set $Name Start          SERVICE_AUTO_START
    # Restart policy: restart 3 s after crash, then 30 s, then 60 s
    nssm set $Name AppRestartDelay 3000
    nssm set $Name AppThrottle    30000
}

# ── Service 1: Katraswing Signal Server ───────────────────────────────────────
Write-Step "Installing katraswing-signals service"
Install-NSSMService `
    -Name       "katraswing-signals" `
    -Exe        $PYTHON `
    -Args       "mt5_signal_server.py --interval 60 --risk-pct 1.0" `
    -WorkDir    $INSTALL_DIR `
    -StdoutLog  "$LOG_DIR\signals_stdout.log" `
    -StderrLog  "$LOG_DIR\signals_stderr.log"
Write-OK "katraswing-signals installed"

# ── Service 2: Katraswing Streamlit Dashboard ─────────────────────────────────
Write-Step "Installing katraswing-streamlit service"
Install-NSSMService `
    -Name       "katraswing-streamlit" `
    -Exe        $PYTHON `
    -Args       "-m streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --browser.gatherUsageStats false --server.headless true" `
    -WorkDir    $INSTALL_DIR `
    -StdoutLog  "$LOG_DIR\streamlit_stdout.log" `
    -StderrLog  "$LOG_DIR\streamlit_stderr.log"
Write-OK "katraswing-streamlit installed"

# ── Start both services ───────────────────────────────────────────────────────
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

Write-Host @"

=============================================================================
  Services installed and started.

  Useful commands:
    Get-Service katraswing-*            # check status
    Restart-Service katraswing-signals  # restart signal server
    nssm edit katraswing-signals        # edit service config in GUI

  Logs:
    $LOG_DIR\signals_stdout.log
    $LOG_DIR\streamlit_stdout.log

  Dashboard:
    http://<your-vps-ip>:8501
=============================================================================
"@ -ForegroundColor Green
