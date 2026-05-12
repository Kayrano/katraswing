# =============================================================================
# Katraswing -- One-time Windows Service installer (run as Administrator)
#
# Creates the katraswing-server Windows Service so the signal server survives
# RDP disconnects. Run this once on the VPS, then use start_all.ps1 normally.
#
# Usage (as Administrator in PowerShell):
#   cd C:\katraswing
#   . deploy\local_config.ps1    # loads $TELEGRAM_TOKEN, $TELEGRAM_CHAT_ID, $VPS_USER, $VPS_PASS
#   .\deploy\install_service.ps1
#
# local_config.ps1 must also contain:
#   $VPS_USER = "your-windows-username"
#   $VPS_PASS = "your-windows-password"
# =============================================================================

$ErrorActionPreference = "Stop"

$INSTALL_DIR  = "C:\katraswing"
$FINNHUB_KEY  = "d7j16r1r01qn2qavovt0d7j16r1r01qn2qavovtg"
$TELEGRAM_TOKEN   = ""
$TELEGRAM_CHAT_ID = ""
$VPS_USER         = ""
$VPS_PASS         = ""

$_localCfg = Join-Path $INSTALL_DIR "deploy\local_config.ps1"
if (Test-Path $_localCfg) { . $_localCfg }

# ── Locate or download NSSM ──────────────────────────────────────────────────
$nssm = Join-Path $INSTALL_DIR "nssm.exe"
if (-not (Test-Path $nssm)) {
    Write-Host "Downloading NSSM..." -ForegroundColor Cyan
    $zip = "$env:TEMP\nssm.zip"
    Invoke-WebRequest "https://nssm.cc/release/nssm-2.24.zip" -OutFile $zip
    Expand-Archive $zip -DestinationPath "$env:TEMP\nssm_extract" -Force
    Copy-Item "$env:TEMP\nssm_extract\nssm-2.24\win64\nssm.exe" $nssm
    Write-Host "NSSM installed at $nssm" -ForegroundColor Green
}

# ── Remove old service if it exists ──────────────────────────────────────────
$existing = Get-Service katraswing-server -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Removing existing katraswing-server service..." -ForegroundColor Yellow
    if ($existing.Status -eq 'Running') { Stop-Service katraswing-server }
    & $nssm remove katraswing-server confirm
    Start-Sleep -Seconds 2
}

# ── Find python path ──────────────────────────────────────────────────────────
$python = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $python) { throw "python not found in PATH" }
Write-Host "Using Python: $python" -ForegroundColor Cyan

# ── Create service ────────────────────────────────────────────────────────────
Write-Host "Creating katraswing-server service..." -ForegroundColor Cyan

& $nssm install katraswing-server $python
& $nssm set katraswing-server AppParameters "mt5_signal_server.py --interval 30 --risk-pct 1.0 --finnhub-key $FINNHUB_KEY --telegram-token $TELEGRAM_TOKEN --telegram-chat-id $TELEGRAM_CHAT_ID"
& $nssm set katraswing-server AppDirectory $INSTALL_DIR
& $nssm set katraswing-server AppEnvironmentExtra "PYTHONIOENCODING=utf-8"
& $nssm set katraswing-server AppStdout "$INSTALL_DIR\logs\signal_server.log"
& $nssm set katraswing-server AppStderr "$INSTALL_DIR\logs\signal_server_err.log"
& $nssm set katraswing-server AppRotateFiles 1
& $nssm set katraswing-server AppRotateOnline 1
& $nssm set katraswing-server AppRotateBytes 10485760   # 10 MB per log file
& $nssm set katraswing-server AppRestartDelay 10000     # restart 10s after crash
& $nssm set katraswing-server Start SERVICE_AUTO_START

# Run as VPS user account (required for MT5 terminal connectivity)
if ($VPS_USER -ne "" -and $VPS_PASS -ne "") {
    & $nssm set katraswing-server ObjectName ".\$VPS_USER" $VPS_PASS
    Write-Host "Service will run as user: $VPS_USER" -ForegroundColor Green
} else {
    Write-Host "WARNING: VPS_USER/VPS_PASS not set in local_config.ps1." -ForegroundColor Yellow
    Write-Host "         Service will run as LocalSystem -- MT5 may not connect." -ForegroundColor Yellow
    Write-Host "         Add `$VPS_USER and `$VPS_PASS to deploy\local_config.ps1 and re-run." -ForegroundColor Yellow
}

# Ensure logs directory exists
New-Item -ItemType Directory -Force -Path "$INSTALL_DIR\logs" | Out-Null

# ── Start service ─────────────────────────────────────────────────────────────
Start-Service katraswing-server
Start-Sleep -Seconds 3
$svc = Get-Service katraswing-server
Write-Host ""
Write-Host "katraswing-server : $($svc.Status)" -ForegroundColor $(if ($svc.Status -eq 'Running') {'Green'} else {'Red'})
Write-Host ""
Write-Host "Done. The signal server will now run on boot and survive RDP disconnects." -ForegroundColor Green
Write-Host "Logs: $INSTALL_DIR\logs\signal_server.log" -ForegroundColor Cyan
