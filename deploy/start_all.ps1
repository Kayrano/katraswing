# =============================================================================
# Katraswing -- Start signal server + GitHub auto-update watcher
# Run this once in the interactive session (same session as MT5).
# - Launches the signal server in a minimized window
# - Starts a watcher that polls GitHub every 5 min and auto-updates on new commits
# =============================================================================

$ErrorActionPreference = "Continue"   # never let a single error kill the loop

$INSTALL_DIR  = "C:\katraswing"
$FINNHUB_KEY  = "d7j16r1r01qn2qavovt0d7j16r1r01qn2qavovtg"

# ── GitHub authentication ─────────────────────────────────────────────────────
# Put your PAT in deploy\local_config.ps1 (gitignored, never committed):
#   $GITHUB_PAT = "ghp_your_token_here"
# That file is loaded below if it exists — start_all.ps1 stays clean.
$GITHUB_PAT   = ""
$GITHUB_USER  = "Kayrano"
$GITHUB_REPO  = "katraswing"
$_localCfg    = Join-Path $INSTALL_DIR "deploy\local_config.ps1"
if (Test-Path $_localCfg) { . $_localCfg }

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

# Kill only the python signal server process — the PowerShell window auto-closes
# when python exits (no -NoExit), so no parent-kill needed (and parent-kill was
# killing the watcher script itself due to Windows process-tree reporting).
function Stop-SignalServer {
    Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

function Start-SignalServer {
    # No -NoExit: window closes automatically when python exits/is killed
    # Tee to a log file so history survives window restarts
    Start-Process powershell -ArgumentList @(
        "-Command",
        "`$env:PYTHONIOENCODING='utf-8'; cd $INSTALL_DIR; python mt5_signal_server.py --interval 30 --risk-pct 1.0 --finnhub-key $FINNHUB_KEY 2>&1 | Tee-Object -FilePath '$INSTALL_DIR\logs\signal_server.log' -Append"
    ) -WindowStyle Minimized
}

Set-Location $INSTALL_DIR

# ── Embed PAT in remote URL so git never prompts for credentials ──────────────
if ($GITHUB_PAT -ne "") {
    $remoteUrl = "https://${GITHUB_USER}:${GITHUB_PAT}@github.com/${GITHUB_USER}/${GITHUB_REPO}.git"
    git remote set-url origin $remoteUrl
    Write-OK "GitHub remote URL configured with PAT"
} else {
    Write-Host "  WARNING: GITHUB_PAT is empty -- watcher may fail if HTTPS credentials expire." -ForegroundColor Yellow
}

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

    # Fetch — capture output and check exit code; do NOT use 2>$null (swallows errors silently)
    $fetchOut = git fetch origin main 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[$checkTime] FETCH FAILED (exit $LASTEXITCODE) -- check PAT/network. Will retry in 5 min." -ForegroundColor Red
        Write-Host "  $fetchOut" -ForegroundColor Red
        Start-Sleep -Seconds 300
        continue
    }

    $local  = git rev-parse HEAD 2>&1
    $remote = git rev-parse origin/main 2>&1

    if ($local -ne $remote) {
        Write-Host ""
        Write-Host "[$checkTime] New commits detected -- updating..." -ForegroundColor Yellow
        try {
            Write-Host "  [1/6] Stopping signal server..." -ForegroundColor DarkGray
            Stop-SignalServer

            Write-Host "  [2/6] Stopping Streamlit..." -ForegroundColor DarkGray
            Stop-Service katraswing-streamlit -ErrorAction SilentlyContinue
            Start-Sleep -Seconds 2

            Write-Host "  [3/6] git pull (auto-resolving local data files)..." -ForegroundColor DarkGray
            # Discard local changes to files that conflict — PAT lives in deploy\local_config.ps1 (gitignored)
            git checkout -- data/strategy_params.json 2>&1 | Out-Null
            git checkout -- deploy/start_all.ps1 2>&1 | Out-Null
            $pullOut = git pull 2>&1
            if ($LASTEXITCODE -ne 0) {
                Write-Host "[$checkTime] git pull FAILED -- aborting update cycle." -ForegroundColor Red
                Write-Host "  $pullOut" -ForegroundColor Red
                Start-SignalServer
                Start-StreamlitService
                Start-Sleep -Seconds 300
                continue
            }

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
