# =============================================================================
# Katraswing — Windows VPS One-Time Setup Script
# Run this in PowerShell as Administrator on a fresh Windows Server 2022 VPS.
# =============================================================================
#Requires -RunAsAdministrator

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$REPO_URL   = "https://github.com/Kayrano/katraswing.git"
$INSTALL_DIR = "C:\katraswing"
$PYTHON_VER  = "3.12.9"
$NSSM_URL    = "https://nssm.cc/release/nssm-2.24.zip"

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-OK($msg)   { Write-Host "    OK: $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "    WARN: $msg" -ForegroundColor Yellow }

# ── 1. Install Python 3.12 ────────────────────────────────────────────────────
Write-Step "Installing Python $PYTHON_VER"
$pyInstaller = "$env:TEMP\python_installer.exe"
$pyUrl = "https://www.python.org/ftp/python/$PYTHON_VER/python-$PYTHON_VER-amd64.exe"
Invoke-WebRequest -Uri $pyUrl -OutFile $pyInstaller -UseBasicParsing
Start-Process -FilePath $pyInstaller -ArgumentList `
    "/quiet InstallAllUsers=1 PrependPath=1 Include_pip=1" `
    -Wait -NoNewWindow
Remove-Item $pyInstaller
$env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + `
            [System.Environment]::GetEnvironmentVariable("Path", "User")
Write-OK "Python installed: $(python --version)"

# ── 2. Install Git ────────────────────────────────────────────────────────────
Write-Step "Installing Git"
$gitInstaller = "$env:TEMP\git_installer.exe"
$gitUrl = "https://github.com/git-for-windows/git/releases/download/v2.44.0.windows.1/Git-2.44.0-64-bit.exe"
Invoke-WebRequest -Uri $gitUrl -OutFile $gitInstaller -UseBasicParsing
Start-Process -FilePath $gitInstaller -ArgumentList "/SILENT /NORESTART" -Wait -NoNewWindow
Remove-Item $gitInstaller
$env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + `
            [System.Environment]::GetEnvironmentVariable("Path", "User")
Write-OK "Git installed"

# ── 3. Clone the repository ───────────────────────────────────────────────────
Write-Step "Cloning repository to $INSTALL_DIR"
if (Test-Path $INSTALL_DIR) {
    Write-Warn "$INSTALL_DIR already exists — pulling latest instead"
    Set-Location $INSTALL_DIR
    git pull
} else {
    git clone $REPO_URL $INSTALL_DIR
}
Set-Location $INSTALL_DIR
Write-OK "Repository ready at $INSTALL_DIR"

# ── 4. Install Python dependencies ───────────────────────────────────────────
Write-Step "Installing Python packages"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Write-OK "requirements.txt installed"

# Install MetaTrader5 (Windows-only, not in requirements.txt)
Write-Step "Installing MetaTrader5 Python package"
python -m pip install MetaTrader5
Write-OK "MetaTrader5 package installed"

# ── 5. Create data directories ────────────────────────────────────────────────
Write-Step "Creating data directories"
New-Item -ItemType Directory -Force -Path "$INSTALL_DIR\data\reports" | Out-Null
New-Item -ItemType Directory -Force -Path "$INSTALL_DIR\logs"          | Out-Null
Write-OK "Directories created"

# ── 6. Create secrets template ───────────────────────────────────────────────
Write-Step "Creating .streamlit/secrets.toml template"
$secretsDir  = "$INSTALL_DIR\.streamlit"
$secretsFile = "$secretsDir\secrets.toml"
if (-not (Test-Path $secretsFile)) {
    New-Item -ItemType Directory -Force -Path $secretsDir | Out-Null
    @"
# ──────────────────────────────────────────────────────────────────────────────
# Fill in your real values then save. This file is NOT committed to git.
# ──────────────────────────────────────────────────────────────────────────────

# Finnhub free key — get one at https://finnhub.io/register
FINNHUB_KEY = "your_finnhub_key_here"
"@ | Out-File -FilePath $secretsFile -Encoding utf8
    Write-Warn "Created secrets template at $secretsFile — EDIT THIS FILE before starting"
} else {
    Write-OK "Secrets file already exists"
}

# ── 7. Download NSSM (service manager) ───────────────────────────────────────
Write-Step "Downloading NSSM (Windows service manager)"
$nssmZip = "$env:TEMP\nssm.zip"
$nssmDir = "C:\nssm"
Invoke-WebRequest -Uri $NSSM_URL -OutFile $nssmZip -UseBasicParsing
Expand-Archive -Path $nssmZip -DestinationPath $env:TEMP\nssm_extract -Force
Copy-Item "$env:TEMP\nssm_extract\nssm-2.24\win64\nssm.exe" -Destination "C:\Windows\System32\nssm.exe" -Force
Remove-Item $nssmZip
Write-OK "NSSM installed to C:\Windows\System32\nssm.exe"

# ── 8. Open firewall port 8501 for Streamlit ─────────────────────────────────
Write-Step "Opening firewall port 8501 (Streamlit)"
$ruleName = "Katraswing-Streamlit"
if (-not (Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule -DisplayName $ruleName `
        -Direction Inbound -Protocol TCP -LocalPort 8501 -Action Allow | Out-Null
    Write-OK "Firewall rule created for port 8501"
} else {
    Write-OK "Firewall rule already exists"
}

# ── Done ──────────────────────────────────────────────────────────────────────
Write-Host @"

=============================================================================
  Setup complete!

  NEXT STEPS:
  1. Install MetaTrader 5 terminal:
     Download from https://www.metatrader5.com/en/download
     Log in with your broker credentials.
     Enable: Tools → Options → Expert Advisors → Allow DLL imports

  2. Edit the secrets file:
     notepad $secretsFile

  3. Install and start the Windows services:
     powershell -ExecutionPolicy Bypass $INSTALL_DIR\deploy\install_services.ps1

  4. Access the dashboard:
     http://<your-vps-ip>:8501
=============================================================================
"@ -ForegroundColor Green
