# =============================================================================
# Katraswing — Pull latest code and restart services
# Run as Administrator whenever you push new changes from your local machine.
# =============================================================================
#Requires -RunAsAdministrator

$INSTALL_DIR = "C:\katraswing"

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-OK($msg)   { Write-Host "    OK: $msg" -ForegroundColor Green }

Set-Location $INSTALL_DIR

Write-Step "Stopping services"
Stop-Service katraswing-signals   -ErrorAction SilentlyContinue
Stop-Service katraswing-streamlit -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
Write-OK "Services stopped"

Write-Step "Pulling latest code"
git pull
Write-OK "Code updated"

Write-Step "Updating Python packages"
python -m pip install -r requirements.txt -q
Write-OK "Packages updated"

Write-Step "Starting services"
Start-Service katraswing-signals
Start-Sleep -Seconds 2
Start-Service katraswing-streamlit
Start-Sleep -Seconds 2

$sig = Get-Service katraswing-signals
$stl = Get-Service katraswing-streamlit
Write-Host ""
Write-Host "  katraswing-signals   : $($sig.Status)" -ForegroundColor $(if ($sig.Status -eq 'Running') {'Green'} else {'Red'})
Write-Host "  katraswing-streamlit : $($stl.Status)" -ForegroundColor $(if ($stl.Status -eq 'Running') {'Green'} else {'Red'})
Write-OK "Update complete"
