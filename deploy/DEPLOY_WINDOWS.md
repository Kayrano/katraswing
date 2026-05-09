# Katraswing — Windows VPS Deployment Guide

Full 24/7 deployment: MT5 terminal + signal server + Streamlit dashboard, all auto-restarting on crash and reboot.

---

## What runs on the VPS

| Process | Description |
|---|---|
| **MetaTrader 5 terminal** | Executes orders, holds open positions |
| **katraswing-signals** (Windows service) | Polls signals every 60 s, sends orders to MT5, runs learning loop hourly |
| **katraswing-streamlit** (Windows service) | Web dashboard on port 8501 |

---

## Step 1 — Get the VPS

1. Go to **https://gratisvps.net/free-vps-windows.html**
2. Select **Windows Server 2022**
3. Pick any region close to your broker's servers (EU for most FX brokers)
4. Register and wait for the email with:
   - VPS IP address
   - Administrator password

---

## Step 2 — Connect via RDP

**On Windows (your local machine):**
1. Press `Win + R` → type `mstsc` → Enter
2. Enter the VPS IP address
3. Username: `Administrator`
4. Password: from the email

**On Mac:**
- Install **Microsoft Remote Desktop** from the App Store

---

## Step 3 — Install MetaTrader 5

Inside the VPS (via RDP):

1. Open Edge browser on the VPS
2. Download MT5: **https://www.metatrader5.com/en/download**
3. Install it (default settings)
4. Launch MT5 and log in with your broker account
5. Go to **Tools → Options → Expert Advisors**:
   - ✅ Allow automated trading
   - ✅ Allow DLL imports
   - ✅ Allow imports from external experts
6. Leave MT5 running — **do not close it**

---

## Step 4 — Run the setup script

Open **PowerShell as Administrator** on the VPS:

```powershell
# Right-click Start → Windows PowerShell (Admin)

# Allow scripts to run
Set-ExecutionPolicy Bypass -Scope Process -Force

# Run the one-time setup (installs Python, Git, clones repo, installs packages)
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/Kayrano/katraswing/main/deploy/setup_windows.ps1" -OutFile "$env:TEMP\setup.ps1" -UseBasicParsing
powershell -ExecutionPolicy Bypass -File "$env:TEMP\setup.ps1"
```

This takes about 3–5 minutes. When it finishes you'll see a green success message.

---

## Step 5 — Edit the secrets file

```powershell
notepad C:\katraswing\.streamlit\secrets.toml
```

Fill in your Finnhub API key (free at https://finnhub.io/register):

```toml
FINNHUB_KEY = "your_key_here"
```

Save and close.

---

## Step 6 — Install and start the services

```powershell
powershell -ExecutionPolicy Bypass -File "C:\katraswing\deploy\install_services.ps1"
```

Both services will install and start. You'll see:
```
  katraswing-signals   : Running
  katraswing-streamlit : Running
```

---

## Step 7 — Keep MT5 running when you disconnect

> Critical: MT5 must stay running even after you close RDP.

When you're done configuring, **disconnect without logging off**:

```powershell
# Run this before closing RDP — it disconnects your session but keeps it alive
%windir%\System32\tscon.exe %SESSIONNAME% /dest:console
```

Or simply close the RDP window (X button) — **don't click Log Off**.

To make MT5 auto-start on every reboot:
1. Right-click the MT5 shortcut on the desktop
2. Copy
3. Press `Win + R` → type `shell:startup` → Enter
4. Paste the shortcut into that folder

---

## Step 8 — Access the dashboard

Open in your browser:

```
http://<your-vps-ip>:8501
```

---

## Ongoing: deploy code updates from your local machine

After pushing changes to GitHub from your local machine, run on the VPS:

```powershell
powershell -ExecutionPolicy Bypass -File "C:\katraswing\deploy\update.ps1"
```

---

## Useful commands

```powershell
# Check service status
Get-Service katraswing-*

# Restart signal server (e.g. after config change)
Restart-Service katraswing-signals

# View live signal server logs
Get-Content C:\katraswing\logs\signals_stdout.log -Wait -Tail 50

# View live Streamlit logs
Get-Content C:\katraswing\logs\streamlit_stdout.log -Wait -Tail 50

# Stop everything
Stop-Service katraswing-signals, katraswing-streamlit

# Emergency: close all open MT5 positions
cd C:\katraswing
python mt5_signal_server.py --close-all
```

---

## What happens automatically (no action needed)

| Cadence | What runs |
|---|---|
| Every 60 s | Signal scan across all tickers, orders sent to MT5 |
| Every hour | Strategy params adapted, pattern stats refreshed, calibration refit |
| Every day | Daily performance report generated in `data/reports/` |
| Every week | Weak strategies pruned, paper-only strategies promoted, ML model retrained |
| On crash | NSSM automatically restarts the crashed service within 3 seconds |
| On reboot | Both services auto-start, MT5 auto-starts from Startup folder |

---

## Troubleshooting

**Signal server not connecting to MT5:**
- Make sure MT5 terminal is open and logged in
- Check Tools → Options → Expert Advisors → "Allow automated trading" is checked
- Restart the service: `Restart-Service katraswing-signals`

**Dashboard not accessible:**
- Check the firewall: `Get-NetFirewallRule -DisplayName "Katraswing-Streamlit"`
- Check the service is running: `Get-Service katraswing-streamlit`
- Try from inside the VPS: `http://localhost:8501`

**Services show Stopped after reboot:**
- Run: `Start-Service katraswing-signals, katraswing-streamlit`
- If it keeps failing, check logs in `C:\katraswing\logs\`
