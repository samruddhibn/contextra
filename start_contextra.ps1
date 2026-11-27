# Starts the webcam monitor and the dashboard in two windows.
# Usage (PowerShell, from the project folder):  .\start_contextra.ps1
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = Join-Path $here ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$here'; & '$py' contextra_monitor.py"
Start-Sleep -Seconds 2
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$here'; & '$py' -m streamlit run app.py"
