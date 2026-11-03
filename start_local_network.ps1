# Start intuB-ML on the local network (same Wi-Fi or LAN), so colleagues can try it.
#   powershell -ExecutionPolicy Bypass -File .\start_local_network.ps1
# The first time it asks for the invite code and creates .env.lan (ignored by git: it holds the secrets).
# To stop the app: Ctrl+C in this window.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$port = 8000
$envFile = ".env.lan"

if (-not (Test-Path $envFile)) {
    $secret = -join ((1..48) | ForEach-Object { "{0:x}" -f (Get-Random -Maximum 16) })
    $invite = Read-Host "Choose the invite code that colleagues will use to sign up"
    "SECRET_KEY=$secret`nINVITE_CODE=$invite" | Set-Content -Encoding ascii $envFile
    Write-Host "Created $envFile (never share or commit it)."
}
Get-Content $envFile | ForEach-Object {
    $k, $v = $_ -split "=", 2
    if ($k) { Set-Item -Path "env:$k" -Value $v }
}

$env:FLASK_APP = "wsgi.py"
& venv\Scripts\flask db upgrade
if ($LASTEXITCODE -ne 0) { throw "Database migration failed." }

$ips = Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" -and $_.PrefixOrigin -ne "WellKnown" } |
    Select-Object -ExpandProperty IPAddress
Write-Host ""
Write-Host "intuB-ML is listening on port $port. Colleagues on the same network open:"
$ips | ForEach-Object { Write-Host "   http://$($_):$port" }
Write-Host "Invite code: $env:INVITE_CODE"
Write-Host "If Windows asks for network access, allow PRIVATE networks only."
Write-Host "Only cases without identifying data (no names, dates of birth or record numbers)."
Write-Host ""

& venv\Scripts\waitress-serve --listen="0.0.0.0:$port" wsgi:app
