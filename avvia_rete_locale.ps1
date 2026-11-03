# Avvia intuB-ML sulla rete locale (stessa Wi-Fi o LAN), per farla provare ai colleghi.
#   powershell -ExecutionPolicy Bypass -File .\avvia_rete_locale.ps1
# La prima volta chiede il codice invito e crea .env.lan (ignorato da git: contiene i segreti).
# Per fermare l'app: Ctrl+C in questa finestra.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$port = 8000
$envFile = ".env.lan"

if (-not (Test-Path $envFile)) {
    $secret = -join ((1..48) | ForEach-Object { "{0:x}" -f (Get-Random -Maximum 16) })
    $invite = Read-Host "Scegli il codice invito da dare ai colleghi per registrarsi"
    "SECRET_KEY=$secret`nINVITE_CODE=$invite" | Set-Content -Encoding ascii $envFile
    Write-Host "Creato $envFile (non va mai condiviso ne' committato)."
}
Get-Content $envFile | ForEach-Object {
    $k, $v = $_ -split "=", 2
    if ($k) { Set-Item -Path "env:$k" -Value $v }
}

$env:FLASK_APP = "wsgi.py"
& venv\Scripts\flask db upgrade
if ($LASTEXITCODE -ne 0) { throw "Migrazione del database fallita." }

$ips = Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" -and $_.PrefixOrigin -ne "WellKnown" } |
    Select-Object -ExpandProperty IPAddress
Write-Host ""
Write-Host "intuB-ML in ascolto sulla porta $port. I colleghi sulla stessa rete aprono:"
$ips | ForEach-Object { Write-Host "   http://$($_):$port" }
Write-Host "Codice invito: $env:INVITE_CODE"
Write-Host "Se Windows chiede l'accesso alla rete, consenti solo le reti PRIVATE."
Write-Host "Solo casi senza dati identificativi (niente nomi, date di nascita, numeri di cartella)."
Write-Host ""

& venv\Scripts\waitress-serve --listen="0.0.0.0:$port" wsgi:app
