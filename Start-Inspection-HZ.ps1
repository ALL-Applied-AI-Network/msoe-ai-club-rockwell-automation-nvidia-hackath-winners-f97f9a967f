param(
    [int]$Port = 8000,
    [switch]$Install,
    [switch]$NoBrowser
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if ($Port -lt 1024 -or $Port -gt 65535) { throw 'Choose a port from 1024 through 65535.' }
if (-not (Get-Command python -ErrorAction SilentlyContinue)) { throw 'Python 3.10+ is required.' }
$webEnvironment = Join-Path $PSScriptRoot 'outputs\hz-web-env'
$webPython = Join-Path $webEnvironment 'Scripts\python.exe'
if ($Install) {
    if (-not (Test-Path -LiteralPath $webPython)) {
        python -m venv --system-site-packages $webEnvironment
        if ($LASTEXITCODE -ne 0) { throw 'Cannot create the local web environment.' }
    }
    & $webPython -m pip install -r inspection_app/requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
}
if (-not (Test-Path -LiteralPath $webPython)) { throw 'Run .\Start-Inspection-HZ.ps1 -Install once to prepare the web environment.' }
& $webPython -c "import fastapi,starlette,uvicorn,multipart,httpx,cv2,PIL,torch,torchvision; from fastapi import FastAPI; FastAPI()"
if ($LASTEXITCODE -ne 0) { throw 'Dependencies are missing. Run .\Start-Inspection-HZ.ps1 -Install once.' }
$inspectionUrl = "http://127.0.0.1:$Port/"
Write-Host "Vial Inspection HZ: $inspectionUrl"
Write-Host 'Press Ctrl+C to stop. Cosmos uses bounded SSH requests to the DGX; connect CSI VPN when needed.'
$browserJob = $null
if (-not $NoBrowser) {
    $browserJob = Start-Job -ArgumentList $inspectionUrl -ScriptBlock {
        param($url)
        for ($attempt = 0; $attempt -lt 60; $attempt++) {
            try {
                $response = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 1
                if ($response.StatusCode -eq 200) { Start-Process $url; return }
            } catch { }
            Start-Sleep -Milliseconds 500
        }
    }
}
try {
    & $webPython -m uvicorn inspection_app.server:app --host 127.0.0.1 --port $Port
    if ($LASTEXITCODE -ne 0) { throw 'The inspection server stopped with an error. If the port is occupied, choose -Port 8001.' }
} finally {
    if ($null -ne $browserJob) {
        Stop-Job -Job $browserJob -ErrorAction SilentlyContinue
        Remove-Job -Job $browserJob -Force -ErrorAction SilentlyContinue
    }
}
