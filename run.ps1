$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host "Se creó .env a partir de .env.example. Completá OPENAI_API_KEY y volvé a ejecutar .\run.ps1"
    exit 1
}
if (-not (Select-String -Path .env -Pattern '^OPENAI_API_KEY=.+' -Quiet)) {
    Write-Host "Falta OPENAI_API_KEY en .env (los embeddings del RAG la necesitan)"
    exit 1
}

docker compose up --build -d
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host -NoNewline "Esperando a la API y al worker"
for ($i = 0; $i -lt 90; $i++) {
    try {
        $salud = Invoke-RestMethod http://localhost:8000/health -TimeoutSec 3
        if ($salud.redis -and $salud.worker -ne "sin worker activo") {
            Write-Host ""
            $salud | Format-List
            Write-Host "API:     http://localhost:8000/docs"
            Write-Host "Phoenix: http://localhost:6006"
            Write-Host "Pruebas: python scripts/pruebas_sistema.py"
            exit 0
        }
    } catch {}
    Write-Host -NoNewline "."
    Start-Sleep -Seconds 2
}
Write-Host ""
Write-Host "El sistema no respondió a tiempo. Revisá: docker compose logs worker api"
exit 1
