param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ModeArgs
)

$ErrorActionPreference = "Stop"

Set-Location -LiteralPath (Split-Path -Parent $MyInvocation.MyCommand.Path)

$venvDir = "venv313"
$python = Join-Path $PWD "$venvDir\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    Write-Host "Creando el entorno virtual $venvDir ..."
    python -m venv $venvDir
}

Write-Host "Instalando los paquetes necesarios ..."
& $python -m pip install --upgrade pip
& $python -m pip install -e ".[dev]"

Write-Host "Iniciando the-machine ..."
& $python main.py @ModeArgs