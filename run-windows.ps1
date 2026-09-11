$ErrorActionPreference = "Stop"

Set-Location -LiteralPath (Split-Path -Parent $MyInvocation.MyCommand.Path)

$python = Join-Path $PWD "venv313\Scripts\python.exe"
if (-not (Test-Path $python)) {
    python -m venv venv313
}

& $python -m pip install -r requirements.txt
& $python main.py @args