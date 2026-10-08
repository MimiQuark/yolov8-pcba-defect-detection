$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $Root
& '.venv\Scripts\python.exe' -m pcba_defect train --config configs/smoke_cpu.yaml
