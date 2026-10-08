$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $Root
& '.venv\Scripts\python.exe' -m pcba_defect prepare --lighting auto --overwrite
& '.venv\Scripts\python.exe' -m pcba_defect analyze
