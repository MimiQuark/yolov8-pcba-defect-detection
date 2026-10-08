$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $Root
& '.venv\Scripts\python.exe' -m pcba_defect app --host 127.0.0.1 --port 7860
