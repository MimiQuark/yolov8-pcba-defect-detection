$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = 'C:\Users\27800\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
Set-Location $Root
& $Python 'scripts\build_report.py'
