$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $Root
& '.venv\Scripts\python.exe' -m pcba_defect eval --split test --imgsz 960 --batch 2 --device cpu
