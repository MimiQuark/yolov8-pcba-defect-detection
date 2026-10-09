$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $Root
if (-not $env:NODE_PATH) { $env:NODE_PATH = (npm root -g) }
$env:CHROME_PATH = 'C:\Program Files\Google\Chrome\Application\chrome.exe'
& 'node' 'scripts\capture_web_demo.cjs' 'reports\assets\web_demo.png'
