$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $Root
$env:NODE_PATH = 'C:\Users\27800\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules'
$env:CHROME_PATH = 'C:\Program Files\Google\Chrome\Application\chrome.exe'
& 'C:\Users\27800\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe' 'scripts\capture_web_demo.cjs' 'reports\assets\web_demo.png'
