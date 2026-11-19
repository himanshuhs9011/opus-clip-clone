# Builds dist\ClipForge-Setup.exe: a Next > Next > Finish setup wizard (Inno Setup 6).
#   powershell -ExecutionPolicy Bypass -File installer\build.ps1
# Needs Inno Setup 6 on the build PC only:  winget install JRSoftware.InnoSetup
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Build = Join-Path $env:TEMP "clipforge_build"
$Stage = Join-Path $Build "stage"
$Dist = Join-Path $Root "dist"
Set-Location $Root

$iscc = @("$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
          "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
          "$env:ProgramFiles\Inno Setup 6\ISCC.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $iscc) { throw "Inno Setup 6 not found - install it with:  winget install JRSoftware.InnoSetup" }

Write-Host "1/4 building the React UI" -ForegroundColor Cyan
if (Get-Command npm -ErrorAction SilentlyContinue) {
    Push-Location frontend
    if (-not (Test-Path node_modules)) { npm install --no-audit --no-fund | Out-Null }
    npm run build | Out-Null
    Pop-Location
}
if (-not (Test-Path "frontend\dist\index.html")) { throw "frontend\dist missing - install Node.js and run npm run build in frontend" }

Write-Host "2/4 app icon" -ForegroundColor Cyan
python -c @"
from PIL import Image, ImageDraw
im = Image.new('RGBA', (256, 256), (0, 0, 0, 0))
d = ImageDraw.Draw(im)
d.rounded_rectangle((8, 8, 248, 248), radius=56, fill=(163, 168, 240, 255))
d.polygon([(98, 70), (98, 186), (196, 128)], fill=(20, 22, 40, 255))
im.save(r'$Root\clipforge.ico', sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
"@

Write-Host "3/4 staging files" -ForegroundColor Cyan
if (Test-Path $Build) { Remove-Item $Build -Recurse -Force }
New-Item -ItemType Directory -Force -Path $Stage, $Dist | Out-Null
foreach ($f in "app.py", "run.py", "ClipForge.pyw", "requirements.txt", "README.md", ".env.example", "clipforge.ico") {
    Copy-Item $f $Stage
}
if (Test-Path "THIRD_PARTY.md") { Copy-Item "THIRD_PARTY.md" $Stage }
New-Item -ItemType Directory -Force -Path "$Stage\opusclip", "$Stage\config", "$Stage\models", "$Stage\frontend" | Out-Null
Copy-Item "opusclip\*.py" "$Stage\opusclip"
Copy-Item "config\*" "$Stage\config"
Copy-Item "models\*.onnx" "$Stage\models"
Copy-Item "frontend\dist" "$Stage\frontend\dist" -Recurse

Write-Host "4/4 compiling the setup wizard" -ForegroundColor Cyan
& $iscc /Q "/DStageDir=$Stage" "/DOutDir=$Dist" "$Root\installer\ClipForge.iss"
if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed" }
$out = Join-Path $Dist "ClipForge-Setup.exe"
$mb = [math]::Round((Get-Item $out).Length / 1MB, 1)
Write-Host ""
Write-Host "Built $out ($mb MB)" -ForegroundColor Green
