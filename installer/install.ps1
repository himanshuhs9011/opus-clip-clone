# ClipForge component installer (Windows 10/11).
# The setup wizard (ClipForge.iss) copies the app files and then runs this with -Wizard;
# it downloads everything else: Python, packages, GPU libraries, FFmpeg, Deno, Ollama + model.
#
#   powershell -ExecutionPolicy Bypass -File install.ps1 -AppDir <dir> [-Wizard]
#              [-NoOllama] [-NoModel] [-OpenRouterKey sk-or-...] [-NoGpu]
param(
    [string]$AppDir = "$env:LOCALAPPDATA\Programs\ClipForge",
    [switch]$Wizard,          # run from the setup wizard: no questions, files already copied
    [switch]$NoOllama,        # do not install Ollama
    [switch]$NoModel,         # do not download the Ollama model
    [string]$OpenRouterKey = "",
    [switch]$NoGpu
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is 10x faster without the bar
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$Bin = Join-Path $AppDir "bin"
$Version = "2.0.0"
$Model = "qwen2.5:7b"
$Warnings = @()
New-Item -ItemType Directory -Force -Path $AppDir, $Bin | Out-Null
try { Start-Transcript -Path "$AppDir\install.log" -Force | Out-Null } catch {}
$Host.UI.RawUI.WindowTitle = "ClipForge setup - downloading components"

function Step($n, $text) { Write-Host ""; Write-Host "[$n/8] $text" -ForegroundColor Cyan }
function Ok($text) { Write-Host "   OK  $text" -ForegroundColor Green }
function Note($text) { Write-Host "   ..  $text" -ForegroundColor Gray }
function Warn($text) { Write-Host "   !!  $text" -ForegroundColor Yellow; $script:Warnings += $text }
function Download($url, $dest) {
    Note "downloading $(Split-Path -Leaf $dest) ..."
    for ($i = 1; $i -le 3; $i++) {
        try { Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing; return }
        catch { if ($i -eq 3) { throw } ; Start-Sleep -Seconds (3 * $i) }
    }
}
function Has($cmd) { return [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }
# Windows PowerShell 5.1 turns any stderr text from a program into a fatal error under "Stop";
# programs like ollama/uv log to stderr, so run them with "Continue" and check the exit code.
function Native([scriptblock]$block) {
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & $block } finally { $ErrorActionPreference = $old }
}
function FirstFile($dir, $name) { (Get-ChildItem $dir -Recurse -Filter $name | Select-Object -First 1).FullName }

Write-Host ""
Write-Host "  ClipForge $Version - long videos in, viral shorts out" -ForegroundColor White
Write-Host "  Setting up components in $AppDir"
Write-Host "  This window closes by itself when everything is ready."
Write-Host "  ------------------------------------------------------"

try {
    # -------------------------------------------------------------- 1. Python
    Step 1 "Python runtime"
    $uv = "$Bin\uv.exe"
    if (-not (Test-Path $uv)) {
        if (Has "uv") { $uv = (Get-Command uv).Source }
        else {
            $zip = "$env:TEMP\cf_uv.zip"
            Download "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip" $zip
            Expand-Archive $zip -DestinationPath "$env:TEMP\cf_uv" -Force
            Copy-Item (FirstFile "$env:TEMP\cf_uv" "uv.exe") "$Bin\uv.exe" -Force
            $uv = "$Bin\uv.exe"
        }
    }
    $Py = "$AppDir\.venv\Scripts\python.exe"
    if (-not (Test-Path $Py)) {
        Native { & $uv venv "$AppDir\.venv" --python 3.12 --quiet 2>&1 | Out-Host }
        if (-not (Test-Path $Py)) { throw "could not create the Python environment" }
    }
    Ok "Python 3.12"

    # -------------------------------------------------------------- 2. packages
    Step 2 "App components (Whisper, face + speaker detection, app window)"
    Native { & $uv pip install --python $Py -r "$AppDir\requirements.txt" --quiet 2>&1 | Out-Host }
    if ($LASTEXITCODE -ne 0) { throw "installing the app components failed (check your internet connection)" }
    Ok "components installed"

    # -------------------------------------------------------------- 3. NVIDIA GPU
    Step 3 "Graphics card"
    $gpu = $false
    if (-not $NoGpu -and (Has "nvidia-smi")) {
        $name = Native { & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1 }
        if ($name) {
            Note "NVIDIA $name found - adding CUDA libraries for fast transcription (~700 MB)"
            Native { & $uv pip install --python $Py "nvidia-cublas-cu12" "nvidia-cudnn-cu12>=9,<10" --quiet 2>&1 | Out-Host }
            if ($LASTEXITCODE -eq 0) { $gpu = $true; Ok "GPU transcription enabled" }
            else { Warn "CUDA libraries could not be installed - transcription will use the CPU" }
        }
    }
    if (-not $gpu) { Note "no NVIDIA GPU - transcription runs on the CPU (slower, still works)" }

    # -------------------------------------------------------------- 4. FFmpeg
    Step 4 "FFmpeg (video engine)"
    if (Test-Path "$Bin\ffmpeg.exe") { Ok "FFmpeg ready" }
    else {
        $zip = "$env:TEMP\cf_ffmpeg.zip"
        Download "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip" $zip
        Expand-Archive $zip -DestinationPath "$env:TEMP\cf_ffmpeg" -Force
        foreach ($exe in "ffmpeg.exe", "ffprobe.exe") { Copy-Item (FirstFile "$env:TEMP\cf_ffmpeg" $exe) "$Bin\$exe" -Force }
        Remove-Item "$env:TEMP\cf_ffmpeg" -Recurse -Force -ErrorAction SilentlyContinue
        Ok "FFmpeg installed (GPU encoding included)"
    }

    # -------------------------------------------------------------- 5. YouTube support
    Step 5 "YouTube link support"
    if ((Has "deno") -or (Test-Path "$Bin\deno.exe")) { Ok "ready" }
    else {
        $zip = "$env:TEMP\cf_deno.zip"
        Download "https://github.com/denoland/deno/releases/latest/download/deno-x86_64-pc-windows-msvc.zip" $zip
        Expand-Archive $zip -DestinationPath $Bin -Force
        Ok "Deno installed (yt-dlp needs it for YouTube)"
    }

    # -------------------------------------------------------------- 6. Ollama
    Step 6 "Ollama - free local AI that picks the clips"
    $ollama = $null
    $known = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
    if (Has "ollama") { $ollama = (Get-Command ollama).Source } elseif (Test-Path $known) { $ollama = $known }
    if ($ollama) { Ok "Ollama is installed" }
    elseif ($NoOllama) { Note "skipped (you chose not to install it)" }
    else {
        $setup = "$env:TEMP\OllamaSetup.exe"
        Download "https://ollama.com/download/OllamaSetup.exe" $setup
        Note "installing Ollama silently ..."
        Start-Process -FilePath $setup -ArgumentList "/VERYSILENT /NORESTART /SUPPRESSMSGBOXES" -Wait
        if (Test-Path $known) { $ollama = $known; Ok "Ollama installed" }
        else { Warn "Ollama could not be installed - get it later from ollama.com, or use OpenRouter" }
    }
    if ($ollama -and -not $NoModel) {
        # make sure the Ollama service is up before asking it for the model
        if (-not (Native { & $ollama list 2>$null | Out-String })) {
            Start-Process -FilePath $ollama -ArgumentList "serve" -WindowStyle Hidden
            Start-Sleep -Seconds 5
        }
        $have = Native { & $ollama list 2>$null | Out-String }
        if ($have -match [regex]::Escape($Model)) { Ok "AI model $Model ready" }
        else {
            Note "downloading AI model $Model (~4.7 GB, one time) ..."
            Native { & $ollama pull $Model }
            if ($LASTEXITCODE -eq 0) { Ok "AI model downloaded" }
            else { Warn "AI model download failed - ClipForge retries it when you first use Ollama" }
        }
    }

    # -------------------------------------------------------------- 7. OpenRouter
    Step 7 "OpenRouter (optional cloud AI)"
    $orKey = $OpenRouterKey.Trim()
    if (-not $orKey -and (Test-Path "$AppDir\.env")) {
        $m = Select-String -Path "$AppDir\.env" -Pattern "^OPENROUTER_API_KEY=(.+)$" | Select-Object -First 1
        if ($m) { $orKey = $m.Matches[0].Groups[1].Value.Trim() }
    }
    if ($orKey) {
        try {
            Invoke-RestMethod -Uri "https://openrouter.ai/api/v1/key" -Headers @{ Authorization = "Bearer $orKey" } | Out-Null
            Ok "OpenRouter key works"
        } catch { Warn "OpenRouter did not accept the key - change it in ClipForge > Settings" }
    } else { Note "no key - add one any time in ClipForge > Settings" }

    # -------------------------------------------------------------- 8. settings
    Step 8 "Settings"
    if (Test-Path "$AppDir\.env") { Ok "kept your existing settings" }
    else {
        $provider = "none"
        if ($ollama) { $provider = "ollama" } elseif ($orKey) { $provider = "openrouter" }
        $whisper = "small"
        if ($gpu) { $whisper = "large-v3-turbo" }
        $lines = @("LLM_PROVIDER=$provider", "OLLAMA_MODEL=$Model", "OPENROUTER_MODEL=google/gemini-2.5-flash",
                   "WHISPER_MODEL=$whisper", "DOWNLOAD_MODE=smart")
        if ($orKey) { $lines += "OPENROUTER_API_KEY=$orKey" }
        Set-Content -Path "$AppDir\.env" -Value $lines -Encoding UTF8
        Ok "AI: $provider - transcription: Whisper $whisper"
    }
    # warm-up: verify the app imports (catches broken installs now, not on first launch)
    $env:CLIPFORGE_DATA = "$env:USERPROFILE\Videos\ClipForge"
    $env:PATH = "$Bin;$AppDir\.venv\Scripts;$env:PATH"
    Push-Location $AppDir
    Native { & $Py -c "import app" 2>&1 | Out-Null }
    Pop-Location
    if ($LASTEXITCODE -ne 0) { throw "ClipForge did not start in a self-test - see install.log" }
    Ok "self-test passed"
}
catch {
    Write-Host ""
    Write-Host "  Setup could not finish: $_" -ForegroundColor Red
    Write-Host "  Log: $AppDir\install.log"
    try { Stop-Transcript | Out-Null } catch {}
    if (-not $Wizard) { Read-Host "Press Enter to close" | Out-Null }
    exit 1
}

Write-Host ""
Write-Host "  All components are ready." -ForegroundColor Green
foreach ($w in $Warnings) { Write-Host "  note: $w" -ForegroundColor Yellow }
try { Stop-Transcript | Out-Null } catch {}
if ($Warnings.Count -gt 0) { exit 2 }   # finished, with something worth telling the user
exit 0
