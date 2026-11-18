$ErrorActionPreference = "Stop"

$RepoUrl = "https://github.com/Arxdevmm/deep-live-cam-arx.git"
$Branch = "main"
$ScriptRoot = $PSScriptRoot
$InstallDir = if ($env:DLC_INSTALL_DIR) {
    $env:DLC_INSTALL_DIR
} else {
    Join-Path $env:LOCALAPPDATA "Deep-Live-Cam-Arx"
}

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Add-UserAndMachinePathsToSession {
    $paths = @(
        [Environment]::GetEnvironmentVariable("Path", "User"),
        [Environment]::GetEnvironmentVariable("Path", "Machine")
    )
    foreach ($path in $paths) {
        if ($path) {
            foreach ($entry in $path.Split(";")) {
                if ($entry -and -not (($env:Path.Split(";")) -contains $entry)) {
                    $env:Path += ";" + $entry
                }
            }
        }
    }
}

function Install-WingetPackage([string]$Id) {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw "winget is required to install $Id. Install App Installer from Microsoft Store, then rerun this script."
    }
    winget install --id $Id --exact --source winget --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        throw "winget failed to install $Id (exit code $LASTEXITCODE)."
    }
    Add-UserAndMachinePathsToSession
}

function Try-InstallOptionalPackage([string]$Id, [string]$Name) {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        Write-Warning "$Name was not installed because winget is unavailable."
        return
    }
    Write-Step "Installing $Name"
    winget install --id $Id --exact --source winget --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "$Name could not be installed automatically. Continue or install it manually."
    }
    Add-UserAndMachinePathsToSession
}

Write-Step "Preparing Deep-Live-Cam"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Step "Installing Git"
    Install-WingetPackage "Git.Git"
}

if (-not (Test-Path (Join-Path $ScriptRoot "run.py"))) {
    if (-not (Test-Path $InstallDir)) {
        New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    }
    if (-not (Test-Path (Join-Path $InstallDir ".git"))) {
        Write-Step "Cloning repository to $InstallDir"
        git clone --depth 1 --branch $Branch $RepoUrl $InstallDir
        if ($LASTEXITCODE -ne 0) {
            throw "Repository clone failed."
        }
    }
    $ProjectDir = $InstallDir
} else {
    $ProjectDir = $ScriptRoot
}

Set-Location $ProjectDir

$Launcher = $null
$LauncherArgs = @()
$PythonVersions = @("3.14", "3.13", "3.12", "3.11")

if (Get-Command py -ErrorAction SilentlyContinue) {
    foreach ($version in $PythonVersions) {
        & py "-$version" -c "import sys; print(sys.executable)" *> $null
        if ($LASTEXITCODE -eq 0) {
            $Launcher = "py"
            $LauncherArgs = @("-$version")
            break
        }
    }
}

if (-not $Launcher) {
    if (Get-Command python -ErrorAction SilentlyContinue) {
        & python -c "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] <= (3, 14) else 1)" *> $null
        if ($LASTEXITCODE -eq 0) {
            $Launcher = "python"
        }
    }
}

if (-not $Launcher) {
    Write-Step "Installing Python 3.14"
    Install-WingetPackage "Python.Python.3.14"
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
        throw "Python 3.14 was installed but the Python launcher is still unavailable. Restart Windows and rerun install.bat."
    }
    $Launcher = "py"
    $LauncherArgs = @("-3.14")
}

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Try-InstallOptionalPackage "Gyan.FFmpeg" "FFmpeg"
}

Try-InstallOptionalPackage "Microsoft.VCRedist.2015+.x64" "Visual C++ 2022 runtime"

Write-Step "Creating virtual environment"
$VenvDir = Join-Path $ProjectDir "venv"
if (-not (Test-Path (Join-Path $VenvDir "Scripts\python.exe"))) {
    & $Launcher @LauncherArgs -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        throw "Virtual environment creation failed."
    }
}

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Virtual environment Python was not created."
}

Write-Step "Updating pip and installing dependencies"
& $VenvPython -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) {
    throw "pip bootstrap failed."
}
& $VenvPython -m pip install -r (Join-Path $ProjectDir "requirements.txt")
if ($LASTEXITCODE -ne 0) {
    throw "Dependency installation failed."
}

Write-Step "Creating launcher"
$LauncherBat = Join-Path $ProjectDir "launch.bat"
@"
@echo off
setlocal
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo Virtual environment is missing. Run install.bat first.
    pause
    exit /b 1
)
"venv\Scripts\python.exe" "run.py" %*
set EXIT_CODE=%ERRORLEVEL%
if not "%EXIT_CODE%"=="0" (
    echo.
    echo Deep-Live-Cam exited with code %EXIT_CODE%.
    pause
)
exit /b %EXIT_CODE%
"@ | Set-Content -Encoding ASCII $LauncherBat

$Desktop = [Environment]::GetFolderPath("Desktop")
$ShortcutPath = Join-Path $Desktop "Deep-Live-Cam-Arx.lnk"
$Shell = New-Object -ComObject WScript.Shell
$Shortcut = $Shell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = $LauncherBat
$Shortcut.WorkingDirectory = $ProjectDir
$Shortcut.IconLocation = "$env:SystemRoot\System32\shell32.dll,13"
$Shortcut.Description = "Launch Deep-Live-Cam Arx"
$Shortcut.Save()

$StartMenu = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"
if (-not (Test-Path $StartMenu)) {
    New-Item -ItemType Directory -Path $StartMenu -Force | Out-Null
}
$StartShortcutPath = Join-Path $StartMenu "Deep-Live-Cam-Arx.lnk"
$StartShortcut = $Shell.CreateShortcut($StartShortcutPath)
$StartShortcut.TargetPath = $LauncherBat
$StartShortcut.WorkingDirectory = $ProjectDir
$StartShortcut.IconLocation = "$env:SystemRoot\System32\shell32.dll,13"
$StartShortcut.Description = "Launch Deep-Live-Cam Arx"
$StartShortcut.Save()

Write-Host ""
Write-Host "Installation complete." -ForegroundColor Green
Write-Host "Launcher: $LauncherBat"
Write-Host "Desktop shortcut: $ShortcutPath"
Write-Host "Project: $ProjectDir"
Write-Host ""
Write-Host "The app will download its required ONNX/InsightFace models on first use." -ForegroundColor Yellow
