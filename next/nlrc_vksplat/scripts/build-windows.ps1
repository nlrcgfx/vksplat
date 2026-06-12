param(
  [ValidateSet(
    "windows-debug",
    "windows-release",
    "windows-debug-emulated-int64",
    "windows-release-emulated-int64"
  )]
  [string]$Preset = "windows-debug",

  [ValidateSet("x64", "arm64")]
  [string]$Arch = "x64",

  [string]$PythonExe = "",

  [string]$Target = "",

  [switch]$RunTests
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Resolve-Path (Join-Path $ScriptDir "..")

$VsWhereCandidates = @(
  "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe",
  "${env:ProgramFiles}\Microsoft Visual Studio\Installer\vswhere.exe"
)

$VsWhere = $VsWhereCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $VsWhere) {
  throw "vswhere.exe not found. Install Visual Studio Build Tools with the C++ workload."
}

$VsInstall = & $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $VsInstall) {
  throw "Visual Studio C++ tools were not found."
}

$VsDevCmd = Join-Path $VsInstall "Common7\Tools\VsDevCmd.bat"
if (-not (Test-Path -LiteralPath $VsDevCmd)) {
  throw "VsDevCmd.bat not found under $VsInstall."
}

function Test-PythonExe {
  param([string]$Path)

  if (-not (Test-Path -LiteralPath $Path)) {
    return $false
  }

  try {
    & $Path --version *> $null
    return $LASTEXITCODE -eq 0
  } catch {
    return $false
  }
}

function Find-PythonExe {
  $PythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
  if ($PythonCommand -and (Test-PythonExe $PythonCommand.Source)) {
    return $PythonCommand.Source
  }

  $PyLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
  if ($PyLauncher) {
    $PythonFromLauncher = & $PyLauncher.Source -3 -c "import sys; print(sys.executable)"
    if ($LASTEXITCODE -eq 0 -and $PythonFromLauncher) {
      return $PythonFromLauncher.Trim()
    }
  }

  $VersionDirs = @("Python313", "Python312", "Python311", "Python310", "Python39")
  $BaseDirs = @(
    "$env:LOCALAPPDATA\Programs\Python",
    "$env:ProgramFiles",
    "${env:ProgramFiles(x86)}"
  )

  foreach ($BaseDir in $BaseDirs) {
    foreach ($VersionDir in $VersionDirs) {
      $Candidate = Join-Path (Join-Path $BaseDir $VersionDir) "python.exe"
      if (Test-PythonExe $Candidate) {
        return $Candidate
      }
    }
  }

  $Candidates = @(
    "$env:LOCALAPPDATA\Programs\Python\Python*\python.exe",
    "$env:ProgramFiles\Python*\python.exe",
    "${env:ProgramFiles(x86)}\Python*\python.exe"
  )

  foreach ($Pattern in $Candidates) {
    $Match = Get-ChildItem -Path $Pattern -ErrorAction SilentlyContinue |
      Sort-Object -Property FullName -Descending |
      Select-Object -First 1
    if ($Match -and (Test-PythonExe $Match.FullName)) {
      return $Match.FullName
    }
  }

  throw "Python 3 not found. Install Python or add python.exe to PATH."
}

if ($PythonExe) {
  if (-not (Test-PythonExe $PythonExe)) {
    throw "Python executable cannot be run: $PythonExe"
  }
} else {
  $PythonExe = Find-PythonExe
}

$Commands = @(
  "call `"$VsDevCmd`" -arch=$Arch -host_arch=x64",
  "cmake --preset $Preset -DPython_EXECUTABLE=`"$PythonExe`""
)

if ($Target) {
  $Commands += "cmake --build --preset $Preset --target $Target"
} else {
  $Commands += "cmake --build --preset $Preset"
}

if ($RunTests) {
  $Commands += "ctest --preset $Preset --output-on-failure"
}

Push-Location $ProjectDir
try {
  cmd.exe /d /s /c ($Commands -join " && ")
  if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
  }
} finally {
  Pop-Location
}
