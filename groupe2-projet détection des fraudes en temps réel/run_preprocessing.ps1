# Run preprocessing with a robust Python selection.
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$pythonCandidates = @(
    $env:FRAUD_RT_PYTHON,
    "C:\venvs\fraud-rt\Scripts\python.exe",
    (Join-Path $projectRoot ".venv\Scripts\python.exe")
) | Where-Object { $_ -and (Test-Path $_) }

if ($pythonCandidates.Count -eq 0) {
    throw "No Python environment found. Run run_all.ps1 once to bootstrap the environment."
}

$py = $pythonCandidates[0]
Write-Host "Using Python: $py"
& $py run_preprocessing.py