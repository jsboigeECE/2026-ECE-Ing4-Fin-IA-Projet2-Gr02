Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

param(
    [switch]$SkipInstall
)

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

function Get-BasePythonCommand {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return @("py", "-3.12")
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return @("python")
    }
    throw "Python launcher not found (py/python). Install Python 3.12+."
}

function Test-ImportStatus([string]$pythonExe) {
    $code = @'
import importlib.util
mods = ["pandas", "numpy", "sklearn", "imblearn", "matplotlib"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
print("MISSING=" + ",".join(missing))
'@
    $output = & $pythonExe -c $code 2>&1
    if ($LASTEXITCODE -ne 0) {
        return @("pandas", "numpy", "scikit-learn", "imbalanced-learn", "matplotlib")
    }
    $line = ($output | Select-Object -Last 1)
    if ($line -like "MISSING=*") {
        $raw = $line.Substring(8)
        if ([string]::IsNullOrWhiteSpace($raw)) {
            return @()
        }
        $mapped = @()
        foreach ($m in ($raw -split ",")) {
            switch ($m.Trim()) {
                "sklearn" { $mapped += "scikit-learn" }
                "imblearn" { $mapped += "imbalanced-learn" }
                default { $mapped += $m.Trim() }
            }
        }
        return $mapped
    }
    return @()
}

function Ensure-PythonEnvironment {
    $localVenv = Join-Path $projectRoot ".venv\Scripts\python.exe"
    $shortVenv = "C:\venvs\fraud-rt\Scripts\python.exe"

    if ($env:FRAUD_RT_PYTHON -and (Test-Path $env:FRAUD_RT_PYTHON)) {
        Write-Host "Using FRAUD_RT_PYTHON: $($env:FRAUD_RT_PYTHON)"
        return $env:FRAUD_RT_PYTHON
    }

    if (Test-Path $shortVenv) {
        Write-Host "Found short-path venv: $shortVenv"
        return $shortVenv
    }
    if (Test-Path $localVenv) {
        Write-Host "Found local venv: $localVenv"
        return $localVenv
    }

    Write-Host "No venv detected. Creating one in C:\\venvs\\fraud-rt ..."
    New-Item -ItemType Directory -Force -Path "C:\venvs\fraud-rt" | Out-Null

    $baseCmd = Get-BasePythonCommand
    if ($baseCmd.Count -eq 2) {
        & $baseCmd[0] $baseCmd[1] -m venv "C:\venvs\fraud-rt"
    } else {
        & $baseCmd[0] -m venv "C:\venvs\fraud-rt"
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create venv at C:\venvs\fraud-rt"
    }

    if (-not (Test-Path $shortVenv)) {
        throw "Python executable not found after venv creation: $shortVenv"
    }
    return $shortVenv
}

function Install-Requirements([string]$pythonExe) {
    if ($SkipInstall) {
        Write-Host "Skipping dependency installation (-SkipInstall)."
        return
    }

    Write-Host "Installing dependencies from requirements.txt ..."
    & $pythonExe -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to upgrade pip"
    }

    & $pythonExe -m pip install -r (Join-Path $projectRoot "requirements.txt")
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install dependencies with $pythonExe"
    }
}

function Run-Script([string]$pythonExe, [string]$scriptPath) {
    Write-Host "`n=== Running: $scriptPath ==="
    & $pythonExe $scriptPath
    return $LASTEXITCODE
}

$py = Ensure-PythonEnvironment
$env:FRAUD_RT_PYTHON = $py

$missingBefore = Test-ImportStatus -pythonExe $py
if ($missingBefore.Count -gt 0 -and -not $SkipInstall) {
    Write-Host "Missing packages detected: $($missingBefore -join ', ')"
}

try {
    Install-Requirements -pythonExe $py
} catch {
    $shortVenv = "C:\venvs\fraud-rt\Scripts\python.exe"
    if ($py -ne $shortVenv) {
        Write-Host "Primary environment failed. Falling back to short-path venv: $shortVenv"
        New-Item -ItemType Directory -Force -Path "C:\venvs\fraud-rt" | Out-Null
        if (-not (Test-Path $shortVenv)) {
            $baseCmd = Get-BasePythonCommand
            if ($baseCmd.Count -eq 2) {
                & $baseCmd[0] $baseCmd[1] -m venv "C:\venvs\fraud-rt"
            } else {
                & $baseCmd[0] -m venv "C:\venvs\fraud-rt"
            }
            if ($LASTEXITCODE -ne 0) {
                throw "Fallback venv creation failed"
            }
        }
        $py = $shortVenv
        $env:FRAUD_RT_PYTHON = $py
        Install-Requirements -pythonExe $py
    } else {
        throw
    }
}

$missingAfter = Test-ImportStatus -pythonExe $py
if ($missingAfter.Count -gt 0) {
    throw "Environment still missing: $($missingAfter -join ', ')"
}

$scripts = @(
    "check_dataset.py",
    "run_data_loader.py",
    "run_isolation_forest.py",
    "run_isolation_forest_evaluation.py",
    "run_isolation_forest_threshold_analysis.py",
    "notebooks/05_autoencoder_detection.py",
    "notebooks/06_method_comparison.py",
    "notebooks/07_class_weights_impact.py",
    "notebooks/08_focal_loss_experiment.py",
    "notebooks/09_gnn_detection.py",
    "notebooks/10_comprehensive_comparison.py",
    "notebooks/11_false_positive_cost_analysis.py"
)

$failed = @()
foreach ($script in $scripts) {
    $exitCode = Run-Script -pythonExe $py -scriptPath $script
    if ($exitCode -ne 0) {
        $failed += [PSCustomObject]@{ Script = $script; ExitCode = $exitCode }
        Write-Host "FAILED: $script (exit=$exitCode)"
    } else {
        Write-Host "OK: $script"
    }
}

Write-Host "`n========================================"
Write-Host "Python used: $py"
if ($failed.Count -eq 0) {
    Write-Host "All scripts executed successfully."
    exit 0
}

Write-Host "Some scripts failed:"
$failed | Format-Table -AutoSize | Out-String | Write-Host
exit 1
