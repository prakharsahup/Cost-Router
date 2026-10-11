# PowerShell Wrapper for Apache JMeter Performance Testing
Param(
    [string]$PythonExe = "python"
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RunnerScript = Join-Path $ScriptDir "run_tests.py"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "       Executing JMeter Performance Testing Suite         " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

& $PythonExe $RunnerScript
exit $LASTEXITCODE
