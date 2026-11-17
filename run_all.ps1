# run_all.ps1 - starts the whole experiment (see run_all.py for what it does).
# Safe to stop (Ctrl+C) and start again: finished runs are skipped, a training
# run that was interrupted continues from its last save.
#
#   .\run_all.ps1                  everything, in order
#   .\run_all.ps1 --tiny           10-minute test of the whole pipeline with tiny models
#   .\run_all.ps1 --list           the stages and what is done
#   .\run_all.ps1 --stages gold    only one stage (tables pilot main gold context loo analyze report)
#   .\run_all.ps1 --report         rebuild paper\results.md
#
# If Windows refuses to run scripts:
#   powershell -ExecutionPolicy Bypass -File .\run_all.ps1

Set-Location -Path $PSScriptRoot
$venv = Join-Path $PSScriptRoot ".venv-c4\Scripts\Activate.ps1"
if (Test-Path $venv) { . $venv }

python -c "import matplotlib" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "installing matplotlib (for the charts) ..."
    python -m pip install --quiet matplotlib
}

python run_all.py @args
exit $LASTEXITCODE
