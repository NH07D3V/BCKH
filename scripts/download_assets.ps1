$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:PYTHONIOENCODING = 'utf-8'
python -m pip install -r environment\requirements-download.txt
python code\download_assets.py

