# tunnel-drawdown : Windows setup (PowerShell 5+).  From the bundle root:
#     powershell -ExecutionPolicy Bypass -File .\setup.ps1
# Installs Python 3.12 (winget) if missing, creates .venv, installs OGS/gmsh/pyvista/... from pip wheels.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Find-Python {
    foreach ($cand in @("py -3.14", "py -3.13", "py -3.12", "py -3.11", "py -3", "python3.14", "python3.12", "python")) {
        try {
            $v = & cmd /c "$cand --version 2>&1"
            if ($LASTEXITCODE -eq 0 -and $v -match "Python 3\.(1[1-4])") { return $cand }
        } catch {}
    }
    return $null
}

$py = Find-Python
if (-not $py) {
    Write-Host "Python 3.11+ not found - installing Python 3.12 with winget ..."
    winget install --id Python.Python.3.12 -e --accept-source-agreements --accept-package-agreements
    Write-Host ""
    Write-Host ">>> Python installed.  Close this window, open a NEW PowerShell, and run setup.ps1 again. <<<"
    exit 0
}
Write-Host "== Using: $py  ($(& cmd /c "$py --version 2>&1")) =="

if (-not (Test-Path ".venv")) { & cmd /c "$py -m venv .venv" }
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel
pip install -r requirements.txt

Write-Host "== OpenGeoSys =="; ogs --version
python -c "import ogstools, gmsh, pyvista, sympy; print('ogstools', ogstools.__version__, '| gmsh ok | pyvista', pyvista.__version__, '| sympy', sympy.__version__)"
Write-Host ""
Write-Host "Setup complete."
Write-Host "  Verify the install :  powershell -ExecutionPolicy Bypass -File .\verify.ps1   (~5 min)"
Write-Host "  Every new session  :  .\.venv\Scripts\Activate.ps1"
