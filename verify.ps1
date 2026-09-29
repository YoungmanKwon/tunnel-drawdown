# Verification chain — reproduces the numbers in Appendix B of the manuscript
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
& .\.venv\Scripts\Activate.ps1
Write-Host "== [1] OGS regression suite (Mandel 2D, ~2 min) =="
Push-Location benchmarks\ogs\HydroMechanics\Verification
New-Item -ItemType Directory -Force -Path ..\..\..\..\proj\_verif_out | Out-Null
ogs -o ..\..\..\..\proj\_verif_out -r . -l error hm2_1Dbiot.prj; if ($LASTEXITCODE -ne 0) { throw "hm2_1Dbiot regression FAILED" } else { Write-Host "hm2_1Dbiot PASS" }
Pop-Location
Write-Host "== [2] Terzaghi + convergence study (~30 s) =="
python proj\terzaghi\convergence.py
Write-Host "== [3] Einstein-Schwartz lined tunnel, reference case (~15 s) =="
python proj\es_verif\es_verif.py A_itasca A_external
Write-Host "== [4] Exact solid-ring solution vs FE (~30 s) =="
python paper\theory\exact_ring.py
Write-Host ""
Write-Host "Verification done.  Full drawdown runs:  python proj\drawdown_3b\drawdown_3b.py  (7 min)"
Write-Host "                                        python proj\drawdown_k0\drawdown_k0.py  (5 min)"
Write-Host "                                        python proj\drawdown_3d\drawdown_3d.py  (4 min)"
